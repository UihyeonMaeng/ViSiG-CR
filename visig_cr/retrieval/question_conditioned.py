"""Question-conditioned contrast retrieval over the existing visual index.

Only case/event rows are read from GraphStore. Stored visual/relation edges,
degrees, path priors and selector relation priorities are never consulted.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from itertools import combinations, product
from typing import TYPE_CHECKING, Any

from ..schemas import Evidence, EvidencePackage, Event, QuestionSpec, SemanticContrastGroup
from ..extraction import OntologyExtractorAgent
from ..graph.relations import has_verified_semantic_support
from ..normality import ACTIVE_CLINICAL_TEMPORAL_STATUSES, scope_safe_normal_findings
from .comparison import pair_non_target_context_score
from .attribute_selector import laterality_predicate_plan, normalize_laterality_predicate_spec

if TYPE_CHECKING:
    import numpy as np
    from .engine import ViSiGCRRetriever


POLICY_VERSION = "question_conditioned_contrast_v2"
IMPLEMENTATION_VERSION = "3.2"
MEMORY_IDENTITY_KEYS = (
    "protocol_id", "memory_size", "case_manifest_sha256", "events_sha256",
    "ontology_sha256", "visual_index_config_sha256", "normality_config_sha256",
)


def uses_question_conditioned(retrieval_cfg: dict[str, Any], method: str | None) -> bool:
    if method != "visig_cr":
        return False
    strategy = retrieval_cfg.get("visig_cr_strategy", "question_conditioned")
    if strategy not in {"question_conditioned", "legacy_relation_graph"}:
        raise ValueError("retrieval.visig_cr_strategy must be question_conditioned or legacy_relation_graph")
    return strategy == "question_conditioned"


def _settings(cfg: dict[str, Any]) -> dict[str, Any]:
    result = {
        "initial_pool_size": 64,
        "max_pool_size": 512,
        "pool_growth_factor": 2,
        "candidates_per_value": 8,
        "max_contrast_groups": 1,
        "minimum_non_target_match": 0.60,
    }
    result.update(cfg.get("question_conditioned", {}))
    for key in ("initial_pool_size", "max_pool_size", "candidates_per_value", "max_contrast_groups"):
        if isinstance(result[key], bool) or int(result[key]) != result[key] or int(result[key]) < 1:
            raise ValueError(f"question_conditioned.{key} must be a positive integer")
        result[key] = int(result[key])
    growth = float(result["pool_growth_factor"])
    if not math.isfinite(growth) or growth <= 1:
        raise ValueError("question_conditioned.pool_growth_factor must exceed one")
    result["pool_growth_factor"] = growth
    if result["max_pool_size"] < result["initial_pool_size"]:
        raise ValueError("question_conditioned.max_pool_size must be >= initial_pool_size")
    threshold = float(result["minimum_non_target_match"])
    if not 0 <= threshold <= 1:
        raise ValueError("question_conditioned.minimum_non_target_match must be in [0, 1]")
    result["minimum_non_target_match"] = threshold
    return result


def _finding_matches(span: str, finding: str, retriever: ViSiGCRRetriever):
    spec = retriever.ontology.finding_spec(finding)
    aliases = [finding.replace("_", " "), str(spec.get("label", "")), *spec.get("synonyms", [])]
    matches = []
    for alias in aliases:
        pattern = OntologyExtractorAgent._flexible_phrase_pattern(str(alias))
        if pattern is not None:
            matches.extend(pattern.finditer(span))
    return matches


def _has_other_clinical_finding(span: str, finding: str, retriever: ViSiGCRRetriever) -> bool:
    # An exclusive-side or negative cue for another finding must never prove
    # the target predicate merely because the shortest stored span was too wide.
    for alias, canonical in retriever.ontology.finding_aliases.items():
        if canonical == finding or retriever.ontology.retrieval_policy(canonical) != "clinical":
            continue
        pattern = OntologyExtractorAgent._flexible_phrase_pattern(alias)
        if pattern is not None and pattern.search(span):
            return True
    return False


def _has_multiple_assertions(span: str) -> bool:
    text = span.strip().rstrip(".!? ")
    return bool(re.search(r"[.!?;]|\b(?:and|or|with|but|however)\b", text, re.I))


def _unilateral_support(
    item: Evidence, case_events: tuple[Event, ...], retriever: ViSiGCRRetriever | None,
) -> dict[str, Any] | None:
    """Prove one-sided involvement; missing opposite events never prove absence."""

    if retriever is None or item.event.get("uncertainty") != "definite":
        return None
    side = item.event.get("laterality")
    finding = str(item.event.get("finding"))
    span = item.evidence_span.strip()
    if side not in {"left", "right"} or not span:
        return None
    # Narrow wording avoids transferring 'only' or 'unilateral' from another
    # finding, severity modifier, negated clause, or uncertain assertion.
    ambiguous = re.search(
        r"\b(?:no|not|without|absent|possible|possibly|probable|probably|may|might|could|"
        r"cannot|history|prior|previous|resolved|and|or|but|with)\b|[;]", span, re.I,
    )
    explicit = re.search(
        rf"\bunilateral(?:ly)?\b|\bonly\s+(?:(?:on|in|at)\s+)?(?:the\s+)?{side}\b|"
        rf"\b{side}\s+(?:(?:side|lung|hemithorax)\s+)?only\b|"
        rf"\b(?:confined|limited)\s+to\s+(?:the\s+)?{side}\b", span, re.I,
    )
    if (explicit and not ambiguous and not _has_multiple_assertions(span)
            and not _has_other_clinical_finding(span, finding, retriever)
            and OntologyExtractorAgent._infer_laterality(
                span, default_anatomy=retriever.ontology.default_anatomy(finding)) == side
            and _finding_matches(span, finding, retriever)):
        return {
            "basis": "explicit_unilateral_source_assertion",
            "supporting_event_ids": [item.event_id],
            "positive_evidence_span": span, "supporting_evidence_spans": [span],
        }
    opposite = "right" if side == "left" else "left"
    minimum_confidence = max(0.70, float(retriever.cfg.get("minimum_event_confidence", 0.70)))
    tolerated = {
        "absence_not_in_pilot_allowlist", "unsectioned_report", "section_corrected",
        "laterality_corrected_from_evidence", "uncertainty_corrected_from_evidence",
        "uncertainty_filled_from_evidence", "severity_corrected_from_evidence",
        "unsupported_severity_removed", "inapplicable_severity_corrected",
    }
    for event in case_events:
        proof = event.evidence_span.strip()
        if (event.finding != finding or event.laterality != opposite
                or event.presence != "absent" or event.temporal_status != "current"
                or event.uncertainty != "definite" or event.confidence < minimum_confidence
                or not proof or event.section not in {"findings", "impression", "other"}
                or not has_verified_semantic_support(event)
                or set(event.validation_errors) - tolerated):
            continue
        marker = event.characteristics.get("_semantic_support", {})
        if marker.get("expected_presence", "absent") != "absent":
            continue
        if not event.retrieval_eligible and "absence_not_in_pilot_allowlist" not in event.validation_errors:
            continue
        default_anatomy = retriever.ontology.default_anatomy(finding)
        if (not retriever.ontology.anatomy_allowed(finding, event.anatomy)
                or not (event.anatomy == default_anatomy
                        or retriever.ontology.is_ancestor(event.anatomy, default_anatomy, "anatomy"))):
            continue
        # New/acute/regional absence cannot establish absence of all opposite-
        # side involvement. A positive span mislabelled absent is also rejected.
        if re.search(r"\b(?:acute|new|increased|decreased|change|worsening|possible|possibly|"
                     r"probable|likely|cannot|may|might|could|history|prior|previous|significant|substantial|"
                     r"appreciable|definite|large|small|tiny|minimal|trace|mild|moderate|severe|obvious|"
                     r"convincing|gross|frank|upper|lower|apical|basal|basilar|lobe|zone|region|base|apex)\b|[;]", proof, re.I):
            continue
        if (_has_multiple_assertions(proof)
                or _has_other_clinical_finding(proof, finding, retriever)):
            continue
        if OntologyExtractorAgent._infer_laterality(proof, default_anatomy=default_anatomy) != opposite:
            continue
        matches = _finding_matches(proof, finding, retriever)
        if not matches or not all(OntologyExtractorAgent._anchor_is_negated(proof, m.start(), m.end()) for m in matches):
            continue
        return {
            "basis": "explicit_current_opposite_side_absence",
            "supporting_event_ids": [item.event_id, event.event_id],
            "positive_evidence_span": span, "opposite_absence_evidence_span": proof,
            "supporting_evidence_spans": [span, proof],
        }
    return None


def _buckets(
    candidates: list[Evidence], spec: QuestionSpec, cap: int,
    retriever: ViSiGCRRetriever | None = None,
) -> dict[str, list[Evidence]]:
    required = set(spec.required_attribute_values)
    predicate = (laterality_predicate_plan(spec.question_text, spec.options)
                 if spec.target_attribute == "laterality" else None)
    buckets: dict[str, list[Evidence]] = defaultdict(list)
    known_sides: dict[tuple[str, str], set[str]] = defaultdict(set)
    raw_case_events: dict[str, tuple[Event, ...]] = {}
    if predicate and predicate["supported"]:
        for item in candidates:
            side = item.event.get("laterality")
            if item.event.get("presence") == "present" and side in {"left", "right", "bilateral"}:
                known_sides[(item.case_id, str(item.event.get("finding")))].add(side)
        if retriever is not None:
            # Eligibility/confidence filters decide which event can be an
            # exemplar. They cannot erase active contradictory context when
            # deciding that a whole case is unilateral. Read only case events,
            # never graph edges, reports, query labels, or missing-side priors.
            candidate_cases = {item.case_id for item in candidates}
            for case_id in sorted(candidate_cases):
                raw_case_events[case_id] = retriever.graph.get_events(case_id)
                for event in raw_case_events[case_id]:
                    if (event.presence == "present"
                            and event.temporal_status in ACTIVE_CLINICAL_TEMPORAL_STATUSES
                            and event.laterality in {"left", "right", "bilateral"}):
                        known_sides[(case_id, event.finding)].add(event.laterality)
    seen: set[tuple[str, str]] = set()
    # Prefer explicit assertions; similarity is primary within each basis tier.
    for item in sorted(candidates, key=lambda e: (
        e.event.get("presence_basis") == "inferred_absent",
        -e.visual_similarity_query, -e.target_match, -(e.event_confidence or 0),
        e.case_id, e.event_id,
    )):
        value = item.attribute_value
        if predicate and predicate["supported"]:
            # Retain the extraction label in event.laterality. Only the
            # comparison bucket joins explicit left/right into unilateral.
            if value not in predicate["raw_value_mapping"]:
                continue
            value = predicate["raw_value_mapping"].get(item.event.get("laterality"))
            if value is None or item.event.get("presence") != "present":
                continue
            case_sides = known_sides[(item.case_id, str(item.event.get("finding")))]
            if value == "unilateral" and ("bilateral" in case_sides or {"left", "right"} <= case_sides):
                continue
            support = None
            if value == "unilateral":
                support = _unilateral_support(item, raw_case_events.get(item.case_id, ()), retriever)
                if support is None:
                    continue
            item = item.model_copy(update={
                "event": {**item.event, **({"laterality_predicate_support": support} if support else {})},
                "attribute_value": value, "comparison_role": "VALUE_EXEMPLAR",
                "matched_option_ids": spec.option_ids_for_value(value),
            })
        if value in {"", "unknown", "not_mentioned", "not_applicable", "unspecified"}:
            continue
        if len(required) >= 2 and value not in required:
            continue
        key = (value, item.case_id)
        if key in seen or len(buckets[value]) >= cap:
            continue
        seen.add(key)
        buckets[value].append(item)
    return dict(buckets)


def _pairs(
    retriever: ViSiGCRRetriever,
    buckets: dict[str, list[Evidence]],
    spec: QuestionSpec,
    settings: dict[str, Any],
) -> tuple[list[tuple[float, Evidence, Evidence, float | None]], int]:
    pairs = []
    checks = 0
    for value_a, value_b in combinations(sorted(buckets), 2):
        for a, b in product(buckets[value_a], buckets[value_b]):
            if a.case_id == b.case_id:
                continue
            subject_a = retriever.graph.get_case(a.case_id).subject_id
            subject_b = retriever.graph.get_case(b.case_id).subject_id
            if subject_a and subject_b and subject_a == subject_b:
                continue
            checks += 1
            context = None
            if spec.target_scope != "case_normality":
                context = pair_non_target_context_score(
                    (retriever.presence_index.event(a.event_id, retriever.graph, retriever.ontology)
                     if retriever.presence_index is not None else retriever.graph.get_event(a.event_id)),
                    (retriever.presence_index.event(b.event_id, retriever.graph, retriever.ontology)
                     if retriever.presence_index is not None else retriever.graph.get_event(b.event_id)),
                    spec.target_attribute, retriever.ontology,
                )
                if context <= 0 or context < settings["minimum_non_target_match"]:
                    continue
            qa, qb = (max(0.0, min(1.0, e.visual_similarity_query)) for e in (a, b))
            confidence = sum(e.event_confidence if e.event_confidence is not None else 1.0 for e in (a, b)) / 2
            # Query relevance dominates. No seed, edge, endpoint-margin or
            # evidence-to-evidence similarity reward is present.
            score = (
                0.45 * (qa + qb) / 2 + 0.25 * min(qa, qb)
                + 0.15 * (a.target_match + b.target_match) / 2
                + 0.10 * (context if context is not None else 1.0)
                + 0.05 * confidence
            )
            pairs.append((score, a, b, context))
    pairs.sort(key=lambda row: (-row[0], row[1].case_id, row[2].case_id, row[1].event_id, row[2].event_id))
    return pairs, checks


def _select(retriever, pairs, spec, max_groups):
    selected: list[Evidence] = []
    groups: list[SemanticContrastGroup] = []
    used: set[str] = set()
    covered: set[str] = set()
    required = set(spec.required_attribute_values)
    remaining = list(pairs)
    while remaining and len(groups) < max_groups:
        viable = [p for p in remaining if p[1].case_id not in used and p[2].case_id not in used]
        if not viable:
            break
        # Cover required values, prefer explicit pairs, then maximize score
        # within the same fixed evidence budget.
        best = max(viable, key=lambda p: (
            len(({p[1].attribute_value, p[2].attribute_value} & required) - covered),
            -sum(e.event.get("presence_basis") == "inferred_absent" for e in p[1:3]), p[0],
        ))
        score, a, b, context = best
        remaining.remove(best)
        group_id = f"qcr{len(groups) + 1:03d}"
        values = [a.attribute_value, b.attribute_value]
        group = SemanticContrastGroup(
            group_id=group_id, target_attribute=spec.target_attribute,
            member_case_ids=[a.case_id, b.case_id], member_event_ids=[a.event_id, b.event_id],
            member_attribute_values=values,
            member_comparison_roles=[a.comparison_role, b.comparison_role],
            required_values_covered=[v for v in values if v in required],
            group_quality=score, non_target_match=context,
            # Descriptive audit only; does not gate or score the pair.
            visual_locality=max(-1.0, min(1.0, retriever.visual.similarity_between(a.case_id, b.case_id))),
            pair_source="question_conditioned_pool",
            query_endpoint_margin=min(1.0, abs(a.visual_similarity_query - b.visual_similarity_query)),
        )
        groups.append(group)
        for item in (a, b):
            selected.append(item.model_copy(update={
                "semantic_contrast_group_id": group_id, "group_quality": score,
                "selection_gain": score,
            }))
            used.add(item.case_id)
            covered.add(item.attribute_value)
        if retriever._coverage_status(selected, spec)["required_value_complete"]:
            break
    complete = bool(groups) and retriever._coverage_status(selected, spec)["required_value_complete"]
    return (selected, groups) if complete else ([], [])


def retrieve_question_conditioned(
    retriever: ViSiGCRRetriever,
    query_id: str,
    query_image: str,
    question: str,
    spec: QuestionSpec,
    query_embedding: np.ndarray,
) -> EvidencePackage:
    from .engine import BoundaryEntry

    question_stem = spec.question_text or question
    spec = normalize_laterality_predicate_spec(spec, question_stem)
    predicate = (laterality_predicate_plan(question_stem, spec.options)
                 if spec.target_scope == "finding" and spec.comparison_mode == "attribute_axis" else None)
    unsupported_predicate = bool(predicate and not predicate["supported"])
    if predicate and predicate["supported"]:
        predicate = {**predicate, "unilateral_support_policy": "explicit_unilateral_or_opposite_absence",
                     "case_coherence_source": "all_active_case_events",
                     "case_coherence_temporal_statuses": sorted(ACTIVE_CLINICAL_TEMPORAL_STATUSES)}
    normality_scope_unavailable = False
    if spec.target_scope == "case_normality":
        if retriever.cfg.get("case_normality_reference_policy", "explicit_event_supported") != "explicit_event_supported":
            raise ValueError(
                "Question-conditioned v2 requires scope-safe explicit_event_supported "
                "normality; closed_world_proxy is a legacy control only."
            )
        normality_scope_unavailable = not scope_safe_normal_findings(
            retriever.ontology,
            retriever.cfg.get("case_normality_explicit_normal_findings", []),
        )
    settings = _settings(retriever.cfg)
    budget = max(0, int(retriever.cfg.get("evidence_k", 6)))
    max_groups = min(settings["max_contrast_groups"], budget // 2)
    max_pool = min(settings["max_pool_size"], len(retriever.visual.case_ids))
    pool_size = min(settings["initial_pool_size"], max_pool)
    candidates: list[Evidence] = []
    seen: set[str] = set()
    history: list[dict[str, Any]] = []
    selected: list[Evidence] = []
    groups: list[SemanticContrastGroup] = []
    buckets: dict[str, list[Evidence]] = {}
    pairs = []
    total_checks = 0
    required = set(spec.required_attribute_values)
    budget_insufficient = max_groups == 0 or len(required) > 2 * max_groups
    while pool_size and not budget_insufficient and not unsupported_predicate and not normality_scope_unavailable:
        neighbors = sorted(retriever.visual.search(query_embedding, pool_size), key=lambda p: (-p[1], p[0]))
        boundary = []
        for rank, (case_id, score) in enumerate(neighbors, 1):
            if case_id in seen:
                continue
            seen.add(case_id)
            case = retriever.graph.get_case(case_id)
            if case.split != "train":
                raise ValueError(f"Question-conditioned memory contains non-train case: {case_id}")
            boundary.append(BoundaryEntry(
                case_id=case_id, seed_case_id=case_id, query_similarity=float(score),
                direct_seed_similarity=1.0, path_score=1.0, hop_count=0,
                path_nodes=(case_id,),
            ))
        new_candidates = retriever._scan_boundary(boundary, spec, query_conditioned=True)
        ranks = {cid: rank for rank, (cid, _) in enumerate(neighbors, 1)}
        for item in new_candidates:
            item.boundary_path = ["query_visual_pool", f"case:{item.case_id}"]
            item.retrieval_signals["query_visual_rank"] = float(ranks[item.case_id])
        candidates.extend(new_candidates)
        buckets = _buckets(candidates, spec, settings["candidates_per_value"], retriever)
        pairs, checks = _pairs(retriever, buckets, spec, settings)
        total_checks += checks
        selected, groups = _select(retriever, pairs, spec, max_groups)
        history.append({
            "requested_pool_size": pool_size, "returned_case_count": len(neighbors),
            "scanned_case_count": len(seen), "target_event_count": len(candidates),
            "retained_value_counts": {v: len(items) for v, items in buckets.items()},
            "valid_pair_count": len(pairs),
        })
        target_values = required or set(buckets)
        enough = len(target_values) >= 2 and all(
            len(buckets.get(v, [])) >= settings["candidates_per_value"] for v in target_values
        )
        inferred_selected = any(e.event.get("presence_basis") == "inferred_absent" for e in selected)
        if (groups and enough and not inferred_selected) or pool_size >= max_pool or len(neighbors) < pool_size:
            break
        pool_size = min(max_pool, max(pool_size + 1, math.ceil(pool_size * settings["pool_growth_factor"])))

    ordered_candidates = sorted((e for items in buckets.values() for e in items), key=lambda e: (
        -e.visual_similarity_query, -e.target_match, -(e.event_confidence or 0), e.case_id, e.event_id,
    ))
    seed = ordered_candidates[0] if ordered_candidates else None
    for item in selected:
        item.seed_case_id = seed.case_id if seed else item.case_id
        item.visual_similarity_seed = (
            retriever.visual.similarity_between(item.case_id, seed.case_id) if seed else 0.0
        )
    fallback = not groups
    missing = sorted(required - set(buckets))
    reason = ""
    if fallback:
        reason = (
            "unsupported_laterality_predicate" if unsupported_predicate else
            "no_scope_safe_normal_descriptor" if normality_scope_unavailable else
            "insufficient_evidence_budget" if budget_insufficient else
            "no_target_candidates" if not candidates else
            "missing_required_values" if missing else "no_compatible_complete_contrast"
        )
    coverage = retriever._coverage_status(selected, spec)
    coverage.update({
        "selection_mode": "semantic_contrast_group", "selection_policy_version": POLICY_VERSION,
        "configured_evidence_budget": budget, "max_contrast_groups": max_groups,
        "fill_ungrouped_evidence": False, "require_complete_contrast_groups": True,
        "candidate_count": len(candidates), "candidate_attribute_values": sorted(buckets),
        "candidate_group_count": len(pairs), "selected_group_count": len(groups),
        "group_co_presentation_count": len(groups), "grouped_attribute_values": sorted({v for g in groups for v in g.member_attribute_values}),
        "missing_grouped_required_attribute_values": coverage["missing_required_attribute_values"],
        "selective_query_only_fallback": fallback, "relation_routing": False,
        "relation_routed_evidence_count": 0, "no_matched_event_evidence_count": 0,
        "non_target_context_applicable": spec.target_scope != "case_normality",
    })
    return EvidencePackage(
        query_id=query_id, query_image_path=query_image, question=question,
        retrieval_question=spec.question_text or question, options=list(spec.options),
        question_spec=spec, method="visig_cr",
        seed_cases=([{
            "case_id": seed.case_id, "event_id": seed.event_id,
            "visual_similarity": seed.visual_similarity_query, "image_path": seed.image_path,
            "attribute_value": seed.attribute_value, "role": "first_target_matched_candidate_audit_only",
        }] if seed else []),
        evidence=selected, semantic_contrast_groups=groups, coverage_status=coverage,
        boundary_stats={
            "implementation_version": IMPLEMENTATION_VERSION, "retrieval_policy": POLICY_VERSION,
            "graph_used": False, "relation_routing": False, "boundary_hops": 0,
            "boundary_case_count": len(seen), "group_pair_mode": "question_conditioned_pool",
            "stored_case_edges_examined": 0, "stored_event_relations_examined": 0,
            "candidate_pair_checks": total_checks, "pool_search_history": history,
            "pool_expansion_count": max(0, len(history) - 1), "pool_limit": max_pool,
            "pool_limit_reached": bool(history and pool_size >= max_pool),
            "retained_value_counts": {v: len(items) for v, items in buckets.items()},
            "missing_candidate_values": missing,
            "normality_scope_unavailable": normality_scope_unavailable,
        },
        warnings=([f"Complete contrast missing ({reason}); using the exact no-RAG reader input."] if fallback else []),
        audit_metadata={
            "retrieval_policy": POLICY_VERSION,
            "question_conditioned": {
                "settings": settings, "graph_used": False,
                "laterality_predicate": predicate,
            },
            "contrast_fallback": {
                "used": fallback, "reason_code": reason,
                "strategy": "query_only" if fallback else "not_applicable",
                "reader_input": "no_rag" if fallback else "structured_contrast",
                "candidate_group_count": len(pairs), "selected_group_count": len(groups),
                "selection_policy_version": POLICY_VERSION,
            },
            "pair_construction": {
                "complete_contrast_only": True,
                "selected_pair_sources": ["question_conditioned_pool"] if groups else [],
            },
        },
    )
