"""Method-independent selected-case evaluation, isolated from reader input.

Shared annotations are built offline from frozen query specifications and the
same case-event store for every method. They assess selected-case content, not
which assertions a reader saw, cited, or actually used. All automatic clinical
metrics remain ontology proxies; independent group audits are separate.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import math
from typing import Any

from .clinical_matching import attribute_value, best_target_match, pair_non_target_context_score
from .graph.relations import has_verified_semantic_support
from .io_utils import stable_json_sha256
from .normality import derive_normality_metadata
from .ontology import Ontology
from .schemas import Event, EvidencePackage, QuestionSpec


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def _mean(values: Iterable[float | None]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _package(value: Any) -> EvidencePackage:
    return value if isinstance(value, EvidencePackage) else EvidencePackage.model_validate(value)


def shared_case_annotation(case_id: str, spec: QuestionSpec, graph: Any,
                           ontology: Ontology, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Annotate one selected case independently of its method's metadata."""
    # A missing case is an identity error; an existing case with no events is not.
    graph.get_case(case_id)
    events = graph.get_events(case_id)
    if spec.target_scope == "case_normality":
        stored = graph.case_normalities_in_cases([case_id]).get(case_id)
        provenance = graph.normality_provenance(case_id)
        return {
            "values": [{"attribute_value": "present" if stored == "abnormal" else "absent",
                        "event_id": f"case_normality:{case_id}", "event": None}]
                      if stored in {"normal", "abnormal"} else [],
            "normality": stored, "normality_provenance": provenance,
        }
    values = []
    minimum_confidence = float(cfg.get("minimum_event_confidence", 0.70))
    minimum_match = float(cfg.get("minimum_target_match", 0.60))
    for event in events:
        if event.temporal_status != "current" or event.presence == "not_mentioned":
            continue
        if event.confidence < minimum_confidence:
            continue
        if cfg.get("require_semantic_support", False) and not has_verified_semantic_support(event):
            continue
        if not event.retrieval_eligible:
            # Mirror explicitly exposed candidate predicates without making
            # unmentioned findings into negatives or exposing reference answers.
            matched = [candidate for candidate in spec.answer_candidates if candidate.finding == event.finding]
            tolerated = {"section_corrected", "unsectioned_report"}
            if ontology.retrieval_policy(event.finding) == "context_only":
                tolerated.add("context_only_finding")
            if any(candidate.presence == "absent" for candidate in matched):
                tolerated.add("absence_not_in_pilot_allowlist")
            if spec.comparison_mode != "candidate_predicate" or not matched or set(event.validation_errors) - tolerated:
                continue
        target, score = best_target_match(event, spec, ontology)
        value = attribute_value(event, spec.target_attribute)
        if target is None or score <= 0 or score < minimum_match or value is None:
            continue
        if spec.comparison_mode == "candidate_predicate" and value not in spec.answer_space:
            continue
        values.append({"attribute_value": value, "event_id": event.event_id,
                       "event": event.model_dump(), "target_match": score})
    return {"values": values}


def build_shared_annotations(
    packages: Iterable[EvidencePackage | Mapping[str, Any]],
    question_specs: Mapping[str, QuestionSpec | Mapping[str, Any]],
    *, graph: Any, ontology: Ontology, visual: Any, encoder: Any,
    retrieval_cfg: Mapping[str, Any], k: int = 6,
) -> Iterable[dict[str, Any]]:
    """Build common annotations once for the union of selected cases.

    Only compact per-query case sets are retained; model inputs are not rewritten.
    The query encoder is the frozen visual encoder, never an answer/selector LLM.
    Pair cosines are computed only for pairs delivered together by a method.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    records: dict[str, dict[str, Any]] = {}
    for raw in packages:
        package = _package(raw)
        if package.input_mode == "no_rag":
            continue
        fallback_audit = package.audit_metadata.get("selector_fallback", {})
        if (
            package.question_spec is None
            and isinstance(fallback_audit, Mapping)
            and fallback_audit.get("used") is True
        ):
            continue
        qid = package.query_id
        if qid not in question_specs:
            raise ValueError(f"Missing frozen evaluation QuestionSpec for {qid!r}")
        raw_spec = question_specs[qid]
        spec = raw_spec if isinstance(raw_spec, QuestionSpec) else QuestionSpec.model_validate(raw_spec)
        if spec.abstain:
            # Keep abstentions as such, rather than selecting a target after seeing retrieval.
            pass
        if spec.options and list(spec.options) != list(package.options):
            raise ValueError(f"Visible options differ from the shared specification: {qid}")
        current = records.setdefault(qid, {"image": package.query_image_path,
                                          "question": package.retrieval_question or package.question,
                                          "options": list(package.options),
                                          "case_ids": set(), "pairs": set(), "spec": spec})
        if current["image"] != package.query_image_path or current["options"] != list(package.options):
            raise ValueError(f"Query image/options disagree across methods: {qid}")
        # The canonical plain question, not a baseline-specific added context, is shared.
        if current["question"].strip() != (package.retrieval_question or package.question).strip():
            raise ValueError(f"Question text disagrees across methods: {qid}")
        ids = sorted({item.case_id for item in package.evidence[:k]})
        current["case_ids"].update(ids)
        current["pairs"].update(combinations(ids, 2))
    from tqdm import tqdm
    graph_identity = graph.artifact_metadata()
    for qid in tqdm(sorted(records), desc="Common retrieval annotations"):
        record = records[qid]
        spec = record["spec"]
        query_embedding = encoder.encode_one(record["image"])
        cases = {}
        for case_id in sorted(record["case_ids"]):
            if case_id not in visual.id_to_row:
                raise ValueError(f"Selected case is outside the active visual memory: {case_id}")
            value = shared_case_annotation(case_id, spec, graph, ontology, retrieval_cfg)
            value["visual_similarity_query"] = float(visual.similarity_to_query(query_embedding, case_id))
            cases[case_id] = value
        pair_visual = [
            {"case_ids": [a, b], "similarity": float(visual.similarity_between(a, b))}
            for a, b in sorted(record["pairs"])
        ]
        yield {
            "query_id": qid, "query_image_path": record["image"],
            "question": record["question"], "options": record["options"],
            "question_spec": spec.model_dump(), "cases": cases,
            "pair_visual_similarities": pair_visual,
            "provenance": {
                "kind": "shared_case_ontology_proxy", "project_version": "2.0",
                "ontology_sha256": ontology.ontology_hash,
                "question_spec_sha256": stable_json_sha256(spec.model_dump()),
                "graph_identity": graph_identity,
                "retrieval_annotation_config_sha256": stable_json_sha256(dict(retrieval_cfg)),
                "reader_inputs_modified": False,
                "annotation_scope": "selected_case_content_not_reader_use",
            },
        }


METRICS = (
    "visual_preservation_at_k", "required_value_coverage_at_k", "required_value_complete_at_k",
    "semantic_contrast_completeness_at_k", "non_target_context_preservation_at_k",
    "delivered_semantic_contrast_validity", "semantic_group_co_presentation_rate",
    "attribute_value_redundancy_at_k", "attribute_value_entropy", "minority_value_fraction",
    "relation_routed_evidence_fraction", "no_matched_event_use_rate", "relation_route_coverage",
)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {name: _mean(row.get(name) for row in rows) for name in METRICS}
    total_groups = sum(row["selected_group_count"] for row in rows)
    audited = sum(row["audited_group_count"] for row in rows)
    result.update({
        "query_count": len(rows),
        "mean_evidence_count": _mean(row["evidence_count"] for row in rows),
        "mean_selected_group_count": _mean(row["selected_group_count"] for row in rows),
        "mean_boundary_case_count": _mean(row["boundary_case_count"] for row in rows),
        "metric_denominators": {name: sum(row.get(name) is not None for row in rows) for name in METRICS},
        "metric_provenance": (next(iter({row["metric_provenance"] for row in rows}))
                              if len({row["metric_provenance"] for row in rows}) == 1 else "mixed_or_empty"),
        "audited_group_count": audited, "unaudited_group_count": total_groups - audited,
        "group_audit_coverage": audited / total_groups if total_groups else None,
        "independently_audited_group_validity": (
            sum(row["audited_valid_group_count"] for row in rows) / audited if audited else None
        ),
        "independent_audit_does_not_validate_unaudited_groups": True,
    })
    return result


def evaluate_packages(
    packages: Sequence[EvidencePackage | Mapping[str, Any]],
    reference_annotations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *, case_annotations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ontology: Ontology | None = None,
    k: int = 6, tau_query: float = 0.20, tau_non_target: float = 0.60,
    tau_group_visual: float = 0.20,
) -> dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    if case_annotations is not None and ontology is None:
        raise ValueError("Common case evaluation requires the frozen ontology")
    annotations: dict[str, Any] = {}
    if isinstance(case_annotations, Mapping):
        annotations = dict(case_annotations)
    elif case_annotations is not None:
        for row in case_annotations:
            qid = str(row["query_id"])
            if qid in annotations:
                raise ValueError(f"Duplicate common annotation query_id: {qid}")
            annotations[qid] = dict(row)
    refs = {}
    if isinstance(reference_annotations, Mapping):
        refs = dict(reference_annotations)
    else:
        for row in reference_annotations or []:
            key = f"{row.get('query_id', '')}:{row.get('group_id', '')}"
            if row.get("method"):
                key = f"{row['query_id']}:{row['method']}:{row.get('group_id', '')}"
            if key in refs:
                raise ValueError(f"Duplicate group audit key: {key}")
            refs[key] = dict(row)
    rows = []
    for raw in packages:
        package = _package(raw)
        delivered = package.evidence[:k]
        ids = list(dict.fromkeys(item.case_id for item in delivered))
        if len(ids) != len(delivered):
            raise ValueError(f"Duplicate delivered case IDs in {package.query_id}")
        fallback_audit = package.audit_metadata.get("selector_fallback", {})
        is_selector_fallback = (
            package.question_spec is None
            and isinstance(fallback_audit, Mapping)
            and fallback_audit.get("used") is True
        )
        common = None if is_selector_fallback else annotations.get(package.query_id)
        if (
            case_annotations is not None
            and common is None
            and package.input_mode != "no_rag"
            and not is_selector_fallback
        ):
            raise ValueError(f"Missing common annotations for {package.query_id}")
        if common is not None:
            if common.get("query_image_path", package.query_image_path) != package.query_image_path:
                raise ValueError(f"Annotation query image mismatch: {package.query_id}")
            if list(common.get("options", package.options)) != list(package.options):
                raise ValueError(f"Annotation option mismatch: {package.query_id}")
            actual_question = (package.retrieval_question or package.question).strip()
            if str(common.get("question", actual_question)).strip() != actual_question:
                raise ValueError(f"Annotation question mismatch: {package.query_id}")
            stored_hash = common.get("provenance", {}).get("ontology_sha256")
            if stored_hash and stored_hash != ontology.ontology_hash:
                raise ValueError("Common annotations were made with a different ontology")
        spec = (QuestionSpec.model_validate(common["question_spec"]) if common is not None
                else package.question_spec)
        is_normality = spec is not None and spec.target_scope == "case_normality"
        evaluable = (package.input_mode != "no_rag" and not is_selector_fallback
                     and spec is not None and not spec.abstain
                     and (common is not None or package.reader_presentation == "visig_cr_structured"))
        values_by_case: dict[str, list[dict[str, Any]]] = {}
        query_similarities = {}
        pair_visual = {}
        if common is not None:
            for case_id in ids:
                if case_id not in common["cases"]:
                    raise ValueError(f"Common annotations lack selected case {case_id!r} for {package.query_id}")
                values_by_case[case_id] = list(common["cases"][case_id]["values"])
                query_similarities[case_id] = float(common["cases"][case_id]["visual_similarity_query"])
            pair_visual = {_pair_key(*item["case_ids"]): float(item["similarity"])
                           for item in common.get("pair_visual_similarities", [])}
        else:
            for item in delivered:
                values_by_case[item.case_id] = ([{"attribute_value": item.attribute_value,
                                                "event_id": item.event_id, "event": None}]
                                              if item.attribute_value not in {"", "unknown"} else [])
                query_similarities[item.case_id] = item.visual_similarity_query
        case_values = {case_id: {v["attribute_value"] for v in values if v["attribute_value"] not in {"", "unknown"}}
                       for case_id, values in values_by_case.items()}
        observed = set().union(*case_values.values()) if case_values else set()
        required = set(spec.required_attribute_values) if spec else set()
        needed = spec.minimum_distinct_attribute_values if spec else 2
        covered = len(required & observed) / len(required) if required else float(len(observed) >= needed)
        value_complete = float(required.issubset(observed) if required else len(observed) >= needed)
        matched_values: set[str] = set()
        matched_cases: set[str] = set()
        if common is not None and evaluable:
            for a, b in combinations(ids, 2):
                key = _pair_key(a, b)
                if key not in pair_visual:
                    raise ValueError(f"Common annotations lack pair cosine for {key}; rebuild with all compared packages")
                if pair_visual[key] < tau_group_visual:
                    continue
                if min(query_similarities[a], query_similarities[b]) < tau_query:
                    continue
                for va in values_by_case[a]:
                    for vb in values_by_case[b]:
                        if va["attribute_value"] == vb["attribute_value"]:
                            continue
                        if not is_normality:
                            context = pair_non_target_context_score(Event.model_validate(va["event"]),
                                                                   Event.model_validate(vb["event"]),
                                                                   spec.target_attribute, ontology)
                            if context <= 0 or context < tau_non_target:
                                continue
                        matched_values.update([va["attribute_value"], vb["attribute_value"]])
                        matched_cases.update([a, b])
            complete = float(required.issubset(matched_values) if required else len(matched_values) >= needed)
            ntcp = None if is_normality else len(matched_cases) / k
        else:
            complete = float(package.coverage_status.get("semantic_contrast_complete", False))
            ntcp = None if is_normality else sum(
                item.non_target_match is not None and item.non_target_match >= tau_non_target
                for item in delivered
            ) / k
        # Fractional per-case value counts avoid counting multi-assertion cases as extra images.
        counts: Counter[str] = Counter()
        for values in case_values.values():
            for value in values:
                counts[value] += 1.0 / len(values)
        total = sum(counts.values())
        distribution = [value / total for value in counts.values()] if total else []
        entropy = (-sum(p * math.log(p) for p in distribution) / math.log(max(2, len(required | observed)))) if distribution else 0.0
        minority = min(counts.get(v, 0.0) / total for v in (required or observed)) if total else 0.0
        redundancy = max(0.0, total - len(counts)) / total if total else 0.0
        group_proxy, co_present = [], []
        audit_count = audited_valid = 0
        for group in package.semantic_contrast_groups:
            delivered_group = set(group.member_case_ids).issubset(set(ids))
            co_present.append(float(delivered_group))
            if not is_normality:
                group_proxy.append(float(delivered_group and len(set(group.member_attribute_values)) >= 2
                                         and group.non_target_match is not None
                                         and group.non_target_match >= tau_non_target
                                         and group.visual_locality >= tau_group_visual))
            ref = refs.get(f"{package.query_id}:{package.method}:{group.group_id}",
                           refs.get(f"{package.query_id}:{group.group_id}"))
            if ref is not None and "valid" in ref:
                if not isinstance(ref["valid"], bool):
                    raise ValueError("Group reference 'valid' must be a JSON boolean")
                audit_count += 1
                audited_valid += int(ref["valid"] and delivered_group)
        routed = [item for item in delivered if item.relation_types]
        routed_values = set().union(*(case_values.get(item.case_id, set()) for item in routed)) if routed else set()
        route_coverage = len(required & routed_values) / len(required) if required else float(len(routed_values) >= needed)
        provenance = ("not_applicable" if package.input_mode == "no_rag" else
                      "shared_case_ontology_proxy" if common is not None else
                      "package_metadata_proxy" if evaluable else "unavailable_without_common_annotations")
        row = {
            "query_id": package.query_id, "method": package.method,
            "target_scope": spec.target_scope if spec else "not_applicable",
            "evidence_count": len(ids), "metric_provenance": provenance,
            "visual_preservation_at_k": sum(v >= tau_query for v in query_similarities.values()) / k if evaluable else None,
            "required_value_coverage_at_k": covered if evaluable else None,
            "required_value_complete_at_k": value_complete if evaluable else None,
            "semantic_contrast_completeness_at_k": complete if evaluable else None,
            "non_target_context_preservation_at_k": ntcp if evaluable else None,
            "delivered_semantic_contrast_validity": _mean(group_proxy),
            "semantic_group_co_presentation_rate": _mean(co_present),
            "attribute_value_redundancy_at_k": redundancy if evaluable else None,
            "attribute_value_entropy": entropy if evaluable else None,
            "minority_value_fraction": minority if evaluable else None,
            "relation_routed_evidence_fraction": (
                None if is_selector_fallback else len(routed) / len(ids) if ids else 0.0
            ),
            "no_matched_event_use_rate": (
                None
                if is_selector_fallback
                else sum(bool(item.used_no_matched_event_ids) for item in delivered) / len(ids)
                if ids
                else 0.0
            ),
            "relation_route_coverage": (
                None if is_selector_fallback else route_coverage if evaluable else None
            ),
            "observed_attribute_values": sorted(observed), "required_attribute_values": sorted(required),
            "attribute_value_counts": dict(sorted(counts.items())),
            "cases_without_supported_target_value": sum(not values for values in case_values.values()),
            "selected_group_count": len(package.semantic_contrast_groups),
            "audited_group_count": audit_count, "audited_valid_group_count": audited_valid,
            "boundary_case_count": int(package.boundary_stats.get("boundary_case_count", 0)),
            "boundary_candidate_count": int(package.coverage_status.get("candidate_count", 0)),
            "coverage_warning_count": len(package.warnings),
        }
        if package.input_mode == "no_rag":
            row.update({name: None for name in METRICS})
        rows.append(row)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_scope[row["target_scope"]].append(row)
    return {"aggregate": _aggregate(rows), "per_query": rows,
            "by_method": {key: _aggregate(value) for key, value in by_method.items()},
            "by_target_scope": {key: _aggregate(value) for key, value in by_scope.items()},
            "notes": [
                "Common annotations evaluate selected-case content, not reader use or visible assertion fidelity.",
                "Normality is a closed-world event proxy; its event-context metrics are null, not 1.0.",
                "Independent group audits have separate counts and never relabel unaudited ontology proxies.",
                "Package-metadata diagnostics without common annotations are not fair cross-method retrieval metrics.",
            ]}
