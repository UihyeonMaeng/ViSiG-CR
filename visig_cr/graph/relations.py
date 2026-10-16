"""Offline event-level relation materialization for visual-neighbor case pairs.

The visual kNN topology remains the bounded substrate.  Each directed visual
edge receives a relation payload containing:

* event-pair relations for clinically comparable assertions; and
* ``no_matched_events`` keyed by the two exact case IDs.

The payload is question-independent.  At query time a deterministic relation
router prioritizes the stored atomic relations selected by the Attribute
Selector Agent and performs final event validation before packaging evidence.
"""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Any, Iterable

from ..ontology import Ontology
from ..schemas import (
    CaseRelationEdge,
    CaseNormalityRelation,
    Event,
    MatchedEventRelation,
    RelationType,
    TargetAttribute,
    VisualEdge,
)

from ..version import RELATION_SCHEMA_VERSION

from ..clinical_matching import pair_non_target_context_score

UNCERTAIN_ASSERTIONS = frozenset(
    {"probable", "possible", "cannot_exclude", "indeterminate", "unlikely"}
)
MISSING = frozenset({"", "unknown", "unspecified", "not_applicable", "not_mentioned"})
GENERIC_ANATOMY = frozenset({"thorax", "lung", "lung_zone", "bone", "unspecified"})
SEMANTIC_SUPPORT_MARKER = "_semantic_support"


def has_verified_semantic_support(event: Event) -> bool:
    """Return whether extraction attached a verified semantic-support marker."""

    if not isinstance(event.characteristics, dict):
        return False
    marker = event.characteristics.get(SEMANTIC_SUPPORT_MARKER)
    return (
        isinstance(marker, dict)
        and marker.get("verified") is True
        and marker.get("schema_version") == "visig_cr_semantic_support_v1"
    )


def evidence_presence_value(event: Event) -> str | None:
    if event.presence == "absent":
        return "absent"
    if event.presence == "present" and event.uncertainty in UNCERTAIN_ASSERTIONS:
        return "uncertain"
    if event.presence == "present":
        return "present"
    return None


def _mean(values: Iterable[float | None], *, default: float = 0.0) -> float:
    observed = [max(0.0, min(1.0, float(value))) for value in values if value is not None]
    return sum(observed) / len(observed) if observed else default


def _relation_label(value: float | None, *, exact: float = 0.999) -> str:
    if value is None:
        return "unknown"
    if value >= exact:
        return "same"
    if value >= 0.75:
        return "compatible"
    if value > 0.0:
        return "broad"
    return "different"


def _laterality_relation(left: Event, right: Event, ontology: Ontology) -> str:
    if left.laterality == "not_applicable" and right.laterality == "not_applicable":
        return "not_applicable"
    score = ontology.laterality_similarity(left.laterality, right.laterality)
    if score is None:
        return "unknown"
    if score >= 0.999:
        return "same"
    if score > 0.0:
        return "compatible"
    return "different"


def _severity_relation(left: Event, right: Event, ontology: Ontology) -> str:
    if left.severity == "not_applicable" and right.severity == "not_applicable":
        return "not_applicable"
    score = ontology.severity_similarity(left.severity, right.severity)
    if score is None:
        return "unknown"
    if score >= 0.999:
        return "same"
    if score >= 0.5:
        return "compatible"
    return "different"


def _uncertainty_relation(left: Event, right: Event, ontology: Ontology) -> str:
    score = ontology.uncertainty_similarity(left, right)
    if score is None:
        return "unknown"
    if score >= 0.999:
        return "same"
    if score >= 0.5:
        return "compatible"
    return "different"


def _presence_relation(left: Event, right: Event) -> str:
    a = evidence_presence_value(left)
    b = evidence_presence_value(right)
    if a is None or b is None:
        return "unknown"
    if a != b:
        return "contrast"
    if a == "present":
        return "same_present"
    if a == "absent":
        return "same_absent"
    return "same_uncertain"


def _finding_relation(left: Event, right: Event, ontology: Ontology) -> str:
    if left.finding == right.finding:
        return "same"
    if ontology.is_differential_pair(left.finding, right.finding):
        return "differential"
    if ontology.finding_similarity(left.finding, right.finding) > 0.0:
        return "same_family"
    return "different"


def _non_target_score(
    left: Event,
    right: Event,
    target_attribute: TargetAttribute,
    ontology: Ontology,
) -> float:
    return pair_non_target_context_score(left, right, target_attribute, ontology)


def _eligible_attributes(
    left: Event,
    right: Event,
    *,
    finding_relation: str,
    anatomy_relation: str,
) -> list[TargetAttribute]:
    attrs: list[TargetAttribute] = []
    left_presence = evidence_presence_value(left)
    right_presence = evidence_presence_value(right)
    if finding_relation == "same" and left_presence and right_presence:
        attrs.append("presence")
        if left_presence != "absent" and right_presence != "absent":
            if left.laterality not in MISSING and right.laterality not in MISSING:
                attrs.append("laterality")
            if left.severity not in MISSING and right.severity not in MISSING:
                attrs.append("severity")
            if left.anatomy not in MISSING and right.anatomy not in MISSING:
                attrs.append("anatomy")
    if (
        finding_relation in {"differential", "same_family", "different"}
        and anatomy_relation in {"same", "compatible", "broad"}
        and left_presence in {"present", "uncertain"}
        and right_presence in {"present", "uncertain"}
    ):
        attrs.append("finding_identity")
    return list(dict.fromkeys(attrs))


def build_event_relation(
    left: Event,
    right: Event,
    ontology: Ontology,
    cfg: dict[str, Any],
) -> MatchedEventRelation | None:
    finding_relation = _finding_relation(left, right, ontology)
    anatomy_similarity = ontology.anatomy_similarity(left.anatomy, right.anatomy)
    anatomy_relation = _relation_label(anatomy_similarity)
    laterality_relation = _laterality_relation(left, right, ontology)
    presence_relation = _presence_relation(left, right)
    severity_relation = _severity_relation(left, right, ontology)
    uncertainty_relation = _uncertainty_relation(left, right, ontology)

    allow_same_anatomy_diff = bool(
        cfg.get("allow_same_anatomy_different_finding", True)
    )
    specific_same_anatomy = (
        left.anatomy == right.anatomy
        and left.anatomy not in GENERIC_ANATOMY
        and evidence_presence_value(left) in {"present", "uncertain"}
        and evidence_presence_value(right) in {"present", "uncertain"}
    )
    both_positive_or_uncertain = (
        evidence_presence_value(left) in {"present", "uncertain"}
        and evidence_presence_value(right) in {"present", "uncertain"}
    )
    meaningful = (
        finding_relation == "same"
        or (
            finding_relation in {"differential", "same_family"}
            and both_positive_or_uncertain
            and (anatomy_similarity or 0.0) >= 0.35
        )
        or (
            allow_same_anatomy_diff
            and finding_relation == "different"
            and specific_same_anatomy
        )
    )
    if not meaningful:
        return None

    relation_types: list[RelationType] = []
    if finding_relation == "same":
        relation_types.append("same_finding")
        if presence_relation == "same_absent":
            relation_types.append("ruleout_absence_weak")
        elif presence_relation == "contrast":
            relation_types.append("presence_contrast")
        else:
            if anatomy_relation == "different":
                relation_types.append("anatomy_contrast")
            if laterality_relation == "different":
                relation_types.append("laterality_contrast")
            if severity_relation == "different":
                relation_types.append("severity_contrast")
            if uncertainty_relation == "different":
                relation_types.append("uncertainty_contrast")

            same_context = (
                anatomy_relation in {"same", "compatible", "unknown"}
                and laterality_relation
                in {"same", "compatible", "unknown", "not_applicable"}
                and severity_relation
                in {"same", "compatible", "unknown", "not_applicable"}
            )
            if same_context and presence_relation in {
                "same_present",
                "same_uncertain",
            }:
                if (
                    anatomy_relation == "same"
                    and laterality_relation in {"same", "not_applicable"}
                    and severity_relation in {"same", "compatible", "unknown", "not_applicable"}
                ):
                    relation_types.append("same_event")
                else:
                    relation_types.append("partial_same_event")
    elif finding_relation == "same_family":
        relation_types.append("same_family")
        if anatomy_relation in {"same", "compatible", "broad"}:
            relation_types.append("differential_same_anatomy")
    else:
        relation_types.append("differential_same_anatomy")

    relation_types = list(dict.fromkeys(relation_types))
    if not relation_types:
        return None

    finding_component = {
        "same": 1.0,
        "differential": 0.75,
        "same_family": 0.55,
        "different": 0.35 if specific_same_anatomy else 0.0,
    }[finding_relation]
    anatomy_component = 0.5 if anatomy_similarity is None else anatomy_similarity
    laterality_similarity = ontology.laterality_similarity(left.laterality, right.laterality)
    laterality_component = 0.5 if laterality_similarity is None else laterality_similarity
    severity_similarity = ontology.severity_similarity(left.severity, right.severity)
    severity_component = 0.5 if severity_similarity is None else severity_similarity
    uncertainty_similarity = ontology.uncertainty_similarity(left, right)
    uncertainty_component = 0.5 if uncertainty_similarity is None else uncertainty_similarity
    presence_component = {
        "same_present": 1.0,
        "same_uncertain": 0.8,
        "contrast": 0.7,
        "same_absent": 0.45,
        "unknown": 0.35,
    }[presence_relation]

    weights = cfg.get("event_relation_weights", {})
    score = (
        float(weights.get("finding", 0.30)) * finding_component
        + float(weights.get("anatomy", 0.20)) * anatomy_component
        + float(weights.get("laterality", 0.12)) * laterality_component
        + float(weights.get("presence", 0.16)) * presence_component
        + float(weights.get("severity", 0.10)) * severity_component
        + float(weights.get("uncertainty", 0.12)) * uncertainty_component
    )
    score *= sqrt(max(0.0, left.confidence) * max(0.0, right.confidence))
    if "same_event" in relation_types:
        score = max(score, 0.85 * min(left.confidence, right.confidence))
    elif "presence_contrast" in relation_types:
        score = max(score, 0.72 * min(left.confidence, right.confidence))
    elif "differential_same_anatomy" in relation_types:
        score = max(score, 0.62 * min(left.confidence, right.confidence))
    score = max(0.0, min(1.0, score))

    minimum = float(cfg.get("minimum_event_relation_score", 0.35))
    if score < minimum and finding_relation != "same":
        return None

    target_attributes: tuple[TargetAttribute, ...] = (
        "presence",
        "laterality",
        "severity",
        "anatomy",
        "finding_identity",
    )
    return MatchedEventRelation(
        source_event_id=left.event_id,
        target_event_id=right.event_id,
        source_finding=left.finding,
        target_finding=right.finding,
        relation_types=relation_types,
        eligible_target_attributes=_eligible_attributes(
            left,
            right,
            finding_relation=finding_relation,
            anatomy_relation=anatomy_relation,
        ),
        finding_relation=finding_relation,  # type: ignore[arg-type]
        anatomy_relation=anatomy_relation,  # type: ignore[arg-type]
        laterality_relation=laterality_relation,  # type: ignore[arg-type]
        presence_relation=presence_relation,  # type: ignore[arg-type]
        severity_relation=severity_relation,  # type: ignore[arg-type]
        uncertainty_relation=uncertainty_relation,  # type: ignore[arg-type]
        non_target_scores={
            attribute: _non_target_score(left, right, attribute, ontology)
            for attribute in target_attributes
        },
        relation_score=score,
    )


def relation_eligible_events(events: Iterable[Event], cfg: dict[str, Any]) -> list[Event]:
    minimum_confidence = float(cfg.get("minimum_event_confidence", 0.70))
    include_conditional = bool(cfg.get("include_conditional_events", False))
    require_semantic_support = bool(cfg.get("require_semantic_support", False))
    tolerated = {
        "section_corrected",
        "unsectioned_report",
        "context_only_finding",
        "absence_not_in_pilot_allowlist",
    }
    selected: list[Event] = []
    for event in events:
        if require_semantic_support and not has_verified_semantic_support(event):
            continue
        if event.confidence < minimum_confidence:
            continue
        if event.temporal_status == "prior_only" or event.presence == "not_mentioned":
            continue
        if event.retrieval_eligible:
            selected.append(event)
            continue
        if include_conditional and not (set(event.validation_errors) - tolerated):
            selected.append(event)
    return selected


def build_case_relation_edge(
    visual_edge: VisualEdge,
    source_events: Iterable[Event],
    target_events: Iterable[Event],
    ontology: Ontology,
    cfg: dict[str, Any],
    *,
    source_normality: str | None = None,
    target_normality: str | None = None,
) -> CaseRelationEdge:
    left_events = relation_eligible_events(source_events, cfg)
    right_events = relation_eligible_events(target_events, cfg)

    candidates: list[MatchedEventRelation] = []
    for left in left_events:
        for right in right_events:
            relation = build_event_relation(left, right, ontology, cfg)
            if relation is not None:
                candidates.append(relation)
    candidates.sort(
        key=lambda item: (
            item.relation_score,
            len(item.eligible_target_attributes),
            len(item.relation_types),
        ),
        reverse=True,
    )

    max_per_event = max(1, int(cfg.get("max_matches_per_event", 3)))
    max_per_edge = max(1, int(cfg.get("max_event_relations_per_edge", 64)))
    left_counts: defaultdict[str, int] = defaultdict(int)
    right_counts: defaultdict[str, int] = defaultdict(int)
    selected: list[MatchedEventRelation] = []
    for relation in candidates:
        if len(selected) >= max_per_edge:
            break
        if left_counts[relation.source_event_id] >= max_per_event:
            continue
        if right_counts[relation.target_event_id] >= max_per_event:
            continue
        selected.append(relation)
        left_counts[relation.source_event_id] += 1
        right_counts[relation.target_event_id] += 1

    # ``no_matched_events`` means that an eligible event has no valid relation
    # candidate at all on this case pair.  It must not be polluted by the
    # materialization caps below: an event with a valid candidate relation is
    # related even when a lower-scoring duplicate pair is omitted for storage.
    candidate_left = {item.source_event_id for item in candidates}
    candidate_right = {item.target_event_id for item in candidates}
    no_matched_events = {
        visual_edge.source_case_id: [
            event.event_id
            for event in left_events
            if event.event_id not in candidate_left
        ],
        visual_edge.target_case_id: [
            event.event_id
            for event in right_events
            if event.event_id not in candidate_right
        ],
    }

    relation_types: list[RelationType] = []
    for relation in selected:
        relation_types.extend(relation.relation_types)
    if any(no_matched_events.values()):
        relation_types.append("no_matched_event")
    clinical_far_threshold = float(cfg.get("clinical_far_relation_score", 0.25))
    best_score = max((item.relation_score for item in selected), default=0.0)
    if not selected or best_score < clinical_far_threshold:
        relation_types.append("visual_near_clinical_far")
    relation_types = list(dict.fromkeys(relation_types))

    relation_score = max((item.relation_score for item in selected), default=0.0)
    if not selected and any(no_matched_events.values()):
        relation_score = float(cfg.get("unmatched_only_relation_score", 0.20))

    known = {"normal", "abnormal"}
    source_normality = source_normality if source_normality in known else None
    target_normality = target_normality if target_normality in known else None
    normality_relation = CaseNormalityRelation(
        source_value=source_normality,
        target_value=target_normality,
        relation=(
            "same" if source_normality == target_normality else "different"
        ) if source_normality is not None and target_normality is not None else "unknown",
    )
    if normality_relation.relation == "same":
        relation_types.append("normality_same")
    elif normality_relation.relation == "different":
        relation_types.append("normality_contrast")

    return CaseRelationEdge(
        source_case_id=visual_edge.source_case_id,
        target_case_id=visual_edge.target_case_id,
        visual_similarity=visual_edge.similarity,
        source_rank=visual_edge.source_rank,
        is_mutual=visual_edge.is_mutual,
        is_backfill=visual_edge.is_backfill,
        relation_types=relation_types,
        matched_events=selected,
        no_matched_events=no_matched_events,
        relation_score=max(0.0, min(1.0, relation_score)),
        ontology_sha256=ontology.ontology_hash,
        relation_schema_version=RELATION_SCHEMA_VERSION,
        normality_relation=normality_relation,
    )
