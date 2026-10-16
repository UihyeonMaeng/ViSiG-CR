"""Question-conditioned routing over offline relation-annotated visual edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..graph.store import GraphStore
from ..ontology import Ontology
from ..schemas import CaseRelationEdge, Event, QuestionSpec, RelationType, TargetAttribute


DEFAULT_RELATION_PRIORITIES: dict[str, list[RelationType]] = {
    "presence": [
        "presence_contrast",
        "uncertainty_contrast",
        "same_event",
        "partial_same_event",
        "same_finding",
        "no_matched_event",
        "ruleout_absence_weak",
        "visual_near_clinical_far",
    ],
    "laterality": [
        "laterality_contrast",
        "uncertainty_contrast",
        "same_event",
        "partial_same_event",
        "same_finding",
        "no_matched_event",
        "visual_near_clinical_far",
    ],
    "severity": [
        "severity_contrast",
        "uncertainty_contrast",
        "same_event",
        "partial_same_event",
        "same_finding",
        "no_matched_event",
    ],
    "anatomy": [
        "anatomy_contrast",
        "same_event",
        "partial_same_event",
        "same_finding",
        "no_matched_event",
    ],
    "finding_identity": [
        "differential_same_anatomy",
        "same_family",
        "no_matched_event",
        "same_event",
        "partial_same_event",
        "visual_near_clinical_far",
    ],
}


def default_relation_priority(
    target_attribute: TargetAttribute | str,
    *,
    comparison_mode: str = "attribute_axis",
) -> list[RelationType]:
    values = list(DEFAULT_RELATION_PRIORITIES[str(target_attribute)])
    if comparison_mode == "candidate_predicate" and "no_matched_event" in values:
        values.remove("no_matched_event")
        values.insert(0, "no_matched_event")
    return values


@dataclass(frozen=True)
class RelationRoute:
    score: float
    relation_types: tuple[str, ...]
    target_event_ids: tuple[str, ...]
    no_matched_target_event_ids: tuple[str, ...]


class RelationRouter:
    """Score a relation edge for one frozen QuestionSpec.

    The Attribute Selector may expose a validated ``relation_priority`` list,
    but it never invents new relation labels.  This router maps that list to
    deterministic edge scores and identifies the target-side events that should
    be inspected during final validation.
    """

    def __init__(self, graph: GraphStore, ontology: Ontology):
        self.graph = graph
        self.ontology = ontology

    def priorities(self, spec: QuestionSpec) -> list[RelationType]:
        if spec.target_scope == "case_normality":
            return ["normality_contrast", "normality_same"]
        return list(
            spec.relation_priority
            or default_relation_priority(
                spec.target_attribute,
                comparison_mode=spec.comparison_mode,
            )
        )

    @staticmethod
    def _priority_score(relation_type: str, priorities: list[RelationType]) -> float:
        try:
            index = priorities.index(relation_type)  # type: ignore[arg-type]
        except ValueError:
            return 0.0
        return max(0.05, 1.0 - index / max(1, len(priorities)))

    @staticmethod
    def _observed(value: str) -> bool:
        return value not in {"", "unknown", "unspecified", "not_applicable", "not_mentioned"}

    def _candidate_findings(self, spec: QuestionSpec) -> set[str]:
        findings = {
            target.finding
            for target in spec.effective_targets()
            if self._observed(target.finding)
        }
        if spec.target_attribute == "finding_identity":
            findings.update(
                value
                for value in spec.answer_space
                if value in self.ontology.findings
            )
        return findings

    def _event_relevant(self, event: Event, spec: QuestionSpec) -> bool:
        findings = self._candidate_findings(spec)
        if findings:
            if event.finding not in findings and not any(
                self.ontology.finding_similarity(event.finding, finding) >= 0.4
                for finding in findings
            ):
                return False
        if self._observed(spec.target_anatomy):
            similarity = self.ontology.anatomy_similarity(
                event.anatomy,
                spec.target_anatomy,
            )
            if similarity is not None and similarity <= 0.0:
                return False
        if (
            spec.target_attribute != "laterality"
            and self._observed(spec.target_laterality)
        ):
            similarity = self.ontology.laterality_similarity(
                event.laterality,
                spec.target_laterality,
            )
            if similarity is not None and similarity <= 0.0:
                return False
        return True

    def score_edge(self, edge: CaseRelationEdge, spec: QuestionSpec) -> RelationRoute:
        if spec.target_scope == "case_normality":
            normality = edge.normality_relation
            if normality.relation == "unknown":
                return RelationRoute(0.0, (), (), ())
            label = "normality_contrast" if normality.relation == "different" else "normality_same"
            priority = 1.0 if normality.relation == "different" else 0.65
            score = 0.8 * priority + 0.2 * max(0.0, min(1.0, edge.visual_similarity))
            return RelationRoute(score, (label,), (), ())
        priorities = self.priorities(spec)
        relevant_types: list[str] = []
        target_event_ids: list[str] = []
        no_matched_ids: list[str] = []
        best_priority = 0.0

        for relation in edge.matched_events:
            try:
                target_event = self.graph.get_event(relation.target_event_id)
            except KeyError:
                continue
            if not self._event_relevant(target_event, spec):
                continue
            if (
                relation.eligible_target_attributes
                and spec.target_attribute not in relation.eligible_target_attributes
                and spec.target_attribute != "finding_identity"
            ):
                continue
            relation_priority = 0.0
            for relation_type in relation.relation_types:
                score = self._priority_score(relation_type, priorities)
                if score > 0.0:
                    relevant_types.append(relation_type)
                    relation_priority = max(relation_priority, score)
            if relation_priority > 0.0:
                target_event_ids.append(relation.target_event_id)
                best_priority = max(
                    best_priority,
                    relation_priority * max(0.1, relation.relation_score),
                )

        unmatched_priority = self._priority_score("no_matched_event", priorities)
        if unmatched_priority > 0.0:
            for event_id in edge.target_unmatched_event_ids():
                try:
                    event = self.graph.get_event(event_id)
                except KeyError:
                    continue
                if self._event_relevant(event, spec):
                    no_matched_ids.append(event_id)
                    target_event_ids.append(event_id)
                    relevant_types.append("no_matched_event")
                    best_priority = max(best_priority, unmatched_priority)

        # Visual fallback is handled explicitly by the retriever, never here.
        visual = max(0.0, min(1.0, edge.visual_similarity))
        relation = max(0.0, min(1.0, edge.relation_score))
        score = (0.55 * best_priority + 0.25 * relation + 0.20 * visual) if best_priority > 0.0 else 0.0
        return RelationRoute(
            score=max(0.0, min(1.0, score)),
            relation_types=tuple(dict.fromkeys(relevant_types)),
            target_event_ids=tuple(dict.fromkeys(target_event_ids)),
            no_matched_target_event_ids=tuple(dict.fromkeys(no_matched_ids)),
        )


def sanitize_relation_priority(
    values: Iterable[str] | None,
    target_attribute: TargetAttribute | str,
    *,
    comparison_mode: str = "attribute_axis",
) -> list[RelationType]:
    allowed = {
        relation
        for priorities in DEFAULT_RELATION_PRIORITIES.values()
        for relation in priorities
    }
    normalized: list[RelationType] = []
    for value in values or []:
        token = str(value).strip().lower()
        if token in allowed and token not in normalized:
            normalized.append(token)  # type: ignore[arg-type]
    defaults = default_relation_priority(
        target_attribute,
        comparison_mode=comparison_mode,
    )
    for value in defaults:
        if value not in normalized:
            normalized.append(value)
    return normalized
