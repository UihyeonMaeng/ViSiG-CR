"""Deterministic clinical comparison utilities used during online retrieval.

Shared by offline relation construction, online final checks, and evaluator-only
case annotations. They never modify extraction results or infer missing assertions.
"""

from __future__ import annotations

from typing import Iterable

from .ontology import Ontology
from .schemas import ClinicalTarget, ComparisonRole, Event, QuestionSpec, TargetAttribute


UNCERTAIN_ASSERTIONS = frozenset(
    {
        "probable",
        "possible",
        "cannot_exclude",
        "indeterminate",
        "unlikely",
    }
)
MISSING_ATTRIBUTE_VALUES = frozenset(
    {"", "unknown", "unspecified", "not_applicable", "not_mentioned"}
)


def attribute_value(event: Event, target_attribute: TargetAttribute | str) -> str | None:
    """Return the memory event's evidence-grounded value for one attribute."""

    target_attribute = str(target_attribute).strip().lower()
    if target_attribute == "presence":
        if event.presence == "absent":
            return "absent"
        if event.presence == "present" and event.uncertainty in UNCERTAIN_ASSERTIONS:
            return "uncertain"
        if event.presence == "present":
            return "present"
        return None

    if target_attribute == "laterality":
        if event.presence == "absent":
            return "none"
        if event.uncertainty in UNCERTAIN_ASSERTIONS:
            return "uncertain"
        return None if event.laterality in MISSING_ATTRIBUTE_VALUES else event.laterality

    if target_attribute == "severity":
        if event.presence != "present":
            return None
        if event.uncertainty in UNCERTAIN_ASSERTIONS:
            return "uncertain"
        return None if event.severity in MISSING_ATTRIBUTE_VALUES else event.severity

    if target_attribute == "anatomy":
        if event.presence != "present":
            return None
        return None if event.anatomy in MISSING_ATTRIBUTE_VALUES else event.anatomy

    if target_attribute == "finding_identity":
        # Finding-candidate questions may explicitly ask which finding is
        # absent or uncertain.  In that setting the answer value is still the
        # finding ID; polarity/certainty are enforced by ``ClinicalTarget`` in
        # ``target_context_score``.  Only a sparse ``not_mentioned`` row is not
        # a usable finding assertion.
        if event.presence == "not_mentioned":
            return None
        return None if event.finding in MISSING_ATTRIBUTE_VALUES else event.finding

    raise ValueError(f"Unsupported target attribute: {target_attribute}")


def _mean_observed(values: Iterable[float | None]) -> float:
    observed = [max(0.0, min(1.0, float(value))) for value in values if value is not None]
    return sum(observed) / len(observed) if observed else 0.0


def target_context_score(
    event: Event,
    target: ClinicalTarget,
    ontology: Ontology,
) -> float:
    """Match one event to question-defined context while masking the target attribute."""

    attribute = target.target_attribute
    checks: list[float | None] = []

    # Candidate predicates are conjunctions, not soft hints.  For example, in
    # "Which finding is absent?" a present pleural-effusion assertion must not
    # survive merely because its finding/anatomy fields match strongly.
    if (
        target.presence not in {"", "unspecified", "not_applicable"}
        and event.presence != target.presence
    ):
        return 0.0
    if (
        target.uncertainty not in {"", "unspecified", "not_applicable"}
        and event.uncertainty != target.uncertainty
    ):
        return 0.0

    if attribute != "finding_identity":
        checks.append(ontology.finding_similarity(event.finding, target.finding))
    else:
        # Finding identity is the value to compare, but a weak semantic relation to
        # the question concept prevents arbitrary same-anatomy findings from
        # entering the candidate pool.
        relation = ontology.finding_similarity(event.finding, target.finding)
        if ontology.is_differential_pair(event.finding, target.finding):
            relation = max(relation, 0.75)
        checks.append(relation)

    if attribute != "anatomy" and target.anatomy not in {
        "",
        "unspecified",
        "not_applicable",
    }:
        checks.append(ontology.anatomy_similarity(event.anatomy, target.anatomy))

    if attribute != "laterality" and target.laterality not in {
        "",
        "unspecified",
        "not_applicable",
    }:
        checks.append(ontology.laterality_similarity(event.laterality, target.laterality))

    if attribute != "presence":
        if target.presence not in {"", "unspecified", "not_applicable"}:
            checks.append(1.0)
        elif event.presence == "not_mentioned":
            checks.append(0.0)
        elif attribute in {"severity", "anatomy", "finding_identity", "laterality"}:
            checks.append(float(event.presence == "present"))

    if target.uncertainty not in {"", "unspecified", "not_applicable"}:
        checks.append(1.0)

    # An explicit incompatibility cannot be averaged away by matching fields.
    # Missing fields remain unknown (None), not negative assertions.
    if any(value is not None and value <= 0.0 for value in checks):
        return 0.0
    return _mean_observed(checks)


def best_target_match(
    event: Event,
    spec: QuestionSpec,
    ontology: Ontology,
) -> tuple[ClinicalTarget | None, float]:
    best_target: ClinicalTarget | None = None
    best_score = 0.0
    for target in spec.effective_targets():
        score = target_context_score(event, target, ontology) * max(0.0, target.score)
        if best_target is None or score > best_score:
            best_target = target
            best_score = score
    return best_target, max(0.0, min(1.0, best_score))


def pair_non_target_context_score(
    left: Event,
    right: Event,
    target_attribute: TargetAttribute | str,
    ontology: Ontology,
) -> float:
    """Match two memory events after masking the question-defined attribute."""

    target_attribute = str(target_attribute)
    checks: list[float | None] = []
    if target_attribute != "finding_identity":
        checks.append(ontology.finding_similarity(left.finding, right.finding))
    elif not ontology.is_differential_pair(left.finding, right.finding):
        # Same identity is not contrastive, but related finding families remain a
        # weak match for open differential questions.
        checks.append(ontology.finding_similarity(left.finding, right.finding))

    if target_attribute != "anatomy":
        checks.append(ontology.anatomy_similarity(left.anatomy, right.anatomy))

    if target_attribute != "laterality":
        checks.append(ontology.laterality_similarity(left.laterality, right.laterality))

    if target_attribute != "presence":
        if left.presence != "not_mentioned" and right.presence != "not_mentioned":
            checks.append(float(left.presence == right.presence))

    if target_attribute != "severity":
        checks.append(ontology.severity_similarity(left.severity, right.severity))

    # Finding identity is a non-target predicate for attribute-axis groups.
    if target_attribute != "finding_identity" and left.finding != right.finding:
        return 0.0
    if any(value is not None and value <= 0.0 for value in checks):
        return 0.0
    return _mean_observed(checks)


def comparison_role(value: str, spec: QuestionSpec) -> ComparisonRole:
    if value == "uncertain" or value in spec.optional_attribute_values:
        return "UNCERTAINTY_CALIBRATOR"
    if spec.target_attribute == "finding_identity":
        return "DIFFERENTIAL_COMPARATOR"
    return "VALUE_EXEMPLAR"


def online_relation_label(
    left_value: str,
    right_value: str,
    target_attribute: TargetAttribute | str,
) -> str:
    """Human-readable relation label computed for package provenance only."""

    if left_value == right_value:
        return f"matched_{target_attribute}_value"
    return f"different_{target_attribute}_values"
