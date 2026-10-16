"""Post-extraction normality provenance. Never mutates or re-extracts events."""
from __future__ import annotations

from typing import Any, Sequence

from .io_utils import stable_json_sha256
from .ontology import Ontology
from .schemas import Event

NORMALITY_POLICY_ID = "clinical_present_else_normal_v1"
CLINICAL_NORMALITY_REFERENCE_POLICY_ID = "scope_safe_event_supported_normality_v2"
# No current canonical descriptor establishes absence of all clinical findings.
# In particular, no acute cardiopulmonary abnormality has a narrower scope.
DEFAULT_GLOBAL_NORMAL_FINDINGS: frozenset[str] = frozenset()
ACTIVE_CLINICAL_TEMPORAL_STATUSES = frozenset(
    {"current", "new", "increased", "decreased", "stable"}
)
BROAD_NORMALITY_ASSERTION_SCOPE = "any_abnormality"
NARROW_NORMALITY_FINDINGS = frozenset({
    "no_acute_cardiopulmonary_abnormality", "normal_lung_transparency",
    "normal_cardiac_silhouette", "normal_hilar_contour", "intact_osseous_structures",
})


def derive_normality_metadata(
    events: Sequence[Event], ontology: Ontology, abnormal_findings: Any = "clinical"
) -> tuple[str, dict[str, Any]]:
    """Retain the registered binary definition; expose its closed-world limits.

    Possible/probable current present findings count as abnormal, as before.
    Empty extraction => normal *proxy*, not report-confirmed clinical normality.
    No synthetic event, confidence, or source span is created.
    """
    findings = ontology.resolve_abnormality_findings(abnormal_findings)
    eligible = [event for event in events if event.finding in findings
                and event.temporal_status == "current" and event.retrieval_eligible]
    positive_ids = sorted(event.event_id for event in eligible if event.presence == "present")
    value = "abnormal" if positive_ids else "normal"
    return value, {
        "policy_id": NORMALITY_POLICY_ID,
        "definition_scope": "closed_world_extracted_event_proxy",
        "ontology_sha256": ontology.ontology_hash,
        "abnormal_findings_sha256": stable_json_sha256(sorted(findings)),
        "supporting_present_event_ids": positive_ids,
        "eligible_current_event_count": len(eligible),
        "empty_event_record": not bool(events),
        "is_clinical_normality_reference": False,
    }


def scope_safe_normal_findings(
    ontology: Ontology,
    explicit_normal_findings: Sequence[str] | None = None,
) -> frozenset[str]:
    """Resolve configured descriptors that can prove broad clinical normality.

    An empty result means a complete case-normality contrast cannot be built
    under this evidence policy. Callers may avoid scanning the candidate pool
    in that case. Configuration cannot widen an established descriptor's scope.
    """
    requested = {
        str(value).strip()
        for value in (
            explicit_normal_findings
            if explicit_normal_findings is not None
            else DEFAULT_GLOBAL_NORMAL_FINDINGS
        )
        if str(value).strip()
    }
    return frozenset(
        finding
        for finding in requested & set(ontology.extractable_findings)
        if finding not in NARROW_NORMALITY_FINDINGS
        and ontology.is_ancestor("normal_descriptor", finding, "finding")
        and ontology.retrieval_policy(finding) != "clinical"
        and ontology.finding_spec(finding).get("normality_scope")
        == BROAD_NORMALITY_ASSERTION_SCOPE
    )


def derive_clinical_normality_reference(
    events: Sequence[Event],
    ontology: Ontology,
    abnormal_findings: Any = "clinical",
    *,
    explicit_normal_findings: Sequence[str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Build a reader-safe normality exemplar from already extracted events.

    The durable case-level ``normality`` field intentionally remains the broad
    closed-world graph label used by existing artifacts. Reader exemplars need
    a stronger contract: an abnormal case must have an existing positive clinical
    event, while a normal case must have an explicit descriptor covering every
    clinical abnormality. Stable or changing findings still exist in the image;
    resolved or prior-only findings do not support its positive endpoint.
    Cases with neither form of evidence return ``None`` instead of silently
    turning extraction absence into clinical normality.

    A configured descriptor cannot widen its ontology assertion scope. Future
    broad normal descriptors must explicitly declare
    ``normality_scope: any_abnormality`` in their ontology finding metadata.
    A broad negative endpoint additionally requires a definite assertion and a
    nonempty supporting report span. The existing no-acute descriptor is
    deliberately rejected, irrespective of its report span or configuration. This helper consumes the existing event
    store and never invokes extraction or changes durable graph labels.
    """

    abnormal_set = ontology.resolve_abnormality_findings(abnormal_findings)
    requested_normal = {
        str(value).strip()
        for value in (
            explicit_normal_findings
            if explicit_normal_findings is not None
            else DEFAULT_GLOBAL_NORMAL_FINDINGS
        )
        if str(value).strip()
    }
    known_requested_normal = requested_normal & set(ontology.extractable_findings)
    known_normal = scope_safe_normal_findings(
        ontology, explicit_normal_findings=sorted(requested_normal)
    )
    narrow_normal_findings = {
        finding
        for finding in ontology.extractable_findings
        if ontology.is_ancestor("normal_descriptor", finding, "finding")
        and finding not in known_normal
    }

    active = [
        event for event in events
        if event.temporal_status in ACTIVE_CLINICAL_TEMPORAL_STATUSES
    ]
    abnormal_events = [
        event
        for event in active
        if event.finding in abnormal_set
        and event.presence == "present"
        and event.retrieval_eligible
    ]
    explicit_normal_events = [
        event
        for event in active
        if event.finding in known_normal
        and event.presence == "present"
        and event.uncertainty == "definite"
        and bool(event.evidence_span.strip())
    ]

    ignored_narrow_normal_events = [
        event for event in active
        if event.finding in narrow_normal_findings and event.presence == "present"
    ]

    # A directly asserted abnormal finding overrides a simultaneously extracted
    # generic-normal descriptor. The conflict remains visible for audit.
    if abnormal_events:
        value: str | None = "abnormal"
        supporting = abnormal_events
        basis = "positive_clinical_event"
    elif explicit_normal_events:
        value = "normal"
        supporting = explicit_normal_events
        basis = "explicit_broad_normal_descriptor"
    else:
        value = None
        supporting = []
        basis = (
            "insufficient_assertion_scope"
            if ignored_narrow_normal_events
            else "insufficient_explicit_evidence"
        )

    return value, {
        "policy_id": CLINICAL_NORMALITY_REFERENCE_POLICY_ID,
        "definition_scope": "explicit_existing_ontology_events",
        "assertion_scope": BROAD_NORMALITY_ASSERTION_SCOPE,
        "active_temporal_statuses": sorted(ACTIVE_CLINICAL_TEMPORAL_STATUSES),
        "ontology_sha256": ontology.ontology_hash,
        "abnormal_findings_sha256": stable_json_sha256(sorted(abnormal_set)),
        "explicit_normal_findings": sorted(known_normal),
        "rejected_explicit_normal_findings": sorted(
            known_requested_normal - known_normal
        ),
        "ignored_narrow_normal_findings": sorted(
            {event.finding for event in ignored_narrow_normal_events}
        ),
        "ignored_narrow_normal_event_ids": sorted(
            event.event_id for event in ignored_narrow_normal_events
        ),
        "reference_basis": basis,
        "supporting_event_ids": sorted(event.event_id for event in supporting),
        "supporting_event_assertions": [
            {
                "finding": event.finding,
                "anatomy": event.anatomy,
                "laterality": event.laterality,
                "presence": event.presence,
                "uncertainty": event.uncertainty,
                "temporal_status": event.temporal_status,
                "evidence_span": event.evidence_span,
            }
            for event in supporting
        ],
        "supporting_findings": sorted({event.finding for event in supporting}),
        "supporting_temporal_statuses": sorted(
            {event.temporal_status for event in supporting}
        ),
        "supporting_evidence_spans": [
            event.evidence_span for event in supporting if event.evidence_span.strip()
        ],
        "conflicting_explicit_normal_event_ids": sorted(
            event.event_id for event in explicit_normal_events
        )
        if abnormal_events
        else [],
        "is_clinical_normality_reference": value is not None,
    }


__all__ = [
    "ACTIVE_CLINICAL_TEMPORAL_STATUSES",
    "BROAD_NORMALITY_ASSERTION_SCOPE",
    "CLINICAL_NORMALITY_REFERENCE_POLICY_ID",
    "DEFAULT_GLOBAL_NORMAL_FINDINGS",
    "NORMALITY_POLICY_ID",
    "derive_clinical_normality_reference",
    "derive_normality_metadata",
    "scope_safe_normal_findings",
]
