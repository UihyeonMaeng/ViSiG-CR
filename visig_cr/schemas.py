"""Core data contracts for ViSiG-CR.

The contracts intentionally separate four concepts:
1. ontology events attached to memory cases,
2. the question-defined target clinical attribute,
3. the evidence-grounded value of each retrieved memory case, and
4. reader-visible semantic contrast group membership.

The canonical runtime store remains normalized at assertion level.  The public
JSONL representation groups assertions by finding under a per-case ``events``
mapping so one finding can safely own multiple laterality/location assertions.
Event-level clinical relations are materialized offline on visual-neighbor
case pairs. Each relation edge stores matched event pairs and a
``no_matched_events`` map, while question-time routing still validates the
selected events against the frozen ontology.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .io_utils import normalize_token


TargetAttribute = Literal[
    "presence",
    "laterality",
    "severity",
    "anatomy",
    "finding_identity",
]

CaseNormality = Literal["normal", "abnormal"]
TargetScope = Literal["finding", "case_normality"]

ComparisonRole = Literal[
    "VALUE_EXEMPLAR",
    "UNCERTAINTY_CALIBRATOR",
    "DIFFERENTIAL_COMPARATOR",
]

ComparisonMode = Literal["attribute_axis", "candidate_predicate"]
ReaderPresentation = Literal["visig_cr_structured", "retrieved_references"]
CandidateType = Literal[
    "attribute_value",
    "finding",
    "normal_state",
    "compound_predicate",
    "unsupported",
]

RelationType = Literal[
    "same_event",
    "partial_same_event",
    "same_finding",
    "same_family",
    "differential_same_anatomy",
    "presence_contrast",
    "laterality_contrast",
    "severity_contrast",
    "anatomy_contrast",
    "uncertainty_contrast",
    "ruleout_absence_weak",
    "no_matched_event",
    "visual_near_clinical_far",
    "normality_same",
    "normality_contrast",
]


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    dataset: str = "MIMIC-CXR"
    subject_id: str
    study_id: str
    dicom_id: str
    image_path: str
    report_path: str
    view: str
    split: str


class Event(BaseModel):
    """Normalized evidence-grounded assertion attached to one memory case.

    ``finding`` is retained in the normalized row because SQLite/index lookup is
    assertion-oriented.  In the public per-case JSON view it becomes the key in
    ``events`` and is therefore omitted from the nested assertion body.
    """

    model_config = ConfigDict(extra="ignore")

    case_id: str
    event_id: str
    finding: str
    anatomy: str = "unspecified"
    laterality: str = "unspecified"
    presence: Literal["present", "absent", "not_mentioned"]
    severity: str = "unspecified"
    uncertainty: str = "unspecified"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    characteristics: dict[str, Any] = Field(default_factory=dict)
    evidence_span: str = ""
    section: str = "other"
    start_char: int | None = None
    end_char: int | None = None
    temporal_status: str = "current"
    retrieval_eligible: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    @field_validator(
        "finding", "anatomy", "laterality", "severity", "uncertainty", "section"
    )
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return normalize_token(value)


class AssertionEvidence(BaseModel):
    """One exact report span supporting a nested finding assertion."""

    model_config = ConfigDict(extra="ignore")

    span: str = ""
    section: str = "other"
    start_char: int | None = None
    end_char: int | None = None

    @field_validator("section")
    @classmethod
    def normalize_section(cls, value: str) -> str:
        return normalize_token(value)


class EventAssertion(BaseModel):
    """Finding-keyed public representation of one normalized ``Event`` row."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    event_id: str
    anatomy: str = "unspecified"
    laterality: str = "unspecified"
    presence: Literal["present", "absent", "not_mentioned"]
    severity: str = "unspecified"
    characteristics: dict[str, Any] = Field(default_factory=dict)
    uncertainty: str = "unspecified"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[AssertionEvidence] = Field(default_factory=list)
    temporal_status: str = "current"
    retrieval_eligible: bool = True
    validation_errors: list[str] = Field(default_factory=list)

    @field_validator("anatomy", "laterality", "severity", "uncertainty")
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return normalize_token(value)

    @classmethod
    def from_event(cls, event: Event) -> "EventAssertion":
        evidence: list[AssertionEvidence] = []
        if (
            event.evidence_span
            or event.start_char is not None
            or event.end_char is not None
        ):
            evidence.append(
                AssertionEvidence(
                    span=event.evidence_span,
                    section=event.section,
                    start_char=event.start_char,
                    end_char=event.end_char,
                )
            )
        return cls(
            event_id=event.event_id,
            anatomy=event.anatomy,
            laterality=event.laterality,
            presence=event.presence,
            severity=event.severity,
            characteristics=event.characteristics,
            uncertainty=event.uncertainty,
            confidence=event.confidence,
            evidence=evidence,
            temporal_status=event.temporal_status,
            retrieval_eligible=event.retrieval_eligible,
            validation_errors=event.validation_errors,
        )

    def to_event(self, *, case_id: str, finding: str) -> Event:
        primary = self.evidence[0] if self.evidence else AssertionEvidence()
        return Event(
            case_id=case_id,
            event_id=self.event_id,
            finding=finding,
            anatomy=self.anatomy,
            laterality=self.laterality,
            presence=self.presence,
            severity=self.severity,
            uncertainty=self.uncertainty,
            confidence=self.confidence,
            characteristics=self.characteristics,
            evidence_span=primary.span,
            section=primary.section,
            start_char=primary.start_char,
            end_char=primary.end_char,
            temporal_status=self.temporal_status,
            retrieval_eligible=self.retrieval_eligible,
            validation_errors=self.validation_errors,
        )


class CaseEvents(BaseModel):
    """Per-case JSON representation with optional derived ``normality``.

    The v3 representation intentionally removes the redundant ``assertions``
    wrapper.  The validator remains backward compatible with v2.1 rows of the
    form ``events[finding].assertions`` and with ``assertion_id``.
    """

    model_config = ConfigDict(extra="ignore")

    case_id: str
    normality: CaseNormality | None = None
    events: dict[str, list[EventAssertion]] = Field(default_factory=dict)

    @field_validator("events", mode="before")
    @classmethod
    def normalize_finding_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized: dict[str, list[dict[str, Any]]] = {}
        for finding, raw_group in value.items():
            key = normalize_token(str(finding))
            if not key:
                continue
            if isinstance(raw_group, Mapping):
                raw_items = raw_group.get("assertions", [])
            else:
                raw_items = raw_group
            if not isinstance(raw_items, list):
                continue
            items: list[dict[str, Any]] = []
            for raw in raw_items:
                if isinstance(raw, EventAssertion):
                    item = raw.model_dump()
                elif isinstance(raw, Mapping):
                    item = dict(raw)
                else:
                    continue
                if "event_id" not in item and "assertion_id" in item:
                    item["event_id"] = item.pop("assertion_id")
                items.append(item)
            normalized.setdefault(key, []).extend(items)
        return normalized

    @classmethod
    def from_events(
        cls,
        case_id: str,
        events: Iterable[Event],
        *,
        normality: CaseNormality | None = None,
    ) -> "CaseEvents":
        grouped: dict[str, list[EventAssertion]] = defaultdict(list)
        for event in events:
            if event.case_id != case_id:
                raise ValueError(
                    f"Event {event.event_id!r} belongs to {event.case_id!r}, "
                    f"not requested case {case_id!r}"
                )
            grouped[event.finding].append(EventAssertion.from_event(event))
        return cls(
            case_id=case_id,
            normality=normality,
            events={
                finding: sorted(items, key=lambda item: item.event_id)
                for finding, items in sorted(grouped.items())
            },
        )

    def to_events(self) -> list[Event]:
        flattened: list[Event] = []
        for finding, items in self.events.items():
            flattened.extend(
                item.to_event(case_id=self.case_id, finding=finding)
                for item in items
            )
        flattened.sort(key=lambda event: event.event_id)
        return flattened


EventRecord = Event | CaseEvents | Mapping[str, Any]


def events_from_record(record: EventRecord) -> list[Event]:
    """Parse one legacy flat row or one finding-keyed per-case row."""

    if isinstance(record, Event):
        return [record]
    if isinstance(record, CaseEvents):
        return record.to_events()
    if not isinstance(record, Mapping):
        raise TypeError(f"Unsupported event record type: {type(record)!r}")
    data = dict(record)
    if isinstance(data.get("events"), Mapping):
        return CaseEvents.model_validate(data).to_events()
    return [Event.model_validate(data)]


def flatten_event_records(records: Iterable[EventRecord]) -> list[Event]:
    """Flatten mixed legacy/new event JSON records into normalized assertions."""

    flattened: list[Event] = []
    for record in records:
        flattened.extend(events_from_record(record))
    return flattened


class ClinicalTarget(BaseModel):
    """One text-selected ontology target used as a retrieval instruction."""

    finding: str
    anatomy: str = "unspecified"
    laterality: str = "unspecified"
    presence: str = "unspecified"
    uncertainty: str = "unspecified"
    target_attribute: TargetAttribute
    score: float = Field(default=1.0, ge=-1.0, le=1.0)
    source: Literal["rule", "medsiglip", "llm", "hierarchy_fallback"] = "rule"
    matched_text: str = ""

    @field_validator("finding", "anatomy", "laterality", "presence", "uncertainty")
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return normalize_token(value)


class AnswerCandidate(BaseModel):
    """One visible answer option mapped without choosing the correct answer."""

    model_config = ConfigDict(extra="ignore")

    option_id: str
    text: str
    candidate_type: CandidateType
    canonical_value: str | None = None
    finding: str | None = None
    anatomy: str = "unspecified"
    laterality: str = "unspecified"
    presence: str | None = None
    uncertainty: str | None = None
    severity: str | None = None
    mapping_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator(
        "canonical_value",
        "finding",
        "anatomy",
        "laterality",
        "presence",
        "uncertainty",
        "severity",
        mode="before",
    )
    @classmethod
    def normalize_optional_tokens(cls, value: Any) -> Any:
        if value is None:
            return None
        token = normalize_token(str(value))
        return token or None


class QuestionSpec(BaseModel):
    """Output of the Attribute Selector Agent.

    Required and optional values define what the retrieved *memory cases* should
    cover. They are not predictions of the query image label.  In
    ``candidate_predicate`` mode, all mapped options are retained symmetrically;
    no option is promoted because it happens to occur in a visual seed case.
    """

    model_config = ConfigDict(extra="ignore")

    target_finding: str = "unspecified"
    target_anatomy: str = "unspecified"
    target_laterality: str = "unspecified"
    target_attribute: TargetAttribute
    target_scope: TargetScope = "finding"
    answer_space: list[str]
    clinical_targets: list[ClinicalTarget] = Field(default_factory=list)
    required_attribute_values: list[str] = Field(default_factory=list)
    optional_attribute_values: list[str] = Field(default_factory=list)
    minimum_distinct_attribute_values: int = Field(default=2, ge=1, le=8)
    selector_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    abstain: bool = False
    comparison_mode: ComparisonMode = "attribute_axis"
    question_text: str = ""
    options: list[str] = Field(default_factory=list)
    answer_candidates: list[AnswerCandidate] = Field(default_factory=list)
    relation_priority: list[RelationType] = Field(default_factory=list)

    @field_validator("target_finding", "target_anatomy", "target_laterality")
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return normalize_token(value)

    @field_validator(
        "answer_space", "required_attribute_values", "optional_attribute_values"
    )
    @classmethod
    def normalize_lists(cls, values: list[str]) -> list[str]:
        return [normalize_token(str(value)) for value in values if str(value).strip()]

    def effective_targets(self) -> list[ClinicalTarget]:
        if self.clinical_targets:
            return self.clinical_targets
        if self.target_finding in {"", "unspecified", "not_applicable"}:
            return []
        return [
            ClinicalTarget(
                finding=self.target_finding,
                anatomy=self.target_anatomy,
                laterality=self.target_laterality,
                target_attribute=self.target_attribute,
                score=self.selector_confidence,
                source="rule",
            )
        ]

    def option_ids_for_value(self, value: str) -> list[str]:
        value = normalize_token(value)
        return [
            candidate.option_id
            for candidate in self.answer_candidates
            if candidate.canonical_value == value
        ]


class VisualEdge(BaseModel):
    source_case_id: str
    target_case_id: str
    similarity: float
    source_rank: int = Field(ge=1)
    is_mutual: bool = False
    is_backfill: bool = False


class MatchedEventRelation(BaseModel):
    """Atomic relation between one source event and one target event."""

    model_config = ConfigDict(extra="ignore")

    source_event_id: str
    target_event_id: str
    source_finding: str
    target_finding: str
    relation_types: list[RelationType] = Field(default_factory=list)
    eligible_target_attributes: list[TargetAttribute] = Field(default_factory=list)
    finding_relation: Literal["same", "same_family", "differential", "different"]
    anatomy_relation: Literal["same", "compatible", "broad", "different", "unknown"]
    laterality_relation: Literal["same", "compatible", "different", "unknown", "not_applicable"]
    presence_relation: Literal[
        "same_present",
        "same_absent",
        "same_uncertain",
        "contrast",
        "unknown",
    ]
    severity_relation: Literal["same", "compatible", "different", "unknown", "not_applicable"]
    uncertainty_relation: Literal["same", "compatible", "different", "unknown"]
    non_target_scores: dict[str, float] = Field(default_factory=dict)
    relation_score: float = Field(default=0.0, ge=0.0, le=1.0)


class CaseNormalityRelation(BaseModel):
    """Case-level closed-world proxy; independent of event-pair assertions."""

    source_value: CaseNormality | None = None
    target_value: CaseNormality | None = None
    relation: Literal["same", "different", "unknown"] = "unknown"
    policy_id: str = "clinical_present_else_normal_v1"
    is_clinical_normality_reference: bool = False


class CaseRelationEdge(BaseModel):
    """Relation annotation attached to one directed visual-neighbor edge.

    ``matched_events`` stores event-pair relations. ``no_matched_events``
    preserves events unique to either endpoint using the exact case IDs as keys.
    """

    model_config = ConfigDict(extra="ignore")

    source_case_id: str
    target_case_id: str
    visual_similarity: float
    source_rank: int = Field(ge=1)
    is_mutual: bool = False
    is_backfill: bool = False
    relation_types: list[RelationType] = Field(default_factory=list)
    matched_events: list[MatchedEventRelation] = Field(default_factory=list)
    no_matched_events: dict[str, list[str]] = Field(default_factory=dict)
    relation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ontology_sha256: str = ""
    relation_schema_version: str = "visig_cr_relation_v2.0"
    normality_relation: CaseNormalityRelation = Field(default_factory=CaseNormalityRelation)

    def target_unmatched_event_ids(self) -> list[str]:
        return list(self.no_matched_events.get(self.target_case_id, []))

    def source_unmatched_event_ids(self) -> list[str]:
        return list(self.no_matched_events.get(self.source_case_id, []))


class VisualPath(BaseModel):
    seed_case_id: str
    target_case_id: str
    nodes: list[str]
    hop_count: int = Field(ge=0, le=3)
    bottleneck_similarity: float
    mean_similarity: float
    path_score: float


class SemanticContrastGroup(BaseModel):
    """Reader-visible matched group spanning distinct target-attribute values."""

    model_config = ConfigDict(extra="ignore")

    group_id: str
    target_attribute: TargetAttribute
    member_case_ids: list[str]
    member_event_ids: list[str]
    member_attribute_values: list[str]
    member_comparison_roles: list[ComparisonRole]
    required_values_covered: list[str] = Field(default_factory=list)
    group_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    visual_locality: float = Field(default=0.0, ge=-1.0, le=1.0)
    non_target_match: float | None = Field(default=0.0, ge=0.0, le=1.0)
    relation_types: list[RelationType] = Field(default_factory=list)
    relation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    # Audit-only provenance for how the pair was constructed. The reader does
    # not render this field, so a bounded retrieval fallback cannot become an
    # answer hint. A default keeps previously generated packages loadable.
    pair_source: Literal[
        "stored_edge",
        "candidate_pair_fallback",
        "all_pairs_control",
        "question_conditioned_pool",
    ] = "stored_edge"
    # Retrieval-only diagnostics. They are persisted for validation/calibration,
    # but the reader renderer deliberately does not expose them. This prevents a
    # similarity-derived preference from becoming a hidden answer hint.
    query_endpoint_margin: float = Field(default=0.0, ge=0.0, le=1.0)
    directional_score: float = Field(default=0.0, ge=0.0, le=1.0)
    query_preferred_attribute_value: str = ""

    @field_validator("member_attribute_values", "required_values_covered")
    @classmethod
    def normalize_values(cls, values: list[str]) -> list[str]:
        return [normalize_token(str(value)) for value in values if str(value).strip()]

    @field_validator("query_preferred_attribute_value")
    @classmethod
    def normalize_preferred_value(cls, value: str) -> str:
        return normalize_token(value) if str(value).strip() else ""


class Evidence(BaseModel):
    """One memory case selected from the query-defined visual boundary."""

    model_config = ConfigDict(extra="ignore")

    case_id: str
    event_id: str
    image_path: str
    seed_case_id: str
    visual_similarity_query: float
    visual_similarity_seed: float
    boundary_path: list[str]
    boundary_hop_count: int = Field(default=0, ge=0, le=3)
    boundary_path_score: float = 0.0
    relation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relation_types: list[RelationType] = Field(default_factory=list)
    relation_event_ids: list[str] = Field(default_factory=list)
    used_no_matched_event_ids: list[str] = Field(default_factory=list)
    target_match: float = Field(default=0.0, ge=0.0, le=1.0)
    non_target_match: float | None = Field(default=0.0, ge=0.0, le=1.0)
    event_confidence: float | None = Field(default=0.0, ge=0.0, le=1.0)
    candidate_score: float = 0.0
    # Published comparison retrievers operate on their method-native relevance
    # signals and do not select an ontology attribute. Structured ViSiG-CR
    # evidence continues to populate this field with a TargetAttribute.
    target_attribute: TargetAttribute | None = None
    attribute_value: str
    comparison_role: ComparisonRole
    event: dict[str, Any]
    evidence_span: str
    # Generic comparison baselines must not inherit ViSiG-CR's ontology/event
    # presentation merely to satisfy the shared package schema.  These fields
    # carry exactly the reference material and retrieval signals that a
    # published baseline is allowed to expose to the frozen reader.
    reference_text: str = ""
    retrieval_signals: dict[str, float] = Field(default_factory=dict)
    reader_visible_image: bool = True
    matched_option_ids: list[str] = Field(default_factory=list)
    semantic_contrast_group_id: str | None = None
    group_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    selection_gain: float = 0.0

    @field_validator("attribute_value")
    @classmethod
    def normalize_attribute_value(cls, value: str) -> str:
        return normalize_token(value)


class EvidencePackage(BaseModel):
    """Frozen reader input for either a RAG or pure no-RAG condition.

    ``audit_metadata`` is persisted for stratified evaluation but is never added
    to a model message.  A no-RAG package is deliberately unable to carry a
    QuestionSpec, coverage status, seeds, graph paths, or retrieved evidence.
    Baseline-native ``retrieved_references`` packages may omit QuestionSpec
    because they do not use ViSiG-CR ontology query interpretation.
    """

    model_config = ConfigDict(extra="ignore")

    input_mode: Literal["rag", "no_rag"] = "rag"
    reader_presentation: ReaderPresentation = "visig_cr_structured"
    query_id: str
    query_image_path: str
    question: str
    retrieval_question: str = ""
    # Optional baseline-native query context (for example MOTOR's offline
    # MAIRA-2 grounded query caption). It is rendered only by the generic
    # retrieved-reference presentation.
    query_context: str = ""
    options: list[str] = Field(default_factory=list)
    question_spec: QuestionSpec | None = None
    method: str
    seed_cases: list[dict[str, Any]] = Field(default_factory=list)
    boundary_stats: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    semantic_contrast_groups: list[SemanticContrastGroup] = Field(default_factory=list)
    coverage_status: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    audit_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_input_mode(self) -> "EvidencePackage":
        if self.input_mode == "no_rag":
            if self.method != "no_rag":
                raise ValueError("input_mode=no_rag requires method=no_rag")
            if self.question_spec is not None:
                raise ValueError("no-RAG packages must not contain QuestionSpec")
            if self.seed_cases or self.evidence or self.semantic_contrast_groups:
                raise ValueError("no-RAG packages must not contain retrieval evidence")
            if self.query_context:
                raise ValueError("no-RAG packages must not contain baseline query context")
            if self.boundary_stats or self.coverage_status or self.warnings:
                raise ValueError(
                    "no-RAG packages must not contain graph, coverage, or retrieval state"
                )
        elif self.reader_presentation == "visig_cr_structured":
            if self.question_spec is None:
                raise ValueError("Structured ViSiG-CR RAG packages require question_spec")
            if any(item.target_attribute is None for item in self.evidence):
                raise ValueError(
                    "Structured ViSiG-CR evidence requires target_attribute"
                )
        return self
