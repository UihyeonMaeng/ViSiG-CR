"""Ontology Extractor Agent and report-grounded event validation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import re
from typing import Any

from pydantic import ValidationError

from .clients import LocalChatClient, ModelResponseParseError
from .io_utils import normalize_token, stable_json_sha256
from .ontology import Ontology
from .prompts import ONTOLOGY_EXTRACTOR_SYSTEM_PROMPT
from .schemas import CaseRecord, Event


SECTION_RE = re.compile(
    r"(?im)^[ \t]*(?P<header>"
    # Standard current-study sections
    r"findings?(?:\s*(?:and|/|&)\s*impressions?)?|"
    r"impressions?|conclusions?|"
    # Prior / metadata sections
    r"comparisons?|"
    r"clinical\s+history|clinical\s+information|clinical\s+indication|"
    r"history|indication|reason\s+for\s+exam(?:ination)?|"
    r"technique|examination|exam|study|type\s+of\s+examination|"
    # MIMIC-CXR current-study aliases
    r"(?:pa|ap|frontal)\s+and\s+lateral\s+chest\s+radiographs?|"
    r"(?:pa|ap|frontal)\s+and\s+lateral\s+views?\s+of\s+the\s+chest|"
    r"chest,\s*(?:pa|ap)\s+and\s+lateral|"
    r"two\s+views?\s+of\s+the\s+chest|"
    r"frontal\s+supine\s+chest\s+radiographs?|"
    r"portable\s+frontal\s+views?\s+of\s+the\s+chest|"
    r"chest,\s*ap\s+semi-erect|"
    r"chest,\s*single\s+ap\s+portable\s+view|"
    r"today'?s\s+chest\s+radiographs?|"
    # Sections that should terminate findings/impression
    r"recommendations?(?:\(s\))?|notification|addendum"
    r")[ \t]*:[ \t]*"
)

SECTION_HEADER_MAP = {
    "finding": "findings",
    "findings": "findings",
    "finding_and_impression": "impression",
    "findings_and_impression": "impression",
    "finding_impression": "impression",
    "findings_impression": "impression",
    "finding_impressions": "impression",
    "findings_impressions": "impression",
    "impression": "impression",
    "impressions": "impression",
    "conclusion": "impression",
    "conclusions": "impression",
    "comparison": "comparison",
    "clinical_history": "history",
    "history": "history",
    "indication": "history",
    "reason_for_exam": "history",
    "technique": "technique",
    "examination": "technique",
    "exam": "technique",
    # Metadata aliases
    "comparisons": "comparison",
    "clinical_information": "history",
    "clinical_indication": "history",
    "reason_for_examination": "history",
    "study": "technique",
    "type_of_examination": "technique",
    # Current-study FINDINGS aliases
    "pa_and_lateral_chest_radiograph": "findings",
    "pa_and_lateral_chest_radiographs": "findings",
    "ap_and_lateral_chest_radiograph": "findings",
    "ap_and_lateral_chest_radiographs": "findings",
    "frontal_and_lateral_chest_radiograph": "findings",
    "frontal_and_lateral_chest_radiographs": "findings",
    "pa_and_lateral_view_of_the_chest": "findings",
    "pa_and_lateral_views_of_the_chest": "findings",
    "ap_and_lateral_view_of_the_chest": "findings",
    "ap_and_lateral_views_of_the_chest": "findings",
    "frontal_and_lateral_view_of_the_chest": "findings",
    "frontal_and_lateral_views_of_the_chest": "findings",
    "chest_pa_and_lateral": "findings",
    "chest_ap_and_lateral": "findings",
    "two_view_of_the_chest": "findings",
    "two_views_of_the_chest": "findings",
    "frontal_supine_chest_radiograph": "findings",
    "frontal_supine_chest_radiographs": "findings",
    "portable_frontal_view_of_the_chest": "findings",
    "portable_frontal_views_of_the_chest": "findings",
    "chest_ap_semi_erect": "findings",
    "chest_single_ap_portable_view": "findings",
    "todays_chest_radiograph": "findings",
    "todays_chest_radiographs": "findings",
    "today_s_chest_radiograph": "findings",
    "today_s_chest_radiographs": "findings",
    # Stop sections
    "recommendation": "other",
    "recommendations": "other",
    "recommendation_s": "other",
    "notification": "other",
    "addendum": "other",
}

CLAUSE_RE = re.compile(r"[^.!?;]+(?:[.!?;]+|$)", flags=re.MULTILINE)
PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n+")
WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
NEGATION_PREFIX_RE = re.compile(
    r"\b(?:no\s+evidence\s+of|no|without|absence\s+of|absent|free\s+of|"
    r"clear(?:ed)?\s+of|negative\s+for|neither|resolution\s+of|removal\s+of)\b",
    flags=re.IGNORECASE,
)
NEGATION_SUFFIX_RE = re.compile(
    r"\b(?:"
    r"(?:is|are|was|were|remains?|appear(?:s|ed)?)?\s*"
    r"not\s+(?:seen|identified|visualized|demonstrated|present|evident)|"
    r"(?:is|are|was|were|has|have|had)?\s*(?:absent|resolved|removed|discontinued)|"
    r"no\s+longer\s+(?:seen|present|visualized|identified)"
    r")\b",
    flags=re.IGNORECASE,
)
NON_ABSENCE_CHANGE_RE = re.compile(
    r"\bno\s+(?:(?:significant|substantial|appreciable)\s+)?"
    r"(?:interval\s+)?(?:change|increase|decrease|worsening|progression|improvement)\b",
    flags=re.IGNORECASE,
)
CONTRAST_BOUNDARY_RE = re.compile(
    r"\b(?:but|however|although|though|except|yet)\b",
    flags=re.IGNORECASE,
)
HISTORICAL_PREFIX_RE = re.compile(
    r"\b(?:history\s+of|remote|prior|previous|previously\s+seen)\b",
    flags=re.IGNORECASE,
)
NON_ASSERTIVE_FINDING_CONTEXT_RE = re.compile(
    r"(?:"
    r"\bif\b[^.!?;]{0,180}\b(?:concern|suspicion)\s+(?:for|of)\b[^.!?;]{0,180}|"
    r"\b(?:continued|further|persistent|strong|clinical)\s+"
    r"(?:concern|suspicion)\s+(?:for|of)\b[^.!?;]{0,180}|"
    r"\b(?:recommend(?:ed)?|consider|suggest(?:ed)?|repeat(?:ed)?|"
    r"obtain(?:ed)?|perform(?:ed)?|follow[- ]?up|followed)\b"
    r"[^.!?;]{0,180}\b(?:to|for)\s+"
    r"(?:exclude|evaluate|assess|rule\s+out)\b[^.!?;]{0,180}|"
    r"\bevaluation\s+(?:for|of)\b[^.!?;]{0,180}\b"
    r"(?:is|was|remains?)\s+limited\b|"
    r"\bnot\s+sensitive\s+for\s+(?:the\s+)?"
    r"(?:detection|evaluation)\s+of\b[^.!?;]{0,180}|"
    r"\b(?:not\s+designed|limited)\s+(?:for|in)\s+(?:the\s+)?"
    r"evaluation\s+of\b[^.!?;]{0,180}"
    r")",
    flags=re.IGNORECASE,
)

PLEURAL_EFFUSION_CONFOUNDER_RE = re.compile(
    r"\bpericardial\s+effusions?\b",
    flags=re.IGNORECASE,
)


def is_non_assertive_finding_context(
    span: str,
    anchor_start: int,
    anchor_end: int,
) -> bool:
    """Return whether the finding anchor is inside a hypothetical/recommendation cue.

    The regex match must overlap the actual finding anchor. This prevents a valid
    positive finding earlier in the same sentence from being rejected merely
    because a separate follow-up recommendation appears later in that sentence.
    """

    for match in NON_ASSERTIVE_FINDING_CONTEXT_RE.finditer(span):
        if match.start() < anchor_end and anchor_start < match.end():
            return True
    return False


TEMPORAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("resolved", re.compile(r"\b(?:resolved|resolution|no\s+longer\s+seen)\b", re.I)),
    ("new", re.compile(r"\bnew(?:ly)?\b", re.I)),
    ("increased", re.compile(r"\b(?:increased|worsened|progressed)\b", re.I)),
    ("decreased", re.compile(r"\b(?:decreased|improved|improving)\b", re.I)),
    ("stable", re.compile(r"\b(?:stable|unchanged|no\s+interval\s+change)\b", re.I)),
)


@dataclass(frozen=True)
class EvidenceAlignment:
    start: int
    end: int
    mode: str


@dataclass(frozen=True)
class SemanticSupportResult:
    valid: bool
    mode: str
    score: float
    reason: str
    anchor: str = ""
    expected_presence: str = ""


class OntologyOutputValidationError(ValueError):
    """The ontology model exhausted all JSON/contract correction attempts."""

    def __init__(self, message: str, *, audit_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit_record = audit_record


def _prompt_finding_specs(prompt_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten v1 flat, v2 keyed, or v3 sparse role/family vocabularies."""

    flat = prompt_spec.get("findings")
    if isinstance(flat, dict):
        return {
            str(finding_id): dict(spec)
            for finding_id, spec in flat.items()
            if isinstance(spec, dict)
        }

    flattened: dict[str, dict[str, Any]] = {}
    finding_groups = prompt_spec.get("finding_groups", {})
    if not isinstance(finding_groups, dict):
        return flattened
    for role_groups in finding_groups.values():
        if not isinstance(role_groups, dict):
            continue
        for family_findings in role_groups.values():
            if isinstance(family_findings, dict):
                for finding_id, spec in family_findings.items():
                    if isinstance(spec, dict):
                        flattened[str(finding_id)] = dict(spec)
                continue
            if isinstance(family_findings, list):
                for raw_spec in family_findings:
                    if not isinstance(raw_spec, dict):
                        continue
                    finding_id = str(raw_spec.get("id", "")).strip()
                    if not finding_id:
                        continue
                    spec = dict(raw_spec)
                    spec.pop("id", None)
                    flattened[finding_id] = spec
    return flattened


def _prompt_group_statistics(prompt_spec: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic model-facing role/family counts for status records."""

    groups = prompt_spec.get("finding_groups")
    if not isinstance(groups, dict):
        count = len(_prompt_finding_specs(prompt_spec))
        return {
            "finding_count": count,
            "family_count": 0,
            "role_family_group_count": 0,
            "role_counts": {},
        }

    role_counts: dict[str, int] = {}
    family_ids: set[str] = set()
    role_family_group_count = 0
    for role, role_groups in groups.items():
        if not isinstance(role_groups, dict):
            continue
        family_ids.update(str(family_id) for family_id in role_groups)
        role_family_group_count += len(role_groups)
        role_counts[str(role)] = sum(
            len(family_findings)
            for family_findings in role_groups.values()
            if isinstance(family_findings, (dict, list))
        )
    return {
        "finding_count": sum(role_counts.values()),
        "family_count": len(family_ids),
        "role_family_group_count": role_family_group_count,
        "role_counts": role_counts,
    }


def ontology_extraction_response_format(
    ontology: Ontology,
    prompt_spec: dict[str, Any],
    *,
    max_events_per_case: int = 16,
) -> dict[str, Any]:
    """Build the strict sparse JSON Schema used for ontology extraction.

    Findings are values of one bounded enum rather than 57 object-property keys.
    This removes the checklist-shaped response contract that caused the model to
    emit one event for many unrelated ontology entries. Anatomy compatibility and
    one-span/per-case cardinality remain deterministic runtime invariants because
    the pinned XGrammar path does not reliably enforce conditional or array-size
    keywords.
    """
    
    max_events_per_case = max(1, int(max_events_per_case))
    allowed_values = prompt_spec.get("allowed_values", {})
    prompt_findings = _prompt_finding_specs(prompt_spec)

    def enum_values(field: str) -> list[str]:
        values = [str(value) for value in allowed_values.get(field, [])]
        if not values:
            values = [str(value) for value in ontology.allowed_values.get(field, [])]
        return values

    finding_enum = [
        finding_id
        for finding_id in ontology.extractable_findings
        if finding_id in prompt_findings
    ]
    if set(finding_enum) != set(ontology.extractable_findings):
        missing = sorted(set(ontology.extractable_findings) - set(finding_enum))
        raise ValueError(
            "Model-facing ontology prompt is missing extractable findings: "
            f"{missing}"
        )

    anatomy_enum = sorted(
        {
            str(anatomy_id)
            for anatomy_id in prompt_spec.get("anatomy", {})
            if str(anatomy_id) in ontology.anatomy
        }
        | {"unspecified", "not_applicable"}
    )

    event_schema = {
        "type": "object",
        "properties": {
            "finding": {"type": "string", "enum": finding_enum},
            "anatomy": {"type": "string", "enum": anatomy_enum},
            "laterality": {
                "type": "string",
                "enum": enum_values("laterality"),
            },
            "presence": {
                "type": "string",
                "enum": ["present", "absent"],
            },
            "severity": {
                "type": "string",
                "enum": enum_values("severity"),
            },
            "uncertainty": {
                "type": "string",
                "enum": enum_values("uncertainty"),
            },
            "characteristics": {
                "type": "object",
                "properties": {
                    "size": {"type": "string"},
                    "morphology": {"type": "string"},
                    "distribution": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"span": {"type": "string"}},
                    "required": ["span"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "finding",
            "anatomy",
            "laterality",
            "presence",
            "severity",
            "uncertainty",
            "evidence",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visig_cr_ontology_events_v1_3",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "events": {
                        "type": "array",
                        "items": event_schema,
                        "maxItems": max_events_per_case,
                    }
                },
                "required": ["events"],
                "additionalProperties": False,
            },
        },
    }


def section_spans(report: str) -> list[tuple[int, int, str]]:
    matches = list(SECTION_RE.finditer(report))
    if not matches:
        return [(0, len(report), "other")]
    spans: list[tuple[int, int, str]] = []
    if matches[0].start() > 0:
        spans.append((0, matches[0].start(), "other"))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(report)
        normalized_header = normalize_token(match.group("header"))
        header = SECTION_HEADER_MAP.get(normalized_header, "other")
        spans.append((start, end, header))
    return spans


def section_for_offset(report: str, start: int) -> str:
    for sec_start, sec_end, name in section_spans(report):
        if sec_start <= start < sec_end:
            return name
    return "other"


def find_exact_span(
    report: str,
    span: str,
    proposed_start: int | None = None,
    *,
    preferred_sections: set[str] | None = None,
    preferred_ranges: list[tuple[int, int]] | None = None,
) -> tuple[int, int] | None:
    if not span:
        return None

    def in_preferred_range(start: int, end: int) -> bool:
        return bool(
            preferred_ranges
            and any(
                range_start <= start and end <= range_end
                for range_start, range_end in preferred_ranges
            )
        )

    if proposed_start is not None and proposed_start >= 0:
        end = proposed_start + len(span)
        if report[proposed_start:end] == span:
            return proposed_start, end
    occurrences = [m.start() for m in re.finditer(re.escape(span), report)]
    if occurrences:
        if preferred_ranges:
            for start in occurrences:
                end = start + len(span)
                if in_preferred_range(start, end):
                    return start, end
        if preferred_sections:
            for start in occurrences:
                if section_for_offset(report, start) in preferred_sections:
                    return start, start + len(span)
        start = occurrences[0]
        return start, start + len(span)
    return None


def _whitespace_flexible_pattern(span: str) -> re.Pattern[str] | None:
    parts = [part for part in re.split(r"\s+", span.strip()) if part]
    if not parts:
        return None
    # Only physical whitespace is flexible. Words, punctuation, spelling, and
    # case remain exact so this cannot turn into semantic/fuzzy matching.
    return re.compile(r"\s+".join(re.escape(part) for part in parts))


def align_evidence_span(
    report: str,
    span: str,
    proposed_start: int | None = None,
    *,
    preferred_sections: set[str] | None = None,
    preferred_ranges: list[tuple[int, int]] | None = None,
    allow_whitespace_fallback: bool = True,
) -> EvidenceAlignment | None:
    """Align model evidence to the immutable raw report.

    Strict literal matching remains the first choice.  The fallback permits only
    whitespace-equivalent source text, which handles MIMIC physical line wraps
    while preserving raw-report provenance.  The returned offsets always point
    into ``report``; callers must persist ``report[start:end]`` rather than the
    model-normalized span.
    """

    exact = find_exact_span(
        report,
        span,
        proposed_start,
        preferred_sections=preferred_sections,
        preferred_ranges=preferred_ranges,
    )
    if exact is not None:
        return EvidenceAlignment(exact[0], exact[1], "exact")
    if not allow_whitespace_fallback:
        return None

    pattern = _whitespace_flexible_pattern(span)
    if pattern is None:
        return None
    matches = list(pattern.finditer(report))
    if not matches:
        return None

    def same_section(match: re.Match[str]) -> bool:
        start_section = section_for_offset(report, match.start())
        end_section = section_for_offset(report, max(match.start(), match.end() - 1))
        return start_section == end_section

    matches = [match for match in matches if same_section(match)]
    if not matches:
        return None

    if preferred_ranges:
        preferred = [
            match
            for match in matches
            if any(
                range_start <= match.start() and match.end() <= range_end
                for range_start, range_end in preferred_ranges
            )
        ]
        if preferred:
            matches = preferred
    elif preferred_sections:
        preferred = [
            match
            for match in matches
            if section_for_offset(report, match.start()) in preferred_sections
        ]
        if preferred:
            matches = preferred

    if proposed_start is not None and proposed_start >= 0:
        match = min(matches, key=lambda item: abs(item.start() - proposed_start))
    else:
        match = matches[0]
    return EvidenceAlignment(match.start(), match.end(), "whitespace_normalized")


class OntologyExtractorAgent:
    """Extract and validate report-grounded ontology events for each case."""

    def __init__(
        self,
        ontology: Ontology,
        llm_cfg: dict[str, Any],
        extraction_cfg: dict[str, Any],
    ):
        self.ontology = ontology
        self.client = LocalChatClient(llm_cfg)
        self.cfg = extraction_cfg
        self.text_handling_policy = str(
            extraction_cfg.get("text_handling_policy", "logical_whitespace_v1")
        ).strip().lower()
        if self.text_handling_policy not in {"raw_v1", "logical_whitespace_v1"}:
            raise ValueError(
                "ontology_extraction.text_handling_policy must be raw_v1 or "
                "logical_whitespace_v1"
            )
        self.prompt_logical_whitespace = bool(
            extraction_cfg.get(
                "prompt_logical_whitespace",
                self.text_handling_policy == "logical_whitespace_v1",
            )
        )
        self.allow_whitespace_equivalent_evidence = bool(
            extraction_cfg.get(
                "allow_whitespace_equivalent_evidence",
                self.text_handling_policy == "logical_whitespace_v1",
            )
        )
        self.presence_validation_context = str(
            extraction_cfg.get(
                "presence_validation_context",
                "logical_clause_v1"
                if self.text_handling_policy == "logical_whitespace_v1"
                else "evidence_only",
            )
        ).strip().lower()
        if self.presence_validation_context not in {
            "evidence_only",
            "logical_clause_v1",
        }:
            raise ValueError(
                "ontology_extraction.presence_validation_context must be "
                "evidence_only or logical_clause_v1"
            )
        self.retrieval_eligible_absence = set(
            extraction_cfg.get("retrieval_eligible_absence_findings", [])
        )
        self.system_prompt = ONTOLOGY_EXTRACTOR_SYSTEM_PROMPT
        self.max_events_per_case = max(
            1,
            int(extraction_cfg.get("max_events_per_case", 16)),
        )
        self.event_cap_policy = str(
            extraction_cfg.get("event_cap_policy", "clinical_priority")
        ).strip().lower().replace("-", "_")
        if self.event_cap_policy not in {"clinical_priority", "reject"}:
            raise ValueError(
                "ontology_extraction.event_cap_policy must be clinical_priority "
                "or reject"
            )
        self.selection_strategy = str(
            extraction_cfg.get("selection_strategy", "evidence_first")
        ).strip().lower().replace("-", "_")
        if self.selection_strategy != "evidence_first":
            raise ValueError(
                "ontology_extraction.selection_strategy must be evidence_first"
            )
        self.context_only_direct_mention_only = bool(
            extraction_cfg.get("context_only_direct_mention_only", True)
        )
        self.forbid_family_expansion = bool(
            extraction_cfg.get("forbid_family_expansion", True)
        )
        self.emit_empty_finding_arrays = bool(
            extraction_cfg.get("emit_empty_finding_arrays", False)
        )
        if self.emit_empty_finding_arrays:
            raise ValueError(
                "ontology_extraction.emit_empty_finding_arrays must remain false"
            )
        default_generic_fallbacks = {
            "lung_opacity",
            "increased_density",
            "infiltrate",
            "alveolar_pattern",
            "chronic_lung_change",
            "lung_lesion",
            "support_device",
        }
        configured_generic_fallbacks = extraction_cfg.get(
            "generic_fallback_finding_ids",
            sorted(default_generic_fallbacks),
        )
        self.generic_fallback_findings = {
            str(value) for value in configured_generic_fallbacks
        }
        unknown_generic_fallbacks = sorted(
            self.generic_fallback_findings - set(self.ontology.extractable_findings)
        )
        if unknown_generic_fallbacks:
            raise ValueError(
                "ontology_extraction.generic_fallback_finding_ids contains "
                f"unknown/non-extractable IDs: {unknown_generic_fallbacks}"
            )
        semantic_cfg = extraction_cfg.get("semantic_validation", {})
        if semantic_cfg is None:
            semantic_cfg = {}
        if not isinstance(semantic_cfg, dict):
            raise ValueError(
                "ontology_extraction.semantic_validation must be an object"
            )
        self.semantic_cfg = dict(semantic_cfg)
        self.semantic_validation_enabled = bool(
            self.semantic_cfg.get("enabled", True)
        )
        self.reject_unsupported_events = bool(
            self.semantic_cfg.get("reject_unsupported_events", True)
        )
        self.recover_explicit_named_events = bool(
            self.semantic_cfg.get("recover_explicit_named_events", True)
        )
        self.recover_context_only_findings = bool(
            self.semantic_cfg.get("recover_context_only_findings", True)
        )
        self.require_semantic_support_for_retrieval = bool(
            self.semantic_cfg.get("require_verified_marker_for_retrieval", True)
        )
        self.max_findings_per_evidence_span = max(
            1,
            int(
                self.semantic_cfg.get(
                    "max_distinct_findings_per_evidence_span", 6
                )
            ),
        )
        self.reject_span_overexpansion = bool(
            self.semantic_cfg.get("reject_span_overexpansion", True)
        )
        self.semantic_marker_key = "_semantic_support"
        self.present_only_findings = {
            str(value)
            for value in self.ontology.semantic_support.get(
                "present_only_findings", []
            )
        }
        self.device_findings = {
            finding_id
            for finding_id in self.ontology.extractable_findings
            if self.ontology.finding_family(finding_id) == "support_device_family"
        }
        self._finding_anchor_patterns = self._compile_finding_anchor_patterns()
        self._curated_support_patterns = self._compile_curated_support_patterns()
        self.prompt_spec = self.ontology.extraction_prompt_spec(
            profile=str(
                extraction_cfg.get("prompt_vocabulary_mode", "minimal_v1")
            ),
            max_finding_synonyms=int(
                extraction_cfg.get("max_finding_synonyms_per_concept", 6)
            ),
            max_anatomy_synonyms=int(
                extraction_cfg.get("max_anatomy_synonyms_per_concept", 4)
            ),
        )
        self.prompt_group_statistics = _prompt_group_statistics(self.prompt_spec)
        self.prompt_spec_sha256 = stable_json_sha256(self.prompt_spec)
        self.prompt_spec_char_count = len(
            json.dumps(
                self.prompt_spec,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.response_format = ontology_extraction_response_format(
            self.ontology,
            self.prompt_spec,
            max_events_per_case=self.max_events_per_case,
        )
        self.response_schema_name = str(
            self.response_format["json_schema"]["name"]
        )
        self.response_schema_sha256 = stable_json_sha256(
            self.response_format["json_schema"]["schema"]
        )
        self.response_schema_char_count = len(
            json.dumps(
                self.response_format["json_schema"]["schema"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.ontology_backend = str(llm_cfg.get("backend", "unspecified"))
        self.ontology_model_family = str(
            llm_cfg.get("model_family", "unspecified")
        )
        self.ontology_checkpoint_id = str(
            llm_cfg.get("checkpoint_id", "unspecified")
        )
        self.ontology_runtime_mode = str(
            llm_cfg.get("runtime_mode", "unspecified")
        )
        self.ontology_api_model = str(llm_cfg.get("model", "unspecified"))

    @staticmethod
    def _flexible_phrase_pattern(value: str) -> re.Pattern[str] | None:
        tokens = WORD_RE.findall(str(value).lower())
        if not tokens:
            return None
        body = r"[\s\-/]+".join(re.escape(token) for token in tokens)
        return re.compile(
            rf"(?<![a-z0-9]){body}(?![a-z0-9])",
            flags=re.IGNORECASE,
        )

    def _compile_finding_anchor_patterns(
        self,
    ) -> dict[str, list[tuple[str, re.Pattern[str]]]]:
        compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        direct_anchor_patterns = self.ontology.semantic_support.get(
            "direct_anchor_patterns", {}
        )
        if not isinstance(direct_anchor_patterns, dict):
            direct_anchor_patterns = {}

        for finding_id in self.ontology.extractable_findings:
            spec = self.ontology.finding_spec(finding_id)
            raw_values = [
                finding_id.replace("_", " "),
                str(spec.get("label", "")),
                *[str(value) for value in spec.get("synonyms", [])],
            ]
            seen: set[str] = set()
            values: list[tuple[str, re.Pattern[str]]] = []
            for raw_value in sorted(
                raw_values,
                key=lambda value: (-len(WORD_RE.findall(value)), -len(value), value),
            ):
                normalized = normalize_token(raw_value)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                pattern = self._flexible_phrase_pattern(raw_value)
                if pattern is not None:
                    values.append((raw_value, pattern))

            # High-precision validator/recovery-only lexical variants. These do
            # not enter the model-facing synonym vocabulary, so validator recall
            # can improve without turning the ontology prompt into a checklist.
            raw_patterns = direct_anchor_patterns.get(finding_id, [])
            if isinstance(raw_patterns, list):
                for raw_pattern in raw_patterns:
                    pattern_text = str(raw_pattern).strip()
                    if not pattern_text:
                        continue
                    values.append(
                        (
                            f"regex:{pattern_text}",
                            re.compile(pattern_text, flags=re.IGNORECASE),
                        )
                    )

            compiled[finding_id] = values
        return compiled

    def _compile_curated_support_patterns(
        self,
    ) -> dict[tuple[str, str], list[re.Pattern[str]]]:
        compiled: dict[tuple[str, str], list[re.Pattern[str]]] = {}
        implications = self.ontology.semantic_support.get(
            "curated_implications", {}
        )
        if not isinstance(implications, dict):
            return compiled
        for finding_id, by_presence in implications.items():
            if not isinstance(by_presence, dict):
                continue
            for presence, patterns in by_presence.items():
                if not isinstance(patterns, list):
                    continue
                compiled[(str(finding_id), str(presence))] = [
                    re.compile(str(pattern), flags=re.IGNORECASE)
                    for pattern in patterns
                ]
        return compiled

    @staticmethod
    def _anchor_is_negated(
        span: str,
        anchor_start: int,
        anchor_end: int,
    ) -> bool:
        prefix = span[max(0, anchor_start - 180) : anchor_start]
        contrast_parts = CONTRAST_BOUNDARY_RE.split(prefix)
        if contrast_parts:
            prefix = contrast_parts[-1]
        prefix = NON_ABSENCE_CHANGE_RE.sub(" ", prefix)
        suffix = span[anchor_end : anchor_end + 120]
        return bool(
            NEGATION_PREFIX_RE.search(prefix)
            or NEGATION_SUFFIX_RE.search(suffix)
        )

    @staticmethod
    def _anchor_is_historical(
        span: str,
        anchor_start: int,
    ) -> bool:
        prefix = span[max(0, anchor_start - 90) : anchor_start]
        # A comma normally closes a comparison lead-in ("Compared with the
        # prior study, ...") and begins the current assertion.
        prefix = re.split(r"[,;]", prefix)[-1]
        return bool(HISTORICAL_PREFIX_RE.search(prefix))

    @staticmethod
    def _anchor_is_non_assertive(
        span: str,
        anchor_start: int,
        anchor_end: int,
    ) -> bool:
        return is_non_assertive_finding_context(span, anchor_start, anchor_end)

    @staticmethod
    def _anchor_is_semantic_confounder(
        finding: str,
        span: str,
        anchor_start: int,
        anchor_end: int,
    ) -> bool:
        if finding != "pleural_effusion":
            return False
        return any(
            match.start() < anchor_end and anchor_start < match.end()
            for match in PLEURAL_EFFUSION_CONFOUNDER_RE.finditer(span)
        )

    def _semantic_support_for_span(
        self,
        *,
        finding: str,
        presence: str,
        evidence_span: str,
        context_span: str | None = None,
        evidence_range_in_context: tuple[int, int] | None = None,
    ) -> SemanticSupportResult:
        """Verify finding grounding and polarity with local clause context.

        The finding anchor itself must remain inside the persisted evidence span.
        When a raw logical-clause context is supplied, negation and historical
        scope are evaluated around that same anchor in the wider clause.  This
        prevents physical line wrapping from dropping a preceding ``no`` while
        avoiding cross-finding leakage for laterality/severity attributes.
        """

        if not getattr(self, "semantic_validation_enabled", False):
            return SemanticSupportResult(
                valid=True,
                mode="validation_disabled",
                score=0.75,
                reason="semantic_validation_disabled",
                expected_presence=presence,
            )

        # Direct finding grounding is always required inside the evidence span.
        # Context is used only to recover semantic scope around that same anchor.
        evidence_direct_matches: list[tuple[str, re.Match[str]]] = []
        for alias, pattern in self._finding_anchor_patterns.get(finding, []):
            match = pattern.search(evidence_span)
            if match is not None:
                evidence_direct_matches.append((alias, match))

        direct_matches: list[tuple[str, re.Match[str]]] = evidence_direct_matches
        support_text = evidence_span
        if (
            evidence_direct_matches
            and context_span is not None
            and evidence_range_in_context is not None
        ):
            evidence_start, evidence_end = evidence_range_in_context
            contextual_matches: list[tuple[str, re.Match[str]]] = []
            for alias, pattern in self._finding_anchor_patterns.get(finding, []):
                for match in pattern.finditer(context_span):
                    if match.start() < evidence_end and evidence_start < match.end():
                        contextual_matches.append((alias, match))
            if contextual_matches:
                direct_matches = contextual_matches
                support_text = context_span

        direct_matches = [
            (alias, match)
            for alias, match in direct_matches
            if not self._anchor_is_semantic_confounder(
                finding,
                support_text,
                match.start(),
                match.end(),
            )
        ]

        if direct_matches:
            alias, match = sorted(
                direct_matches,
                key=lambda item: (
                    -(item[1].end() - item[1].start()),
                    item[1].start(),
                    item[0],
                ),
            )[0]
            if (
                presence == "present"
                and self._anchor_is_non_assertive(
                    support_text,
                    match.start(),
                    match.end(),
                )
            ):
                return SemanticSupportResult(
                    valid=False,
                    mode="direct_anchor",
                    score=0.0,
                    reason="non_assertive_hypothetical_or_recommendation_context",
                    anchor=alias,
                    expected_presence="none",
                )
            if finding in self.present_only_findings:
                # Normal-descriptor concepts encode the normal/negative clinical
                # meaning in the concept itself (for example, ``no acute
                # cardiopulmonary abnormality``). Their concept presence is
                # therefore ``present`` even when lexical negation is present.
                normal_descriptor = (
                    self.ontology.finding_family(finding) == "normal_descriptor"
                )
                negated = self._anchor_is_negated(
                    support_text,
                    match.start(),
                    match.end(),
                )
                if presence != "present":
                    return SemanticSupportResult(
                        valid=False,
                        mode="direct_anchor",
                        score=0.0,
                        reason="present_only_finding_cannot_be_absent",
                        anchor=alias,
                        expected_presence="present",
                    )
                if negated and not normal_descriptor:
                    return SemanticSupportResult(
                        valid=False,
                        mode="direct_anchor",
                        score=0.0,
                        reason="negated_present_only_context_is_not_emitted",
                        anchor=alias,
                        expected_presence="none",
                    )
                return SemanticSupportResult(
                    valid=True,
                    mode="direct_anchor",
                    score=1.0,
                    reason="direct_present_only_anchor",
                    anchor=alias,
                    expected_presence="present",
                )

            negated = self._anchor_is_negated(
                support_text,
                match.start(),
                match.end(),
            )
            expected_presence = "absent" if negated else "present"
            if self._anchor_is_historical(support_text, match.start()) and not negated:
                return SemanticSupportResult(
                    valid=False,
                    mode="direct_anchor",
                    score=0.0,
                    reason="historical_anchor_not_current_evidence",
                    anchor=alias,
                    expected_presence=expected_presence,
                )
            if presence != expected_presence:
                return SemanticSupportResult(
                    valid=False,
                    mode="direct_anchor",
                    score=0.0,
                    reason="presence_does_not_match_anchor_polarity",
                    anchor=alias,
                    expected_presence=expected_presence,
                )
            return SemanticSupportResult(
                valid=True,
                mode="direct_anchor",
                score=1.0,
                reason="direct_anchor_with_matching_polarity",
                anchor=alias,
                expected_presence=expected_presence,
            )

        if (
            finding == "pleural_effusion"
            and PLEURAL_EFFUSION_CONFOUNDER_RE.search(evidence_span)
        ):
            return SemanticSupportResult(
                valid=False,
                mode="semantic_confounder",
                score=0.0,
                reason="pericardial_effusion_is_not_pleural_effusion",
                expected_presence=presence,
            )

        # Curated implications stay evidence-local. Widening them to the whole
        # clause could let a neighboring finding satisfy the wrong event.
        for pattern in self._curated_support_patterns.get((finding, presence), []):
            match = pattern.search(evidence_span)
            if match is not None:
                return SemanticSupportResult(
                    valid=True,
                    mode="curated_implication",
                    score=0.95,
                    reason="curated_ontology_implication",
                    anchor=match.group(0),
                    expected_presence=presence,
                )

        return SemanticSupportResult(
            valid=False,
            mode="unsupported",
            score=0.0,
            reason="finding_not_supported_by_direct_anchor_or_curated_implication",
            expected_presence=presence,
        )

    @staticmethod
    def _infer_temporal_status(span: str, section: str) -> str:
        if section in {"comparison", "history"}:
            return "prior_only"
        for status, pattern in TEMPORAL_PATTERNS:
            if pattern.search(span):
                return status
        return "current"

    def _current_source_spans(
        self,
        report: str,
    ) -> tuple[list[tuple[int, int, str]], str]:
        """Resolve report spans that can contain current-study findings.
        Policy:
        1. Explicit FINDINGS remains authoritative.
        2. When FINDINGS is absent but IMPRESSION exists, a trailing paragraph accidentally absorbed into COMPARISON/TECHNIQUE may be promoted to implicit FINDINGS.
        3. Fully unsectioned reports retain the historical full-report behavior.
        4. When no explicit current section exists, retain the existing conservative trailing-paragraph fallback.
        """
        spans = section_spans(report)
        include_sections = set(self.cfg.get("include_sections", ["findings", "impression"]))
        has_named_sections = any(name != "other" for _, _, name in spans)
        explicit = [item for item in spans if item[2] in include_sections]
        explicit_findings = [item for item in explicit if item[2] == "findings"]
        explicit_impressions = [item for item in explicit if item[2] == "impression"]
        fallback_enabled = bool(self.cfg.get("implicit_current_section_fallback", True))
        min_words = max(1, int(self.cfg.get("implicit_current_section_min_words", 5)))
        def implicit_tail_from_span(
            span: tuple[int, int, str],
        ) -> tuple[int, int, str] | None:
            span_start, span_end, _span_name = span
            body = report[span_start:span_end]
            boundary = PARAGRAPH_BREAK_RE.search(body)
            if boundary is None:
                return None
            tail_start = span_start + boundary.end()
            while tail_start < span_end and report[tail_start].isspace():
                tail_start += 1
            if tail_start >= span_end:
                return None
            tail = report[tail_start:span_end]
            if len(WORD_RE.findall(tail)) < min_words:
                return None
            return (tail_start, span_end, "findings")
        # 1. Explicit FINDINGS exists: trust FINDINGS / IMPRESSION exactly as before.
        if explicit_findings:
            return explicit, "explicit_current_sections"
        # 2. FINDINGS absent, but IMPRESSION exists: recover a current-study narrative accidentally swallowed by the immediately preceding COMPARISON or TECHNIQUE section.
        # HISTORY is deliberately NOT promoted because a blank-line separated history paragraph may still describe prior disease.
        if explicit_impressions:
            if fallback_enabled and "findings" in include_sections:
                first_impression_index: int | None = None
                for index, item in enumerate(spans):
                    if item[2] == "impression":
                        first_impression_index = index
                        break
                if first_impression_index is not None and first_impression_index > 0:
                    previous_span = spans[first_impression_index - 1]
                    previous_section = previous_span[2]
                    if previous_section in {"comparison", "technique"}:
                        implicit_findings = implicit_tail_from_span(previous_span)
                        if implicit_findings is not None:
                            return [implicit_findings, *explicit], "implicit_current_tail_plus_impression"
            return explicit, "explicit_current_sections"
        # 3. Completely unsectioned report: preserve existing behavior.
        if not has_named_sections:
            return [(0, len(report), "other")], "unsectioned_full_report"
        # 4. Existing fallback when FINDINGS/IMPRESSION are both absent.
        if not fallback_enabled:
            return [], "no_current_text"
        named_spans = [item for item in spans if item[2] != "other"]
        if not named_spans:
            return [], "no_current_text"
        last_span = named_spans[-1]
        last_name = last_span[2]
        if last_name not in {"history", "comparison", "technique"}:
            return [], "no_current_text"
        implicit_findings = implicit_tail_from_span(last_span)
        if implicit_findings is None:
            return [], "no_current_text"
        return [implicit_findings], "implicit_current_tail"

    def _effective_section_for_offset(
        self,
        report: str,
        start: int | None,
    ) -> str:
        if start is None:
            return "other"
        raw_section = section_for_offset(report, start)
        source_spans, source_mode = self._current_source_spans(report)
        if source_mode in {"implicit_current_tail", "implicit_current_tail_plus_impression"}:
            for span_start, span_end, section in source_spans:
                if span_start <= start < span_end:
                    return section
        return raw_section

    def _iter_current_clauses(
        self,
        report: str,
    ) -> list[tuple[str, int, int, str]]:
        source_spans, _source_mode = self._current_source_spans(report)
        clauses: list[tuple[str, int, int, str]] = []
        for section_start, section_end, section in source_spans:
            body = report[section_start:section_end]
            for match in CLAUSE_RE.finditer(body):
                raw = match.group(0)
                if not raw.strip():
                    continue
                left_trim = len(raw) - len(raw.lstrip())
                right_trimmed = raw.rstrip()
                start = section_start + match.start() + left_trim
                end = section_start + match.start() + len(right_trimmed)
                text = report[start:end]
                if text:
                    clauses.append((text, start, end, section))
        return clauses

    @staticmethod
    def _logicalize_prompt_text(text: str) -> str:
        """Collapse physical whitespace inside logical clauses for the LLM.

        Sentence/clause punctuation remains the semantic boundary.  This view is
        model-facing only; raw report text is never mutated and remains the sole
        source for persisted evidence spans and offsets.
        """

        clauses: list[str] = []
        for match in CLAUSE_RE.finditer(text):
            clause = re.sub(r"\s+", " ", match.group(0)).strip()
            if clause:
                clauses.append(clause)
        return "\n".join(clauses)

    def _logical_clause_context_for_offsets(
        self,
        report: str,
        start: int | None,
        end: int | None,
    ) -> tuple[str | None, tuple[int, int] | None]:
        if start is None or end is None:
            return None, None
        for clause, clause_start, clause_end, _section in self._iter_current_clauses(
            report
        ):
            if clause_start <= start and end <= clause_end:
                return clause, (start - clause_start, end - clause_start)
        return None, None

    @staticmethod
    def _infer_laterality(span: str, *, default_anatomy: str) -> str:
        if default_anatomy == "not_applicable":
            return "not_applicable"
        text = span.lower()
        bilateral = bool(
            re.search(r"\b(?:bilateral|bilaterally|both|bibasilar|biapical)\b", text)
        )
        left = bool(re.search(r"\bleft\b", text))
        right = bool(re.search(r"\bright\b", text))
        if bilateral or (left and right):
            return "bilateral"
        if re.search(r"\bmidline\b", text):
            return "midline"
        if left:
            return "left"
        if right:
            return "right"
        return "unspecified"

    def _infer_severity(self, span: str, finding: str, presence: str) -> str:
        family = self.ontology.finding_family(finding)
        if family in {
            "normal_descriptor",
            "support_device_family",
            "technical_quality_family",
        }:
            return "not_applicable"
        text = span.lower()
        if re.search(r"\b(?:trace|tiny|minimal)\b", text):
            return "trace"
        if re.search(r"\b(?:mild|small)\b", text):
            return "mild"
        if re.search(r"\bmoderate\b", text):
            return "moderate"
        if re.search(r"\b(?:severe|large|marked)\b", text):
            return "severe"
        return "unspecified" if presence in {"present", "absent"} else "not_applicable"

    @staticmethod
    def _infer_uncertainty(span: str) -> str:
        text = span.lower()
        if re.search(
            r"\b(?:"
            r"(?:cannot|can't|could\s+not)\s+"
            r"(?:(?:entirely|completely)\s+)?(?:be\s+)?exclude(?:d)?|"
            r"(?:difficult|hard|impossible)\s+to\s+"
            r"(?:(?:entirely|completely)\s+)?(?:be\s+)?exclude(?:d)?|"
            r"not\s+possible\s+to\s+(?:be\s+)?exclude(?:d)?"
            r")\b",
            text,
        ):
            return "cannot_exclude"
        if re.search(r"\b(?:possible|possibly|may\s+(?:represent|reflect|be))\b", text):
            return "possible"
        if re.search(r"\b(?:probable|probably|likely)\b", text):
            return "probable"
        if re.search(r"\bunlikely\b", text):
            return "unlikely"
        if re.search(r"\bindeterminate\b", text):
            return "indeterminate"
        return "definite"

    @staticmethod
    def _events_overlap_clause(event: Event, start: int, end: int) -> bool:
        if event.start_char is None or event.end_char is None:
            return False
        return event.start_char < end and start < event.end_char

    @staticmethod
    def _deduplicate_validated_events(
        events: list[Event],
    ) -> tuple[list[Event], list[Event]]:
        """Drop exact semantic duplicates after deterministic corrections.

        A model may emit bilateral, left, and right variants for one sentence.
        Once evidence-based correction normalizes all three to bilateral they
        represent one assertion and must not create duplicate graph events.
        """

        kept: list[Event] = []
        dropped: list[Event] = []
        seen: set[tuple[Any, ...]] = set()
        for event in events:
            key = (
                event.finding,
                event.anatomy,
                event.laterality,
                event.presence,
                event.severity,
                event.uncertainty,
                event.start_char,
                event.end_char,
                event.evidence_span,
            )
            if key in seen:
                dropped.append(event)
                continue
            seen.add(key)
            kept.append(event)
        return kept, dropped

    def _recover_explicit_events(
        self,
        case: CaseRecord,
        report: str,
        existing_events: list[Event],
        *,
        start_index: int,
    ) -> tuple[list[Event], Counter[str]]:
        if not self.semantic_validation_enabled or not self.recover_explicit_named_events:
            return [], Counter()

        candidates: list[tuple[int, int, str, str, str]] = []
        seen_candidates: set[tuple[int, int, str, str]] = set()
        for clause, start, end, _section in self._iter_current_clauses(report):
            for finding in self.ontology.extractable_findings:
                if (
                    self.ontology.retrieval_policy(finding) != "clinical"
                    and not self.recover_context_only_findings
                ):
                    continue
                direct_match: re.Match[str] | None = None
                for _alias, pattern in self._finding_anchor_patterns.get(finding, []):
                    direct_match = pattern.search(clause)
                    if direct_match is not None:
                        break
                if direct_match is None:
                    continue
                if finding in self.present_only_findings:
                    normal_descriptor = (
                        self.ontology.finding_family(finding) == "normal_descriptor"
                    )
                    negated = self._anchor_is_negated(
                        clause,
                        direct_match.start(),
                        direct_match.end(),
                    )
                    if negated and not normal_descriptor:
                        continue
                    presence = "present"
                else:
                    presence = (
                        "absent"
                        if self._anchor_is_negated(
                            clause,
                            direct_match.start(),
                            direct_match.end(),
                        )
                        else "present"
                    )
                key = (start, end, finding, presence)
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidates.append((start, end, clause, finding, presence))

            for (finding, presence), patterns in self._curated_support_patterns.items():
                if (
                    self.ontology.retrieval_policy(finding) != "clinical"
                    and not self.recover_context_only_findings
                ):
                    continue
                if not any(pattern.search(clause) for pattern in patterns):
                    continue
                key = (start, end, finding, presence)
                if key not in seen_candidates:
                    seen_candidates.add(key)
                    candidates.append((start, end, clause, finding, presence))

        recovered: list[Event] = []
        rejected: Counter[str] = Counter()
        for candidate_index, (start, end, span, finding, presence) in enumerate(
            sorted(candidates, key=lambda item: (item[0], item[3], item[4]))
        ):
            if any(
                event.finding == finding
                and event.presence == presence
                and self._events_overlap_clause(event, start, end)
                for event in [*existing_events, *recovered]
            ):
                continue
            default_anatomy = self.ontology.default_anatomy(finding)
            raw_event = {
                "finding": finding,
                "anatomy": default_anatomy,
                "laterality": self._infer_laterality(
                    span,
                    default_anatomy=default_anatomy,
                ),
                "presence": presence,
                "severity": self._infer_severity(span, finding, presence),
                "uncertainty": self._infer_uncertainty(span),
                "characteristics": {},
                "evidence_span": span,
                "start_char": start,
            }
            try:
                event = self._validate_raw_event(
                    case,
                    report,
                    raw_event,
                    start_index + candidate_index,
                    event_source="deterministic_recovery",
                )
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                rejected[f"{type(exc).__name__}: {exc}"] += 1
                continue
            if event is not None:
                recovered.append(event)
        return recovered, rejected

    def _current_report_for_prompt(self, report: str) -> str:
        """Expose current-study text through a model-facing logical view.

        Validation always uses the untouched full report.  When logical
        whitespace is enabled, physical line wrapping inside current sections is
        collapsed for the LLM while persisted evidence is later aligned back to
        exact raw-report substrings and offsets.
        """

        if not bool(self.cfg.get("prompt_current_sections_only", True)):
            return report

        logicalize = getattr(
            self,
            "prompt_logical_whitespace",
            bool(self.cfg.get("prompt_logical_whitespace", False)),
        )
        source_spans, source_mode = self._current_source_spans(report)
        if not source_spans:
            return ""
        if source_mode == "unsectioned_full_report":
            return self._logicalize_prompt_text(report) if logicalize else report

        parts: list[str] = []
        for start, end, name in source_spans:
            body = report[start:end].strip()
            if not body:
                continue
            prompt_body = self._logicalize_prompt_text(body) if logicalize else body
            if prompt_body:
                parts.append(f"{name.upper()}:\n{prompt_body}")
        return "\n\n".join(parts)

    def _extraction_prompt_spec(self) -> dict[str, Any]:
        """Return the frozen, bounded model-facing extraction vocabulary."""

        return self.prompt_spec

    def _messages(
        self,
        report: str,
        *,
        prompt_report: str | None = None,
    ) -> list[dict[str, Any]]:
        max_events = self.max_events_per_case
        prompt_report = (
            self._current_report_for_prompt(report)
            if prompt_report is None
            else prompt_report
        )
        schema = {
            "events": [
                {
                    "finding": "one allowed extractable finding ID",
                    "anatomy": "one allowed anatomy ID",
                    "laterality": "left|right|bilateral|midline|unspecified|not_applicable",
                    "presence": "present|absent",
                    "severity": "trace|mild|moderate|severe|unspecified|not_applicable",
                    "uncertainty": "definite|probable|possible|cannot_exclude|unlikely|indeterminate|unspecified",
                    "characteristics": {
                        "size": "optional; omit the field when unsupported",
                        "morphology": "optional; omit the field when unsupported",
                        "distribution": "optional; omit the field when unsupported",
                    },
                    "evidence": [
                        {"span": "one shortest exact verbatim source span"}
                    ],
                }
            ]
        }
        # Dict insertion order is preserved by json.dumps.  SOURCE_REPORT is
        # intentionally first so the model reads evidence before the 57-choice
        # vocabulary and does not treat the ontology as a completion checklist.
        user = {
            "source_report": prompt_report,
            "source_report_scope": {
                "model_input": (
                    "configured current-study sections only when recognized; "
                    "otherwise the full report"
                ),
                "clinical_evidence_source": "source_report_only",
                "image_input": False,
                "external_label_input": False,
            },
            "extraction_task": {
                "selection_strategy": self.selection_strategy,
                "internal_steps": [
                    "read_source_report_before_ontology",
                    "identify_shortest_explicit_current_study_evidence_spans",
                    "map_each_span_to_the_minimum_directly_expressed_finding_ids",
                    "build_complete_events_only_for_selected_finding_ids",
                    "remove_events_without_exact_direct_support",
                ],
                "emit_only_final_json": True,
            },
            "ontology": self._extraction_prompt_spec(),
            "extraction_policy": {
                "max_events_per_case": max_events,
                "max_events_per_case_is_upper_bound_not_target": True,
                "stop_when_no_new_unique_event_remains": True,
                "duplicate_event_signature": [
                    "finding",
                    "anatomy",
                    "laterality",
                    "presence",
                    "evidence.span",
                ],
                "emit_not_mentioned": False,
                "response_representation": "sparse_event_array",
                "emit_only_selected_events": True,
                "generic_normality_expands_to_absences": False,
                "explicit_named_negation_only": True,
                "one_evidence_span_per_event": True,
                "forbid_parent_child_sibling_expansion": self.forbid_family_expansion,
                "generic_fallback_finding_ids": sorted(
                    self.generic_fallback_findings
                ),
                "generic_fallback_requires_explicit_generic_wording": True,
                "context_only_direct_mention_only": (
                    self.context_only_direct_mention_only
                ),
                "normal_descriptor_presence_policy": (
                    "A normal/no-acute descriptor is a concept that is PRESENT "
                    "when its dedicated normal phrase is explicitly stated. "
                    "Never emit a normal descriptor with presence=absent."
                ),
                "present_only_context_policy": (
                    "Acquisition, technical-quality, generic support-device, and "
                    "low-volume context concepts are emitted only when positively "
                    "and directly stated; do not create their absent forms."
                ),
                "finding_role_meaning": {
                    "clinical": (
                        "ordinary clinical finding; emit only when directly "
                        "supported by source_report"
                    ),
                    "context_only": (
                        "stored for audit/context preservation and not an ordinary "
                        "clinical retrieval target; emit only from a direct "
                        "dedicated source phrase"
                    ),
                },
                "attribute_missing_value_policy": {
                    "finding_not_mentioned": "omit_event",
                    "applicable_attribute_not_explicit": "unspecified",
                    "attribute_semantically_inapplicable": "not_applicable",
                    "unhedged_assertion_uncertainty": "definite",
                },
            },
            "output_schema": schema,
        }
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]

    def _cap_validated_events(
        self,
        events: list[Event],
    ) -> tuple[list[Event], list[Event]]:
        """Enforce the configured per-case event budget deterministically.

        The prompt cap is advisory for the model.  This runtime cap is the actual
        invariant.  It never turns overflow into ordinary extraction rejection:
        dropped, otherwise-valid events are recorded separately for auditing.
        """

        if len(events) <= self.max_events_per_case:
            return list(events), []
        if self.event_cap_policy == "reject":
            raise ValueError(
                "Validated ontology event count exceeds max_events_per_case: "
                f"{len(events)} > {self.max_events_per_case}"
            )

        def priority(event: Event) -> tuple[Any, ...]:
            policy = self.ontology.retrieval_policy(event.finding)
            if policy == "clinical" and event.retrieval_eligible:
                tier = 0 if event.presence == "present" else 1
            elif policy == "clinical":
                tier = 2 if event.presence == "present" else 3
            elif policy == "context_only":
                tier = 4
            else:
                tier = 5
            source_order = (
                event.start_char if event.start_char is not None else 10**12
            )
            return (
                tier,
                -float(event.confidence),
                source_order,
                event.event_id,
            )

        ranked = sorted(events, key=priority)
        kept_ids = {
            event.event_id for event in ranked[: self.max_events_per_case]
        }
        # Preserve source/model order in the persisted event list so stable event
        # IDs and report traces remain easy to inspect.
        kept = [event for event in events if event.event_id in kept_ids]
        dropped = [event for event in events if event.event_id not in kept_ids]
        return kept, dropped

    def extract(self, case: CaseRecord, report: str) -> tuple[list[Event], dict[str, Any]]:
        validation_retries = max(
            0,
            min(5, int(self.cfg.get("validation_retries", 0))),
        )
        record_raw = bool(self.cfg.get("record_raw_responses", False))
        record_parsed = bool(self.cfg.get("record_parsed_responses", True))
        raw_limit = max(0, int(self.cfg.get("raw_response_max_chars", 0)))
        current_source_spans, current_source_mode = self._current_source_spans(report)
        prompt_report = self._current_report_for_prompt(report)
        base_messages = self._messages(
            report,
            prompt_report=prompt_report,
        )
        messages = list(base_messages)
        attempts: list[dict[str, Any]] = []
        raw_rows: list[dict[str, Any]] | None = None
        raw_event_group_count = 0
        raw_empty_event_group_count = 0
        raw_emitted_finding_keys: list[str] = []
        last_error: ValueError | None = None
        model_call_skipped_reason: str | None = None

        report_sections = section_spans(report)
        recognized_report_sections = sorted(
            {name for _, _, name in report_sections if name != "other"}
        )
        report_is_unsectioned = not recognized_report_sections

        def audited_raw(value: str) -> str:
            if raw_limit and len(value) > raw_limit:
                omitted = len(value) - raw_limit
                return value[:raw_limit] + f"...[truncated {omitted} chars]"
            return value

        if not prompt_report.strip():
            # A report containing recognized headings but no configured current
            # section is a valid zero-event case. Do not ask the model to infer
            # current findings from history, comparison, or technique text.
            raw_rows = []
            model_call_skipped_reason = "no_current_study_findings_or_impression"
        else:
            for attempt_index in range(validation_retries + 1):
                completion = None
                previous_raw = ""
                try:
                    completion = self.client.complete_json_response(
                        messages,
                        response_format=self.response_format,
                        allow_unconstrained_fallback=False,
                    )
                    raw = completion.parsed
                    if "events" not in raw:
                        raise ValueError(
                            "Ontology Extractor Agent output must contain a top-level "
                            "'events' array"
                        )
                    raw_events = raw["events"]
                    raw_rows = self._flatten_raw_assertions(raw_events)
                    raw_emitted_finding_keys = list(
                        dict.fromkeys(
                            str(row.get("finding", ""))
                            for row in raw_rows
                            if str(row.get("finding", "")).strip()
                        )
                    )
                    raw_event_group_count = len(raw_emitted_finding_keys)
                    raw_empty_event_group_count = 0
                    if isinstance(raw_events, dict):
                        # Legacy keyed traces remain parseable and retain their
                        # historical empty-group audit signal.
                        raw_event_group_count = len(raw_events)
                        raw_empty_event_group_count = sum(
                            int(isinstance(group, list) and len(group) == 0)
                            for group in raw_events.values()
                        )
                        raw_emitted_finding_keys = [
                            str(finding_id) for finding_id in raw_events
                        ]
                except ModelResponseParseError as exc:
                    last_error = exc
                    previous_raw = exc.raw_content
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_index + 1,
                        "finish_reason": exc.finish_reason,
                        "validation_error": str(exc),
                    }
                    if record_parsed:
                        attempt_record["parsed_output"] = None
                    if record_raw:
                        attempt_record["raw_content"] = audited_raw(previous_raw)
                    attempts.append(attempt_record)
                except ValueError as exc:
                    # Transport/API failures have no completion and are surfaced
                    # to the caller. Parsed contract failures are eligible for a
                    # bounded correction attempt.
                    last_error = exc
                    if completion is None:
                        raise
                    previous_raw = completion.raw_content
                    attempt_record = {
                        "attempt": attempt_index + 1,
                        "finish_reason": completion.finish_reason,
                        "validation_error": str(exc),
                    }
                    if record_parsed:
                        attempt_record["parsed_output"] = completion.parsed
                    if record_raw:
                        attempt_record["raw_content"] = audited_raw(previous_raw)
                    attempts.append(attempt_record)
                else:
                    attempt_record = {
                        "attempt": attempt_index + 1,
                        "finish_reason": completion.finish_reason,
                        "validation_error": None,
                    }
                    if record_parsed:
                        attempt_record["parsed_output"] = completion.parsed
                    if record_raw:
                        attempt_record["raw_content"] = audited_raw(
                            completion.raw_content
                        )
                    attempts.append(attempt_record)
                    break

                length_capped = bool(
                    attempts and attempts[-1].get("finish_reason") == "length"
                )
                retry_on_length = bool(self.cfg.get("retry_on_length", False))
                if length_capped and not retry_on_length:
                    break

                if attempt_index < validation_retries:
                    if length_capped:
                        correction = (
                            "The previous generation reached max_tokens before closing "
                            "the JSON. Start over from SOURCE_REPORT; do not continue, "
                            "quote, or imitate the previous output. Return at most "
                            f"{self.max_events_per_case} UNIQUE events. This is an upper "
                            "bound, not a target. Never emit the same "
                            "(finding, anatomy, laterality, presence, evidence.span) "
                            "signature more than once. When no new directly supported "
                            "unique event remains, close the events array and JSON "
                            "immediately. Use an empty array when no event is supported. "
                            "Return JSON only."
                        )

                        messages = list(base_messages) + [
                            {
                                "role": "user",
                                "content": correction,
                            },
                        ]

                    else:
                        correction = (
                            "The previous response failed the ontology JSON contract: "
                            + str(last_error)
                            + ". Re-read source_report first, retain only events with "
                            "direct exact evidence, and return one corrected JSON object "
                            "with exactly one top-level `events` array. Every array item "
                            "must contain one allowed `finding` ID and one exact evidence "
                            "span. Use an empty array when no event is supported. Do not add "
                            "prose or Markdown fences."
                        )

                        messages = list(messages) + [
                            {
                                "role": "assistant",
                                "content": previous_raw,
                            },
                            {
                                "role": "user",
                                "content": correction,
                            },
                        ]

        audit_record = {
            "attempt_count": len(attempts),
            "validation_retries_configured": validation_retries,
            "model_call_skipped_reason": model_call_skipped_reason,
            "attempts": attempts,
        }
        if raw_rows is None:
            raise OntologyOutputValidationError(
                "Ontology output validation failed after "
                f"{len(attempts)} attempt(s): {last_error}",
                audit_record=audit_record,
            ) from last_error

        model_events: list[Event] = []
        rejected = 0
        rejected_error_counts: Counter[str] = Counter()
        rejected_event_samples: list[dict[str, Any]] = []
        rejected_sample_limit = max(
            0,
            int(self.cfg.get("rejected_event_sample_limit", 20)),
        )
        for idx, raw_event in enumerate(raw_rows):
            try:
                event = self._validate_raw_event(
                    case,
                    report,
                    raw_event,
                    idx,
                    event_source="model",
                )
            except (ValidationError, ValueError, TypeError, KeyError) as exc:
                rejected += 1
                error_key = f"{type(exc).__name__}: {exc}"
                rejected_error_counts[error_key] += 1
                if len(rejected_event_samples) < rejected_sample_limit:
                    rejected_event_samples.append(
                        {
                            "raw_event_index": idx,
                            "finding": str(raw_event.get("finding", "")),
                            "anatomy": str(raw_event.get("anatomy", "")),
                            "laterality": str(raw_event.get("laterality", "")),
                            "presence": str(raw_event.get("presence", "")),
                            "severity": str(raw_event.get("severity", "")),
                            "uncertainty": str(raw_event.get("uncertainty", "")),
                            "evidence_span": str(
                                raw_event.get("evidence_span", "")
                            ),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                continue
            if event is not None:
                model_events.append(event)
            else:
                rejected += 1

        model_events, duplicate_model_events = self._deduplicate_validated_events(
            model_events
        )

        recovered_events, recovery_rejected_error_counts = self._recover_explicit_events(
            case,
            report,
            model_events,
            start_index=len(raw_rows),
        )
        events = [*model_events, *recovered_events]

        # Quarantine an evidence span that still fans out to an implausibly broad
        # finding set after semantic validation. Unlike the historical cap, this
        # guard acts on one evidence statement and cannot hide a 16-way expansion.
        span_to_findings: dict[tuple[int | None, int | None, str], set[str]] = defaultdict(set)
        for event in events:
            span_to_findings[(event.start_char, event.end_char, event.evidence_span)].add(
                event.finding
            )
        max_distinct_findings_per_span = max(
            (len(values) for values in span_to_findings.values()),
            default=0,
        )
        overexpanded_span_keys = {
            key
            for key, values in span_to_findings.items()
            if len(values) > self.max_findings_per_evidence_span
        }
        overexpanded_span_records = [
            {
                "start_char": key[0],
                "end_char": key[1],
                "evidence_span": key[2],
                "finding_ids": sorted(span_to_findings[key]),
            }
            for key in sorted(
                overexpanded_span_keys,
                key=lambda value: (
                    value[0] if value[0] is not None else 10**12,
                    value[2],
                ),
            )
        ]
        span_guard_dropped_events: list[Event] = []
        span_guard_quarantined_events: list[Event] = []
        if overexpanded_span_keys:
            kept: list[Event] = []
            for event in events:
                key = (event.start_char, event.end_char, event.evidence_span)
                if key not in overexpanded_span_keys:
                    kept.append(event)
                    continue
                if self.reject_span_overexpansion:
                    span_guard_dropped_events.append(event)
                    continue
                marker = dict(
                    event.characteristics.get(self.semantic_marker_key, {})
                    if isinstance(event.characteristics, dict)
                    else {}
                )
                marker.update(
                    {
                        "verified": False,
                        "score": 0.0,
                        "reason": "evidence_span_overexpansion",
                    }
                )
                characteristics = dict(event.characteristics)
                characteristics[self.semantic_marker_key] = marker
                quarantined = event.model_copy(
                    update={
                        "characteristics": characteristics,
                        "retrieval_eligible": False,
                        "validation_errors": list(
                            dict.fromkeys(
                                [*event.validation_errors, "evidence_span_overexpansion"]
                            )
                        ),
                    }
                )
                kept.append(quarantined)
                span_guard_quarantined_events.append(quarantined)
            events = kept

        uncapped_validated_event_count = len(events)
        events, cap_dropped_events = self._cap_validated_events(events)

        semantic_mode_counts: Counter[str] = Counter()
        semantic_source_counts: Counter[str] = Counter()
        semantic_verified_count = 0
        for event in events:
            marker = (
                event.characteristics.get(self.semantic_marker_key, {})
                if isinstance(event.characteristics, dict)
                else {}
            )
            if isinstance(marker, dict):
                semantic_mode_counts[str(marker.get("mode", "missing"))] += 1
                semantic_source_counts[str(marker.get("source", "unknown"))] += 1
                semantic_verified_count += int(marker.get("verified") is True)
            else:
                semantic_mode_counts["missing"] += 1
                semantic_source_counts["unknown"] += 1

        status = {
            "case_id": case.case_id,
            "ok": True,
            "raw_event_count": len(raw_rows),
            "raw_event_group_count": raw_event_group_count,
            "raw_empty_event_group_count": raw_empty_event_group_count,
            "raw_emitted_finding_keys": raw_emitted_finding_keys,
            "raw_clinical_event_count": sum(
                int(
                    str(raw_event.get("finding", ""))
                    in self.ontology.clinical_findings
                )
                for raw_event in raw_rows
            ),
            "raw_context_event_count": sum(
                int(
                    str(raw_event.get("finding", ""))
                    in self.ontology.context_findings
                )
                for raw_event in raw_rows
            ),
            "model_validated_event_count": len(model_events),
            "duplicate_model_event_count": len(duplicate_model_events),
            "duplicate_model_event_ids": [
                event.event_id for event in duplicate_model_events
            ],
            "deterministically_recovered_event_count": len(recovered_events),
            "deterministically_recovered_finding_ids": sorted(
                {event.finding for event in recovered_events}
            ),
            "recovery_rejected_event_error_counts": dict(
                recovery_rejected_error_counts
            ),
            "event_group_count": len({event.finding for event in events}),
            "uncapped_validated_event_count": uncapped_validated_event_count,
            "validated_event_count": len(events),
            "persisted_event_count": len(events),
            "max_events_per_case": self.max_events_per_case,
            "event_cap_policy": self.event_cap_policy,
            "event_cap_applied": bool(cap_dropped_events),
            "event_cap_dropped_count": len(cap_dropped_events),
            "event_cap_dropped_event_ids": [
                event.event_id for event in cap_dropped_events
            ],
            "semantic_validation_enabled": self.semantic_validation_enabled,
            "semantic_verified_event_count": semantic_verified_count,
            "semantic_support_mode_counts": dict(semantic_mode_counts),
            "semantic_event_source_counts": dict(semantic_source_counts),
            "semantic_support_required_for_retrieval": (
                self.require_semantic_support_for_retrieval
            ),
            "max_distinct_findings_per_evidence_span": (
                max_distinct_findings_per_span
            ),
            "max_allowed_distinct_findings_per_evidence_span": (
                self.max_findings_per_evidence_span
            ),
            "overexpanded_evidence_span_count": len(overexpanded_span_keys),
            "overexpanded_evidence_spans": overexpanded_span_records,
            "span_guard_dropped_event_count": len(span_guard_dropped_events),
            "span_guard_quarantined_event_count": len(
                span_guard_quarantined_events
            ),
            "retrieval_eligible_event_count": sum(
                int(event.retrieval_eligible) for event in events
            ),
            "clinical_event_count": sum(
                int(self.ontology.retrieval_policy(event.finding) == "clinical")
                for event in events
            ),
            "context_event_count": sum(
                int(
                    self.ontology.retrieval_policy(event.finding) == "context_only"
                )
                for event in events
            ),
            "anatomy_incompatible_event_count": sum(
                int("anatomy_not_allowed" in event.validation_errors)
                for event in events
            ),
            "rejected_event_count": rejected,
            "rejected_event_error_counts": dict(rejected_error_counts),
            "rejected_event_samples": rejected_event_samples,
            "model_attempt_count": len(attempts),
            "model_validation_retry_count": max(0, len(attempts) - 1),
            "model_finish_reason": (
                attempts[-1].get("finish_reason") if attempts else None
            ),
            "model_call_skipped_reason": model_call_skipped_reason,
            "current_source_mode": current_source_mode,
            "implicit_current_section_fallback_used": (
                current_source_mode in {"implicit_current_tail", "implicit_current_tail_plus_impression"}
            ),
            "current_source_span_count": len(current_source_spans),
            "current_source_char_count": sum(
                end - start for start, end, _section in current_source_spans
            ),
            "ontology_output_audit": audit_record,
            "ontology_sha256": self.ontology.ontology_hash,
            "ontology_prompt_profile": self.prompt_spec.get(
                "prompt_profile", "unknown"
            ),
            "ontology_prompt_spec_sha256": self.prompt_spec_sha256,
            "ontology_prompt_spec_char_count": self.prompt_spec_char_count,
            "ontology_prompt_finding_count": self.prompt_group_statistics[
                "finding_count"
            ],
            "ontology_prompt_family_count": self.prompt_group_statistics[
                "family_count"
            ],
            "ontology_prompt_role_family_group_count": (
                self.prompt_group_statistics["role_family_group_count"]
            ),
            "ontology_prompt_finding_role_counts": self.prompt_group_statistics[
                "role_counts"
            ],
            "ontology_prompt_anatomy_count": len(
                self.prompt_spec.get("anatomy", {})
            ),
            "ontology_selection_strategy": self.selection_strategy,
            "full_report_char_count": len(report),
            "model_source_report_char_count": len(prompt_report),
            "recognized_report_sections": recognized_report_sections,
            "report_is_unsectioned": report_is_unsectioned,
            "prompt_current_sections_only": bool(
                self.cfg.get("prompt_current_sections_only", True)
            ),
            "text_handling_policy": self.text_handling_policy,
            "prompt_logical_whitespace": self.prompt_logical_whitespace,
            "allow_whitespace_equivalent_evidence": (
                self.allow_whitespace_equivalent_evidence
            ),
            "presence_validation_context": self.presence_validation_context,
            "whitespace_aligned_event_count": sum(
                int("evidence_whitespace_aligned" in event.validation_errors)
                for event in events
            ),
            "ontology_response_schema_name": self.response_schema_name,
            "ontology_response_schema_sha256": self.response_schema_sha256,
            "ontology_response_schema_char_count": self.response_schema_char_count,
            "ontology_backend": self.ontology_backend,
            "ontology_model_family": self.ontology_model_family,
            "ontology_checkpoint_id": self.ontology_checkpoint_id,
            "ontology_runtime_mode": self.ontology_runtime_mode,
            "ontology_api_model": self.ontology_api_model,
        }
        return events, status

    @staticmethod
    def _flatten_raw_assertions(raw_events: Any) -> list[dict[str, Any]]:
        """Normalize the sparse v1.2 array and legacy finding-keyed forms.

        Preferred model-facing form::

            {"events": [{"finding": "pleural_effusion", ...}]}

        The durable public store remains finding-keyed, and the parser continues
        to accept historical keyed model outputs so old traces and unit fixtures
        remain readable.
        """

        def normalize_row(value: dict[str, Any], *, finding: str | None = None) -> dict[str, Any]:
            row = dict(value)
            if finding is not None:
                row["finding"] = finding
            evidence = row.get("evidence")
            if isinstance(evidence, list) and len(evidence) != 1:
                # The pinned XGrammar path does not reliably enforce array
                # cardinality. Preserve the observed count for deterministic
                # runtime rejection.
                row["_invalid_evidence_item_count"] = len(evidence)
            if isinstance(evidence, list) and evidence:
                primary = evidence[0]
                if isinstance(primary, dict):
                    row.setdefault(
                        "evidence_span",
                        primary.get("span", primary.get("evidence_span", "")),
                    )
                    if "start_char" in primary:
                        row.setdefault("start_char", primary.get("start_char"))
                    if "section" in primary:
                        row.setdefault("section", primary.get("section"))
            return row

        if isinstance(raw_events, list):
            return [
                normalize_row(item)
                for item in raw_events
                if isinstance(item, dict)
            ]
        if not isinstance(raw_events, dict):
            raise ValueError(
                "Ontology Extractor Agent output field 'events' must be an array"
            )

        flattened: list[dict[str, Any]] = []
        for finding_key, raw_group in raw_events.items():
            if isinstance(raw_group, dict):
                assertions = raw_group.get("assertions", [])
            else:
                assertions = raw_group
            if not isinstance(assertions, list):
                continue
            for assertion in assertions:
                if isinstance(assertion, dict):
                    flattened.append(
                        normalize_row(assertion, finding=str(finding_key))
                    )
        return flattened

    def _validate_raw_event(
        self,
        case: CaseRecord,
        report: str,
        raw_event: dict[str, Any],
        idx: int,
        *,
        event_source: str = "model",
    ) -> Event | None:
        """Validate one candidate and attach deterministic semantic provenance.

        Structural validity (schema, allowed values, exact span, anatomy policy)
        and semantic support are intentionally separate.  An exact substring is
        not sufficient: the substring must directly name the finding with the
        matching polarity, or match one of the small, ontology-versioned curated
        implications.
        """

        errors: list[str] = []

        invalid_evidence_item_count = raw_event.get(
            "_invalid_evidence_item_count"
        )
        if invalid_evidence_item_count is not None:
            raise ValueError("Event must contain exactly one evidence span")

        finding_raw = str(raw_event.get("finding", ""))
        finding = (
            finding_raw
            if finding_raw in self.ontology.extractable_findings
            else self.ontology.canonical_finding(finding_raw)
        )
        if not finding or finding not in self.ontology.extractable_findings:
            raise ValueError("Unknown or non-extractable finding")

        anatomy_raw = str(raw_event.get("anatomy", "unspecified"))
        anatomy = (
            anatomy_raw
            if anatomy_raw in self.ontology.anatomy
            else self.ontology.canonical_anatomy(anatomy_raw)
        )
        anatomy = anatomy or self.ontology.default_anatomy(finding)

        laterality = normalize_token(str(raw_event.get("laterality", "unspecified")))
        presence = normalize_token(str(raw_event.get("presence", "not_mentioned")))
        severity = normalize_token(str(raw_event.get("severity", "unspecified")))
        uncertainty = normalize_token(str(raw_event.get("uncertainty", "unspecified")))

        for field, value in (
            ("anatomy", anatomy),
            ("laterality", laterality),
            ("presence", presence),
            ("severity", severity),
            ("uncertainty", uncertainty),
        ):
            if not self.ontology.allowed(field, value):
                raise ValueError(f"Invalid {field}: {value}")
        if presence == "not_mentioned":
            raise ValueError("not_mentioned events are forbidden in sparse extraction")

        model_evidence_span = str(raw_event.get("evidence_span", "")).strip()
        proposed_start = raw_event.get("start_char")
        proposed_start = (
            int(proposed_start)
            if isinstance(proposed_start, (int, float, str))
            and str(proposed_start).isdigit()
            else None
        )
        include_sections = set(
            self.cfg.get("include_sections", ["findings", "impression"])
        )
        current_spans, _current_source_mode = self._current_source_spans(report)
        preferred_ranges = [
            (start, end) for start, end, _section in current_spans
        ]
        alignment = align_evidence_span(
            report,
            model_evidence_span,
            proposed_start,
            preferred_sections=include_sections,
            preferred_ranges=preferred_ranges,
            allow_whitespace_fallback=getattr(
                self,
                "allow_whitespace_equivalent_evidence",
                bool(self.cfg.get("allow_whitespace_equivalent_evidence", False)),
            ),
        )
        if self.cfg.get("require_exact_span", True) and alignment is None:
            raise ValueError("Required exact span not found")
        if alignment is not None:
            start_char, end_char = alignment.start, alignment.end
            evidence_span = report[start_char:end_char]
            offsets: tuple[int, int] | None = (start_char, end_char)
            if alignment.mode == "whitespace_normalized":
                errors.append("evidence_whitespace_aligned")
        else:
            start_char, end_char = None, None
            evidence_span = model_evidence_span
            offsets = None

        detected_section = self._effective_section_for_offset(
            report, start_char
        )
        proposed_section = raw_event.get("section")
        section = (
            normalize_token(str(proposed_section))
            if proposed_section is not None
            else detected_section
        )
        if section not in self.ontology.allowed_values.get("section", []):
            section = detected_section
        if section != detected_section and detected_section != "other":
            errors.append("section_corrected")
            section = detected_section

        support_context: str | None = None
        evidence_range_in_context: tuple[int, int] | None = None
        if getattr(
            self,
            "presence_validation_context",
            str(self.cfg.get("presence_validation_context", "evidence_only")),
        ) == "logical_clause_v1":
            support_context, evidence_range_in_context = (
                self._logical_clause_context_for_offsets(
                    report,
                    start_char,
                    end_char,
                )
            )
        support = self._semantic_support_for_span(
            finding=finding,
            presence=presence,
            evidence_span=evidence_span,
            context_span=support_context,
            evidence_range_in_context=evidence_range_in_context,
        )
        if not support.valid:
            if getattr(self, "reject_unsupported_events", True):
                raise ValueError(
                    "Semantic support failed "
                    f"[{support.reason}] for {finding}:{presence}"
                )
            errors.append(f"semantic_support_failed:{support.reason}")

        # Explicit attribute cues override unsupported model guesses.  This is
        # deliberately conservative: a side/severity/uncertainty value is kept
        # only when it is visible in the exact evidence span or is the ontology's
        # semantically inapplicable value.
        question_attributes = set(
            self.ontology.question_attributes_for_finding(finding)
        )
        if "laterality" in question_attributes and laterality != "not_applicable":
            inferred_laterality = self._infer_laterality(
                evidence_span,
                default_anatomy=self.ontology.default_anatomy(finding),
            )
            if inferred_laterality != laterality:
                laterality = inferred_laterality
                errors.append("laterality_corrected_from_evidence")

        inferred_uncertainty = self._infer_uncertainty(evidence_span)
        if inferred_uncertainty != "definite" and uncertainty != inferred_uncertainty:
            uncertainty = inferred_uncertainty
            errors.append("uncertainty_corrected_from_evidence")
        elif uncertainty == "unspecified":
            uncertainty = inferred_uncertainty
            errors.append("uncertainty_filled_from_evidence")

        inferred_severity = self._infer_severity(evidence_span, finding, presence)
        if "severity" in question_attributes:
            if inferred_severity not in {"unspecified", "not_applicable"}:
                if severity != inferred_severity:
                    severity = inferred_severity
                    errors.append("severity_corrected_from_evidence")
            elif severity not in {"unspecified", "not_applicable"}:
                severity = "unspecified"
                errors.append("unsupported_severity_removed")
        elif severity not in {"unspecified", "not_applicable"}:
            severity = "not_applicable"
            errors.append("inapplicable_severity_corrected")

        temporal_status = self._infer_temporal_status(evidence_span, section)
        retrieval_eligible = support.valid

        # Coverage and online-comparison eligibility are deliberately separate.
        # Context/device/technical descriptors remain auditable but do not enter
        # ordinary clinical retrieval.
        retrieval_policy = self.ontology.retrieval_policy(finding)
        if retrieval_policy == "context_only":
            retrieval_eligible = False
            errors.append("context_only_finding")
        elif retrieval_policy == "excluded":
            retrieval_eligible = False
            errors.append("excluded_finding")

        anatomy_compatible = self.ontology.anatomy_allowed(finding, anatomy)
        if not anatomy_compatible:
            retrieval_eligible = False
            errors.append("anatomy_not_allowed")

        if presence == "absent" and finding not in self.retrieval_eligible_absence:
            retrieval_eligible = False
            errors.append("absence_not_in_pilot_allowlist")
        if self.cfg.get("reject_prior_only", True) and temporal_status == "prior_only":
            retrieval_eligible = False
            errors.append("prior_only")

        spans = section_spans(report)
        has_named_sections = any(name != "other" for _, _, name in spans)
        if (
            self.cfg.get("enforce_include_sections", True)
            and has_named_sections
            and section not in include_sections
        ):
            retrieval_eligible = False
            errors.append("section_not_in_allowlist")
        elif not has_named_sections:
            errors.append("unsectioned_report")

        if getattr(self, "require_semantic_support_for_retrieval", False) and not support.valid:
            retrieval_eligible = False
            errors.append("semantic_support_required_for_retrieval")

        # Confidence is now dominated by evidence-to-concept support rather than
        # by mere JSON/span validity.  This prevents an unrelated exact sentence
        # from receiving the same confidence as a correctly grounded event.
        section_score = 1.0 if section in include_sections else (0.35 if section == "other" else 0.0)
        terms: list[tuple[float, float]] = [
            (0.30, 1.0 if offsets else 0.0),
            (0.40, support.score if support.valid else 0.0),
            (0.15, section_score),
            (0.10, 1.0 if anatomy_compatible else 0.0),
            (0.05, 1.0),
        ]
        confidence = sum(weight * value for weight, value in terms)
        non_penalized = {
            "unsectioned_report",
            "context_only_finding",
            "excluded_finding",
            "absence_not_in_pilot_allowlist",
            "laterality_corrected_from_evidence",
            "uncertainty_filled_from_evidence",
            "uncertainty_corrected_from_evidence",
            "severity_corrected_from_evidence",
            "unsupported_severity_removed",
            "inapplicable_severity_corrected",
            "evidence_whitespace_aligned",
        }
        if any(error not in non_penalized for error in errors):
            confidence *= 0.95

        raw_characteristics = raw_event.get("characteristics") or {}
        characteristics = (
            dict(raw_characteristics)
            if isinstance(raw_characteristics, dict)
            else {"raw": str(raw_characteristics)}
        )
        semantic_marker_key = getattr(
            self,
            "semantic_marker_key",
            "_semantic_support",
        )
        characteristics[semantic_marker_key] = {
            "verified": bool(support.valid),
            "mode": support.mode,
            "score": float(support.score),
            "reason": support.reason,
            "anchor": support.anchor,
            "expected_presence": support.expected_presence,
            "source": event_source,
            "schema_version": "visig_cr_semantic_support_v1",
        }

        event_prefix = "r" if event_source == "deterministic_recovery" else "a"
        return Event(
            case_id=case.case_id,
            event_id=f"{case.case_id}:{finding}:{event_prefix}{idx:03d}",
            finding=finding,
            anatomy=anatomy,
            laterality=laterality,
            presence=presence,  # type: ignore[arg-type]
            severity=severity,
            uncertainty=uncertainty,
            confidence=float(max(0.0, min(1.0, confidence))),
            characteristics=characteristics,
            evidence_span=evidence_span,
            section=section,
            start_char=start_char,
            end_char=end_char,
            temporal_status=temporal_status,
            retrieval_eligible=retrieval_eligible,
            validation_errors=list(dict.fromkeys(errors)),
        )
