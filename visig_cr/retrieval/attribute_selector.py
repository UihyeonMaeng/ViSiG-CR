"""Question-and-option interpretation for semantic comparison retrieval."""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, Sequence

import numpy as np

from ..clients import LocalChatClient, ModelResponseParseError
from ..io_utils import normalize_token
from ..ontology import Ontology
from ..prompts import ATTRIBUTE_SELECTOR_SYSTEM_PROMPT
from ..schemas import (
    AnswerCandidate,
    ClinicalTarget,
    QuestionSpec,
    TargetAttribute,
)
from .relation_router import DEFAULT_RELATION_PRIORITIES, sanitize_relation_priority


class TextEncoder(Protocol):
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray: ...


_OPTION_RE = re.compile(r"^\s*([A-Za-z]|\d+)\s*[\.)\]:-]\s*(.*)$")


def laterality_predicate_plan(
    question: str, options: Sequence[str] = (),
) -> dict[str, Any] | None:
    """Interpret a narrow binary laterality predicate from visible language.

    ``unilateral`` is a comparison bucket for explicit left/right events, not
    an ontology extraction label. Missing laterality and absent findings are
    never evidence that a finding is unilateral. Other axes and non-binary
    option sets retain their ordinary selector contract.
    """

    stem = str(question).split("\n\nOPTIONS", 1)[0]
    q = " ".join(stem.lower().split())
    bilateral = bool(re.search(r"\b(?:non[- ]?)?bilateral(?:ly)?\b|\bboth (?:sides?|lungs?)\b", q))
    unilateral = bool(re.search(r"\bunilateral(?:ly)?\b", q))
    if not (bilateral or unilateral):
        return None
    if options:
        bodies = []
        for option in options:
            match = _OPTION_RE.match(str(option))
            bodies.append(normalize_token(match.group(2) if match else str(option)))
        if len(bodies) != 2 or set(bodies) not in ({"yes", "no"}, {"true", "false"}):
            return None
    if not re.match(r"^(?:is|are|was|were|does|do|has|have|can)\b", q):
        return None
    ambiguous = (
        bilateral and unilateral
        or re.search(r"\b(?:not|no|neither|without|and|or|left|right|only|exclusively|"
                     r"predominantly|mainly|more|less|larger|greater|worse|asymmetric|symmetric|with|plus|"
                     r"severe|moderate|mild|trace|excluded|exclude|possible|probable|likely|unlikely|"
                     r"acute|chronic|new|stable|large|small|tiny|minimal|significant|definite|recurrent)\b|"
                     r"n't\b|\bnon[- ]?bilateral\b", q)
    )
    predicate = "bilateral" if bilateral else "unilateral"
    return {
        "supported": not bool(ambiguous),
        "predicate": predicate,
        "positive_values": [predicate],
        "negative_values": ["unilateral" if bilateral else "bilateral"],
        "raw_value_mapping": {"left": "unilateral", "right": "unilateral", "bilateral": "bilateral"},
        "derived_bucket_source": "event.laterality",
        "reason_code": "unsupported_laterality_predicate" if ambiguous else "",
    }


def normalize_laterality_predicate_spec(
    spec: QuestionSpec, question: str | None = None,
) -> QuestionSpec:
    """Repair binary predicate plans, including previously cached left/right specs."""

    if spec.target_scope != "finding" or spec.comparison_mode != "attribute_axis":
        return spec
    plan = laterality_predicate_plan(question or spec.question_text, spec.options)
    if not plan or not plan["supported"] or spec.target_attribute not in {"presence", "laterality"}:
        return spec
    values = [*plan["positive_values"], *plan["negative_values"]]
    targets = [target.model_copy(update={"target_attribute": "laterality", "laterality": "unspecified"})
               for target in spec.clinical_targets]
    candidates = []
    for candidate in spec.answer_candidates:
        token = normalize_token(candidate.text)
        value = (values[0] if token in {"yes", "true"}
                 else values[1] if token in {"no", "false"} else None)
        candidates.append(candidate.model_copy(update={
            "canonical_value": value, "candidate_type": "attribute_value" if value else "unsupported",
            "laterality": "unspecified", "mapping_confidence": 1.0 if value else 0.0,
        }))
    return spec.model_copy(update={
        "target_attribute": "laterality", "target_laterality": "unspecified",
        "question_text": question or spec.question_text, "answer_space": values, "required_attribute_values": values,
        "optional_attribute_values": [], "minimum_distinct_attribute_values": 2,
        "clinical_targets": targets, "answer_candidates": candidates,
        "relation_priority": sanitize_relation_priority(None, "laterality", comparison_mode="attribute_axis"),
    })


class AttributeSelectorAbstainError(ValueError):
    """The LLM judged that the visible query is outside the selector contract."""


class AttributeSelectorResolutionError(ValueError):
    """The selector could not resolve a supported retrieval target from visible text."""


class AttributeSelectorAgent:
    """Select the comparison task from question-language input only.

    The agent may inspect the question stem and every visible answer option. It
    never inspects the query image, a memory seed, or the reference answer. In a
    finding-candidate MCQ it retains all mapped findings symmetrically instead of
    selecting one as the presumed target.
    """

    def __init__(
        self,
        ontology: Ontology,
        selector_cfg: dict[str, Any] | None = None,
        llm_cfg: dict[str, Any] | None = None,
        text_encoder: TextEncoder | None = None,
    ):
        self.ontology = ontology
        self.cfg = selector_cfg or {}
        self.llm_cfg = llm_cfg or {}
        self.llm = (
            LocalChatClient(self.llm_cfg)
            if self.llm_cfg and self.llm_cfg.get("enabled", False)
            else None
        )
        self.text_encoder = text_encoder
        self.system_prompt = ATTRIBUTE_SELECTOR_SYSTEM_PROMPT

    def select(
        self,
        question: str,
        options: Sequence[str] | None = None,
    ) -> QuestionSpec:
        """Select a retrieval instruction under an explicit backend policy.

        ``llm`` is intentionally LLM-primary. Deterministic rules are used only
        as post-generation validation/recovery when a structured model response
        is invalid, never as a silent shortcut around a healthy LLM call.
        ``rules_llm`` is retained as an explicit ablation/legacy mode.
        """

        normalized_options = self._normalize_options(options or [])
        top_k = max(1, int(self.cfg.get("top_k_ontology", 3)))
        backend = (
            str(self.cfg.get("backend", "llm"))
            .strip()
            .lower()
            .replace("-", "_")
        )

        if backend in {"llm", "llm_rules", "hybrid"}:
            if self.llm is None:
                raise RuntimeError(
                    "Attribute Selector Agent backend is LLM-primary, but "
                    "models.attribute_selector_llm.enabled is false or missing. "
                    "Start the attribute-selector server (for example: "
                    "bash model.sh start-attribute-selector8) and enable its config."
                )
            return self._select_with_llm(
                question,
                options=normalized_options,
                top_k=top_k,
            )

        if backend == "rules_llm":
            try:
                return self._select_with_rules(
                    question,
                    options=normalized_options,
                    top_k=top_k,
                    semantic=False,
                )
            except ValueError:
                if self.llm is None:
                    raise
                return self._select_with_llm(
                    question,
                    options=normalized_options,
                    top_k=top_k,
                    allow_deterministic_recovery=False,
                )

        if backend in {"rules", "medsiglip", "rules_medsiglip"}:
            return self._select_with_rules(
                question,
                options=normalized_options,
                top_k=top_k,
                semantic=backend in {"medsiglip", "rules_medsiglip"},
            )

        raise ValueError(
            "Unknown Attribute Selector backend "
            f"{backend!r}; use llm, llm_rules, rules_llm, rules, "
            "medsiglip, or rules_medsiglip."
        )

    def _select_with_rules(
        self,
        question: str,
        *,
        options: list[str],
        top_k: int,
        semantic: bool = False,
    ) -> QuestionSpec:
        """Deterministic interpretation used for ablation and LLM recovery.

        Compared with the historical keyword-only path, this combines ontology
        question patterns, robust semantic cue regexes, visible-option value
        structure, and symmetric finding-option mapping.
        """

        q = question.lower().strip()
        target_attribute = self._target_attribute(q, options)
        laterality = self._laterality(q)
        anatomy = self._find_longest_alias(q, self.ontology.anatomy_aliases) or "unspecified"
        ranked_rule = self._rank_alias_matches(q, self.ontology.finding_aliases)
        finding_candidates = self._finding_candidates(
            options,
            question=q,
            anatomy=anatomy,
            laterality=laterality,
        )
        if self._use_candidate_predicate_mode(
            options,
            finding_candidates,
            target_attribute,
            ranked_rule,
        ):
            return self._make_candidate_finding_spec(
                question=question,
                options=options,
                candidates=finding_candidates,
                ranked_rule=ranked_rule,
                anatomy=anatomy,
                laterality=laterality,
            )

        answer_candidates = self._attribute_candidates(
            options,
            target_attribute=target_attribute,
            anatomy=anatomy,
            laterality=laterality,
        )

        # A global abnormality-presence question intentionally has no single
        # finding target. Route it before semantic retrieval so an embedding
        # backend cannot invent an arbitrary nearest finding merely because the
        # question contains the word "abnormality".
        if (
            target_attribute in {None, "presence"}
            and not ranked_rule
            and anatomy == "unspecified"
            and laterality == "unspecified"
            and self._is_global_abnormality_presence_question(q)
        ):
            return self._make_case_normality_spec(
                question=question,
                options=options,
                anatomy=anatomy,
                laterality=laterality,
                confidence=0.95,
            )

        if target_attribute and ranked_rule:
            targets = [
                self._target(
                    finding=finding,
                    anatomy=anatomy,
                    laterality=laterality,
                    target_attribute=target_attribute,
                    score=score,
                    source="rule",
                    matched_text=matched,
                )
                for finding, score, matched in ranked_rule[:top_k]
            ]
            return self._make_spec(
                targets,
                target_attribute,
                confidence=targets[0].score,
                question=question,
                options=options,
                answer_candidates=answer_candidates,
            )

        if target_attribute and semantic and self.text_encoder is not None:
            targets = self._semantic_targets(
                question,
                target_attribute=target_attribute,
                anatomy=anatomy,
                laterality=laterality,
                top_k=top_k,
            )
            if targets:
                return self._make_spec(
                    targets,
                    target_attribute,
                    confidence=targets[0].score,
                    question=question,
                    options=options,
                    answer_candidates=answer_candidates,
                )

        missing: list[str] = []
        if not ranked_rule:
            missing.append("target finding or mapped finding options")
        if not target_attribute:
            missing.append("target clinical attribute")
        raise AttributeSelectorResolutionError(
            "Attribute Selector Agent deterministic recovery could not identify "
            + ", ".join(missing)
            + f" from question: {question}"
        )

    def parse(
        self,
        question: str,
        options: Sequence[str] | None = None,
    ) -> QuestionSpec:
        return self.select(question, options=options)

    @staticmethod
    def _normalize_options(options: Sequence[str]) -> list[str]:
        normalized: list[str] = []
        for index, option in enumerate(options):
            text = str(option).strip()
            if not text:
                continue
            match = _OPTION_RE.match(text)
            if match:
                option_id = match.group(1).upper()
                body = match.group(2).strip()
            else:
                option_id = chr(ord("A") + index)
                body = text
            normalized.append(f"{option_id}. {body}")
        return normalized

    @staticmethod
    def _split_option(option: str, index: int) -> tuple[str, str]:
        match = _OPTION_RE.match(option)
        if match:
            return match.group(1).upper(), match.group(2).strip()
        return chr(ord("A") + index), option.strip()

    def _rank_alias_matches(
        self,
        text: str,
        aliases: dict[str, str],
    ) -> list[tuple[str, float, str]]:
        normalized_text = normalize_token(text)
        best: dict[str, tuple[float, str]] = {}
        for alias_norm, canonical in aliases.items():
            pattern = rf"(?:^|_){re.escape(alias_norm)}(?:_|$)"
            if not re.search(pattern, normalized_text):
                continue
            specificity = min(1.0, 0.80 + len(alias_norm) / 100.0)
            if self.ontology.findings.get(canonical, {}).get("abstract", False):
                specificity -= 0.05
            current = best.get(canonical)
            if current is None or specificity > current[0]:
                best[canonical] = (specificity, alias_norm.replace("_", " "))
        ranked = [
            (canonical, score, matched)
            for canonical, (score, matched) in best.items()
        ]
        ranked.sort(key=lambda item: (item[1], len(item[2])), reverse=True)
        return ranked

    def _find_longest_alias(self, text: str, aliases: dict[str, str]) -> str | None:
        matches = self._rank_alias_matches(text, aliases)
        return matches[0][0] if matches else None

    @staticmethod
    def _laterality(q: str) -> str:
        if re.search(r"\bleft\b", q):
            return "left"
        if re.search(r"\bright\b", q):
            return "right"
        if re.search(r"\bbilateral\b|\bboth sides?\b", q):
            return "bilateral"
        return "unspecified"

    def _target_attribute_from_options(
        self,
        options: Sequence[str],
    ) -> TargetAttribute | None:
        if len(options) < 2:
            return None

        option_bodies = [self._split_option(option, i)[1] for i, option in enumerate(options)]

        finding_values: list[str | None] = []
        for body in option_bodies:
            finding = self.ontology.canonical_finding(body)
            if finding is None:
                matches = self._rank_alias_matches(body, self.ontology.finding_aliases)
                finding = matches[0][0] if matches else None
            finding_values.append(finding)
        if len({value for value in finding_values if value}) >= 2:
            return "finding_identity"

        candidate_attributes: list[TargetAttribute] = [
            "presence",
            "laterality",
            "severity",
            "anatomy",
        ]
        for attribute in candidate_attributes:
            mapped = [self._option_value(body, attribute) for body in option_bodies]
            observed = [value for value in mapped if value is not None]
            if len(observed) >= 2 and len(set(observed)) >= 2:
                return attribute
        return None

    def _target_attribute(
        self,
        q: str,
        options: Sequence[str] | None = None,
    ) -> TargetAttribute | None:
        q = q.lower().strip()

        # A binary "is ... bilateral?" task contrasts bilateral with unilateral;
        # it is not the same task as selecting left versus right.
        if laterality_predicate_plan(q, options or []) is not None:
            return "laterality"

        # Ontology-authored patterns remain the first deterministic evidence.
        ordered: list[TargetAttribute] = [
            "finding_identity",
            "laterality",
            "severity",
            "anatomy",
            "presence",
        ]
        for target_attribute in ordered:
            for phrase in self.ontology.question_patterns.get(target_attribute, []):
                phrase_text = str(phrase).strip().lower()
                if phrase_text and phrase_text in q:
                    return target_attribute

        # Robust paraphrase cues cover common benchmark formulations without
        # requiring an exact hand-written trigger sentence.
        cue_patterns: list[tuple[TargetAttribute, str]] = [
            (
                "finding_identity",
                r"\b(which|what)\s+(finding|abnormalit(?:y|ies)|diagnos(?:is|es)|patholog(?:y|ies)|condition|disease)\b|"
                r"\b(most likely|best explains?|cause of|causing|differential)\b",
            ),
            (
                "laterality",
                r"\b(which|what)\s+side\b|\blaterality\b|\bleft\s+or\s+right\b|"
                r"\bunilateral\s+or\s+bilateral\b",
            ),
            (
                "severity",
                r"\b(how\s+severe|severity|degree|extent|grade|grading)\b|"
                r"\b(trace|mild|moderate|severe)\b\s*(?:or|vs\.?)",
            ),
            (
                "anatomy",
                r"\b(where|location|located|locali[sz]ed|which\s+(?:lobe|zone|region)|"
                r"anatomic(?:al)?\s+(?:site|location)|distribution)\b",
            ),
            (
                "presence",
                r"\b(evidence\s+of|present|absent|seen|visualized|identified|demonstrated|"
                r"detected|excluded|rule\s*out|ruled\s*out)\b",
            ),
        ]
        for attribute, pattern in cue_patterns:
            if re.search(pattern, q):
                return attribute

        option_attribute = self._target_attribute_from_options(options or [])
        if option_attribute is not None:
            return option_attribute

        if q.startswith(
            (
                "is ",
                "are ",
                "does ",
                "do ",
                "can ",
                "has ",
                "have ",
                "was ",
                "were ",
            )
        ):
            return "presence"
        return None

    @staticmethod
    def _is_global_abnormality_presence_question(question: str) -> bool:
        """Recognize explicit global abnormality-presence questions.

        This intentionally excludes open ``What abnormality ...`` questions and
        finding-/region-specific abnormality wording.  The case-normality route
        is a narrow fallback only when no ontology finding can be grounded.
        """

        q = " ".join(str(question).lower().split())
        if not re.search(r"\babnormalit(?:y|ies)\b|\babnormal\s+findings?\b", q):
            return False
        if re.search(r"\b(what|which)\s+(?:abnormalit(?:y|ies)|abnormal\s+findings?)\b", q):
            return False
        abnormality = r"(?:abnormalit(?:y|ies)|abnormal\s+findings?)"
        return bool(
            re.search(rf"\b(?:is|are)\s+there\b.*\b(?:any\s+)?{abnormality}\b", q)
            or re.search(rf"\bevidence\s+of\b.*\b(?:any\s+)?{abnormality}\b", q)
            or re.search(
                rf"\b(?:does|do)\b.*\b(?:show|demonstrate|reveal|contain|have)\b.*"
                rf"\b(?:any\s+)?{abnormality}\b",
                q,
            )
            or re.fullmatch(rf"(?:any|an)\s+{abnormality}\??", q) is not None
        )

    def _target(
        self,
        *,
        finding: str,
        anatomy: str,
        laterality: str,
        target_attribute: TargetAttribute,
        score: float,
        source: str,
        matched_text: str,
        presence: str = "unspecified",
        uncertainty: str = "unspecified",
    ) -> ClinicalTarget:
        target_anatomy = anatomy
        if (
            target_anatomy == "unspecified"
            or not self.ontology.anatomy_allowed(finding, target_anatomy)
        ):
            target_anatomy = self.ontology.default_anatomy(finding)
        return ClinicalTarget(
            finding=finding,
            anatomy=target_anatomy,
            laterality=laterality,
            presence=presence,
            uncertainty=uncertainty,
            target_attribute=target_attribute,
            score=float(max(-1.0, min(1.0, score))),
            source=source,  # type: ignore[arg-type]
            matched_text=matched_text,
        )

    def _semantic_targets(
        self,
        question: str,
        *,
        target_attribute: TargetAttribute,
        anatomy: str,
        laterality: str,
        top_k: int,
    ) -> list[ClinicalTarget]:
        assert self.text_encoder is not None
        candidates = self.ontology.selectable_findings
        template = str(
            self.cfg.get(
                "semantic_template",
                "A chest X-ray question about {label}.",
            )
        )
        prompts = [
            template.format(
                label=self.ontology.label(finding, "finding"),
                description=self.ontology.description(finding, "finding"),
            )
            for finding in candidates
        ]
        embeddings = self.text_encoder.encode_texts([question, *prompts])
        if embeddings.shape[0] != len(prompts) + 1:
            return []
        scores = embeddings[1:] @ embeddings[0]
        minimum = float(self.cfg.get("semantic_min_score", 0.10))
        targets: list[ClinicalTarget] = []
        for index in np.argsort(-scores).tolist():
            score = float(scores[index])
            if score < minimum:
                continue
            finding = candidates[index]
            targets.append(
                self._target(
                    finding=finding,
                    anatomy=anatomy,
                    laterality=laterality,
                    target_attribute=target_attribute,
                    score=score,
                    source="medsiglip",
                    matched_text=self.ontology.label(finding, "finding"),
                )
            )
            if len(targets) >= top_k:
                break
        return targets

    @staticmethod
    def _comparison_values(
        target_attribute: TargetAttribute,
        answer_space: list[str],
    ) -> tuple[list[str], list[str], int]:
        values = [normalize_token(value) for value in answer_space]
        if target_attribute == "presence":
            required = [value for value in ("present", "absent") if value in values]
            optional = [value for value in ("uncertain",) if value in values]
            return required, optional, 2
        if target_attribute == "laterality":
            required = [value for value in ("left", "right") if value in values]
            optional = [
                value
                for value in ("bilateral", "none", "uncertain")
                if value in values
            ]
            return required, optional, 2
        if target_attribute == "severity":
            optional = [
                value
                for value in ("trace", "mild", "moderate", "severe", "uncertain")
                if value in values
            ]
            return [], optional, 2
        return [], [], 2

    def _make_spec(
        self,
        targets: list[ClinicalTarget],
        target_attribute: TargetAttribute,
        confidence: float,
        *,
        question: str,
        options: list[str],
        answer_candidates: list[AnswerCandidate],
        relation_priority: Sequence[str] | None = None,
    ) -> QuestionSpec:
        if not targets:
            raise ValueError("No ontology targets were selected")
        primary = targets[0]
        mapped_option_values: list[str] = []
        seen_option_values: set[str] = set()
        for candidate in answer_candidates:
            value = candidate.canonical_value
            if value and value not in seen_option_values:
                seen_option_values.add(value)
                mapped_option_values.append(value)

        # For MCQ data, visible options define the operative answer space.  The
        # full ontology answer space remains the fallback for open-ended or
        # incompletely mapped questions.
        answer_space = (
            mapped_option_values
            if len(mapped_option_values) >= 2
            else self.ontology.answer_space(target_attribute)
        )
        required_values, optional_values, minimum_distinct = self._comparison_values(
            target_attribute,
            answer_space,
        )
        if len(mapped_option_values) >= 2:
            if target_attribute in {"severity", "anatomy", "finding_identity"}:
                required_values = [
                    value for value in mapped_option_values if value != "uncertain"
                ]
                optional_values = [
                    value for value in mapped_option_values if value == "uncertain"
                ]
                minimum_distinct = min(2, len(mapped_option_values))
        spec = QuestionSpec(
            target_finding=primary.finding,
            target_anatomy=primary.anatomy,
            target_laterality=primary.laterality,
            target_attribute=target_attribute,
            answer_space=answer_space,
            clinical_targets=targets,
            required_attribute_values=required_values,
            optional_attribute_values=optional_values,
            minimum_distinct_attribute_values=minimum_distinct,
            selector_confidence=float(max(0.0, min(1.0, confidence))),
            abstain=False,
            comparison_mode="attribute_axis",
            question_text=question,
            options=options,
            answer_candidates=answer_candidates,
            relation_priority=sanitize_relation_priority(
                relation_priority,
                target_attribute,
                comparison_mode="attribute_axis",
            ),
        )
        return normalize_laterality_predicate_spec(spec, question)

    def _make_case_normality_spec(
        self,
        *,
        question: str,
        options: list[str],
        anatomy: str,
        laterality: str,
        confidence: float,
        relation_priority: Sequence[str] | None = None,
    ) -> QuestionSpec:
        """Create a presence-axis spec backed by case-level normality.

        ``normality=abnormal`` is interpreted as global abnormality presence;
        ``normality=normal`` is interpreted as global abnormality absence.  The
        visible yes/no answer contract therefore remains the ordinary presence
        axis and no pseudo-finding is injected into the ontology.
        """

        answer_candidates = self._attribute_candidates(
            options,
            target_attribute="presence",
            anatomy=anatomy,
            laterality=laterality,
        )
        mapped_values = [
            candidate.canonical_value
            for candidate in answer_candidates
            if candidate.canonical_value
        ]
        answer_space = (
            list(dict.fromkeys(mapped_values))
            if len(set(mapped_values)) >= 2
            else self.ontology.answer_space("presence")
        )
        required_values, optional_values, minimum_distinct = self._comparison_values(
            "presence",
            answer_space,
        )
        return QuestionSpec(
            target_finding="unspecified",
            target_anatomy=anatomy,
            target_laterality=laterality,
            target_attribute="presence",
            target_scope="case_normality",
            answer_space=answer_space,
            clinical_targets=[],
            required_attribute_values=required_values,
            optional_attribute_values=optional_values,
            minimum_distinct_attribute_values=minimum_distinct,
            selector_confidence=float(max(0.0, min(1.0, confidence))),
            abstain=False,
            comparison_mode="attribute_axis",
            question_text=question,
            options=options,
            answer_candidates=answer_candidates,
            relation_priority=sanitize_relation_priority(
                relation_priority,
                "presence",
                comparison_mode="attribute_axis",
            ),
        )

    @staticmethod
    def _question_assertion_constraint(question: str) -> tuple[str, str]:
        q = question.lower()
        if re.search(
            r"\b(absent|absence|not present|not seen|not identified|not visualized)\b",
            q,
        ):
            return "absent", "definite"
        if "cannot be excluded" in q or "cannot exclude" in q:
            return "present", "cannot_exclude"
        if re.search(r"\b(possible|possibly|may represent|may be)\b", q):
            return "present", "possible"
        return "present", "unspecified"

    def _finding_candidates(
        self,
        options: list[str],
        *,
        question: str,
        anatomy: str,
        laterality: str,
    ) -> list[AnswerCandidate]:
        predicate_presence, predicate_uncertainty = self._question_assertion_constraint(
            question
        )
        candidates: list[AnswerCandidate] = []
        for index, option in enumerate(options):
            option_id, body = self._split_option(option, index)
            token = normalize_token(body)
            finding: str | None = None
            candidate_type = "unsupported"
            confidence = 0.0
            if token in {
                "normal",
                "normal_lung",
                "normal_lungs",
                "clear",
                "clear_lung",
                "clear_lungs",
                "no_abnormality",
                "no_acute_abnormality",
            }:
                finding = "normal_lung_transparency"
                candidate_type = "normal_state"
                confidence = 0.95
            else:
                matches = self._rank_alias_matches(body, self.ontology.finding_aliases)
                if matches:
                    finding, confidence, _ = matches[0]
                    candidate_type = "finding"
            candidates.append(
                AnswerCandidate(
                    option_id=option_id,
                    text=body,
                    candidate_type=candidate_type,  # type: ignore[arg-type]
                    canonical_value=finding,
                    finding=finding,
                    anatomy=anatomy,
                    laterality=laterality,
                    presence=predicate_presence if finding else None,
                    uncertainty=predicate_uncertainty if finding else None,
                    mapping_confidence=confidence,
                )
            )
        return candidates

    @staticmethod
    def _use_candidate_predicate_mode(
        options: list[str],
        candidates: list[AnswerCandidate],
        target_attribute: TargetAttribute | None,
        ranked_rule: list[tuple[str, float, str]],
    ) -> bool:
        if len(options) < 2:
            return False
        mapped = [candidate for candidate in candidates if candidate.finding]
        distinct = {candidate.finding for candidate in mapped if candidate.finding}
        if len(distinct) < 2:
            return False
        mapped_fraction = len(mapped) / max(1, len(options))
        return (
            target_attribute == "finding_identity"
            or not ranked_rule
            or mapped_fraction >= 0.5
        )

    def _make_candidate_finding_spec(
        self,
        *,
        question: str,
        options: list[str],
        candidates: list[AnswerCandidate],
        ranked_rule: list[tuple[str, float, str]],
        anatomy: str,
        laterality: str,
    ) -> QuestionSpec:
        mapped = [candidate for candidate in candidates if candidate.finding]
        values: list[str] = []
        targets: list[ClinicalTarget] = []
        seen: set[str] = set()
        for candidate in mapped:
            assert candidate.finding is not None
            if candidate.finding in seen:
                continue
            seen.add(candidate.finding)
            values.append(candidate.finding)
            targets.append(
                self._target(
                    finding=candidate.finding,
                    anatomy=anatomy,
                    laterality=laterality,
                    target_attribute="finding_identity",
                    score=max(0.5, candidate.mapping_confidence),
                    source="rule",
                    matched_text=candidate.text,
                    presence=candidate.presence or "unspecified",
                    uncertainty=candidate.uncertainty or "unspecified",
                )
            )
        if len(values) < 2:
            raise ValueError("Candidate-predicate mode requires at least two mapped findings")
        stem_anchor = ranked_rule[0][0] if ranked_rule else "unspecified"
        selector_confidence = sum(
            candidate.mapping_confidence for candidate in mapped
        ) / len(mapped)
        return QuestionSpec(
            target_finding=stem_anchor,
            target_anatomy=anatomy,
            target_laterality=laterality,
            target_attribute="finding_identity",
            answer_space=values,
            clinical_targets=targets,
            required_attribute_values=values,
            optional_attribute_values=[],
            minimum_distinct_attribute_values=min(2, len(values)),
            selector_confidence=max(0.0, min(1.0, selector_confidence)),
            abstain=False,
            comparison_mode="candidate_predicate",
            question_text=question,
            options=options,
            answer_candidates=candidates,
            relation_priority=sanitize_relation_priority(
                None,
                "finding_identity",
                comparison_mode="candidate_predicate",
            ),
        )

    def _attribute_candidates(
        self,
        options: list[str],
        *,
        target_attribute: TargetAttribute | None,
        anatomy: str,
        laterality: str,
    ) -> list[AnswerCandidate]:
        candidates: list[AnswerCandidate] = []
        for index, option in enumerate(options):
            option_id, body = self._split_option(option, index)
            value = self._option_value(body, target_attribute)
            candidates.append(
                AnswerCandidate(
                    option_id=option_id,
                    text=body,
                    candidate_type=(
                        "attribute_value" if value is not None else "unsupported"
                    ),
                    canonical_value=value,
                    anatomy=anatomy,
                    laterality=laterality,
                    mapping_confidence=1.0 if value is not None else 0.0,
                )
            )
        return candidates

    def _option_value(
        self,
        text: str,
        target_attribute: TargetAttribute | None,
    ) -> str | None:
        token = normalize_token(text)
        if target_attribute == "presence":
            mapping = {
                "yes": "present",
                "true": "present",
                "present": "present",
                "positive": "present",
                "detected": "present",
                "seen": "present",
                "identified": "present",
                "no": "absent",
                "false": "absent",
                "absent": "absent",
                "negative": "absent",
                "not_present": "absent",
                "not_seen": "absent",
                "not_detected": "absent",
                "no_evidence": "absent",
                "no_evidence_of": "absent",
                "uncertain": "uncertain",
                "indeterminate": "uncertain",
                "equivocal": "uncertain",
                "cannot_exclude": "uncertain",
                "cannot_be_excluded": "uncertain",
                "possible": "uncertain",
                "possibly": "uncertain",
            }
            direct = mapping.get(token)
            if direct is not None:
                return direct
            lowered = text.lower().strip()
            if re.search(
                r"\b(uncertain|indeterminate|equivocal|possible|possibly|cannot\s+(?:be\s+)?exclude(?:d)?)\b",
                lowered,
            ):
                return "uncertain"
            if re.search(
                r"\b(no|absent|negative|not\s+(?:present|seen|detected|identified)|no\s+evidence)\b",
                lowered,
            ):
                return "absent"
            if re.search(
                r"\b(yes|present|positive|seen|detected|identified|demonstrated)\b",
                lowered,
            ):
                return "present"
            return None
        if target_attribute == "laterality":
            mapping = {
                "left": "left",
                "left_sided": "left",
                "right": "right",
                "right_sided": "right",
                "bilateral": "bilateral",
                "both": "bilateral",
                "both_sides": "bilateral",
                "midline": "midline",
                "none": "none",
                "neither": "none",
                "uncertain": "uncertain",
                "indeterminate": "uncertain",
            }
            mapping["unilateral"] = "unilateral"
            direct = mapping.get(token)
            if direct is not None:
                return direct
            lowered = text.lower()
            if re.search(r"\bbilateral\b|\bboth\b", lowered):
                return "bilateral"
            if re.search(r"\bleft\b", lowered):
                return "left"
            if re.search(r"\bright\b", lowered):
                return "right"
            if re.search(r"\bmidline\b", lowered):
                return "midline"
            return None
        if target_attribute == "severity":
            mapping = {
                "trace": "trace",
                "minimal": "trace",
                "slight": "mild",
                "mild": "mild",
                "moderate": "moderate",
                "marked": "severe",
                "severe": "severe",
                "uncertain": "uncertain",
                "indeterminate": "uncertain",
            }
            direct = mapping.get(token)
            if direct is not None:
                return direct
            lowered = text.lower()
            for pattern, value in (
                (r"\btrace\b|\bminimal\b", "trace"),
                (r"\bmild\b|\bslight\b", "mild"),
                (r"\bmoderate\b", "moderate"),
                (r"\bsevere\b|\bmarked\b", "severe"),
                (r"\buncertain\b|\bindeterminate\b", "uncertain"),
            ):
                if re.search(pattern, lowered):
                    return value
            return None
        if target_attribute == "anatomy":
            return (
                self.ontology.canonical_anatomy(text)
                or self._find_longest_alias(text, self.ontology.anatomy_aliases)
            )
        if target_attribute == "finding_identity":
            canonical = self.ontology.canonical_finding(text)
            if canonical is not None:
                return canonical
            matches = self._rank_alias_matches(text, self.ontology.finding_aliases)
            return matches[0][0] if matches else None
        return None

    def _attribute_selector_response_format(self, top_k: int) -> dict[str, Any]:
        findings = sorted(set(self.ontology.selectable_findings))
        anatomy = sorted(
            set(self.ontology.anatomy) | {"unspecified", "not_applicable"}
        )
        lateralities = sorted(
            set(self.ontology.allowed_values.get("laterality", []))
            | {"unspecified", "not_applicable"}
        )
        relation_labels = sorted(
            {
                relation
                for priorities in DEFAULT_RELATION_PRIORITIES.values()
                for relation in priorities
            }
        )
        target_attributes = [
            "presence",
            "laterality",
            "severity",
            "anatomy",
            "finding_identity",
        ]
        clinical_target_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "finding": {"type": "string", "enum": ["unspecified", *findings]},
                "anatomy": {
                    "type": "string",
                    "enum": anatomy,
                },
                "laterality": {"type": "string", "enum": lateralities},
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "matched_text": {"type": "string"},
            },
            "required": ["finding", "anatomy", "laterality", "score", "matched_text"],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_attribute": {"type": "string", "enum": target_attributes},
                "comparison_mode": {
                    "type": "string",
                    "enum": ["attribute_axis", "candidate_predicate"],
                },
                "target_finding": {
                    "type": "string",
                    "enum": ["unspecified", *findings],
                },
                "target_anatomy": {
                    "type": "string",
                    "enum": anatomy,
                },
                "target_laterality": {"type": "string", "enum": lateralities},
                "clinical_targets": {
                    "type": "array",
                    "items": clinical_target_schema,
                    "maxItems": 8,
                },
                "relation_priority": {
                    "type": "array",
                    "items": {"type": "string", "enum": relation_labels},
                    "maxItems": len(relation_labels),
                },
                "selector_confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "abstain": {"type": "boolean"},
                "abstain_reason": {"type": "string"},
            },
            "required": [
                "target_attribute",
                "comparison_mode",
                "target_finding",
                "target_anatomy",
                "target_laterality",
                "clinical_targets",
                "relation_priority",
                "selector_confidence",
                "abstain",
                "abstain_reason",
            ],
        }
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "visig_cr_attribute_selector_v2",
                "strict": True,
                "schema": schema,
            },
        }

    @staticmethod
    def _safe_confidence(value: Any, default: float = 0.5) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = default
        return max(0.0, min(1.0, score))

    def _spec_from_llm_object(
        self,
        obj: dict[str, Any],
        *,
        question: str,
        options: list[str],
        top_k: int,
    ) -> QuestionSpec:
        q = question.lower().strip()
        rule_laterality = self._laterality(q)
        rule_anatomy = (
            self._find_longest_alias(q, self.ontology.anatomy_aliases)
            or "unspecified"
        )
        ranked_rule = self._rank_alias_matches(q, self.ontology.finding_aliases)

        # For a truly global yes/no abnormality question, the visible query is
        # more authoritative than a model-proposed specific finding/axis. This
        # also turns an LLM abstention on "no finding key" into the supported
        # deterministic case-level route. Region-specific questions are excluded.
        if (
            not ranked_rule
            and rule_anatomy == "unspecified"
            and rule_laterality == "unspecified"
            and self._is_global_abnormality_presence_question(q)
        ):
            return self._make_case_normality_spec(
                question=question,
                options=options,
                anatomy=rule_anatomy,
                laterality=rule_laterality,
                confidence=self._safe_confidence(
                    obj.get("selector_confidence"),
                    0.95,
                ),
                relation_priority=(
                    obj.get("relation_priority")
                    if isinstance(obj.get("relation_priority"), list)
                    else None
                ),
            )

        if bool(obj.get("abstain", False)):
            reason = str(obj.get("abstain_reason", "unsupported or ambiguous query")).strip()
            raise AttributeSelectorAbstainError(
                f"Attribute Selector Agent abstained: {reason}"
            )

        target_attribute_text = normalize_token(str(obj.get("target_attribute", "")))
        valid_attributes = {
            "presence",
            "laterality",
            "severity",
            "anatomy",
            "finding_identity",
        }
        if target_attribute_text not in valid_attributes:
            raise ValueError(
                "Attribute Selector Agent returned invalid target_attribute "
                f"{target_attribute_text!r}"
            )
        target_attribute: TargetAttribute = target_attribute_text  # type: ignore[assignment]
        comparison_mode = normalize_token(str(obj.get("comparison_mode", "attribute_axis")))
        if comparison_mode not in {"attribute_axis", "candidate_predicate"}:
            raise ValueError(
                f"Attribute Selector Agent returned invalid comparison_mode {comparison_mode!r}"
            )

        model_anatomy = self.ontology.canonical_anatomy(str(obj.get("target_anatomy", "")))
        if model_anatomy is None:
            raw_anatomy = normalize_token(str(obj.get("target_anatomy", "unspecified")))
            model_anatomy = raw_anatomy if raw_anatomy in {"unspecified", "not_applicable"} else None
        anatomy = model_anatomy or rule_anatomy

        allowed_lateralities = set(self.ontology.allowed_values.get("laterality", [])) | {
            "unspecified",
            "not_applicable",
        }
        model_laterality = normalize_token(str(obj.get("target_laterality", "unspecified")))
        laterality = (
            model_laterality
            if model_laterality in allowed_lateralities
            else rule_laterality
        )
        if laterality == "unspecified" and rule_laterality != "unspecified":
            laterality = rule_laterality

        finding_candidates = self._finding_candidates(
            options,
            question=q,
            anatomy=anatomy,
            laterality=laterality,
        )
        mapped_findings = {
            candidate.finding for candidate in finding_candidates if candidate.finding
        }
        if (
            comparison_mode == "candidate_predicate"
            or (target_attribute == "finding_identity" and len(mapped_findings) >= 2)
        ):
            if len(mapped_findings) < 2:
                raise ValueError(
                    "LLM requested candidate_predicate mode but fewer than two visible "
                    "options map to ontology findings"
                )
            return self._make_candidate_finding_spec(
                question=question,
                options=options,
                candidates=finding_candidates,
                ranked_rule=ranked_rule,
                anatomy=anatomy,
                laterality=laterality,
            )

        option_attribute = self._target_attribute_from_options(options)
        if (
            option_attribute in {"presence", "laterality", "severity", "anatomy"}
            and option_attribute != target_attribute
            and not (target_attribute == "laterality" and option_attribute == "presence"
                     and laterality_predicate_plan(question, options) is not None)
        ):
            raise ValueError(
                "LLM target_attribute conflicts with the visible option value axis: "
                f"model={target_attribute}, options={option_attribute}"
            )

        raw_targets = obj.get("clinical_targets")
        if not isinstance(raw_targets, list):
            raw_targets = []
        if not raw_targets:
            raw_targets = [
                {
                    "finding": obj.get("target_finding", "unspecified"),
                    "anatomy": anatomy,
                    "laterality": laterality,
                    "score": obj.get("selector_confidence", 0.5),
                    "matched_text": "",
                }
            ]

        targets: list[ClinicalTarget] = []
        seen_findings: set[str] = set()
        for raw in raw_targets[:top_k]:
            if not isinstance(raw, dict):
                continue
            finding = self.ontology.canonical_finding(str(raw.get("finding", "")))
            if finding is None:
                candidate_id = normalize_token(str(raw.get("finding", "")))
                if candidate_id in self.ontology.selectable_findings:
                    finding = candidate_id
            if finding is None or finding not in self.ontology.selectable_findings:
                continue
            if finding in seen_findings:
                continue
            seen_findings.add(finding)

            raw_anatomy = self.ontology.canonical_anatomy(str(raw.get("anatomy", "")))
            target_anatomy = raw_anatomy or anatomy
            raw_laterality = normalize_token(str(raw.get("laterality", laterality)))
            target_laterality = (
                raw_laterality if raw_laterality in allowed_lateralities else laterality
            )
            targets.append(
                self._target(
                    finding=finding,
                    anatomy=target_anatomy,
                    laterality=target_laterality,
                    target_attribute=target_attribute,
                    score=self._safe_confidence(
                        raw.get("score"),
                        self._safe_confidence(obj.get("selector_confidence", 0.5)),
                    ),
                    source="llm",
                    matched_text=str(raw.get("matched_text", "")).strip(),
                )
            )

        # Deterministic recovery is evidence-local to the query language: when
        # the model picked a valid axis but omitted an explicitly named finding,
        # recover only direct ontology aliases from the question stem.
        if not targets and ranked_rule:
            for finding, score, matched in ranked_rule[:top_k]:
                targets.append(
                    self._target(
                        finding=finding,
                        anatomy=anatomy,
                        laterality=laterality,
                        target_attribute=target_attribute,
                        score=score,
                        source="rule",
                        matched_text=matched,
                    )
                )
        if not targets:
            raise ValueError(
                "Attribute Selector Agent returned no valid ontology target and "
                "deterministic direct-alias recovery found none"
            )

        answer_candidates = self._attribute_candidates(
            options,
            target_attribute=target_attribute,
            anatomy=targets[0].anatomy,
            laterality=targets[0].laterality,
        )
        return self._make_spec(
            targets,
            target_attribute,
            confidence=self._safe_confidence(
                obj.get("selector_confidence"), targets[0].score
            ),
            question=question,
            options=options,
            answer_candidates=answer_candidates,
            relation_priority=(
                obj.get("relation_priority")
                if isinstance(obj.get("relation_priority"), list)
                else None
            ),
        )

    def _select_with_llm(
        self,
        question: str,
        *,
        options: list[str],
        top_k: int,
        allow_deterministic_recovery: bool | None = None,
    ) -> QuestionSpec:
        if self.llm is None:
            raise RuntimeError("Attribute Selector Agent LLM client is not configured")

        if allow_deterministic_recovery is None:
            allow_deterministic_recovery = bool(
                self.cfg.get("deterministic_fallback_on_validation_error", True)
            )
        relation_labels = sorted(
            {
                relation
                for priorities in DEFAULT_RELATION_PRIORITIES.values()
                for relation in priorities
            }
        )
        payload = {
            "question": question,
            "options": options,
            "top_k_ontology_targets": top_k,
            "allowed_relation_labels": relation_labels,
            "ontology": self.ontology.compact_prompt_spec(),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response_format = self._attribute_selector_response_format(top_k)
        validation_retries = max(0, int(self.cfg.get("validation_retries", 1)))
        last_error: Exception | None = None

        for attempt in range(validation_retries + 1):
            try:
                completion = self.llm.complete_json_response(
                    messages,
                    response_format=response_format,
                    allow_unconstrained_fallback=False,
                )
                return self._spec_from_llm_object(
                    completion.parsed,
                    question=question,
                    options=options,
                    top_k=top_k,
                )
            except ModelResponseParseError as exc:
                last_error = exc
                raw_content = exc.raw_content
            except AttributeSelectorAbstainError:
                # A deliberate scope abstention is not a malformed completion and
                # must not be overwritten by heuristic recovery.
                raise
            except ValueError as exc:
                # JSON parsed, but the proposal violated semantic/ontology
                # invariants. This is the only model-output failure class that
                # may trigger deterministic recovery. Transport/API failures are
                # deliberately not swallowed.
                last_error = exc
                raw_content = ""
            except Exception:
                raise

            if attempt >= validation_retries:
                break
            if raw_content:
                messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous selector proposal was invalid: "
                        f"{last_error}. Re-read the visible question/options, use only "
                        "canonical ontology IDs and allowed relation labels, preserve all "
                        "candidate options symmetrically, and return one corrected JSON "
                        "object matching the schema. Do not answer the question."
                    ),
                }
            )

        if allow_deterministic_recovery:
            try:
                return self._select_with_rules(
                    question,
                    options=options,
                    top_k=top_k,
                    semantic=False,
                )
            except AttributeSelectorResolutionError as recovery_error:
                raise AttributeSelectorResolutionError(
                    "Attribute Selector Agent LLM proposal remained invalid and "
                    "deterministic recovery also failed. "
                    f"LLM error: {last_error}; recovery error: {recovery_error}"
                ) from recovery_error

        raise ValueError(
            "Attribute Selector Agent LLM proposal remained invalid after "
            f"{validation_retries + 1} attempt(s): {last_error}"
        )
