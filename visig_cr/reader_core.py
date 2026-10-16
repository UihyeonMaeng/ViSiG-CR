"""Frozen multimodal reader and model-visible package construction."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from .answer_utils import (
    normalize_binary_answer,
    normalize_option_answer,
    option_id_to_text,
)
from .clients import LocalChatClient, ModelResponseParseError
from .image_utils import image_normalization_provenance, load_rgb_image
from .io_utils import file_sha256
from .prompts import (
    COMPARISON_RAG_READER_SYSTEM_PROMPT,
    LLAVA_MED_COMPARISON_RAG_SYSTEM_PROMPT,
    LLAVA_MED_NO_RAG_SYSTEM_PROMPT,
    LLAVA_MED_STRUCTURED_RAG_SYSTEM_PROMPT,
    NO_RAG_READER_SYSTEM_PROMPT,
    READER_SYSTEM_PROMPT,
)
from .normality import CLINICAL_NORMALITY_REFERENCE_POLICY_ID
from .schemas import Evidence, EvidencePackage
from .retrieval.question_conditioned import POLICY_VERSION


QUESTION_CONDITIONED_DECISION_GUIDANCE = (
    "Answer the literal question from target-specific features in the QUERY IMAGE. "
    "Form an initial assessment of those features before comparing the references. "
    "Compare both endpoints symmetrically; their labels describe memory cases only. "
    "An absent exemplar or overall visual resemblance to it does not establish "
    "absence in the query. Change your assessment only when the reference helps "
    "identify relevant features actually visible in the query. "
    "Preserve the assertion scope: 'no acute cardiopulmonary abnormality' does not "
    "exclude chronic findings or every abnormality. A stable finding can still be "
    "present. An unrelated finding in a positive reference neither establishes "
    "nor excludes the query target. An inferred absence is not an explicit negative "
    "report assertion. If an annotation disagrees with its supporting source span, "
    "treat that reference as unreliable. If reference evidence is irrelevant, narrower "
    "than the question, or contradictory, answer from the query image."
)


def image_to_data_url(
    path: str | Path,
    max_side: int = 1024,
    jpeg_quality: int = 95,
) -> str:
    """Encode a local image as an OpenAI-compatible data URL."""

    with load_rgb_image(path) as image:
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_identity(path: str | Path) -> str:
    target = Path(path)
    if target.is_file():
        return file_sha256(target)
    # Unit fixtures may intentionally use synthetic paths.  Preserve sensitivity
    # without leaking a real local path into the package payload.
    return "missing:" + hashlib.sha256(str(target).encode("utf-8")).hexdigest()


def evidence_package_sha256(package: EvidencePackage) -> str:
    """Hash exactly the reader-visible package semantics and image payloads."""

    payload = {
        "image_normalization": image_normalization_provenance(),
        "input_mode": package.input_mode,
        "reader_presentation": package.reader_presentation,
        "query_id": package.query_id,
        "query_image_sha256": _image_identity(package.query_image_path),
        "question": package.question,
        "retrieval_question": package.retrieval_question,
        "query_context": package.query_context,
        "options": package.options,
        "reader_answer_space": _package_closed_answer_space(package),
        "reader_answer_space_source": _package_closed_answer_space_source(package),
        "question_spec": (
            package.question_spec.model_dump() if package.question_spec is not None else None
        ),
        "method": package.method,
        "seed_cases": [
            {
                **{key: value for key, value in row.items() if key != "image_path"},
                "image_sha256": (
                    _image_identity(row["image_path"])
                    if row.get("image_path")
                    else None
                ),
            }
            for row in package.seed_cases
        ],
        # Wall-clock instrumentation is not a reader semantic and must not
        # invalidate a frozen-input identity. No-RAG's empty mapping is unchanged.
        "boundary_stats": {key: value for key, value in package.boundary_stats.items()
                           if not key.endswith("_seconds")},
        "evidence": [
            {
                **item.model_dump(exclude={"image_path"}),
                "image_sha256": (
                    _image_identity(item.image_path)
                    if item.reader_visible_image
                    else None
                ),
            }
            for item in package.evidence
        ],
        "semantic_contrast_groups": [
            group.model_dump() for group in package.semantic_contrast_groups
        ],
        "coverage_status": package.coverage_status,
        "warnings": package.warnings,
    }
    if package.audit_metadata.get("retrieval_policy") == POLICY_VERSION:
        # These audit fields control model-visible guidance and fallback routing.
        # Hash their reader semantics so a metadata-only change cannot resume old
        # predictions against newly rendered inputs.
        qcr_audit = package.audit_metadata.get("question_conditioned", {})
        fallback = package.audit_metadata.get("contrast_fallback", {})
        payload["question_conditioned_reader_semantics"] = {
            "retrieval_policy": POLICY_VERSION,
            "laterality_predicate": qcr_audit.get("laterality_predicate") if isinstance(qcr_audit, dict) else None,
            "contrast_fallback": {
                "used": fallback.get("used"),
                "reader_input": fallback.get("reader_input"),
            } if isinstance(fallback, dict) else None,
        }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class ReaderOutputValidationError(ValueError):
    """A reader exhausted all response-validation correction attempts."""

    def __init__(self, message: str, *, audit_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.audit_record = audit_record


def _canonical_closed_answer_space(value: Any) -> list[str]:
    """Return a safe, model-visible finite answer space from query metadata."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    normalized: list[str] = []
    for item in value:
        token = str(item).strip().casefold()
        if token and token not in normalized:
            normalized.append(token)
    # v2.0 exposes only the audited yes/no protocol through this path.  Do not
    # reuse the selector's clinical value space (present/absent/uncertain) as a
    # reader answer enum because the visible question asks for yes/no.
    return ["yes", "no"] if set(normalized) == {"yes", "no"} else []


def _resolved_package_closed_answer_space(
    package: EvidencePackage,
) -> tuple[list[str], str]:
    """Resolve a finite yes/no answer contract without consulting references.

    The preferred source is explicit query metadata emitted by the frozen
    closed-binary preprocessing step.  For backward compatibility with packages
    produced from older normalized binary query files, an option-free
    ``task_type=binary`` row is deterministically interpreted as the project's
    canonical yes/no answer space.  The resolution source is persisted in the
    reader audit and package hash so this compatibility path is never silent.
    """

    explicit = _canonical_closed_answer_space(
        package.audit_metadata.get("answer_space")
    )
    if explicit:
        source = str(
            package.audit_metadata.get(
                "answer_space_source",
                "explicit_query_metadata",
            )
        ).strip()
        return explicit, source or "explicit_query_metadata"

    if package.options:
        return [], "visible_options"

    task_type = (
        str(package.audit_metadata.get("task_type", ""))
        .strip()
        .casefold()
        .replace("-", "_")
    )
    if task_type in {"binary", "yes_no", "yes/no"}:
        return ["yes", "no"], "inferred_from_binary_task_type"
    return [], "none"


def _package_closed_answer_space(package: EvidencePackage) -> list[str]:
    values, _ = _resolved_package_closed_answer_space(package)
    return values


def _package_closed_answer_space_source(package: EvidencePackage) -> str:
    _, source = _resolved_package_closed_answer_space(package)
    return source


def reader_response_format(
    options: list[str] | tuple[str, ...],
    *,
    no_rag: bool,
    answer_space: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return the per-question decode-time JSON Schema."""

    option_ids = list(option_id_to_text(options))
    closed_values = _canonical_closed_answer_space(answer_space)
    answer_schema: dict[str, Any]
    if option_ids:
        answer_schema = {"type": "string", "enum": option_ids}
    elif closed_values:
        answer_schema = {"type": "string", "enum": closed_values}
    else:
        answer_schema = {"type": "string", "minLength": 1}
    used_evidence_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    used_groups_schema: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
    }
    evidence_use_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    if no_rag:
        # Runtime validation still requires both arrays to be empty. Avoid
        # maxItems=0 here because some vLLM 0.11 structured-output backends do
        # not implement that JSON-Schema keyword consistently.
        evidence_use_schema = {"type": "string", "const": "none"}

    schema = {
        "type": "object",
        "properties": {
            "answer": answer_schema,
            "finding": {"type": "string", "minLength": 1},
            "anatomy": {"type": "string", "minLength": 1},
            "laterality": {"type": "string", "minLength": 1},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "used_evidence": used_evidence_schema,
            "used_semantic_contrast_groups": used_groups_schema,
            "evidence_use_type": evidence_use_schema,
            "rationale": {"type": "string", "minLength": 1},
        },
        "required": [
            "answer",
            "finding",
            "anatomy",
            "laterality",
            "confidence",
            "used_evidence",
            "used_semantic_contrast_groups",
            "evidence_use_type",
            "rationale",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "visig_cr_reader_output",
            "strict": True,
            "schema": schema,
        },
    }


def _normalized_option_text(value: str) -> str:
    """Normalize typography for duplicate-option detection only."""

    return " ".join(str(value).strip().casefold().split())


def _llava_med_answer_schema(
    options: list[str] | tuple[str, ...],
    *,
    answer_representation: str = "option_text",
    answer_space: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve the compact answer schema without using reference answers."""

    option_map = option_id_to_text(options)
    requested = (
        str(answer_representation).strip().lower().replace("-", "_")
    )
    if not option_map:
        closed_values = _canonical_closed_answer_space(answer_space)
        if closed_values:
            return {"type": "string", "enum": closed_values}, "closed_binary"
        return {"type": "string", "minLength": 1}, "free_text"
    if requested in {"option_id", "option_letter", "letter", "id"}:
        return {"type": "string", "enum": list(option_map)}, "option_id"

    option_texts = list(option_map.values())
    normalized_texts = [_normalized_option_text(value) for value in option_texts]
    if len(set(normalized_texts)) == len(normalized_texts):
        return {"type": "string", "enum": option_texts}, "option_text"
    return {"type": "string", "enum": list(option_map)}, "option_id"


def llava_med_compact_response_format(
    options: list[str] | tuple[str, ...],
    *,
    answer_representation: str = "option_text",
    answer_space: Sequence[str] | None = None,
    profile: str = "no_rag",
    reference_count: int = 0,
    group_count: int = 0,
) -> tuple[dict[str, Any], str]:
    """Return an adaptive compact schema for the LLaVA-Med reader.

    LLaVA-Med-v1.5 uses a different chat/model stack from Qwen3-VL.  Asking it
    to generate ViSiG-CR's full nine-field audit object made otherwise valid
    decisions fail on descriptor, rationale, or citation fields.  The compact
    adapter asks only for answer/calibration plus bounded numeric citation
    handles that are meaningful for the active presentation.  Runtime code
    maps those handles back to canonical case/group IDs and records every
    post-filled field in ``reader_output_audit``.
    """

    normalized_profile = str(profile).strip().lower().replace("-", "_")
    if normalized_profile not in {
        "no_rag",
        "retrieved_references",
        "structured_rag",
    }:
        raise ValueError(f"Unknown LLaVA-Med compact profile: {profile!r}")

    answer_schema, resolved_representation = _llava_med_answer_schema(
        options,
        answer_representation=answer_representation,
        answer_space=answer_space,
    )
    properties: dict[str, Any] = {
        "answer": answer_schema,
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    }
    required = ["answer", "confidence"]

    reference_count = max(0, int(reference_count))
    group_count = max(0, int(group_count))
    if normalized_profile != "no_rag" and reference_count:
        properties["used_reference_numbers"] = {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": reference_count,
            },
        }
        required.append("used_reference_numbers")
    if normalized_profile == "structured_rag" and group_count:
        properties["used_group_numbers"] = {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": group_count,
            },
        }
        required.append("used_group_numbers")

    schema = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return (
        {
            "type": "json_schema",
            "json_schema": {
                "name": f"visig_cr_llava_med_{normalized_profile}_decision_v1",
                "strict": True,
                "schema": schema,
            },
        },
        resolved_representation,
    )


def llava_med_no_rag_response_format(
    options: list[str] | tuple[str, ...],
    *,
    answer_representation: str = "option_text",
    answer_space: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str]:
    """Backward-compatible wrapper for the query-only compact schema."""

    return llava_med_compact_response_format(
        options,
        answer_representation=answer_representation,
        answer_space=answer_space,
        profile="no_rag",
    )

def _safe_value_repr(value: Any) -> str:
    """Render invalid values without failing on Python's huge-int guard."""

    try:
        return repr(value)
    except (OverflowError, ValueError):
        if isinstance(value, int):
            return f"<int bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"


class VLMReader:
    """Call one frozen local VLM for RAG or a strictly isolated no-RAG control."""

    def __init__(self, model_cfg: dict[str, Any], reader_cfg: dict[str, Any]):
        family = str(
            reader_cfg.get("model_family", model_cfg.get("model_family", "qwen"))
        ).strip().lower().replace("_", "-")
        if family in {"qwen8", "qwen30", "qwen3-vl"}:
            family = "qwen"
        elif family in {"llava", "llava-med"}:
            family = "llava-med"

        self.model_family = family
        self.is_llava_med = family == "llava-med"
        effective_model_cfg = dict(model_cfg)
        effective_model_cfg["model_family"] = family
        if getattr(self, "is_llava_med", False):
            # Qwen-only chat-template kwargs are not meaningful for Mistral/LLaVA.
            effective_model_cfg["disable_qwen_thinking"] = False

        self.client = LocalChatClient(effective_model_cfg)
        self.model_cfg = effective_model_cfg
        self.reader_cfg = dict(reader_cfg)

    def _image_content(self, path: str) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(
                    path,
                    max_side=int(self.model_cfg.get("image_max_side", 1024)),
                    jpeg_quality=int(self.model_cfg.get("jpeg_quality", 95)),
                )
            },
        }

    def _ordered_evidence(self, package: EvidencePackage) -> list[Evidence]:
        evidence = list(package.evidence)
        rng = random.Random(int(self.reader_cfg.get("order_seed", 42)))
        by_case = {item.case_id: item for item in evidence}
        groups = list(package.semantic_contrast_groups)
        if self.reader_cfg.get("randomize_group_order", False):
            rng.shuffle(groups)

        ordered: list[Evidence] = []
        used: set[str] = set()
        for group in groups:
            members = [
                by_case[case_id]
                for case_id in group.member_case_ids
                if case_id in by_case
            ]
            if self.reader_cfg.get("randomize_within_group", False):
                rng.shuffle(members)
            elif package.audit_metadata.get("retrieval_policy") == POLICY_VERSION:
                policy = self.reader_cfg.get("question_conditioned_reference_order", "balanced")
                if policy not in {"balanced", "stored", "reversed"}:
                    raise ValueError(
                        "reader.question_conditioned_reference_order must be "
                        "'balanced', 'stored', or 'reversed'"
                    )
                # Query/group-specific deterministic balancing does not assign
                # one target value permanent first-reference authority.
                key = f"{int(self.reader_cfg.get('order_seed', 42))}:{package.query_id}:{group.group_id}"
                if policy == "reversed" or (
                    policy == "balanced" and hashlib.sha256(key.encode("utf-8")).digest()[0] & 1
                ):
                    members.reverse()
            ordered.extend(members)
            used.update(item.case_id for item in members)

        ungrouped = [item for item in evidence if item.case_id not in used]
        if self.reader_cfg.get("randomize_ungrouped_order", False):
            rng.shuffle(ungrouped)
        ordered.extend(ungrouped)
        return ordered

    def _llava_med_output_profile(self, package: EvidencePackage) -> str:
        """Resolve the model-specific output adapter for one package."""

        if not getattr(self, "is_llava_med", False):
            return "full"
        mode = (
            str(self.reader_cfg.get("llava_med_output_mode", "compact_adaptive"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if mode in {"full", "full_schema", "disabled", "off"}:
            return "full"
        if mode in {"compact_no_rag", "no_rag_only"} and package.input_mode != "no_rag":
            return "full"
        if package.input_mode == "no_rag":
            return "no_rag"
        if package.reader_presentation == "retrieved_references":
            return "retrieved_references"
        return "structured_rag"

    def _llava_med_compact_enabled(self, package: EvidencePackage) -> bool:
        return self._llava_med_output_profile(package) != "full"

    def _llava_med_answer_representation(
        self,
        package: EvidencePackage,
    ) -> str:
        """Resolve the actual answer representation used by the compact schema."""

        _, representation = _llava_med_answer_schema(
            package.options,
            answer_representation=str(
                self.reader_cfg.get(
                    "llava_med_answer_representation",
                    "option_text",
                )
            ),
            answer_space=_package_closed_answer_space(package),
        )
        return representation

    @staticmethod
    def _llava_med_clean_question(package: EvidencePackage) -> str:
        """Render only the clinical question and visible options for LLaVA-Med.

        Dataset adapters can append the generic nine-field output instruction to
        ``package.question``.  The model-specific compact adapter instead uses the
        clean retrieval stem and a deterministic option block, then appends its
        own output contract.
        """

        stem = package.retrieval_question.strip()
        if not stem:
            stem = package.question.strip()
            for marker in ("\n\nOPTIONS\n", "\nOPTIONS\n"):
                if marker in stem:
                    stem = stem.split(marker, 1)[0].strip()
                    break

        option_map = option_id_to_text(package.options)
        if not option_map:
            return "QUESTION\n" + stem

        option_block = "\n".join(
            f"{option_id}. {option_text}"
            for option_id, option_text in option_map.items()
        )
        return "QUESTION\n" + stem + "\n\nOPTIONS\n" + option_block

    def _ordered_evidence_for_presentation(
        self,
        package: EvidencePackage,
    ) -> list[Evidence]:
        if package.reader_presentation == "retrieved_references":
            return list(package.evidence)
        return self._ordered_evidence(package)

    def _llava_med_citation_maps(
        self,
        package: EvidencePackage,
    ) -> tuple[dict[int, str], dict[int, str]]:
        evidence_map = {
            index: item.case_id
            for index, item in enumerate(
                self._ordered_evidence_for_presentation(package),
                1,
            )
        }
        group_map = {
            index: group.group_id
            for index, group in enumerate(package.semantic_contrast_groups, 1)
        }
        return evidence_map, group_map

    def _model_visible_image_count(self, package: EvidencePackage) -> int:
        count = 1  # query image
        if package.input_mode == "no_rag":
            return count
        if package.reader_presentation == "retrieved_references":
            return count + sum(
                int(item.reader_visible_image) for item in package.evidence
            )
        return count + len(package.evidence)

    def _validate_image_budget(self, package: EvidencePackage) -> None:
        raw_limit = self.reader_cfg.get("max_images_per_prompt")
        if raw_limit in {None, "", False, 0, "0"}:
            return
        limit = int(raw_limit)
        if limit < 1:
            raise ValueError("reader.max_images_per_prompt must be at least 1")
        visible = self._model_visible_image_count(package)
        if visible > limit:
            raise ValueError(
                "Reader-visible image count exceeds the configured model-server "
                f"limit: {visible} > {limit}. Rebuild the package with a smaller "
                "evidence budget or raise both reader.max_images_per_prompt and "
                "vLLM --limit-mm-per-prompt; evidence is never silently truncated."
            )

    def _output_contract(self, package: EvidencePackage) -> str:
        """Repeat schema-critical rules as the final user text block."""

        no_rag = package.input_mode == "no_rag"
        llava_profile = self._llava_med_output_profile(package)
        closed_answer_space = _package_closed_answer_space(package)
        if llava_profile != "full":
            representation = self._llava_med_answer_representation(package)
            if closed_answer_space:
                answer_rule = (
                    "Set `answer` to exactly one of "
                    + json.dumps(closed_answer_space, ensure_ascii=False)
                    + ", with no explanation in the `answer` field."
                )
            elif representation == "option_text":
                answer_rule = (
                    "Set `answer` to the exact text of one displayed option, "
                    "without its letter label."
                )
            elif representation == "option_id":
                answer_rule = "Set `answer` to exactly one displayed option letter."
            else:
                answer_rule = "Set `answer` to a non-empty brief answer."

            evidence_map, group_map = self._llava_med_citation_maps(package)
            keys = ["answer", "confidence"]
            citation_rules: list[str] = []
            if llava_profile != "no_rag" and evidence_map:
                keys.append("used_reference_numbers")
                citation_rules.append(
                    "`used_reference_numbers` must be a JSON list containing only "
                    f"integers from 1 through {len(evidence_map)}; use [] when no "
                    "reference materially influenced the decision."
                )
            if llava_profile == "structured_rag" and group_map:
                keys.append("used_group_numbers")
                citation_rules.append(
                    "`used_group_numbers` must be a JSON list containing only "
                    f"integers from 1 through {len(group_map)}; use [] when no "
                    "semantic contrast group materially influenced the decision."
                )
            rendered_keys = ", ".join(f"`{key}`" for key in keys)
            return (
                "OUTPUT CONTRACT\n"
                f"Return one compact JSON object with exactly these keys: {rendered_keys}. "
                + answer_rule
                + " `confidence` must be a bare JSON number from 0 to 1. "
                + " ".join(citation_rules)
                + " Do not add rationale, descriptors, Markdown, or prose. Close "
                "the JSON object immediately."
            )

        option_ids = list(option_id_to_text(package.options))
        if option_ids:
            answer_rule = (
                "Set `answer` to exactly one of "
                + json.dumps(option_ids, ensure_ascii=False)
                + ". Choose one displayed option even when uncertain; express "
                "uncertainty only through `confidence`. Do not answer with "
                "none, unknown, or an explanation unless it is a displayed "
                "option ID."
            )
            example_answer = option_ids[0]
        elif closed_answer_space:
            answer_rule = (
                "Set `answer` to exactly one of "
                + json.dumps(closed_answer_space, ensure_ascii=False)
                + ". Do not put an explanation in the `answer` field."
            )
            example_answer = closed_answer_space[0]
        else:
            answer_rule = "Set `answer` to a non-empty JSON string."
            example_answer = "brief answer"

        if getattr(self, "is_llava_med", False):
            no_rag_rule = ""
            if no_rag:
                no_rag_rule = (
                    " For no-RAG, `used_evidence` and "
                    "`used_semantic_contrast_groups` must be [] and "
                    "`evidence_use_type` must be \"none\"."
                )
            return (
                "OUTPUT CONTRACT (MANDATORY)\n"
                "Return exactly one compact JSON object with these nine keys: "
                "`answer`, `finding`, `anatomy`, `laterality`, `confidence`, "
                "`used_evidence`, `used_semantic_contrast_groups`, "
                "`evidence_use_type`, and `rationale`. "
                + answer_rule
                + " `confidence` must be a bare JSON number in [0,1]. "
                "Use `unspecified` only when a descriptor cannot be determined "
                "or is not applicable. `rationale` must be one brief sentence "
                "based on the query image. Close the final string and JSON "
                "object immediately."
                + no_rag_rule
            )

        example = {
            "answer": example_answer,
            "finding": "unspecified",
            "anatomy": "unspecified",
            "laterality": "unspecified",
            "confidence": 0.73,
            "used_evidence": [],
            "used_semantic_contrast_groups": [],
            "evidence_use_type": "none" if no_rag else "supporting",
            "rationale": "brief reason",
        }
        no_rag_rule = ""
        rag_citation_rule = ""
        if no_rag:
            no_rag_rule = (
                " For no-RAG, `used_evidence` and "
                "`used_semantic_contrast_groups` must be [] and "
                "`evidence_use_type` must be \"none\"."
            )
        else:
            reference_count = len(self._ordered_evidence_for_presentation(package))
            if reference_count:
                rag_citation_rule = (
                    " When citing retrieved material, `used_evidence` may contain "
                    f"only handles `reference_1` through `reference_{reference_count}`; "
                    "use [] if no reference materially influenced the answer."
                )
            else:
                rag_citation_rule = " `used_evidence` must be []."
            if package.semantic_contrast_groups:
                rag_citation_rule += (
                    " `used_semantic_contrast_groups` may contain only handles "
                    f"`group_1` through `group_{len(package.semantic_contrast_groups)}`."
                )
            else:
                rag_citation_rule += (
                    " `used_semantic_contrast_groups` must be []."
                )
        return (
            "OUTPUT CONTRACT (MANDATORY)\n"
            "Return exactly one JSON object with all nine required fields. "
            + answer_rule
            + " `confidence` is the estimated probability that `answer` is "
            "correct and must be a bare JSON number from 0 through 1. "
            "Valid: `\"confidence\": 0.73`. Invalid: `\"confidence\": "
            "\"0.73\"`, `\"confidence\": \"high\"`, "
            "`\"confidence\": \"73%\"`, or `\"confidence\": null`. "
            "Use `unspecified` for finding, anatomy, or laterality when not "
            "applicable."
            + no_rag_rule
            + rag_citation_rule
            + "\nExample shape:\n"
            + json.dumps(example, ensure_ascii=False, separators=(",", ":"))
        )

    def _no_rag_messages(self, package: EvidencePackage) -> list[dict[str, Any]]:
        if getattr(self, "is_llava_med", False):
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": LLAVA_MED_NO_RAG_SYSTEM_PROMPT,
                },
                self._image_content(package.query_image_path),
                {
                    "type": "text",
                    "text": self._llava_med_clean_question(package),
                },
                {
                    "type": "text",
                    "text": self._output_contract(package),
                },
            ]
            # LLaVA-Med-v1.5 is Mistral-instruct based. Keep one multimodal
            # user turn instead of relying on a separately rendered system role.
            return [{"role": "user", "content": content}]

        content = [
            {"type": "text", "text": "QUERY IMAGE"},
            self._image_content(package.query_image_path),
            {
                "type": "text",
                "text": self._llava_med_clean_question(package),
            },
            {"type": "text", "text": self._output_contract(package)},
        ]
        return [
            {"role": "system", "content": NO_RAG_READER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _compact_target_payload(package: EvidencePackage) -> dict[str, Any]:
        """Expose only decision-relevant selector output to the reader."""

        if package.question_spec is None:
            return {}
        spec = package.question_spec
        payload: dict[str, Any] = {
            "target_finding": spec.target_finding,
            "target_anatomy": spec.target_anatomy,
            "target_laterality": spec.target_laterality,
            "target_attribute": spec.target_attribute,
            "target_scope": spec.target_scope,
            "comparison_mode": spec.comparison_mode,
            "memory_values_to_compare": list(spec.required_attribute_values),
        }
        if spec.comparison_mode == "candidate_predicate":
            payload["answer_candidates"] = [
                {
                    "option_id": item.option_id,
                    "canonical_value": item.canonical_value,
                    "finding": item.finding,
                    "anatomy": item.anatomy,
                    "laterality": item.laterality,
                }
                for item in spec.answer_candidates
            ]
        return payload

    @staticmethod
    def _compact_event_payload(event: dict[str, Any]) -> dict[str, Any]:
        """Remove graph/audit provenance while retaining target semantics."""

        if event.get("case_level_attribute") == "normality":
            provenance = event.get("normality_provenance", {})
            return {
                "case_level_attribute": "normality",
                "abnormality_presence": event.get("abnormality_presence"),
                "evidence_basis": (
                    provenance.get("reference_basis")
                    if isinstance(provenance, dict)
                    else event.get("derivation")
                ),
                "assertion_scope": provenance.get("assertion_scope") if isinstance(provenance, dict) else None,
                "supporting_findings": provenance.get("supporting_findings", []) if isinstance(provenance, dict) else [],
                "supporting_temporal_statuses": provenance.get("supporting_temporal_statuses", []) if isinstance(provenance, dict) else [],
                "supporting_event_assertions": [
                    {key: assertion.get(key) for key in (
                        "finding", "anatomy", "laterality", "presence", "uncertainty", "temporal_status"
                    ) if key in assertion}
                    for assertion in provenance.get("supporting_event_assertions", [])
                    if isinstance(assertion, dict)
                ] if isinstance(provenance, dict) else [],
            }
        allowed = (
            "finding",
            "anatomy",
            "laterality",
            "presence",
            "severity",
            "uncertainty",
            "temporal_status",
            "presence_basis",
            "presence_source_finding",
        )
        payload = {key: event.get(key) for key in allowed if key in event}
        support = event.get("laterality_predicate_support")
        if isinstance(support, dict):
            payload["laterality_predicate_support"] = {"basis": support.get("basis")}
        return payload

    @staticmethod
    def _retrieval_origin(item: Evidence) -> str:
        if item.boundary_hop_count == 0 and item.case_id == item.seed_case_id:
            return "DIRECT_VISUAL_SEED"
        return "GRAPH_EXPANDED_CASE"

    def _rag_messages(self, package: EvidencePackage) -> list[dict[str, Any]]:
        if package.question_spec is None:
            raise ValueError("RAG reader package lacks QuestionSpec")

        content: list[dict[str, Any]] = []
        if getattr(self, "is_llava_med", False):
            content.append(
                {"type": "text", "text": LLAVA_MED_STRUCTURED_RAG_SYSTEM_PROMPT}
            )

        # All RAG methods use the same high-level layout: query image/question
        # first, method-native references second, and one shared output contract
        # last. Package-stored formatting instructions are deliberately removed
        # by _llava_med_clean_question so stale packages cannot create a prompt
        # mismatch or duplicate the final contract.
        content.extend(
            [
                {"type": "text", "text": "QUERY IMAGE"},
                self._image_content(package.query_image_path),
                {
                    "type": "text",
                    "text": self._llava_med_clean_question(package),
                },
                {
                    "type": "text",
                    "text": "TARGET ATTRIBUTE\n"
                    + json.dumps(
                        self._compact_target_payload(package),
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ]
        )

        if package.audit_metadata.get("retrieval_policy") == POLICY_VERSION:
            content.append({
                "type": "text",
                "text": "QUERY AND REFERENCE INTERPRETATION\n" + QUESTION_CONDITIONED_DECISION_GUIDANCE,
            })
            predicate = package.audit_metadata.get("question_conditioned", {}).get("laterality_predicate")
            if predicate and predicate.get("supported"):
                content.append({
                    "type": "text",
                    "text": "QUESTION PREDICATE\n" + json.dumps({
                        "predicate": predicate.get("predicate"),
                        "positive_values": predicate.get("positive_values"),
                        "negative_values": predicate.get("negative_values"),
                        "unilateral_bucket": "A derived comparison bucket for explicitly supported one-sided involvement. Raw left/right labels alone are insufficient; this is not an extracted ontology label.",
                    }, ensure_ascii=False),
                })

        ordered_evidence = self._ordered_evidence(package)
        group_numbers = {
            group.group_id: index
            for index, group in enumerate(package.semantic_contrast_groups, 1)
        }
        reference_numbers = {
            item.case_id: index for index, item in enumerate(ordered_evidence, 1)
        }
        group_manifest = {
            group.group_id: {
                "group_number": group_numbers[group.group_id],
                "target_attribute": group.target_attribute,
                "member_reference_numbers": [
                    reference_numbers[case_id]
                    for case_id in group.member_case_ids
                    if case_id in reference_numbers
                ],
                "member_target_values": list(group.member_attribute_values),
                "instruction": (
                    "Compare the query target with every member; the two memory "
                    "values are exemplars, not votes or query labels."
                ),
            }
            for group in package.semantic_contrast_groups
        }
        announced_groups: set[str] = set()
        for index, item in enumerate(ordered_evidence, 1):
            group_id = item.semantic_contrast_group_id
            if group_id and group_id not in announced_groups:
                content.append(
                    {
                        "type": "text",
                        "text": "SEMANTIC CONTRAST GROUP\n"
                        + json.dumps(
                            group_manifest.get(group_id, {}),
                            ensure_ascii=False,
                            indent=2,
                        ),
                    }
                )
                announced_groups.add(group_id)

            metadata: dict[str, Any] = {
                "reference_number": index,
                "target_attribute": item.target_attribute,
                "memory_case_attribute_value": item.attribute_value,
                "comparison_role": item.comparison_role,
                "semantic_contrast_group_number": (
                    group_numbers.get(group_id) if group_id else None
                ),
                "retrieval_origin": self._retrieval_origin(item),
            }
            if package.audit_metadata.get("retrieval_policy") == POLICY_VERSION:
                # Symmetric exemplars: seed status is recorded only in the
                # package audit and never confers model-visible authority.
                metadata.pop("retrieval_origin")
            if self.reader_cfg.get("include_event_metadata", True):
                metadata["target_event"] = self._compact_event_payload(item.event)
            if item.event.get("presence_basis"):
                # This distinction is mandatory even when optional event fields
                # are hidden: an inferred absence is not an explicit assertion.
                metadata["presence_basis"] = item.event["presence_basis"]
                metadata["presence_source_finding"] = item.event.get("presence_source_finding")
            laterality_support = item.event.get("laterality_predicate_support")
            if isinstance(laterality_support, dict):
                # Opposite-side absence is separate evidence, not a missing key.
                # Keep source IDs in audit only and make its basis explicit.
                metadata["laterality_predicate_support"] = {"basis": laterality_support.get("basis")}
                if self.reader_cfg.get("include_evidence_span", True):
                    metadata["laterality_predicate_support"]["supporting_evidence_spans"] = list(dict.fromkeys(
                        str(span).strip() for span in laterality_support.get("supporting_evidence_spans", [])
                        if str(span).strip()
                    ))
            if self.reader_cfg.get("include_evidence_span", True):
                metadata["evidence_span"] = item.evidence_span
                if item.event.get("case_level_attribute") == "normality":
                    provenance = item.event.get("normality_provenance", {})
                    spans = provenance.get("supporting_evidence_spans", []) if isinstance(provenance, dict) else []
                    metadata["supporting_evidence_spans"] = list(dict.fromkeys(
                        str(span).strip() for span in spans if str(span).strip()
                    ))

            content.append(
                {
                    "type": "text",
                    "text": "REFERENCE CASE ANNOTATION\n"
                    + json.dumps(metadata, ensure_ascii=False, indent=2),
                }
            )
            content.append(self._image_content(item.image_path))

        content.append({"type": "text", "text": self._output_contract(package)})
        if getattr(self, "is_llava_med", False):
            # LLaVA-Med-v1.5 is Mistral-instruct based.  Keep one multimodal
            # user turn for every condition instead of relying on a system role.
            return [{"role": "user", "content": content}]
        return [
            {"role": "system", "content": READER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def _retrieved_reference_messages(
        self, package: EvidencePackage
    ) -> list[dict[str, Any]]:
        """Render baseline-native references without ViSiG-CR-only metadata.

        Published comparison packages omit ``question_spec`` entirely. Selector
        output, graph paths, ontology events, coverage, and semantic-group state
        are intentionally absent from model input.
        """

        content: list[dict[str, Any]] = []
        if getattr(self, "is_llava_med", False):
            content.append(
                {"type": "text", "text": LLAVA_MED_COMPARISON_RAG_SYSTEM_PROMPT}
            )
        content.extend(
            [
                {"type": "text", "text": "QUERY IMAGE"},
                self._image_content(package.query_image_path),
                {
                    "type": "text",
                    "text": self._llava_med_clean_question(package),
                },
            ]
        )
        if package.query_context.strip():
            content.append(
                {
                    "type": "text",
                    "text": (
                        "BASELINE-NATIVE QUERY CONTEXT\n"
                        + package.query_context.strip()
                    ),
                }
            )
        for index, item in enumerate(package.evidence, 1):
            content.append(
                {
                    "type": "text",
                    # Retrieval scores, component signals, case IDs, graph
                    # metadata, and OT costs stay in the serialized audit
                    # package. Published baselines do not expose them to the
                    # reader unless their original reader contract requires it.
                    "text": f"RETRIEVED REFERENCE {index}",
                }
            )
            if item.reference_text.strip():
                content.append(
                    {
                        "type": "text",
                        "text": "REFERENCE REPORT\n" + item.reference_text.strip(),
                    }
                )
            if item.reader_visible_image:
                content.append(self._image_content(item.image_path))

        content.append({"type": "text", "text": self._output_contract(package)})
        if getattr(self, "is_llava_med", False):
            return [{"role": "user", "content": content}]
        return [
            {"role": "system", "content": COMPARISON_RAG_READER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    def build_messages(self, package: EvidencePackage) -> list[dict[str, Any]]:
        """Build exact model messages and enforce the declared image budget."""

        package = self._effective_reader_package(package)
        self._validate_image_budget(package)
        if package.input_mode == "no_rag":
            return self._no_rag_messages(package)
        if package.reader_presentation == "retrieved_references":
            return self._retrieved_reference_messages(package)
        return self._rag_messages(package)

    @staticmethod
    def _effective_reader_package(package: EvidencePackage) -> EvidencePackage:
        policy = package.audit_metadata.get("retrieval_policy")
        if package.method == "visig_cr" and policy == "question_conditioned_contrast_v1":
            raise ValueError(
                "Legacy question-conditioned v1 packages use an unsafe evidence contract; "
                "regenerate packages with the current retrieval policy in a fresh run directory."
            )
        if package.method == "visig_cr" and policy == POLICY_VERSION:
            for item in package.evidence:
                if item.event.get("case_level_attribute") != "normality":
                    continue
                provenance = item.event.get("normality_provenance", {})
                if (
                    not isinstance(provenance, dict)
                    or provenance.get("policy_id") != CLINICAL_NORMALITY_REFERENCE_POLICY_ID
                    or provenance.get("assertion_scope") != "any_abnormality"
                    or not item.event.get("is_clinical_normality_reference")
                ):
                    raise ValueError(
                        "Normality evidence lacks the current scope-safe contract; "
                        "regenerate packages instead of reusing old normal endpoints."
                    )
        fallback = package.audit_metadata.get("contrast_fallback", {})
        if (
            package.method == "visig_cr"
            and package.audit_metadata.get("retrieval_policy") == POLICY_VERSION
            and fallback.get("used") is True
            and fallback.get("reader_input") == "no_rag"
        ):
            if package.evidence or package.semantic_contrast_groups:
                raise ValueError("Query-only fallback must not contain evidence")
            return EvidencePackage(
                input_mode="no_rag", method="no_rag", query_id=package.query_id,
                query_image_path=package.query_image_path, question=package.question,
                retrieval_question=package.retrieval_question, options=package.options,
                audit_metadata=package.audit_metadata,
            )
        return package

    @staticmethod
    def _message_layout_name(package: EvidencePackage) -> str:
        package = VLMReader._effective_reader_package(package)
        if package.input_mode == "no_rag":
            return "no_rag_query_only_v1"
        if package.reader_presentation == "retrieved_references":
            return "retrieved_references_query_first_v2"
        if package.audit_metadata.get("retrieval_policy") == POLICY_VERSION:
            return "question_conditioned_query_first_contrast_v2"
        return "structured_query_first_compact_references_v4"

    def reader_protocol_sha256(self, package: EvidencePackage) -> str:
        """Hash reader instructions/layout independently from package content."""

        package = self._effective_reader_package(package)
        if package.input_mode == "no_rag":
            system_prompt = (
                LLAVA_MED_NO_RAG_SYSTEM_PROMPT
                if getattr(self, "is_llava_med", False)
                else NO_RAG_READER_SYSTEM_PROMPT
            )
        else:
            if package.reader_presentation == "retrieved_references":
                system_prompt = (
                    LLAVA_MED_COMPARISON_RAG_SYSTEM_PROMPT
                    if getattr(self, "is_llava_med", False)
                    else COMPARISON_RAG_READER_SYSTEM_PROMPT
                )
            else:
                system_prompt = (
                    LLAVA_MED_STRUCTURED_RAG_SYSTEM_PROMPT
                    if getattr(self, "is_llava_med", False)
                    else READER_SYSTEM_PROMPT
                )
        payload = {
            "model_family": getattr(self, "model_family", "qwen"),
            "message_layout": self._message_layout_name(package),
            "system_prompt": system_prompt,
            "output_contract": self._output_contract(package),
            "include_event_metadata": bool(
                self.reader_cfg.get("include_event_metadata", True)
            ),
            "include_evidence_span": bool(
                self.reader_cfg.get("include_evidence_span", True)
            ),
        }
        if (
            package.audit_metadata.get("retrieval_policy") == POLICY_VERSION
            and package.reader_presentation != "retrieved_references"
            and package.input_mode != "no_rag"
        ):
            payload["question_conditioned_decision_guidance"] = QUESTION_CONDITIONED_DECISION_GUIDANCE
            payload["reference_order"] = {
                "policy": self.reader_cfg.get("question_conditioned_reference_order", "balanced"),
                "seed": int(self.reader_cfg.get("order_seed", 42)),
                "randomize_group_order": bool(self.reader_cfg.get("randomize_group_order", False)),
                "randomize_within_group": bool(self.reader_cfg.get("randomize_within_group", False)),
                "randomize_ungrouped_order": bool(self.reader_cfg.get("randomize_ungrouped_order", False)),
            }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _validated_result(
        raw: dict[str, Any],
        *,
        no_rag: bool,
        options: list[str] | tuple[str, ...] = (),
        answer_space: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        result = dict(raw)
        closed_values = _canonical_closed_answer_space(answer_space)
        if options:
            answer = normalize_option_answer(result.get("answer"), options)
        elif closed_values:
            answer = normalize_binary_answer(result.get("answer"), mode="leading")
        else:
            answer = normalize_option_answer(result.get("answer"), options)

        raw_confidence = result.get("confidence")
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence, (int, float)
        ):
            raise ValueError(
                "Reader confidence must be a bare JSON number; got "
                + _safe_value_repr(raw_confidence)
                + f" ({type(raw_confidence).__name__})"
            )
        try:
            confidence = float(raw_confidence)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "Reader confidence must be finite and within [0, 1]; got "
                + _safe_value_repr(raw_confidence)
            ) from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Reader confidence must be finite and within [0, 1]; got "
                + _safe_value_repr(raw_confidence)
            )

        used_evidence = result.get("used_evidence", [])
        used_groups = result.get("used_semantic_contrast_groups", [])
        if not isinstance(used_evidence, list) or not isinstance(used_groups, list):
            raise ValueError("Reader evidence-use fields must be lists")
        if not all(isinstance(item, str) for item in used_evidence + used_groups):
            raise ValueError("Reader evidence-use list items must be strings")

        raw_evidence_use_type = result.get("evidence_use_type")
        if not isinstance(raw_evidence_use_type, str):
            raise ValueError("Reader evidence_use_type must be a JSON string")
        evidence_use_type = raw_evidence_use_type.strip()
        if not evidence_use_type:
            raise ValueError("Reader evidence_use_type must be non-empty")
        if no_rag and (used_evidence or used_groups):
            raise ValueError("no-RAG reader output must not cite retrieved evidence")
        if no_rag and evidence_use_type != "none":
            raise ValueError("no-RAG evidence_use_type must be 'none'")

        def normalized_descriptor(field: str) -> str:
            value = result.get(field, "unspecified")
            if not isinstance(value, str):
                raise ValueError(f"Reader {field} must be a JSON string")
            return value.strip() or "unspecified"

        raw_rationale = result.get("rationale", result.get("concise rationale"))
        if not isinstance(raw_rationale, str) or not raw_rationale.strip():
            raise ValueError("Reader rationale must be a non-empty JSON string")

        if (
            answer == "A"
            and normalized_descriptor("finding") == "unspecified"
            and normalized_descriptor("anatomy") == "unspecified"
            and normalized_descriptor("laterality") == "unspecified"
            and confidence == 0.73
            and used_evidence == []
            and used_groups == []
            and evidence_use_type == "none"
            and raw_rationale.strip().casefold() == "brief reason"
        ):
            raise ValueError(
                "Reader appears to have copied the legacy output example"
            )

        result.update(
            {
                "answer": answer,
                "confidence": confidence,
                "finding": normalized_descriptor("finding"),
                "anatomy": normalized_descriptor("anatomy"),
                "laterality": normalized_descriptor("laterality"),
                "used_evidence": used_evidence,
                "used_semantic_contrast_groups": used_groups,
                "evidence_use_type": evidence_use_type,
                "rationale": raw_rationale.strip(),
            }
        )
        return result

    def _canonicalize_full_schema_citations(
        self,
        result: dict[str, Any],
        package: EvidencePackage,
    ) -> dict[str, Any]:
        """Map compact model-visible handles back to durable package IDs."""

        if package.input_mode == "no_rag":
            return result
        canonical = dict(result)
        evidence_map = {
            f"reference_{index}": (item.event_id or item.case_id)
            for index, item in enumerate(
                self._ordered_evidence_for_presentation(package),
                1,
            )
        }
        group_map = {
            f"group_{index}": group.group_id
            for index, group in enumerate(package.semantic_contrast_groups, 1)
        }
        canonical["used_evidence"] = [
            evidence_map.get(str(value).strip().casefold(), value)
            for value in canonical.get("used_evidence", [])
        ]
        canonical["used_semantic_contrast_groups"] = [
            group_map.get(str(value).strip().casefold(), value)
            for value in canonical.get("used_semantic_contrast_groups", [])
        ]
        return canonical

    @staticmethod
    def _validated_llava_med_no_rag_result(
        raw: dict[str, Any],
        *,
        options: list[str] | tuple[str, ...] = (),
        answer_space: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validate a compact query-only decision and fill canonical metadata."""

        result = dict(raw)
        closed_values = _canonical_closed_answer_space(answer_space)
        if options:
            answer = normalize_option_answer(result.get("answer"), options)
        elif closed_values:
            answer = normalize_binary_answer(result.get("answer"), mode="leading")
        else:
            answer = normalize_option_answer(result.get("answer"), options)
        raw_confidence = result.get("confidence")
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence,
            (int, float),
        ):
            raise ValueError(
                "LLaVA-Med confidence must be a bare JSON number; got "
                + _safe_value_repr(raw_confidence)
                + f" ({type(raw_confidence).__name__})"
            )
        try:
            confidence = float(raw_confidence)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                "LLaVA-Med confidence must be finite and within [0, 1]; got "
                + _safe_value_repr(raw_confidence)
            ) from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "LLaVA-Med confidence must be finite and within [0, 1]; got "
                + _safe_value_repr(raw_confidence)
            )

        return {
            "answer": answer,
            "finding": "not_collected",
            "anatomy": "not_collected",
            "laterality": "not_collected",
            "confidence": confidence,
            "used_evidence": [],
            "used_semantic_contrast_groups": [],
            "evidence_use_type": "none",
            "rationale": "not_collected",
        }

    @staticmethod
    def _validated_compact_index_list(
        raw_value: Any,
        *,
        field: str,
        maximum: int,
    ) -> list[int]:
        if not isinstance(raw_value, list):
            raise ValueError(f"LLaVA-Med {field} must be a JSON list")
        values: list[int] = []
        for value in raw_value:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"LLaVA-Med {field} items must be integers; got "
                    + _safe_value_repr(value)
                )
            if not 1 <= value <= maximum:
                raise ValueError(
                    f"LLaVA-Med {field} item {value} is outside [1, {maximum}]"
                )
            values.append(value)
        if len(values) != len(set(values)):
            raise ValueError(f"LLaVA-Med {field} must not contain duplicates")
        return values

    def _validated_llava_med_compact_result(
        self,
        raw: dict[str, Any],
        *,
        package: EvidencePackage,
        profile: str,
    ) -> dict[str, Any]:
        """Map a compact model decision to the canonical persisted schema."""

        canonical = self._validated_llava_med_no_rag_result(
            raw,
            options=package.options,
            answer_space=_package_closed_answer_space(package),
        )
        if profile == "no_rag":
            return canonical

        evidence_map, group_map = self._llava_med_citation_maps(package)
        used_reference_numbers: list[int] = []
        if evidence_map:
            used_reference_numbers = self._validated_compact_index_list(
                raw.get("used_reference_numbers"),
                field="used_reference_numbers",
                maximum=len(evidence_map),
            )
        elif "used_reference_numbers" in raw:
            raise ValueError(
                "LLaVA-Med returned used_reference_numbers although no reference "
                "was model-visible"
            )

        used_group_numbers: list[int] = []
        if profile == "structured_rag" and group_map:
            used_group_numbers = self._validated_compact_index_list(
                raw.get("used_group_numbers"),
                field="used_group_numbers",
                maximum=len(group_map),
            )
        elif "used_group_numbers" in raw:
            raise ValueError(
                "LLaVA-Med returned used_group_numbers although no group citation "
                "field was requested"
            )

        canonical["used_evidence"] = [
            evidence_map[number] for number in used_reference_numbers
        ]
        canonical["used_semantic_contrast_groups"] = [
            group_map[number] for number in used_group_numbers
        ]
        if used_group_numbers:
            canonical["evidence_use_type"] = "grouped_comparison"
        elif used_reference_numbers:
            canonical["evidence_use_type"] = "supporting"
        else:
            canonical["evidence_use_type"] = "none"
        return canonical

    def answer(self, package: EvidencePackage) -> dict[str, Any]:
        effective = self._effective_reader_package(package)
        if effective is not package:
            result = self.answer(effective)
            result.update({
                "input_mode": package.input_mode, "retrieval_method": package.method,
                "coverage_status": package.coverage_status,
                "evidence_package_sha256": evidence_package_sha256(package),
                "retrieval_warnings": package.warnings, "audit_metadata": package.audit_metadata,
            })
            result["reader_output_audit"]["effective_input_mode"] = "no_rag"
            return result
        no_rag = package.input_mode == "no_rag"
        llava_profile = self._llava_med_output_profile(package)
        compact_llava = llava_profile != "full"
        evidence_map, group_map = self._llava_med_citation_maps(package)

        answer_representation = "option_id"
        closed_answer_space, answer_space_source = (
            _resolved_package_closed_answer_space(package)
        )
        if compact_llava:
            response_format, answer_representation = llava_med_compact_response_format(
                package.options,
                answer_representation=str(
                    self.reader_cfg.get(
                        "llava_med_answer_representation",
                        "option_text",
                    )
                ),
                answer_space=closed_answer_space,
                profile=llava_profile,
                reference_count=(0 if no_rag else len(evidence_map)),
                group_count=(
                    len(group_map) if llava_profile == "structured_rag" else 0
                ),
            )
        else:
            response_format = reader_response_format(
                package.options,
                no_rag=no_rag,
                answer_space=closed_answer_space,
            )

        retry_key = (
            "llava_med_validation_retries"
            if getattr(self, "is_llava_med", False)
            else "validation_retries"
        )
        default_validation_retries = (
            0 if getattr(self, "is_llava_med", False) else 2
        )
        validation_retries = max(
            0,
            min(
                5,
                int(
                    self.reader_cfg.get(
                        retry_key,
                        default_validation_retries,
                    )
                ),
            ),
        )
        record_raw = bool(self.reader_cfg.get("record_raw_responses", True))
        raw_limit_key = (
            "llava_med_raw_response_max_chars"
            if getattr(self, "is_llava_med", False)
            else "raw_response_max_chars"
        )
        raw_limit = max(0, int(self.reader_cfg.get(raw_limit_key, 0)))
        base_messages = self.build_messages(package)

        request_max_tokens: int | None = None
        if compact_llava:
            if no_rag:
                request_max_tokens = max(
                    16,
                    min(
                        256,
                        int(
                            self.reader_cfg.get(
                                "llava_med_no_rag_max_tokens",
                                128,
                            )
                        ),
                    ),
                )
            else:
                request_max_tokens = max(
                    32,
                    min(
                        768,
                        int(
                            self.reader_cfg.get(
                                "llava_med_rag_max_tokens",
                                256,
                            )
                        ),
                    ),
                )
        elif getattr(self, "is_llava_med", False) and no_rag:
            request_max_tokens = max(
                64,
                min(
                    512,
                    int(
                        self.reader_cfg.get(
                            "llava_med_no_rag_max_tokens",
                            256,
                        )
                    ),
                ),
            )

        attempts: list[dict[str, Any]] = []
        result: dict[str, Any] | None = None
        last_error: ValueError | None = None

        def audited_raw(value: str) -> str:
            if raw_limit and len(value) > raw_limit:
                omitted = len(value) - raw_limit
                return value[:raw_limit] + f"...[truncated {omitted} chars]"
            return value

        required_output_fields = list(
            response_format["json_schema"]["schema"].get("required", [])
        )
        required_output_text = ", ".join(
            f"`{field}`" for field in required_output_fields
        )

        for attempt_index in range(validation_retries + 1):
            # Every retry starts again from the original image/question prompt.
            # A malformed completion is never inserted as an assistant turn.
            messages = copy.deepcopy(base_messages)
            if attempt_index > 0:
                correction = (
                    "The prior response was incomplete or invalid. Return one "
                    "closed JSON object with exactly these fields: "
                    + required_output_text
                    + ". Follow the output contract and add no prose."
                )
                if closed_answer_space:
                    correction += (
                        " Set `answer` to exactly one of "
                        + json.dumps(closed_answer_space, ensure_ascii=False)
                        + "; do not place an explanation in that field."
                    )
                final_content = messages[-1].get("content")
                if isinstance(final_content, list):
                    final_content.append(
                        {
                            "type": "text",
                            "text": "RETRY INSTRUCTION\n" + correction,
                        }
                    )
                else:
                    messages.append({"role": "user", "content": correction})

            completion = None
            try:
                completion = self.client.complete_json_response(
                    messages,
                    max_tokens=request_max_tokens,
                    response_format=response_format,
                    allow_unconstrained_fallback=False,
                )
                if compact_llava:
                    validated = self._validated_llava_med_compact_result(
                        completion.parsed,
                        package=package,
                        profile=llava_profile,
                    )
                else:
                    validated = self._validated_result(
                        completion.parsed,
                        no_rag=no_rag,
                        options=package.options,
                        answer_space=closed_answer_space,
                    )
                    validated = self._canonicalize_full_schema_citations(
                        validated,
                        package,
                    )
            except ModelResponseParseError as exc:
                last_error = exc
                attempt_record: dict[str, Any] = {
                    "attempt": attempt_index + 1,
                    "finish_reason": exc.finish_reason,
                    "parsed_output": None,
                    "validation_error": str(exc),
                }
                if record_raw:
                    attempt_record["raw_content"] = audited_raw(exc.raw_content)
                attempts.append(attempt_record)
            except ValueError as exc:
                # Transport/API exceptions are intentionally not retried.
                last_error = exc
                if completion is None:
                    raise
                attempt_record = {
                    "attempt": attempt_index + 1,
                    "finish_reason": completion.finish_reason,
                    "parsed_output": completion.parsed,
                    "validation_error": str(exc),
                }
                if record_raw:
                    attempt_record["raw_content"] = audited_raw(
                        completion.raw_content
                    )
                attempts.append(attempt_record)
            else:
                attempt_record = {
                    "attempt": attempt_index + 1,
                    "finish_reason": completion.finish_reason,
                    "parsed_output": completion.parsed,
                    "validation_error": None,
                }
                if record_raw:
                    attempt_record["raw_content"] = audited_raw(
                        completion.raw_content
                    )
                attempts.append(attempt_record)
                result = validated
                break

        if compact_llava:
            output_adapter = {
                "no_rag": "llava_med_compact_no_rag_v2",
                "retrieved_references": (
                    "llava_med_compact_retrieved_references_v1"
                ),
                "structured_rag": "llava_med_compact_structured_rag_v1",
            }[llava_profile]
            model_generated_fields = required_output_fields
            postfilled_fields = [
                "finding",
                "anatomy",
                "laterality",
                "used_evidence",
                "used_semantic_contrast_groups",
                "evidence_use_type",
                "rationale",
            ]
            rationale_collected = False
        else:
            output_adapter = "full_reader_schema_v1"
            model_generated_fields = [
                "answer",
                "finding",
                "anatomy",
                "laterality",
                "confidence",
                "used_evidence",
                "used_semantic_contrast_groups",
                "evidence_use_type",
                "rationale",
            ]
            postfilled_fields = []
            rationale_collected = True

        audit_record = {
            "attempt_count": len(attempts),
            "validation_retries_configured": validation_retries,
            "schema_name": response_format["json_schema"]["name"],
            "model_family": getattr(self, "model_family", "qwen"),
            "request_max_tokens": (
                request_max_tokens
                if request_max_tokens is not None
                else int(self.model_cfg.get("max_tokens", 1200))
            ),
            "output_adapter": output_adapter,
            "llava_med_profile": llava_profile if compact_llava else None,
            "answer_representation": answer_representation,
            "resolved_answer_space": list(closed_answer_space),
            "answer_space_source": answer_space_source,
            "answer_schema": response_format["json_schema"]["schema"]
            ["properties"]["answer"],
            "model_generated_fields": model_generated_fields,
            "postfilled_fields": postfilled_fields,
            "rationale_collected": rationale_collected,
            "model_visible_image_count": self._model_visible_image_count(package),
            "multi_image_input": self._model_visible_image_count(package) > 1,
            "multi_image_policy": (
                "llava_med_sensitivity_smoke_required"
                if compact_llava and self._model_visible_image_count(package) > 1
                else "standard"
            ),
            "message_layout": self._message_layout_name(package),
            "reader_protocol_sha256": self.reader_protocol_sha256(package),
            "reference_number_to_case_id": {
                str(index): case_id for index, case_id in evidence_map.items()
            }
            if compact_llava and not no_rag
            else {},
            "group_number_to_group_id": {
                str(index): group_id for index, group_id in group_map.items()
            }
            if compact_llava and llava_profile == "structured_rag"
            else {},
            "attempts": attempts,
        }
        if result is None:
            raise ReaderOutputValidationError(
                "Reader output validation failed after "
                f"{len(attempts)} attempt(s): {last_error}",
                audit_record=audit_record,
            ) from last_error

        result.update(
            {
                "query_id": package.query_id,
                "input_mode": package.input_mode,
                "retrieval_method": package.method,
                "retrieved_case_ids": [
                    item.case_id for item in package.evidence
                ],
                "semantic_contrast_group_ids": [
                    group.group_id
                    for group in package.semantic_contrast_groups
                ],
                "coverage_status": package.coverage_status,
                "evidence_package_sha256": evidence_package_sha256(package),
                "image_normalization": image_normalization_provenance(),
                "retrieval_warnings": package.warnings,
                "audit_metadata": package.audit_metadata,
                "reader_output_audit": audit_record,
            }
        )
        return result
