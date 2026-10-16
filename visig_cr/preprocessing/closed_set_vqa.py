"""Build auditable closed-binary views from normalized VQA artifacts.

This module is intentionally a *post-normalization* protocol layer.  It never
rewrites the source adapters' outputs in place.  Instead, it consumes aligned
``queries/<split>.jsonl`` and ``reference_answers/<split>.jsonl`` files and
materializes a separate binary evaluation view with:

* deterministic yes/no canonicalization,
* raw-reference preservation on the reference side only,
* strict semantic/source-family and question-form eligibility,
* duplicate/conflict handling,
* source/protocol hashes and label distributions,
* stratified manual-audit candidates, and
* explicit exclusions rather than silent row drops.

The resulting query stream remains answer-free and is compatible with the
existing ``batch-retrieve`` / ``batch-reader`` / ``evaluate-reader`` pipeline.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..answer_utils import parse_binary_answer
from ..io_utils import file_sha256, read_jsonl, stable_json_sha256
from .vqa_common import (
    clean_text,
    unique_map,
    write_jsonl,
    write_vqa_split,
)


CLOSED_BINARY_PROTOCOL = "closed_binary_v3"
BINARY_ANSWER_SPACE = ["yes", "no"]


# Main-table binary questions should be phrased as an explicit polar decision.
# The matcher intentionally checks only the beginning of the normalized stem;
# this avoids accepting open questions merely because they contain words such
# as ``present`` or ``seen`` later in the sentence.
_POLAR_QUESTION_PREFIX = re.compile(
    r"^(?:"
    r"is|are|was|were|does|do|did|has|have|had|"
    r"can|could|would|should|will"
    r")\b|^(?:any\s+)?evidence\s+of\b",
    flags=re.IGNORECASE,
)

# Medical-CXR-VQA's source ``abnormality`` family mixes open questions
# (``What abnormalities ...?``), existential global-abnormality questions
# (``Is there evidence of any abnormality?``), and reversed normality
# questions (``Is this image normal?``).  For the main binary-presence view,
# only the existential form has the same yes=present / no=absent direction as
# target-finding presence questions.
_ABNORMALITY_EXISTENTIAL_PREFIX = re.compile(
    r"^(?:is|are)\s+there\b",
    flags=re.IGNORECASE,
)

# A yes/no answer to an explicitly negated predicate reverses the clinical
# interpretation of the selected label (for example, ``yes`` to "Is there no
# edema?").  Those rows are excluded from the default main-table view so the
# target axis remains an ordinary finding-presence decision.  They can still be
# recovered prospectively with ``--allow-negated-question`` for a separately
# audited negation protocol.
_NEGATED_QUESTION = re.compile(
    r"(?:"
    r"\bno\b|\bnot\b|\bwithout\b|\babsent\b|\babsence\s+of\b|"
    r"\bfree\s+of\b|\brule(?:d)?\s+out\b|\bexcluded\b|"
    r"\bcannot\s+exclude\b|\bcan['’]?t\s+exclude\b"
    r")",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ClosedBinarySource:
    """One already-normalized binary artifact pair used by a merge command."""

    name: str
    queries_path: Path
    references_path: Path


@dataclass
class _Candidate:
    query_id: str
    query: dict[str, Any]
    reference: dict[str, Any]
    raw_question: str
    raw_answer: str
    label: str
    parse_rule: str
    dedupe_key: str
    image_identity: str
    original_index: int


def format_binary_question(question: str) -> str:
    """Append a fixed answer contract without changing the retrieval stem."""

    stem = clean_text(question)
    if not stem:
        return stem
    return (
        stem
        + "\n\nIn the required JSON object, set `answer` to exactly `yes` or "
        "`no`; include every other required JSON field."
    )


def _normalized_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _normalized_question_type(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean_text(value).casefold()).strip("_")


def _image_identity(row: Mapping[str, Any]) -> str:
    for key in ("case_id", "dicom_id", "image_id", "study_id"):
        value = clean_text(row.get(key))
        if value:
            return f"{key}:{value}"
    path = clean_text(row.get("image_path", row.get("query_image_path")))
    return f"image_path:{path}"


def _dedupe_key(query: Mapping[str, Any], raw_question: str) -> tuple[str, str]:
    image_identity = _image_identity(query)
    return image_identity, f"{image_identity}|{_normalized_key(raw_question)}"


def _raw_scalar_reference(reference: Mapping[str, Any]) -> tuple[str | None, str | None]:
    for key in ("reference_answers", "answers"):
        if key not in reference:
            continue
        value = reference.get(key)
        if not isinstance(value, (list, tuple, set)):
            return None, "non_array_multi_answer_field"
        values = [clean_text(item) for item in value if clean_text(item)]
        if len(values) != 1:
            return None, "multi_answer_reference"
        return values[0], None
    for key in ("answer", "reference_answer", "reference_answer_text"):
        if key in reference:
            value = reference.get(key)
            if isinstance(value, (list, tuple, set, dict)):
                return None, "non_scalar_reference"
            text = clean_text(value)
            return (text, None) if text else (None, "empty_reference")
    return None, "missing_reference_answer"


def _gold_provenance(dataset: str, override: str | None) -> str:
    if override and clean_text(override):
        return clean_text(override)
    token = _normalized_key(dataset)
    if "expert" in token and "mimic" in token:
        return "radiologist_authored_reference"
    if "medical cxr vqa" in token:
        return "report_derived_llm_generated_reference"
    if "vindr" in token:
        return "api_generated_answer_with_expert_structured_source_labels"
    return "source_dataset_reference"


def _question_family(query: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    return clean_text(
        query.get("question_family", reference.get("question_family", ""))
    ).casefold()


def _question_type(query: Mapping[str, Any], reference: Mapping[str, Any]) -> str:
    return clean_text(
        query.get(
            "source_question_type",
            query.get(
                "question_type",
                reference.get(
                    "source_question_type",
                    reference.get("question_type", ""),
                ),
            ),
        )
    )


def _source_question_family(
    query: Mapping[str, Any], reference: Mapping[str, Any]
) -> str:
    """Return the source dataset's own family before semantic remapping.

    New adapters persist ``source_question_family`` explicitly.  Older
    normalized artifacts did not, so their semantic family is the safest
    backward-compatible fallback.
    """

    return clean_text(
        query.get(
            "source_question_family",
            reference.get(
                "source_question_family",
                query.get("question_family", reference.get("question_family", "")),
            ),
        )
    ).casefold()


def _is_polar_question(question: str) -> bool:
    return bool(_POLAR_QUESTION_PREFIX.match(clean_text(question)))


def _is_explicitly_negated_question(question: str) -> bool:
    return bool(_NEGATED_QUESTION.search(clean_text(question)))


def _is_existential_abnormality_question(question: str) -> bool:
    return bool(_ABNORMALITY_EXISTENTIAL_PREFIX.match(clean_text(question)))


def _binary_question_subtype(source_question_family: str) -> str:
    family = _normalized_question_type(source_question_family)
    if family == "abnormality":
        return "global_abnormality_presence"
    if family == "presence":
        return "target_finding_presence"
    return "other_presence"


def _exclusion_record(
    *,
    query_id: str,
    query: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    reason: str,
    raw_answer: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    query = dict(query or {})
    reference = dict(reference or {})
    record = {
        "query_id": query_id,
        "dataset": clean_text(query.get("dataset", reference.get("dataset"))),
        "source_split": clean_text(
            query.get("source_split", reference.get("source_split"))
        ),
        "case_id": clean_text(query.get("case_id")),
        "study_id": clean_text(query.get("study_id")),
        "dicom_id": clean_text(query.get("dicom_id")),
        "image_id": clean_text(query.get("image_id")),
        "question_type": _question_type(query, reference),
        "source_question_family": _source_question_family(query, reference),
        "question_family": _question_family(query, reference),
        "question": clean_text(
            query.get("retrieval_question", query.get("question", ""))
        ),
        "raw_answer": raw_answer,
        "status": "excluded",
        "reason": reason,
    }
    if detail:
        record["detail"] = detail
    return record


def _canonical_query(
    query: Mapping[str, Any],
    *,
    raw_question: str,
    source_task_type: str,
    source_semantic_question_family: str,
    source_question_family: str,
    polarity_mode: str,
) -> dict[str, Any]:
    result = dict(query)
    result["source_dataset"] = clean_text(
        query.get("source_dataset", query.get("dataset"))
    )
    result["source_task_type"] = source_task_type
    result["source_semantic_question_family"] = source_semantic_question_family
    result["source_question_family"] = source_question_family
    result["binary_question_subtype"] = _binary_question_subtype(
        source_question_family
    )
    result["question"] = format_binary_question(raw_question)
    result["reader_question"] = result["question"]
    result["retrieval_question"] = raw_question
    result["options"] = []
    result["question_family"] = "presence"
    result["task_type"] = "binary"
    result["target_attribute"] = "presence"
    result["reference_format"] = "scalar"
    result["answer_space"] = list(BINARY_ANSWER_SPACE)
    result["closed_set_protocol"] = CLOSED_BINARY_PROTOCOL
    result["conversion_rule_id"] = f"binary_polarity_{polarity_mode}_v3"
    return result


def _canonical_reference(
    reference: Mapping[str, Any],
    *,
    raw_answer: str,
    label: str,
    parse_rule: str,
    source_task_type: str,
    source_semantic_question_family: str,
    source_question_family: str,
    polarity_mode: str,
    gold_provenance: str,
) -> dict[str, Any]:
    result = dict(reference)
    for key in ("answer", "answers", "reference_answers", "reference_answer_text"):
        result.pop(key, None)
    result["source_dataset"] = clean_text(
        reference.get("source_dataset", reference.get("dataset"))
    )
    result["source_task_type"] = source_task_type
    result["source_semantic_question_family"] = source_semantic_question_family
    result["source_question_family"] = source_question_family
    result["binary_question_subtype"] = _binary_question_subtype(
        source_question_family
    )
    result["reference_answer"] = label
    result["reference_answer_raw"] = raw_answer
    result["question_family"] = "presence"
    result["task_type"] = "binary"
    result["target_attribute"] = "presence"
    result["reference_format"] = "scalar"
    result["answer_space"] = list(BINARY_ANSWER_SPACE)
    result["closed_set_protocol"] = CLOSED_BINARY_PROTOCOL
    result["conversion_rule_id"] = f"binary_polarity_{polarity_mode}_v3"
    result["polarity_parse_rule"] = parse_rule
    result["gold_provenance"] = gold_provenance
    return result


def _candidate_audit_record(candidate: _Candidate) -> dict[str, Any]:
    query = candidate.query
    reference = candidate.reference
    return {
        "query_id": candidate.query_id,
        "dataset": query.get("dataset"),
        "source_dataset": query.get("source_dataset"),
        "source_split": query.get("source_split"),
        "case_id": query.get("case_id"),
        "study_id": query.get("study_id"),
        "dicom_id": query.get("dicom_id"),
        "image_id": query.get("image_id"),
        "question_type": query.get("question_type"),
        "source_question_type": query.get("source_question_type"),
        "source_question_family": query.get("source_question_family"),
        "binary_question_subtype": query.get("binary_question_subtype"),
        "source_semantic_question_family": query.get(
            "source_semantic_question_family"
        ),
        "question_family": "presence",
        "question": candidate.raw_question,
        "raw_answer": candidate.raw_answer,
        "canonical_answer": candidate.label,
        "parse_rule": candidate.parse_rule,
        "gold_provenance": reference.get("gold_provenance"),
        "gt_finding": reference.get("gt_finding"),
        "gt_location": reference.get("gt_location"),
        "status": "included",
    }


def _stratified_audit_rows(
    included: Sequence[_Candidate],
    exclusions: Sequence[Mapping[str, Any]],
    *,
    per_stratum: int,
    per_exclusion_reason: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []

    strata: dict[tuple[str, str, str], list[_Candidate]] = defaultdict(list)
    for candidate in included:
        key = (
            clean_text(candidate.query.get("dataset")),
            candidate.label,
            _question_type(candidate.query, candidate.reference) or "unspecified",
        )
        strata[key].append(candidate)
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda item: item.query_id)
        rng.shuffle(rows)
        result.extend(_candidate_audit_record(row) for row in rows[:per_stratum])

    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in exclusions:
        row = dict(raw)
        by_reason[clean_text(row.get("reason")) or "unspecified"].append(row)
    for reason in sorted(by_reason):
        rows = sorted(by_reason[reason], key=lambda item: str(item.get("query_id", "")))
        rng.shuffle(rows)
        result.extend(rows[:per_exclusion_reason])
    return result


def _summarize_candidates(
    candidates: Sequence[_Candidate],
    exclusions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "included_count": len(candidates),
        "excluded_count": len(exclusions),
        "label_counts": dict(sorted(Counter(row.label for row in candidates).items())),
        "dataset_counts": dict(
            sorted(Counter(clean_text(row.query.get("dataset")) for row in candidates).items())
        ),
        "question_type_counts": dict(
            sorted(
                Counter(
                    _question_type(row.query, row.reference) or "unspecified"
                    for row in candidates
                ).items()
            )
        ),
        "source_question_family_counts": dict(
            sorted(
                Counter(
                    clean_text(row.query.get("source_question_family"))
                    or "unspecified"
                    for row in candidates
                ).items()
            )
        ),
        "source_semantic_question_family_counts": dict(
            sorted(
                Counter(
                    clean_text(
                        row.query.get("source_semantic_question_family")
                    )
                    or "unspecified"
                    for row in candidates
                ).items()
            )
        ),
        "binary_question_subtype_counts": dict(
            sorted(
                Counter(
                    clean_text(row.query.get("binary_question_subtype"))
                    or "unspecified"
                    for row in candidates
                ).items()
            )
        ),
        "parse_rule_counts": dict(
            sorted(Counter(row.parse_rule for row in candidates).items())
        ),
        "gold_provenance_counts": dict(
            sorted(
                Counter(
                    clean_text(row.reference.get("gold_provenance")) or "unspecified"
                    for row in candidates
                ).items()
            )
        ),
        "unique_image_count": len({row.image_identity for row in candidates}),
        "exclusion_reason_counts": dict(
            sorted(Counter(str(row.get("reason", "unspecified")) for row in exclusions).items())
        ),
    }


def prepare_closed_binary_vqa(
    *,
    queries_path: str | Path,
    references_path: str | Path,
    output_root: str | Path,
    split: str,
    polarity_mode: str = "exact",
    allowed_question_types: Sequence[str] | None = None,
    allowed_source_question_families: Sequence[str] | None = None,
    require_presence_family: bool = True,
    require_polar_question: bool = False,
    allow_negated_question: bool = False,
    require_gt_finding: bool = False,
    gold_provenance: str | None = None,
    audit_per_stratum: int = 25,
    audit_per_exclusion_reason: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a strict binary presence view from one normalized VQA split."""

    query_path = Path(queries_path)
    reference_path = Path(references_path)
    queries = unique_map(read_jsonl(query_path), label="source query")
    references = unique_map(read_jsonl(reference_path), label="source reference")
    if set(queries) != set(references):
        raise ValueError(
            "Cannot build a closed binary view from misaligned artifacts: "
            f"query_only={sorted(set(queries)-set(references))[:20]}, "
            f"reference_only={sorted(set(references)-set(queries))[:20]}"
        )
    allowed_types = {
        _normalized_question_type(item)
        for item in (allowed_question_types or [])
        if clean_text(item)
    }
    allowed_source_families = {
        _normalized_question_type(item)
        for item in (allowed_source_question_families or [])
        if clean_text(item)
    }
    if audit_per_stratum < 0 or audit_per_exclusion_reason < 0:
        raise ValueError("Audit sample sizes must be non-negative")

    candidates: list[_Candidate] = []
    exclusions: list[dict[str, Any]] = []
    for original_index, query_id in enumerate(queries):
        query = queries[query_id]
        reference = references[query_id]
        options = query.get("options", [])
        if isinstance(options, (str, bytes)) or not isinstance(options, Sequence):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="invalid_options_field",
                )
            )
            continue
        if any(clean_text(item) for item in options):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="has_visible_options",
                )
            )
            continue

        family = _question_family(query, reference)
        if require_presence_family and family != "presence":
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="non_presence_family",
                    detail=f"observed_family={family or 'unspecified'}",
                )
            )
            continue
        source_family = _source_question_family(query, reference)
        if (
            allowed_source_families
            and _normalized_question_type(source_family)
            not in allowed_source_families
        ):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="source_question_family_not_allowed",
                    detail=(
                        "observed_source_question_family="
                        f"{source_family or 'unspecified'}"
                    ),
                )
            )
            continue
        qtype = _question_type(query, reference)
        if allowed_types and _normalized_question_type(qtype) not in allowed_types:
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="question_type_not_allowed",
                    detail=f"observed_question_type={qtype or 'unspecified'}",
                )
            )
            continue
        if require_gt_finding and reference.get("gt_finding") in (None, "", [], {}):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="missing_gt_finding",
                )
            )
            continue

        raw_answer, answer_error = _raw_scalar_reference(reference)
        if answer_error or raw_answer is None:
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason=answer_error or "missing_reference_answer",
                )
            )
            continue
        parsed = parse_binary_answer(raw_answer, mode=polarity_mode)
        if not parsed.ok or parsed.label is None:
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason=parsed.reason or "unmapped_binary_answer",
                    raw_answer=raw_answer,
                )
            )
            continue

        raw_question = clean_text(
            query.get("retrieval_question", query.get("question", ""))
        )
        if not raw_question:
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="missing_question",
                    raw_answer=raw_answer,
                )
            )
            continue
        if require_polar_question and not _is_polar_question(raw_question):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="non_polar_question_form",
                    raw_answer=raw_answer,
                )
            )
            continue
        if (
            require_polar_question
            and _normalized_question_type(source_family) == "abnormality"
            and not _is_existential_abnormality_question(raw_question)
        ):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="non_existential_abnormality_question",
                    raw_answer=raw_answer,
                    detail=(
                        "Main binary abnormality rows must begin with "
                        "'Is there' or 'Are there'; reversed normality and "
                        "other polar forms are excluded."
                    ),
                )
            )
            continue
        if (
            require_polar_question
            and not allow_negated_question
            and _is_explicitly_negated_question(raw_question)
        ):
            exclusions.append(
                _exclusion_record(
                    query_id=query_id,
                    query=query,
                    reference=reference,
                    reason="explicitly_negated_question",
                    raw_answer=raw_answer,
                )
            )
            continue

        source_task_type = clean_text(
            query.get("task_type", reference.get("task_type", "unspecified"))
        ) or "unspecified"
        source_semantic_question_family = family or "unspecified"
        source_question_family = source_family or source_semantic_question_family
        provenance = _gold_provenance(
            clean_text(query.get("dataset", reference.get("dataset"))),
            gold_provenance,
        )
        canonical_query = _canonical_query(
            query,
            raw_question=raw_question,
            source_task_type=source_task_type,
            source_semantic_question_family=source_semantic_question_family,
            source_question_family=source_question_family,
            polarity_mode=polarity_mode,
        )
        canonical_reference = _canonical_reference(
            reference,
            raw_answer=raw_answer,
            label=parsed.label,
            parse_rule=parsed.rule,
            source_task_type=source_task_type,
            source_semantic_question_family=source_semantic_question_family,
            source_question_family=source_question_family,
            polarity_mode=polarity_mode,
            gold_provenance=provenance,
        )
        image_identity, dedupe_key = _dedupe_key(canonical_query, raw_question)
        candidates.append(
            _Candidate(
                query_id=query_id,
                query=canonical_query,
                reference=canonical_reference,
                raw_question=raw_question,
                raw_answer=raw_answer,
                label=parsed.label,
                parse_rule=parsed.rule,
                dedupe_key=dedupe_key,
                image_identity=image_identity,
                original_index=original_index,
            )
        )

    # Resolve exact duplicate image/question pairs before writing.  Identical
    # labels keep one deterministic row; conflicting labels are all excluded.
    dedupe_groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        dedupe_groups[candidate.dedupe_key].append(candidate)
    retained: list[_Candidate] = []
    for key in sorted(dedupe_groups):
        group = dedupe_groups[key]
        labels = {row.label for row in group}
        if len(labels) > 1:
            for row in group:
                exclusions.append(
                    _exclusion_record(
                        query_id=row.query_id,
                        query=row.query,
                        reference=row.reference,
                        reason="duplicate_label_conflict",
                        raw_answer=row.raw_answer,
                        detail=f"labels={sorted(labels)}; dedupe_key={key}",
                    )
                )
            continue
        ordered = sorted(
            group,
            key=lambda row: (
                0 if row.parse_rule.startswith("exact_") else 1,
                row.original_index,
                row.query_id,
            ),
        )
        retained.append(ordered[0])
        for row in ordered[1:]:
            exclusions.append(
                _exclusion_record(
                    query_id=row.query_id,
                    query=row.query,
                    reference=row.reference,
                    reason="duplicate_same_label",
                    raw_answer=row.raw_answer,
                    detail=f"kept_query_id={ordered[0].query_id}; dedupe_key={key}",
                )
            )
    retained.sort(key=lambda row: (row.original_index, row.query_id))
    exclusions.sort(key=lambda row: (str(row.get("reason", "")), str(row.get("query_id", ""))))

    if not retained:
        summary = _summarize_candidates(retained, exclusions)
        raise ValueError(
            "Closed binary preprocessing retained zero rows; "
            f"summary={summary}"
        )

    root = Path(output_root)
    summary = _summarize_candidates(retained, exclusions)
    source_count = len(queries)
    metadata = {
        "dataset": sorted(
            {clean_text(row.query.get("dataset")) for row in retained if clean_text(row.query.get("dataset"))}
        ),
        "closed_set_protocol": CLOSED_BINARY_PROTOCOL,
        "answer_space": list(BINARY_ANSWER_SPACE),
        "polarity_mode": polarity_mode,
        "allowed_question_types": sorted(allowed_types),
        "allowed_source_question_families": sorted(allowed_source_families),
        "require_presence_family": require_presence_family,
        "require_polar_question": require_polar_question,
        "abnormality_question_form_policy": (
            "existential_is_are_there" if require_polar_question else "not_enforced"
        ),
        "allow_negated_question": allow_negated_question,
        "require_gt_finding": require_gt_finding,
        "seed": seed,
        "source_query_count": source_count,
        "acceptance_rate": len(retained) / source_count if source_count else 0.0,
        "source_queries_path": str(query_path),
        "source_references_path": str(reference_path),
        "source_queries_sha256": file_sha256(query_path),
        "source_references_sha256": file_sha256(reference_path),
        **summary,
    }
    manifest = write_vqa_split(
        root,
        split=split,
        queries=[row.query for row in retained],
        references=[row.reference for row in retained],
        metadata=metadata,
    )
    exclusion_path = root / "exclusions" / f"{split}.jsonl"
    audit_path = root / "audit_candidates" / f"{split}.jsonl"
    write_jsonl(exclusion_path, exclusions)
    audit_rows = _stratified_audit_rows(
        retained,
        exclusions,
        per_stratum=audit_per_stratum,
        per_exclusion_reason=audit_per_exclusion_reason,
        seed=seed,
    )
    write_jsonl(audit_path, audit_rows)
    manifest.update(
        {
            "exclusions_path": str(exclusion_path),
            "audit_candidates_path": str(audit_path),
            "audit_candidate_count": len(audit_rows),
            "included_query_id_sha256": stable_json_sha256(
                [row.query_id for row in retained]
            ),
        }
    )
    # write_vqa_split already wrote the manifest; overwrite it with the paths
    # added after exclusion/audit materialization.
    from ..io_utils import write_json

    write_json(root / "manifests" / f"{split}.json", manifest)
    return {
        "ok": True,
        "output_root": str(root),
        "split": split,
        "manifest": manifest,
        "summary": summary,
    }


def _parse_source_tuple(value: Sequence[str]) -> ClosedBinarySource:
    if len(value) != 3:
        raise ValueError("Each binary merge source requires NAME QUERIES REFERENCES")
    name, queries, references = value
    if not clean_text(name):
        raise ValueError("Binary merge source name must be non-empty")
    return ClosedBinarySource(clean_text(name), Path(queries), Path(references))


def merge_closed_binary_vqa(
    *,
    sources: Sequence[ClosedBinarySource | Sequence[str]],
    output_root: str | Path,
    split: str,
    dataset_name: str = "MIMIC-CXR Binary Suite",
    conflict_policy: str = "exclude",
    audit_per_source_label: int = 25,
    seed: int = 42,
) -> dict[str, Any]:
    """Merge already-canonical binary views while preserving source provenance.

    Source order defines duplicate priority.  Exact duplicate image/question
    rows with the same label keep the first source.  Conflicting duplicates are
    excluded by default because silently preferring one provenance source would
    hide a reference disagreement.
    """

    normalized_policy = clean_text(conflict_policy).casefold()
    if normalized_policy not in {"exclude", "error"}:
        raise ValueError("conflict_policy must be 'exclude' or 'error'")
    parsed_sources = [
        item if isinstance(item, ClosedBinarySource) else _parse_source_tuple(item)
        for item in sources
    ]
    if not parsed_sources:
        raise ValueError("At least one binary source is required")

    candidates: list[_Candidate] = []
    exclusions: list[dict[str, Any]] = []
    input_metadata: list[dict[str, Any]] = []
    global_index = 0
    for priority, source in enumerate(parsed_sources):
        queries = unique_map(read_jsonl(source.queries_path), label=f"{source.name} query")
        references = unique_map(
            read_jsonl(source.references_path), label=f"{source.name} reference"
        )
        if set(queries) != set(references):
            raise ValueError(f"Binary source {source.name} has query/reference mismatch")
        input_metadata.append(
            {
                "name": source.name,
                "priority": priority,
                "query_count": len(queries),
                "queries_path": str(source.queries_path),
                "references_path": str(source.references_path),
                "queries_sha256": file_sha256(source.queries_path),
                "references_sha256": file_sha256(source.references_path),
            }
        )
        for source_query_id in queries:
            query = dict(queries[source_query_id])
            reference = dict(references[source_query_id])
            raw_answer, answer_error = _raw_scalar_reference(reference)
            if answer_error or raw_answer is None:
                raise ValueError(
                    f"Binary source {source.name} row {source_query_id} is not scalar: "
                    f"{answer_error}"
                )
            parsed = parse_binary_answer(raw_answer, mode="leading")
            if not parsed.ok or parsed.label is None:
                raise ValueError(
                    f"Binary source {source.name} row {source_query_id} is not canonical "
                    f"yes/no: reason={parsed.reason}, answer={raw_answer!r}"
                )
            answer_space = reference.get("answer_space", query.get("answer_space", []))
            if [clean_text(item).casefold() for item in answer_space] != BINARY_ANSWER_SPACE:
                raise ValueError(
                    f"Binary source {source.name} row {source_query_id} lacks "
                    f"answer_space={BINARY_ANSWER_SPACE}"
                )
            raw_question = clean_text(
                query.get("retrieval_question", query.get("question", ""))
            )
            image_identity, key = _dedupe_key(query, raw_question)
            merged_id = (
                f"closed_binary_suite:{_normalized_question_type(source.name)}:"
                f"{source_query_id}"
            )
            query["query_id"] = merged_id
            query["source_query_id"] = source_query_id
            query["source_dataset"] = clean_text(
                query.get("source_dataset", query.get("dataset", source.name))
            )
            query["source_suite_name"] = source.name
            query["dataset"] = dataset_name
            query["merge_source_priority"] = priority
            reference["query_id"] = merged_id
            reference["source_query_id"] = source_query_id
            reference["source_dataset"] = clean_text(
                reference.get("source_dataset", reference.get("dataset", source.name))
            )
            reference["source_suite_name"] = source.name
            reference["dataset"] = dataset_name
            reference["merge_source_priority"] = priority
            candidates.append(
                _Candidate(
                    query_id=merged_id,
                    query=query,
                    reference=reference,
                    raw_question=raw_question,
                    raw_answer=raw_answer,
                    label=parsed.label,
                    parse_rule=clean_text(reference.get("polarity_parse_rule"))
                    or parsed.rule,
                    dedupe_key=key,
                    image_identity=image_identity,
                    original_index=global_index,
                )
            )
            global_index += 1

    groups: dict[str, list[_Candidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.dedupe_key].append(candidate)
    retained: list[_Candidate] = []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda row: (
                int(row.query.get("merge_source_priority", 10**9)),
                row.original_index,
                row.query_id,
            ),
        )
        labels = {row.label for row in group}
        if len(labels) > 1:
            if normalized_policy == "error":
                raise ValueError(
                    f"Conflicting binary references for dedupe key {key}: "
                    f"{[(row.query_id, row.label) for row in group]}"
                )
            for row in group:
                exclusions.append(
                    _exclusion_record(
                        query_id=row.query_id,
                        query=row.query,
                        reference=row.reference,
                        reason="cross_source_label_conflict",
                        raw_answer=row.raw_answer,
                        detail=f"labels={sorted(labels)}; dedupe_key={key}",
                    )
                )
            continue
        retained.append(group[0])
        for row in group[1:]:
            exclusions.append(
                _exclusion_record(
                    query_id=row.query_id,
                    query=row.query,
                    reference=row.reference,
                    reason="cross_source_duplicate_same_label",
                    raw_answer=row.raw_answer,
                    detail=f"kept_query_id={group[0].query_id}; dedupe_key={key}",
                )
            )
    retained.sort(key=lambda row: (row.original_index, row.query_id))
    if not retained:
        raise ValueError("Binary merge retained zero rows")

    root = Path(output_root)
    summary = _summarize_candidates(retained, exclusions)
    source_counts = Counter(
        clean_text(row.query.get("source_suite_name")) for row in retained
    )
    metadata = {
        "dataset": dataset_name,
        "closed_set_protocol": CLOSED_BINARY_PROTOCOL,
        "merge_protocol": "closed_binary_merge_v1",
        "answer_space": list(BINARY_ANSWER_SPACE),
        "conflict_policy": normalized_policy,
        "input_sources": input_metadata,
        "retained_source_counts": dict(sorted(source_counts.items())),
        "warning": (
            "This pooled artifact mixes reference provenance. Report source-specific "
            "metrics alongside any pooled metric; do not present it as an expert-only set."
        ),
        **summary,
    }
    manifest = write_vqa_split(
        root,
        split=split,
        queries=[row.query for row in retained],
        references=[row.reference for row in retained],
        metadata=metadata,
    )
    exclusion_path = root / "exclusions" / f"{split}.jsonl"
    audit_path = root / "audit_candidates" / f"{split}.jsonl"
    write_jsonl(exclusion_path, exclusions)

    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[_Candidate]] = defaultdict(list)
    for row in retained:
        strata[(clean_text(row.query.get("source_suite_name")), row.label)].append(row)
    audit_rows: list[dict[str, Any]] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda row: row.query_id)
        rng.shuffle(rows)
        audit_rows.extend(
            _candidate_audit_record(row) for row in rows[:audit_per_source_label]
        )
    write_jsonl(audit_path, audit_rows)
    manifest.update(
        {
            "exclusions_path": str(exclusion_path),
            "audit_candidates_path": str(audit_path),
            "audit_candidate_count": len(audit_rows),
        }
    )
    from ..io_utils import write_json

    write_json(root / "manifests" / f"{split}.json", manifest)
    return {
        "ok": True,
        "output_root": str(root),
        "split": split,
        "manifest": manifest,
        "summary": summary,
    }


__all__ = [
    "BINARY_ANSWER_SPACE",
    "CLOSED_BINARY_PROTOCOL",
    "ClosedBinarySource",
    "format_binary_question",
    "merge_closed_binary_vqa",
    "prepare_closed_binary_vqa",
]
