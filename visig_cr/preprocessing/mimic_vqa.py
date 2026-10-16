"""Adapters for expert MIMIC-CXR VQA and Medical-CXR-VQA.

The source adapters normalize dataset rows against the patient-disjoint MIMIC
query-case pool.  Evaluation-specific closed-set conversion remains a separate
step in :mod:`visig_cr.preprocessing.closed_set_vqa` so that source-normalized
artifacts and derived binary views have distinct provenance.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..io_utils import file_sha256, write_json
from ..mimic import build_image_path
from ..protocol import protocol_layout
from .common import canonical_split, iter_json_records
from .vqa_common import (
    CaseIndex,
    canonical_id,
    clean_text,
    format_mcq,
    write_vqa_split,
)


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def _iter_file(path: Path) -> Iterator[dict[str, Any]]:
    lower = path.name.lower()
    if lower.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        delimiter = "\t" if ".tsv" in lower else ","
        with _open_csv(path) as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield dict(row)
        return
    if lower.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        yield dict(value)
        return
    if lower.endswith(".json"):
        for outer_id, row in iter_json_records(path):
            record = dict(row)
            record.setdefault("_outer_id", outer_id)
            yield record
        return


def _source_files(
    source: str | Path,
    *,
    preferred_names: Sequence[str] = (),
) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"VQA source not found: {path}")

    visible_files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part.startswith(".") for part in item.relative_to(path).parts)
    ]
    if preferred_names:
        preferred = {name.lower() for name in preferred_names}
        matches = sorted(
            item for item in visible_files if item.name.lower() in preferred
        )
        if len(matches) == 1:
            return matches
        if len(matches) > 1:
            raise ValueError(
                "Multiple primary VQA tables matched beneath "
                f"{path}: {[str(item) for item in matches]}"
            )
        raise FileNotFoundError(
            "No primary VQA table was found beneath "
            f"{path}; expected one of {list(preferred_names)}. "
            "Pass --source with the exact table path if the local filename differs."
        )

    files = [
        item
        for item in visible_files
        if item.name.lower().endswith(
            (".csv", ".csv.gz", ".tsv", ".tsv.gz", ".json", ".jsonl", ".ndjson")
        )
        and not any(
            token in item.name.lower()
            for token in ("readme", "license", "manifest")
        )
        and not any(
            part.lower() in {"libs", "supplementary"}
            for part in item.relative_to(path).parts[:-1]
        )
    ]
    if not files:
        raise FileNotFoundError(f"No supported VQA table found beneath: {path}")
    return sorted(files)


def _field(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    parsed: Any = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (json.JSONDecodeError, TypeError, ValueError, SyntaxError):
            continue
    if isinstance(parsed, (list, tuple, set)):
        return [clean_text(item) for item in parsed if clean_text(item)]
    if "|" in text or ";" in text:
        return [
            clean_text(item)
            for item in re.split(r"[|;]", text)
            if clean_text(item)
        ]
    return [text]


def _parse_options(row: Mapping[str, Any]) -> list[str]:
    raw = _field(row, "options", "choices", "answer_options")
    options = _parse_list(raw)
    if options:
        return options
    found: list[str] = []
    for letter in "ABCDEFGH":
        value = _field(
            row,
            letter,
            f"option_{letter.lower()}",
            f"choice_{letter.lower()}",
        )
        if clean_text(value):
            found.append(f"{letter}. {clean_text(value)}")
    return found


def _source_question_type(row: Mapping[str, Any]) -> str:
    return clean_text(
        _field(row, "question_family", "question_type", "type", "category", "class")
    )


def _source_question_family(row: Mapping[str, Any]) -> str:
    raw = canonical_id(_source_question_type(row)).lower()
    aliases = {
        "yes_no": "presence",
        "yesno": "presence",
        "level": "severity",
    }
    return aliases.get(raw, raw) or "unspecified"


def _family(row: Mapping[str, Any], question: str) -> str:
    """Map source taxonomy and question wording to a semantic VQA family.

    Wording is checked before the source category.  This is intentional for the
    Medical-CXR-VQA rows whose source category is ``abnormality`` but whose
    actual question is a closed polar question such as "Is there evidence of
    any abnormalities in this image?".  Such rows belong to the presence axis;
    open "What abnormalities ...?" rows remain finding-identity questions.
    """

    source_family = _source_question_family(row)
    question_text = clean_text(question).casefold()
    if question_text.startswith(
        (
            "is ",
            "are ",
            "was ",
            "were ",
            "does ",
            "do ",
            "did ",
            "has ",
            "have ",
            "had ",
            "can ",
            "could ",
            "would ",
            "should ",
            "will ",
        )
    ):
        return "presence"
    if source_family == "presence":
        return "presence"

    text = f"{source_family} {question_text}"
    rules = (
        ("location", ("location", "where", "which side", "laterality")),
        ("severity", ("level", "severity", "how severe")),
        ("finding_identity", ("abnormality", "finding", "what abnormal", "diagnosis")),
        ("type", ("type", "kind")),
        ("view", ("view", "projection")),
        ("count", ("count", "how many")),
    )
    for family, tokens in rules:
        if any(token in text for token in tokens):
            return family
    return source_family or "other"


def _target_attribute(family: str) -> str:
    return {
        "presence": "presence",
        "location": "anatomy",
        "severity": "severity",
        "finding_identity": "finding_identity",
        "type": "finding_identity",
    }.get(family, "unspecified")


def _task_type(answers: Sequence[str], options: Sequence[str]) -> str:
    if options:
        return "multiple_choice"
    if len(answers) > 1:
        return "multi_label"
    if answers and answers[0].strip().lower() in {
        "yes",
        "no",
        "present",
        "absent",
        "true",
        "false",
    }:
        return "binary"
    return "open"


def _normalized_family_set(values: Sequence[Any] | None) -> set[str]:
    return {
        canonical_id(value).lower()
        for value in (values or [])
        if clean_text(value)
    }


def _prepare_mimic_vqa(
    cfg: dict[str, Any],
    *,
    dataset_name: str,
    source: str | Path,
    output_root: str | Path,
    query_cases_jsonl: str | Path,
    allowed_families: set[str] | None,
    excluded_families: set[str] | None,
    max_questions: int | None,
    max_questions_per_case: int | None,
    allowed_source_families: set[str] | None = None,
    preferred_source_names: Sequence[str] = (),
    require_exact_dicom_image: bool = False,
    artifact_profile: str = "source_normalized",
) -> dict[str, Any]:
    case_index = CaseIndex.from_jsonl(query_cases_jsonl)
    query_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reference_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    per_case: Counter[str] = Counter()
    seen_ids: set[str] = set()

    allowed_families = set(allowed_families or []) or None
    excluded_families = set(excluded_families or [])
    allowed_source_families = set(allowed_source_families or []) or None
    source_files = _source_files(source, preferred_names=preferred_source_names)

    stop = False
    for file_index, source_file in enumerate(source_files):
        for record_index, row in enumerate(_iter_file(source_file)):
            counters["source_rows"] += 1
            source_question_type = _source_question_type(row)
            source_question_family = _source_question_family(row)
            counters[f"source_family_seen:{source_question_family}"] += 1
            source_dataset_split = clean_text(
                _field(row, "split", "source_split", "partition")
            )
            if source_dataset_split:
                counters[f"source_dataset_split_seen:{source_dataset_split}"] += 1

            if (
                allowed_source_families
                and source_question_family not in allowed_source_families
            ):
                counters["skipped_source_family_not_allowed"] += 1
                counters[
                    f"skipped_source_family:{source_question_family}"
                ] += 1
                continue
            counters["passed_source_family_filter"] += 1

            question = clean_text(_field(row, "question", "question_text", "query"))
            if not question:
                counters["skipped_missing_question"] += 1
                continue
            answers = _parse_list(
                _field(
                    row,
                    "reference_answers",
                    "answers",
                    "answer",
                    "reference_answer",
                    "gold_answer",
                    "correct_answer",
                    "expected_answer",
                )
            )
            if not answers:
                counters["skipped_missing_answer"] += 1
                continue
            counters["passed_basic_qa_fields"] += 1

            source_dicom_id = clean_text(
                _field(row, "dicom_id", "image_id", "image", "sop_id")
            )
            source_study_id = clean_text(
                _field(row, "study_id", "study", "studyid")
            )
            source_subject_id = clean_text(
                _field(row, "subject_id", "patient_id", "subject")
            )
            case = case_index.resolve(
                case_id=_field(row, "case_id"),
                dicom_id=source_dicom_id,
                study_id=source_study_id,
                subject_id=source_subject_id,
            )
            if case is None:
                counters["skipped_not_in_query_pool"] += 1
                continue
            counters["joined_to_query_pool"] += 1
            counters[f"joined_protocol_split:{canonical_split(case.split)}"] += 1
            if source_dataset_split:
                counters[
                    f"joined_source_dataset_split:{source_dataset_split}"
                ] += 1

            selected_dicom_id = case.dicom_id
            selected_image_path = case.image_path
            if require_exact_dicom_image and source_dicom_id:
                if source_study_id and source_study_id != str(case.study_id):
                    counters["skipped_study_id_mismatch"] += 1
                    continue
                try:
                    exact_image = build_image_path(
                        Path(cfg["paths"]["mimic_jpg_root"]),
                        int(source_subject_id or case.subject_id),
                        int(source_study_id or case.study_id),
                        source_dicom_id,
                    )
                except (KeyError, TypeError, ValueError):
                    counters["skipped_unresolvable_exact_dicom"] += 1
                    continue
                if not exact_image.is_file():
                    counters["skipped_missing_exact_dicom_image"] += 1
                    continue
                selected_dicom_id = source_dicom_id
                selected_image_path = str(exact_image.resolve())
                counters["joined_with_exact_dicom_image"] += 1

            family = _family(row, question)
            if allowed_families and family not in allowed_families:
                counters["skipped_family_not_allowed"] += 1
                counters[f"skipped_semantic_family:{family}"] += 1
                continue
            if excluded_families and family in excluded_families:
                counters["skipped_family_excluded"] += 1
                counters[f"skipped_semantic_family:{family}"] += 1
                continue
            counters["passed_semantic_family_filter"] += 1
            if (
                max_questions_per_case is not None
                and per_case[case.case_id] >= max_questions_per_case
            ):
                counters["skipped_case_question_cap"] += 1
                continue
            if max_questions is not None and counters["written"] >= max_questions:
                counters["stopped_at_max_questions"] += 1
                stop = True
                break

            source_id = clean_text(
                _field(row, "query_id", "question_id", "id", "_outer_id")
            )
            base_id = canonical_id(source_id) or f"{case.study_id}:{record_index:07d}"
            query_id = f"{canonical_id(dataset_name).lower()}:{base_id}"
            if query_id in seen_ids:
                query_id = f"{query_id}:{file_index:02d}:{record_index:07d}"
            seen_ids.add(query_id)

            options = _parse_options(row)
            reader_question = format_mcq(question, options)
            split = canonical_split(case.split)
            task_type = _task_type(answers, options)
            common_metadata = {
                "query_id": query_id,
                "dataset": dataset_name,
                # ``source_split`` remains the patient-disjoint protocol split
                # for compatibility.  The source dataset's own split, when
                # present, is retained separately for audit only.
                "source_split": case.split,
                "source_dataset_split": source_dataset_split or "unspecified",
                "source_question_type": source_question_type or "unspecified",
                "source_question_family": source_question_family,
                "question_family": family,
                "task_type": task_type,
                "target_attribute": _target_attribute(family),
                "artifact_profile": artifact_profile,
            }
            query_record = {
                **common_metadata,
                "source_query_id": source_id,
                "subject_id": case.subject_id,
                "study_id": case.study_id,
                "dicom_id": selected_dicom_id,
                "case_id": case.case_id,
                "image_path": selected_image_path,
                "question": reader_question,
                "reader_question": reader_question,
                "retrieval_question": question,
                "options": options,
                "reference_format": (
                    "answer_set" if len(answers) > 1 else "scalar"
                ),
            }
            reference_record: dict[str, Any] = dict(common_metadata)
            if len(answers) > 1:
                reference_record["reference_answers"] = answers
            else:
                reference_record["reference_answer"] = answers[0]

            query_rows[split].append(query_record)
            reference_rows[split].append(reference_record)
            counters["written"] += 1
            counters[f"written_split:{split}"] += 1
            counters[f"family:{family}"] += 1
            counters[f"written_source_family:{source_question_family}"] += 1
            counters[f"task_type:{task_type}"] += 1
            per_case[case.case_id] += 1
        if stop:
            break

    if not counters["written"]:
        raise ValueError(
            f"No {dataset_name} rows joined to the protocol query pool. "
            f"source_files={[str(path) for path in source_files]}, "
            f"counters={dict(sorted(counters.items()))}, "
            f"query_cases_jsonl={query_cases_jsonl}"
        )

    per_case_values = list(per_case.values())
    summary = {
        "dataset": dataset_name,
        "artifact_profile": artifact_profile,
        "source": str(source),
        "source_files": [str(path) for path in source_files],
        "source_file_sha256": {
            str(path): file_sha256(path) for path in source_files
        },
        "query_cases_jsonl": str(query_cases_jsonl),
        "query_cases_sha256": file_sha256(query_cases_jsonl),
        "query_pool_case_count": len(case_index.by_case),
        "query_pool_study_count": len(case_index.by_study),
        "query_pool_subject_count": len(
            {subject_id for subject_id, _ in case_index.by_subject_study}
        ),
        "query_pool_case_split_counts": dict(
            sorted(
                Counter(
                    canonical_split(case.split)
                    for case in case_index.by_case.values()
                ).items()
            )
        ),
        "allowed_families": sorted(allowed_families or []),
        "excluded_families": sorted(excluded_families),
        "allowed_source_families": sorted(allowed_source_families or []),
        "max_questions": max_questions,
        "max_questions_per_case": max_questions_per_case,
        "require_exact_dicom_image": require_exact_dicom_image,
        "unique_case_count": len(per_case),
        "questions_per_case_min": min(per_case_values) if per_case_values else 0,
        "questions_per_case_max": max(per_case_values) if per_case_values else 0,
        "questions_per_case_mean": (
            sum(per_case_values) / len(per_case_values) if per_case_values else 0.0
        ),
        "counters": dict(sorted(counters.items())),
    }
    root = Path(output_root)
    summary_path = root / "preparation_summary.json"
    write_json(summary_path, summary)

    manifests: dict[str, Any] = {}
    for split in sorted(query_rows):
        manifests[split] = write_vqa_split(
            root,
            split=split,
            queries=query_rows[split],
            references=reference_rows[split],
            metadata={
                "dataset": dataset_name,
                "artifact_profile": artifact_profile,
                "source": str(source),
                "source_files": [str(path) for path in source_files],
                "query_cases_jsonl": str(query_cases_jsonl),
                "require_exact_dicom_image": require_exact_dicom_image,
                "allowed_families": sorted(allowed_families or []),
                "excluded_families": sorted(excluded_families),
                "allowed_source_families": sorted(allowed_source_families or []),
                "max_questions": max_questions,
                "max_questions_per_case": max_questions_per_case,
                "preparation_summary_path": str(summary_path),
                "global_counters": dict(sorted(counters.items())),
            },
        )
    return {
        "ok": True,
        "dataset": dataset_name,
        "artifact_profile": artifact_profile,
        "output_root": str(root),
        "preparation_summary": str(summary_path),
        "splits": manifests,
        "counters": dict(sorted(counters.items())),
    }


def _query_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return (
        cfg.get("dataset_protocol", {})
        .get("in_domain_queries", {})
        .get(name, {})
    )


def _optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    return int(value) if value is not None else None


def prepare_mimic_expert_vqa(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None = None,
    source: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    layout = protocol_layout(cfg, requested_protocol=requested_protocol)
    source = source or cfg["paths"]["mimic_vqa_evaluation_root"]
    output_root = output_root or Path(layout["root"]) / "query" / "mimic_expert_vqa"
    query_cfg = _query_config(cfg, "mimic_expert_vqa")
    return _prepare_mimic_vqa(
        cfg,
        dataset_name="MIMIC-CXR Expert VQA",
        source=source,
        output_root=output_root,
        query_cases_jsonl=layout["query_cases_jsonl"],
        allowed_families=_normalized_family_set(
            query_cfg.get("question_families", [])
        )
        or None,
        excluded_families=_normalized_family_set(
            query_cfg.get("exclude_question_families", [])
        ),
        allowed_source_families=_normalized_family_set(
            query_cfg.get("source_question_families", [])
        )
        or None,
        max_questions=_optional_int(query_cfg, "max_questions"),
        max_questions_per_case=_optional_int(
            query_cfg, "max_questions_per_case"
        ),
        preferred_source_names=("mimic_cxr_vqa.tsv",),
        require_exact_dicom_image=True,
        artifact_profile="expert_source_normalized",
    )


def prepare_medical_cxr_vqa(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None = None,
    source: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    layout = protocol_layout(cfg, requested_protocol=requested_protocol)
    source = source or cfg["paths"]["medical_cxr_vqa_root"]
    output_root = output_root or Path(layout["root"]) / "query" / "medical_cxr_vqa"
    query_cfg = _query_config(cfg, "medical_cxr_vqa")
    return _prepare_mimic_vqa(
        cfg,
        dataset_name="Medical-CXR-VQA",
        source=source,
        output_root=output_root,
        query_cases_jsonl=layout["query_cases_jsonl"],
        allowed_families=_normalized_family_set(
            query_cfg.get("question_families", [])
        )
        or None,
        excluded_families=_normalized_family_set(
            query_cfg.get("exclude_question_families", [])
        ),
        allowed_source_families=_normalized_family_set(
            query_cfg.get("source_question_families", [])
        )
        or None,
        max_questions=_optional_int(query_cfg, "max_questions"),
        max_questions_per_case=_optional_int(
            query_cfg, "max_questions_per_case"
        ),
        preferred_source_names=(
            "medical_cxr_vqa_questions.csv",
            "medical-cxr-vqa-questions.csv",
        ),
        require_exact_dicom_image=False,
        artifact_profile="full_source_normalized",
    )


def prepare_medical_cxr_vqa_binary_source(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None = None,
    source: str | Path | None = None,
    output_root: str | Path | None = None,
    source_question_families: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Prepare uncapped Medical-CXR-VQA candidates for closed-binary conversion.

    The default profile keeps semantic presence questions whose *source* family
    is either ``presence`` or ``abnormality``.  Open abnormality questions are
    still rejected because their wording maps to ``finding_identity``.  Answer
    canonicalization, polar-question validation, duplicate handling, and audit
    materialization are intentionally performed by the subsequent
    ``vqa-closed-binary`` command.
    """

    layout = protocol_layout(cfg, requested_protocol=requested_protocol)
    source = source or cfg["paths"]["medical_cxr_vqa_root"]
    output_root = output_root or (
        Path(layout["root"]) / "query" / "medical_cxr_vqa_binary_source"
    )
    query_cfg = _query_config(cfg, "medical_cxr_vqa_binary")
    if not query_cfg:
        query_cfg = {
            "question_families": ["presence"],
            "source_question_families": ["presence", "abnormality"],
            "exclude_question_families": ["view"],
            "max_questions": None,
            "max_questions_per_case": None,
        }
    return _prepare_mimic_vqa(
        cfg,
        dataset_name="Medical-CXR-VQA",
        source=source,
        output_root=output_root,
        query_cases_jsonl=layout["query_cases_jsonl"],
        allowed_families=_normalized_family_set(
            query_cfg.get("question_families", ["presence"])
        )
        or {"presence"},
        excluded_families=_normalized_family_set(
            query_cfg.get("exclude_question_families", ["view"])
        ),
        allowed_source_families=_normalized_family_set(
            source_question_families
            if source_question_families is not None
            else query_cfg.get(
                "source_question_families", ["presence", "abnormality"]
            )
        )
        or {"presence", "abnormality"},
        max_questions=_optional_int(query_cfg, "max_questions"),
        max_questions_per_case=_optional_int(
            query_cfg, "max_questions_per_case"
        ),
        preferred_source_names=(
            "medical_cxr_vqa_questions.csv",
            "medical-cxr-vqa-questions.csv",
        ),
        require_exact_dicom_image=False,
        artifact_profile="closed_binary_source_v2",
    )


__all__ = [
    "prepare_medical_cxr_vqa",
    "prepare_medical_cxr_vqa_binary_source",
    "prepare_mimic_expert_vqa",
]
