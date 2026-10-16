"""Shared contracts and validation for ViSiG-CR VQA query/reference artifacts."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..io_utils import read_jsonl, stable_json_sha256, write_json
from ..schemas import CaseRecord


FORBIDDEN_QUERY_KEYS = {
    "answer",
    "answers",
    "reference_answer",
    "reference_answers",
    "reference_answer_text",
    "reference_explanation",
    "correct_answer",
    "gold_answer",
    "gt_answer",
    "gt_finding",
    "gt_location",
    "report",
    "report_text",
    "query_report",
}


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_id(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", text).strip("_")


def format_mcq(question: str, options: Sequence[str]) -> str:
    question = clean_text(question)
    visible = [clean_text(item) for item in options if clean_text(item)]
    if not visible:
        return question
    return (
        question
        + "\n\nOPTIONS\n"
        + "\n".join(visible)
        + "\n\nIn the required JSON object, set `answer` to exactly one "
        "displayed option letter; include every other required JSON field."
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def unique_map(
    rows: Iterable[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    missing = 0
    for row in rows:
        record = dict(row)
        query_id = clean_text(record.get("query_id"))
        if not query_id:
            missing += 1
            continue
        if query_id in result:
            duplicates.append(query_id)
        else:
            result[query_id] = record
    if missing or duplicates:
        raise ValueError(
            f"Invalid {label} query IDs: missing={missing}, "
            f"duplicates={sorted(set(duplicates))[:20]}"
        )
    return result


class CaseIndex:
    """Resolve study/dicom/case identifiers to one prepared query image."""

    def __init__(self, cases: Iterable[CaseRecord]):
        self.by_case: dict[str, CaseRecord] = {}
        self.by_study: dict[str, list[CaseRecord]] = defaultdict(list)
        self.by_dicom: dict[str, CaseRecord] = {}
        self.by_subject_study: dict[tuple[str, str], list[CaseRecord]] = defaultdict(list)
        for case in cases:
            self.by_case[case.case_id] = case
            self.by_study[str(case.study_id)].append(case)
            self.by_dicom[str(case.dicom_id)] = case
            self.by_subject_study[(str(case.subject_id), str(case.study_id))].append(case)
        for rows in self.by_study.values():
            rows.sort(key=lambda case: (case.view != "PA", case.view != "AP", case.case_id))
        for rows in self.by_subject_study.values():
            rows.sort(key=lambda case: (case.view != "PA", case.view != "AP", case.case_id))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CaseIndex":
        return cls(CaseRecord.model_validate(row) for row in read_jsonl(path))

    def resolve(
        self,
        *,
        case_id: Any = None,
        dicom_id: Any = None,
        study_id: Any = None,
        subject_id: Any = None,
    ) -> CaseRecord | None:
        case_key = clean_text(case_id)
        if case_key and case_key in self.by_case:
            return self.by_case[case_key]
        dicom_key = clean_text(dicom_id)
        if dicom_key and dicom_key in self.by_dicom:
            return self.by_dicom[dicom_key]
        study_key = clean_text(study_id)
        subject_key = clean_text(subject_id)
        if subject_key and study_key:
            rows = self.by_subject_study.get((subject_key, study_key), [])
            if rows:
                return rows[0]
        if study_key:
            rows = self.by_study.get(study_key, [])
            if rows:
                return rows[0]
        return None


def write_vqa_split(
    output_root: str | Path,
    *,
    split: str,
    queries: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    query_path = root / "queries" / f"{split}.jsonl"
    reference_path = root / "reference_answers" / f"{split}.jsonl"
    query_map = unique_map(queries, label=f"{split} query")
    ref_map = unique_map(references, label=f"{split} reference")
    if set(query_map) != set(ref_map):
        raise ValueError(
            f"VQA query/reference ID mismatch for split={split}: "
            f"query_only={sorted(set(query_map)-set(ref_map))[:20]}, "
            f"reference_only={sorted(set(ref_map)-set(query_map))[:20]}"
        )
    for query_id, row in query_map.items():
        forbidden = sorted(FORBIDDEN_QUERY_KEYS & set(row))
        if forbidden:
            raise ValueError(
                f"Query {query_id} contains forbidden answer/report keys: {forbidden}"
            )
    ordered_ids = list(query_map)
    write_jsonl(query_path, (query_map[query_id] for query_id in ordered_ids))
    write_jsonl(reference_path, (ref_map[query_id] for query_id in ordered_ids))
    manifest = {
        "split": split,
        "query_count": len(ordered_ids),
        "query_path": str(query_path),
        "reference_path": str(reference_path),
        "query_id_sha256": stable_json_sha256(ordered_ids),
        **dict(metadata or {}),
    }
    write_json(root / "manifests" / f"{split}.json", manifest)
    return manifest


def validate_vqa_artifacts(
    *,
    queries_path: str | Path,
    references_path: str | Path,
    require_images: bool = True,
) -> dict[str, Any]:
    queries = unique_map(read_jsonl(queries_path), label="query")
    references = unique_map(read_jsonl(references_path), label="reference")
    query_ids = set(queries)
    reference_ids = set(references)
    missing_images: list[str] = []
    forbidden_rows: dict[str, list[str]] = {}
    for query_id, row in queries.items():
        forbidden = sorted(FORBIDDEN_QUERY_KEYS & set(row))
        if forbidden:
            forbidden_rows[query_id] = forbidden
        image_path = Path(clean_text(row.get("image_path", row.get("query_image_path"))))
        if require_images and not image_path.is_file():
            missing_images.append(query_id)
    result = {
        "ok": (
            query_ids == reference_ids
            and not missing_images
            and not forbidden_rows
        ),
        "query_count": len(query_ids),
        "reference_count": len(reference_ids),
        "query_only_count": len(query_ids - reference_ids),
        "reference_only_count": len(reference_ids - query_ids),
        "query_only_ids": sorted(query_ids - reference_ids)[:50],
        "reference_only_ids": sorted(reference_ids - query_ids)[:50],
        "missing_image_count": len(missing_images),
        "missing_image_query_ids": missing_images[:50],
        "forbidden_query_row_count": len(forbidden_rows),
        "forbidden_query_rows": dict(list(forbidden_rows.items())[:50]),
    }
    return result



def filter_vqa_artifacts(
    *,
    queries_path: str | Path,
    references_path: str | Path,
    output_root: str | Path,
    field: str,
    value: str = "true",
) -> dict[str, Any]:
    """Create an aligned VQA subset selected by one query metadata field.

    This is intended for protocol-scope fields such as ReXVQA's
    ``is_primary_protocol``. Reference answers are joined only after selection,
    so hidden labels never influence which queries are retained.
    """

    field = str(field).strip()
    if not field:
        raise ValueError("VQA filter field must be non-empty")
    queries = unique_map(read_jsonl(queries_path), label="query")
    references = unique_map(read_jsonl(references_path), label="reference")
    if set(queries) != set(references):
        raise ValueError("Cannot filter misaligned query/reference artifacts")

    expected = str(value).strip().lower()

    def normalized(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value).strip().lower()

    selected = [
        query_id
        for query_id, row in queries.items()
        if normalized(row.get(field)) == expected
    ]
    root = Path(output_root)
    query_out = root / "queries.jsonl"
    reference_out = root / "reference_answers.jsonl"
    write_jsonl(query_out, (queries[query_id] for query_id in selected))
    write_jsonl(reference_out, (references[query_id] for query_id in selected))
    manifest = {
        "filter_field": field,
        "filter_value": expected,
        "source_count": len(queries),
        "selected_count": len(selected),
        "excluded_count": len(queries) - len(selected),
        "query_id_sha256": stable_json_sha256(selected),
        "queries": str(query_out),
        "references": str(reference_out),
    }
    write_json(root / "manifest.json", manifest)
    return manifest

def subset_vqa_artifacts(
    *,
    queries_path: str | Path,
    references_path: str | Path,
    output_root: str | Path,
    limit: int,
    seed: int = 42,
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("VQA subset limit must be positive")
    queries = unique_map(read_jsonl(queries_path), label="query")
    references = unique_map(read_jsonl(references_path), label="reference")
    if set(queries) != set(references):
        raise ValueError("Cannot subset misaligned query/reference artifacts")
    ids = sorted(queries)
    random.Random(seed).shuffle(ids)
    selected = ids[: min(limit, len(ids))]
    root = Path(output_root)
    query_out = root / "queries.jsonl"
    reference_out = root / "reference_answers.jsonl"
    write_jsonl(query_out, (queries[query_id] for query_id in selected))
    write_jsonl(reference_out, (references[query_id] for query_id in selected))
    manifest = {
        "seed": seed,
        "requested_limit": limit,
        "selected_count": len(selected),
        "query_id_sha256": stable_json_sha256(selected),
        "queries": str(query_out),
        "references": str(reference_out),
    }
    write_json(root / "manifest.json", manifest)
    return manifest


__all__ = [
    "CaseIndex",
    "FORBIDDEN_QUERY_KEYS",
    "canonical_id",
    "clean_text",
    "format_mcq",
    "filter_vqa_artifacts",
    "subset_vqa_artifacts",
    "unique_map",
    "validate_vqa_artifacts",
    "write_jsonl",
    "write_vqa_split",
]
