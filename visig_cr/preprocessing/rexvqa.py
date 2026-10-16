"""Prepare ReXVQA as compact ViSiG-CR query and reference artifacts."""

from __future__ import annotations

import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

from .common import (
    JSONLWriter,
    as_list,
    assert_real_payload,
    canonical_split,
    clean_text,
    extract_sop_uid,
    iter_json_records,
    normalize_view,
    read_jsonl,
    remove_option_prefix,
    reset_output_dir,
    value_at,
    write_json,
)
from .rexgradient import ImageResolver, _select_representative


SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "validate", "test")
PRIMARY_FAMILIES = {
    "presence",
    "negation_uncertainty",
    "location",
    "laterality",
    "finding_identity",
    "differential",
}


def _field(record: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    lowered = {str(key).lower(): key for key in record}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return record[key]
    return default


def _study_uid_from_case(row: Mapping[str, Any]) -> str:
    direct = clean_text(row.get("study_instance_uid"))
    if direct:
        return direct
    study = row.get("study")
    if isinstance(study, Mapping):
        return clean_text(study.get("study_instance_uid"))
    return ""


def _load_case_index(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load only fields needed by ReXVQA joins.

    ``cases.jsonl`` deliberately retains rich patient/study metadata, but the
    VQA join only needs the representative image and nested image list.  A
    compact in-memory value keeps the full 160K conversion from duplicating
    unnecessary metadata in two lookup dictionaries.
    """

    by_study: dict[str, dict[str, Any]] = {}
    by_uid: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        study_id = clean_text(row.get("study_id"))
        study_uid = _study_uid_from_case(row)
        compact = {
            "case_id": clean_text(row.get("case_id")),
            "study_id": study_id,
            "study_instance_uid": study_uid,
            "image_path": clean_text(row.get("image_path")),
            "view": normalize_view(row.get("view")),
            "images": row.get("images") if isinstance(row.get("images"), list) else [],
        }
        if study_id:
            by_study[study_id] = compact
        if study_uid:
            by_uid[study_uid] = compact
    return by_study, by_uid


def _normalize_options(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        raw_items = [f"{key}. {clean_text(item)}" for key, item in sorted(value.items())]
    else:
        raw_items = [clean_text(item) for item in as_list(value) if clean_text(item)]

    normalized: list[str] = []
    for index, item in enumerate(raw_items):
        default_letter = chr(ord("A") + index)
        match = re.match(r"^\s*([A-Za-z])\s*[\.)\]:-]\s*(.*)$", item)
        if match:
            letter = match.group(1).upper()
            body = match.group(2).strip()
        else:
            letter = default_letter
            body = item.strip()
        normalized.append(f"{letter}. {body}")
    return normalized


def _reference_answer(raw_answer: Any, options: list[str]) -> str:
    text = clean_text(raw_answer)
    if not text:
        return ""
    match = re.match(r"^\s*([A-Za-z])(?:\s*[\.)\]:-]|\s*$)", text)
    if match:
        return match.group(1).upper()
    if text.isdigit():
        number = int(text)
        if 0 <= number < len(options):
            return chr(ord("A") + number)
        if 1 <= number <= len(options):
            return chr(ord("A") + number - 1)

    normalized_answer = re.sub(
        r"\s+", " ", remove_option_prefix(text).lower()
    ).strip()
    for option in options:
        option_text = re.sub(
            r"\s+", " ", remove_option_prefix(option).lower()
        ).strip()
        if option_text == normalized_answer:
            return option[0].upper()
    return text.upper()


def _reference_answer_text(reference_answer: str, options: list[str]) -> str:
    for option in options:
        if option[:1].upper() == reference_answer[:1].upper():
            return remove_option_prefix(option)
    return ""


def _format_mcq(question: str, options: list[str]) -> str:
    question = clean_text(question)
    option_block = "\n".join(options)
    if not option_block:
        return question
    return (
        f"{question}\n\nOPTIONS\n{option_block}\n\n"
        "In the required JSON object, set `answer` to exactly one displayed "
        "option letter; include every other required JSON field."
    )


def _classify_question(record: Mapping[str, Any]) -> tuple[str, str]:
    task_name = clean_text(_field(record, "task_name"))
    class_name = clean_text(_field(record, "class"))
    subcategory = clean_text(_field(record, "subcategory"))
    question = clean_text(_field(record, "question"))
    text = " ".join([task_name, class_name, subcategory, question]).lower()

    if any(token in text for token in ("negation", "absent", "absence", "not present")):
        return "negation_uncertainty", "presence"
    if any(
        token in text
        for token in ("uncertain", "uncertainty", "possible", "cannot exclude", "likely")
    ):
        return "negation_uncertainty", "presence"
    if any(token in text for token in ("laterality", "left or right", "which side")):
        return "laterality", "laterality"
    if any(
        token in text
        for token in ("anatomic location", "anatomical location", "localization", "where is")
    ):
        return "location", "anatomy"
    if any(token in text for token in ("severity", "mild", "moderate", "severe")):
        return "severity", "severity"
    if any(token in text for token in ("differential", "alternative diagnosis")):
        return "differential", "finding_identity"
    if any(
        token in text
        for token in ("identifying finding", "finding identification", "which finding", "what finding")
    ):
        return "finding_identity", "finding_identity"
    if question.lower().startswith(("is ", "are ", "does ", "do ", "can ")):
        return "presence", "presence"
    if "presence" in text:
        return "presence", "presence"

    fallback = re.sub(r"[^a-z0-9]+", "_", task_name.lower()).strip("_") or "other"
    return fallback, "finding_identity"


def _case_source_images(case_entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = case_entry.get("images")
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping) or not raw.get("exists"):
            continue
        result.append(
            {
                "image_index": raw.get("image_index"),
                "source_path": clean_text(raw.get("source_path")),
                "runtime_path": clean_text(raw.get("runtime_path")),
                "sop_instance_uid": clean_text(raw.get("sop_instance_uid")),
                "view": normalize_view(raw.get("view")),
                "selected": bool(raw.get("selected")),
            }
        )
    return result


def _resolve_query_images(
    record: Mapping[str, Any],
    case_entry: Mapping[str, Any] | None,
    resolver: ImageResolver,
    *,
    view_priority: tuple[str, ...],
    unknown_view_policy: str,
    materialize_8bit: bool,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Return selected runtime path/view and all available source-image metadata."""

    if case_entry is not None:
        selected_path = clean_text(case_entry.get("image_path"))
        selected_view = normalize_view(case_entry.get("view"))
        source_images = _case_source_images(case_entry)
        if selected_path and Path(selected_path).is_file():
            return selected_path, selected_view, source_images

        image_rows: list[dict[str, Any]] = []
        for item in source_images:
            source_path = clean_text(item.get("source_path"))
            runtime_path = clean_text(item.get("runtime_path"))
            image_rows.append(
                {
                    "source_path": source_path,
                    "runtime_path": runtime_path,
                    "view": normalize_view(item.get("view")),
                    "sop_instance_uid": clean_text(item.get("sop_instance_uid")),
                    "image_index": item.get("image_index"),
                    "exists": bool(source_path and Path(source_path).is_file()),
                }
            )
        selected = _select_representative(
            image_rows,
            view_priority=view_priority,
            unknown_view_policy=unknown_view_policy,
        )
        if selected is not None:
            source_path = Path(clean_text(selected["source_path"]))
            runtime_path = resolver.runtime_path(
                source_path, materialize_8bit=materialize_8bit
            )
            return str(runtime_path), normalize_view(selected["view"]), source_images

    study_uid = clean_text(
        _field(record, "StudyInstanceUid", "StudyInstanceUID", "study_instance_uid")
    )
    image_refs = as_list(_field(record, "ImagePath", "image_paths", default=[]))
    raw_views = as_list(
        _field(record, "ImageViewPosition", "ViewPosition", "view_positions", default=[])
    )

    image_rows: list[dict[str, Any]] = []
    source_images: list[dict[str, Any]] = []
    for index, image_ref in enumerate(image_refs):
        source = resolver.resolve(image_ref, study_uid=study_uid)
        view = normalize_view(value_at(raw_views, index, ""))
        exists = bool(source and source.is_file())
        row = {
            "source_path": str(source) if source else "",
            "runtime_path": "",
            "view": view,
            "sop_instance_uid": extract_sop_uid(image_ref),
            "image_index": index,
            "exists": exists,
        }
        image_rows.append(row)
        if exists:
            source_images.append(
                {
                    "image_index": index,
                    "source_path": row["source_path"],
                    "runtime_path": "",
                    "sop_instance_uid": row["sop_instance_uid"],
                    "view": view,
                    "selected": False,
                }
            )

    selected = _select_representative(
        image_rows,
        view_priority=view_priority,
        unknown_view_policy=unknown_view_policy,
    )
    if selected is None:
        return "", "", source_images

    source_selected = Path(clean_text(selected["source_path"]))
    selected_path = resolver.runtime_path(
        source_selected, materialize_8bit=materialize_8bit
    )
    selected_index = selected.get("image_index")
    for item in source_images:
        if item.get("image_index") == selected_index:
            item["selected"] = True
            item["runtime_path"] = str(selected_path)
            break
    return str(selected_path), normalize_view(selected["view"]), source_images


def prepare_rexvqa(
    *,
    source_root: str | Path = "/workspace/Datasets/06.CXR/ReXVQA",
    output_root: str | Path = "/workspace/visig_cr/work/datasets/rex/ReXVQA",
    rexgradient_source_root: str | Path = "/workspace/Datasets/06.CXR/ReXGradient-160K",
    rexgradient_output_root: str | Path = "/workspace/visig_cr/work/datasets/rex/ReXGradient-160K",
    overwrite: bool = False,
    path_mode: str = "symlink",
    view_priority: tuple[str, ...] = ("PA", "AP"),
    unknown_view_policy: str = "skip",
    scan_fallback: bool = True,
    materialize_8bit: bool = False,
    limit_per_split: int | None = None,
    allowed_validation_codes: set[str] | None = None,
    evaluator_answer_alias: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Create answer-time queries, retrieval queries, and reference answers.

    ``reference_answers`` is the canonical reference artifact.  When
    ``evaluator_answer_alias`` is enabled, each reference record also receives
    an ``answer`` alias for older evaluators that have not yet switched to
    ``reference_answer``.
    """

    source_dataset_root = Path(source_root).absolute()
    metadata_root = source_dataset_root / "metadata"
    output_dataset_root = reset_output_dir(output_root, overwrite=overwrite)
    rex_source_root = Path(rexgradient_source_root).absolute()
    rex_output_root = Path(rexgradient_output_root).absolute()
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"ReXVQA metadata directory does not exist: {metadata_root}")

    cases_path = rex_output_root / "cases.jsonl"
    if not cases_path.is_file():
        raise FileNotFoundError(
            "ReXGradient must be prepared first; missing case manifest: "
            + str(cases_path)
        )
    cases_by_id, cases_by_uid = _load_case_index(cases_path)

    resolver = ImageResolver(
        rex_source_root,
        rex_output_root,
        path_mode=path_mode,
        scan_fallback=scan_fallback,
    )
    if materialize_8bit and resolver.output_image_root.is_symlink():
        raise ValueError(
            "ReXGradient images/ is a source symlink. Re-run ReXGradient with "
            "--materialize-8bit before preparing ReXVQA with the same option."
        )
    if materialize_8bit:
        resolver.output_image_root.mkdir(parents=True, exist_ok=True)

    for directory in ("queries", "retrieval_queries", "reference_answers"):
        (output_dataset_root / directory).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "ReXVQA",
        "schema_id": "rexvqa_query_manifest",
        "source": {
            "dataset_root": str(source_dataset_root),
            "metadata_root": str(metadata_root),
            "rexgradient_source_root": str(rex_source_root),
            "rexgradient_cases": str(cases_path),
        },
        "output_root": str(output_dataset_root),
        "settings": {
            "view_priority": list(view_priority),
            "unknown_view_policy": unknown_view_policy,
            "materialize_8bit": materialize_8bit,
            "allowed_validation_codes": (
                sorted(allowed_validation_codes)
                if allowed_validation_codes is not None
                else None
            ),
            "evaluator_answer_alias": evaluator_answer_alias,
            "limit_per_split": limit_per_split,
        },
        "splits": {},
    }

    issues_path = output_dataset_root / "issues.jsonl"
    query_ids_seen: set[str] = set()

    with ExitStack() as stack:
        issues_writer = stack.enter_context(JSONLWriter(issues_path))

        for source_split in SOURCE_SPLITS:
            output_split = canonical_split(source_split)
            source_path = assert_real_payload(
                metadata_root / f"{source_split}_vqa_data.json"
            )
            queries_writer = stack.enter_context(
                JSONLWriter(output_dataset_root / "queries" / f"{output_split}.jsonl")
            )
            retrieval_writer = stack.enter_context(
                JSONLWriter(
                    output_dataset_root
                    / "retrieval_queries"
                    / f"{output_split}.jsonl"
                )
            )
            reference_writer = stack.enter_context(
                JSONLWriter(
                    output_dataset_root
                    / "reference_answers"
                    / f"{output_split}.jsonl"
                )
            )

            counters: Counter[str] = Counter()
            family_counts: Counter[str] = Counter()
            selected_view_counts: Counter[str] = Counter()
            reference_answer_counts: Counter[str] = Counter()
            validation_code_counts: Counter[str] = Counter()

            for record_index, (outer_query_id, record) in enumerate(
                iter_json_records(source_path)
            ):
                if limit_per_split is not None and record_index >= limit_per_split:
                    break
                counters["records_seen"] += 1

                query_id = clean_text(outer_query_id or _field(record, "query_id", "id"))
                if not query_id:
                    query_id = f"rexvqa:{output_split}:{record_index:09d}"
                if query_id in query_ids_seen:
                    issues_writer.write(
                        {
                            "dataset": "ReXVQA",
                            "issue_type": "query_skipped",
                            "query_id": query_id,
                            "split": output_split,
                            "reasons": ["duplicate_query_id"],
                        }
                    )
                    counters["skipped_duplicate_query_id"] += 1
                    continue

                validation_code = clean_text(_field(record, "validation_code"))
                validation_code_counts[validation_code or "missing"] += 1
                if (
                    allowed_validation_codes is not None
                    and validation_code not in allowed_validation_codes
                ):
                    counters["skipped_validation_code"] += 1
                    continue

                study_id = clean_text(_field(record, "study_id"))
                study_uid = clean_text(
                    _field(
                        record,
                        "StudyInstanceUid",
                        "StudyInstanceUID",
                        "study_instance_uid",
                    )
                )
                case_entry = cases_by_id.get(study_id) or cases_by_uid.get(study_uid)
                selected_path, selected_view, source_images = _resolve_query_images(
                    record,
                    case_entry,
                    resolver,
                    view_priority=tuple(normalize_view(item) for item in view_priority),
                    unknown_view_policy=unknown_view_policy,
                    materialize_8bit=materialize_8bit,
                )
                if not selected_path or not Path(selected_path).is_file():
                    issues_writer.write(
                        {
                            "dataset": "ReXVQA",
                            "issue_type": "query_skipped",
                            "query_id": query_id,
                            "split": output_split,
                            "study_id": study_id,
                            "study_instance_uid": study_uid,
                            "reasons": ["no_eligible_query_image"],
                            "source_images": source_images,
                        }
                    )
                    counters["skipped_no_query_image"] += 1
                    continue

                question_stem = clean_text(_field(record, "question"))
                options = _normalize_options(_field(record, "options", default=[]))
                reader_question = _format_mcq(question_stem, options)
                reference_answer = _reference_answer(
                    _field(record, "correct_answer"), options
                )
                if not reference_answer:
                    issues_writer.write(
                        {
                            "dataset": "ReXVQA",
                            "issue_type": "query_skipped",
                            "query_id": query_id,
                            "split": output_split,
                            "study_id": study_id,
                            "reasons": ["missing_reference_answer"],
                        }
                    )
                    counters["skipped_missing_reference_answer"] += 1
                    continue

                family, target_attribute = _classify_question(record)
                is_primary = family in PRIMARY_FAMILIES
                case_id = clean_text(
                    case_entry.get("case_id") if case_entry is not None else ""
                ) or f"rexgradient:{study_id or study_uid}"

                taxonomy = {
                    "category": clean_text(_field(record, "category")),
                    "class": clean_text(_field(record, "class")),
                    "subcategory": clean_text(_field(record, "subcategory")),
                    "task_name": clean_text(_field(record, "task_name")),
                }

                query_record = {
                    "query_id": query_id,
                    "image_path": selected_path,
                    "question": reader_question,
                    "retrieval_question": question_stem,
                    "reader_question": reader_question,
                    "options": options,
                    "question_family": family,
                    "target_attribute": target_attribute,
                    "dataset": "ReXVQA",
                    "study_id": study_id,
                    "study_instance_uid": study_uid,
                    "case_id": case_id,
                    "selected_view": selected_view,
                    "source_images": source_images,
                    "is_primary_protocol": is_primary,
                    **taxonomy,
                }
                retrieval_record = {
                    "query_id": query_id,
                    "image_path": selected_path,
                    "question": question_stem,
                    "options": options,
                    "question_family": family,
                    "target_attribute": target_attribute,
                    "dataset": "ReXVQA",
                    "study_id": study_id,
                    "case_id": case_id,
                    "selected_view": selected_view,
                    "is_primary_protocol": is_primary,
                }
                reference_record = {
                    "query_id": query_id,
                    "reference_answer": reference_answer,
                    "reference_answer_text": _reference_answer_text(
                        reference_answer, options
                    ),
                    "reference_explanation": clean_text(
                        _field(record, "correct_answer_explanation")
                    ),
                    "question_family": family,
                    "target_attribute": target_attribute,
                    "dataset": "ReXVQA",
                    "study_id": study_id,
                    "study_instance_uid": study_uid,
                    "case_id": case_id,
                    "options": options,
                    "validation_code": validation_code,
                    "is_primary_protocol": is_primary,
                    **taxonomy,
                }
                if evaluator_answer_alias:
                    reference_record["answer"] = reference_answer

                queries_writer.write(query_record)
                retrieval_writer.write(retrieval_record)
                reference_writer.write(reference_record)

                query_ids_seen.add(query_id)
                counters["records_written"] += 1
                counters["primary_records"] += int(is_primary)
                if case_entry is not None:
                    counters["joined_via_rexgradient_cases"] += 1
                else:
                    counters["resolved_from_vqa_metadata"] += 1
                family_counts[family] += 1
                selected_view_counts[selected_view or "UNKNOWN"] += 1
                reference_answer_counts[reference_answer] += 1

            manifest["splits"][output_split] = {
                **dict(counters),
                "source_metadata": str(source_path),
                "question_families": dict(family_counts),
                "selected_views": dict(selected_view_counts),
                "reference_answers": dict(reference_answer_counts),
                "validation_codes": dict(validation_code_counts),
            }

    manifest["outputs"] = {
        "queries": {
            split: str(output_dataset_root / "queries" / f"{split}.jsonl")
            for split in OUTPUT_SPLITS
        },
        "retrieval_queries": {
            split: str(
                output_dataset_root / "retrieval_queries" / f"{split}.jsonl"
            )
            for split in OUTPUT_SPLITS
        },
        "reference_answers": {
            split: str(
                output_dataset_root / "reference_answers" / f"{split}.jsonl"
            )
            for split in OUTPUT_SPLITS
        },
        "issues_jsonl": str(issues_path),
        "manifest_json": str(output_dataset_root / "manifest.json"),
    }
    write_json(output_dataset_root / "manifest.json", manifest)

    total_written = sum(
        int(manifest["splits"].get(split, {}).get("records_written", 0))
        for split in OUTPUT_SPLITS
    )
    if strict and total_written == 0:
        raise RuntimeError("No ReXVQA queries were generated")
    return manifest
