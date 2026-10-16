"""VinDr-CXR-VQA adapter with answer/reference isolation.

The adapter intentionally separates semantic VQA from localization.  Bounding
boxes, rationales, and other gold fields are written only to the reference
stream.  The first ViSiG-CR protocol evaluates the semantic-compatible subset;
localization remains an explicitly separate task.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..io_utils import read_jsonl
from .common import canonical_split, iter_json_records
from .vqa_common import canonical_id, clean_text, format_mcq, write_vqa_split


_SUPPORTED_TABLE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tsv.gz",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
)
# VinDr-CXR is distributed as DICOM, but the ViSiG-CR reader/visual encoder opens
# images through PIL.  The canonical preprocessing path therefore accepts only the
# deterministic dataset-side PNG derivative created before this adapter runs.
_SUPPORTED_IMAGE_SUFFIXES = {".png"}
_MISSING_SPLIT_POLICIES = {"error", "full_release_test"}


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
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
    if isinstance(parsed, (list, tuple, set)):
        return [clean_text(item) for item in parsed if clean_text(item)]
    if "|" in text or ";" in text:
        return [clean_text(item) for item in text.replace(";", "|").split("|") if clean_text(item)]
    return [text]


def _parse_object(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (Mapping, list, tuple)):
        return value
    text = str(value).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
    return text


def _flatten_vqa_record(record: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Flatten the official ``data_v1.json`` image-level ``vqa`` list.

    Already-flat CSV/JSONL/Parquet records pass through unchanged. Outer image
    identifiers and split metadata are inherited by every question row, while
    child fields take precedence.
    """

    nested = record.get("vqa")
    if isinstance(nested, list):
        inherited = {key: value for key, value in record.items() if key != "vqa"}
        for index, item in enumerate(nested):
            if not isinstance(item, Mapping):
                continue
            flattened = {**inherited, **dict(item)}
            flattened.setdefault("_vqa_index", index)
            yield flattened
        return
    yield dict(record)


def _normalize_question_type(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("_", " ").replace("/", " ").replace("-", " ")
    return " ".join(text.split())


def _normalize_missing_split_policy(value: Any) -> str:
    policy = clean_text(value).lower().replace("-", "_") or "error"
    if policy not in _MISSING_SPLIT_POLICIES:
        raise ValueError(
            "Unsupported VinDr-CXR-VQA missing-split policy: "
            f"{value!r}; expected one of {sorted(_MISSING_SPLIT_POLICIES)}"
        )
    return policy


def _iter_table(path: Path) -> Iterator[dict[str, Any]]:
    lower = path.name.lower()
    if lower.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        opener = gzip.open if lower.endswith(".gz") else open
        delimiter = "\t" if ".tsv" in lower else ","
        with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter=delimiter):
                yield from _flatten_vqa_record(dict(row))
        return
    if lower.endswith((".jsonl", ".ndjson")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if isinstance(value, Mapping):
                    yield from _flatten_vqa_record(dict(value))
        return
    if lower.endswith(".json"):
        for outer_id, row in iter_json_records(path):
            record = dict(row)
            record.setdefault("_outer_id", outer_id)
            yield from _flatten_vqa_record(record)
        return
    if lower.endswith(".parquet"):
        import pandas as pd

        for row in pd.read_parquet(path).to_dict(orient="records"):
            yield from _flatten_vqa_record(dict(row))
        return
    raise ValueError(f"Unsupported VinDr-CXR-VQA table: {path}")


def _source_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"VinDr-CXR-VQA source not found: {path}")
    visible = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and not any(part.startswith(".") for part in item.relative_to(path).parts)
    ]
    official = sorted(item for item in visible if item.name.lower() == "data_v1.json")
    if len(official) == 1:
        return official
    if len(official) > 1:
        raise ValueError(
            "Multiple VinDr-CXR-VQA data_v1.json files were found beneath "
            f"{path}: {[str(item) for item in official]}"
        )
    files = [
        item
        for item in visible
        if item.name.lower().endswith(_SUPPORTED_TABLE_SUFFIXES)
        and not any(token in item.name.lower() for token in ("readme", "license", "manifest"))
    ]
    if not files:
        raise FileNotFoundError(f"No supported VinDr-CXR-VQA table found beneath: {path}")
    return sorted(files)


def _image_index(image_root: str | Path) -> dict[str, Path]:
    root = Path(image_root)
    if not root.is_dir():
        raise FileNotFoundError(f"VinDr-CXR image root not found: {root}")
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES:
            continue
        keys = {path.name.lower(), path.stem.lower()}
        for key in keys:
            if key in index and index[key] != path:
                duplicates.add(key)
            else:
                index[key] = path
    for key in duplicates:
        index.pop(key, None)
    return index


def _image_index_from_manifest(manifest_path: str | Path) -> dict[str, Path]:
    target = Path(manifest_path)
    if not target.is_file():
        raise FileNotFoundError(f"VinDr-CXR image manifest not found: {target}")
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for line_number, row in enumerate(read_jsonl(target), start=1):
        image_id = clean_text(row.get("image_id")).lower()
        image_path = Path(clean_text(row.get("image_path")))
        if (
            not image_id
            or not image_path.is_file()
            or image_path.suffix.lower() not in _SUPPORTED_IMAGE_SUFFIXES
        ):
            raise ValueError(
                f"Invalid VinDr image manifest row at {target}:{line_number}: "
                f"image_id={image_id!r}, image_path={image_path}"
            )
        keys = {image_id, image_path.name.lower(), image_path.stem.lower()}
        for key in keys:
            if key in index and index[key] != image_path:
                duplicates.add(key)
            else:
                index[key] = image_path.resolve()
    for key in duplicates:
        index.pop(key, None)
    if not index:
        raise ValueError(f"No usable image rows found in VinDr image manifest: {target}")
    return index


def _resolve_image(row: Mapping[str, Any], image_root: Path, index: Mapping[str, Path]) -> Path | None:
    raw_path = clean_text(
        _field(row, "image_path", "file_path", "filepath", "path", "image_file", "filename")
    )
    if raw_path:
        path = Path(raw_path)
        candidates = [path, image_root / path]
        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.suffix.lower() in _SUPPORTED_IMAGE_SUFFIXES
            ):
                return candidate.resolve()
    image_id = clean_text(
        _field(
            row,
            "image_id",
            "img_id",
            "image",
            "dicom_id",
            "sop_instance_uid",
            "study_id",
            "id",
        )
    )
    for key in (
        image_id,
        Path(image_id).name if image_id else "",
        Path(image_id).stem if image_id else "",
    ):
        normalized = key.lower()
        if normalized and normalized in index:
            return index[normalized].resolve()
    return None


def _question_type(row: Mapping[str, Any], question: str) -> str:
    raw = clean_text(_field(row, "question_type", "type", "category", "question_category"))
    if raw:
        return raw
    lower = question.lower()
    if lower.startswith(("is there", "are there", "does ", "do ", "has ", "have ")):
        return "Yes/No"
    if lower.startswith("where"):
        return "Where"
    if lower.startswith("what"):
        return "What"
    if lower.startswith("which"):
        return "Which"
    if lower.startswith("how many"):
        return "How many"
    return "Other"


def _family(question_type: str, question: str) -> str:
    text = f"{question_type} {question}".lower()
    if any(token in text for token in ("yes/no", "yes_no", "is there", "are there", "presence")):
        return "presence"
    if any(token in text for token in ("where", "location", "which side", "laterality")):
        return "location"
    if "how many" in text or "count" in text:
        return "count"
    if any(token in text for token in ("what", "which", "finding", "abnormal")):
        return "finding_identity"
    return "other"


def _options(row: Mapping[str, Any]) -> list[str]:
    direct = _parse_list(_field(row, "options", "choices", "answer_options"))
    if direct:
        return direct
    values: list[str] = []
    for letter in "ABCDEFGH":
        value = _field(row, letter, f"option_{letter.lower()}", f"choice_{letter.lower()}")
        if clean_text(value):
            values.append(f"{letter}. {clean_text(value)}")
    return values


def prepare_vindr_cxr_vqa(
    cfg: dict[str, Any],
    *,
    source: str | Path | None = None,
    image_root: str | Path | None = None,
    image_manifest: str | Path | None = None,
    output_root: str | Path | None = None,
    semantic_only: bool | None = None,
    allowed_question_types: Sequence[str] | None = None,
    max_questions: int | None = None,
    missing_split_policy: str | None = None,
) -> dict[str, Any]:
    """Normalize VinDr-CXR-VQA while keeping gold grounding out of queries."""

    external_cfg = cfg.get("dataset_protocol", {}).get("external_queries", {}).get(
        "vindr_cxr_vqa", {}
    )
    source = source or cfg["paths"]["vindr_cxr_vqa_root"]
    explicit_image_root = image_root is not None
    image_root = image_root or cfg["paths"].get(
        "vindr_cxr_png_root", cfg["paths"].get("vindr_cxr_image_root")
    )
    if image_root is None:
        raise KeyError("config.paths.vindr_cxr_png_root is required")
    if image_manifest is None and not explicit_image_root:
        configured_manifest = cfg["paths"].get("vindr_cxr_images_jsonl")
        if configured_manifest:
            # Enforce the intended two-stage contract: first validate/index the
            # dataset-side PNG tree with ``prepare_datasets.py vindr-cxr``, then
            # build VQA artifacts from that immutable registry.
            image_manifest = configured_manifest
    output_root = output_root or Path(cfg["paths"]["work_root"]) / "datasets" / "vindr" / "vindr-cxr-vqa"
    semantic_only = bool(
        external_cfg.get("semantic_only", True) if semantic_only is None else semantic_only
    )
    allowed = {
        _normalize_question_type(item)
        for item in (allowed_question_types or external_cfg.get("question_types", []))
        if clean_text(item)
    }
    if max_questions is None and external_cfg.get("max_questions") is not None:
        max_questions = int(external_cfg["max_questions"])
    missing_split_policy = _normalize_missing_split_policy(
        missing_split_policy
        if missing_split_policy is not None
        else external_cfg.get("missing_split_policy", "error")
    )

    image_root_path = Path(image_root)
    if image_manifest is not None:
        image_lookup = _image_index_from_manifest(image_manifest)
        image_index_source = str(Path(image_manifest))
    else:
        image_lookup = _image_index(image_root_path)
        image_index_source = str(image_root_path)
    queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    seen: set[str] = set()

    stop = False
    for file_index, table in enumerate(_source_files(source)):
        for row_index, row in enumerate(_iter_table(table)):
            counters["source_rows"] += 1
            question = clean_text(_field(row, "question", "question_text", "query"))
            answers = _parse_list(
                _field(row, "answer", "answers", "reference_answer", "gold_answer", "correct_answer")
            )
            if not question or not answers:
                counters["skipped_missing_question_or_answer"] += 1
                continue
            qtype = _question_type(row, question)
            qtype_key = _normalize_question_type(qtype)
            if allowed and qtype_key not in allowed:
                counters["skipped_question_type"] += 1
                continue
            image = _resolve_image(row, image_root_path, image_lookup)
            if image is None:
                counters["skipped_missing_image"] += 1
                continue
            family = _family(qtype, question)
            options = _options(row)
            source_id = clean_text(_field(row, "query_id", "question_id", "qa_id", "id", "_outer_id"))
            image_id = clean_text(_field(row, "image_id", "dicom_id", "study_id")) or image.stem
            nested_index = clean_text(_field(row, "_vqa_index"))
            base = canonical_id(source_id) or (
                f"{canonical_id(image_id)}:{nested_index}"
                if nested_index
                else f"{canonical_id(image_id)}:{row_index:07d}"
            )
            query_id = f"vindr_cxr_vqa:{base}"
            if query_id in seen:
                query_id = f"{query_id}:{file_index:02d}:{row_index:07d}"
            seen.add(query_id)
            source_split_raw = clean_text(
                _field(row, "split", "source_split", "partition")
            )
            if source_split_raw:
                split = canonical_split(source_split_raw)
                split_provenance = "source_record"
            elif missing_split_policy == "full_release_test":
                # The public data_v1.json schema does not expose the paper's
                # internal train/validation partition.  Treating the released
                # file as one frozen external evaluation corpus is permitted
                # only through this explicit policy; it must not be described
                # as the source dataset's official test split.
                split = "test"
                split_provenance = "forced_full_release_external_evaluation"
                counters["source_rows_missing_split"] += 1
            else:
                raise ValueError(
                    "VinDr-CXR-VQA source row has no split metadata. The public "
                    "data_v1.json schema commonly omits a split field, so the "
                    "adapter will not silently label the full release as test. "
                    "Use --missing-split-policy full-release-test only when the "
                    "entire released file has been prospectively frozen as an "
                    "external evaluation corpus. "
                    f"image_id={image_id!r}, question_type={qtype!r}"
                )
            counters[f"split_provenance:{split_provenance}"] += 1
            counters[f"written_split:{split}"] += 1
            reader_question = format_mcq(question, options)
            task_type = "multiple_choice" if options else (
                "binary" if answers[0].lower() in {"yes", "no", "true", "false", "present", "absent"} else "open"
            )
            query = {
                "query_id": query_id,
                "source_query_id": source_id,
                "dataset": "VinDr-CXR-VQA",
                "source_split": split,
                "source_split_raw": source_split_raw or "missing",
                "split_provenance": split_provenance,
                "evaluation_partition": (
                    "full_release_external_evaluation"
                    if split_provenance == "forced_full_release_external_evaluation"
                    else "source_defined_split"
                ),
                "image_id": image_id,
                "image_path": str(image),
                "question": reader_question,
                "reader_question": reader_question,
                "retrieval_question": question,
                "options": options,
                "question_type": qtype,
                "question_family": family,
                "task_type": task_type,
                "target_attribute": "presence" if family == "presence" else (
                    "anatomy" if family == "location" else "finding_identity"
                ),
                "reference_format": "answer_set" if len(answers) > 1 else "scalar",
                "semantic_only": semantic_only,
            }
            reference: dict[str, Any] = {
                "query_id": query_id,
                "dataset": "VinDr-CXR-VQA",
                "source_split": split,
                "source_split_raw": source_split_raw or "missing",
                "split_provenance": split_provenance,
                "evaluation_partition": (
                    "full_release_external_evaluation"
                    if split_provenance == "forced_full_release_external_evaluation"
                    else "source_defined_split"
                ),
                "question_type": qtype,
                "question_family": family,
                "task_type": task_type,
            }
            if len(answers) > 1:
                reference["reference_answers"] = answers
            else:
                reference["reference_answer"] = answers[0]
            boxes = _parse_object(_field(row, "bbox", "bboxes", "bounding_box", "bounding_boxes", "boxes"))
            if boxes not in (None, "", [], {}):
                reference["reference_boxes"] = boxes
                reference["has_grounding"] = True
            rationale = clean_text(_field(row, "reason", "rationale", "explanation"))
            if rationale:
                reference["reference_rationale"] = rationale
            gt_finding = _parse_object(_field(row, "gt_finding", "finding", "findings"))
            gt_location = _parse_object(_field(row, "gt_location", "location", "locations"))
            if gt_finding not in (None, "", [], {}):
                reference["gt_finding"] = gt_finding
                reference["reference_finding"] = gt_finding
            if gt_location not in (None, "", [], {}):
                reference["gt_location"] = gt_location
                reference["reference_location"] = gt_location
                if "reference_boxes" not in reference and isinstance(gt_location, str):
                    import re

                    matches = re.findall(
                        r"<loc_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)>",
                        gt_location,
                    )
                    if matches:
                        reference["reference_boxes"] = [
                            {
                                "x_min": int(x1),
                                "y_min": int(y1),
                                "x_max": int(x2),
                                "y_max": int(y2),
                            }
                            for x1, y1, x2, y2 in matches
                        ]
                        reference["has_grounding"] = True
            queries[split].append(query)
            references[split].append(reference)
            counters["written"] += 1
            counters[f"question_type:{qtype}"] += 1
            counters[f"family:{family}"] += 1
            if max_questions is not None and counters["written"] >= max_questions:
                stop = True
                break
        if stop:
            break

    if not counters["written"]:
        raise ValueError(
            "No VinDr-CXR-VQA rows were written; "
            f"counters={dict(sorted(counters.items()))}, "
            f"source={source}, image_index_source={image_index_source}. "
            "Run `prepare_datasets.py vindr-cxr --config config.yaml` first and "
            "verify that VQA image_id values occur in its images.jsonl."
        )
    manifests: dict[str, Any] = {}
    for split in sorted(queries):
        manifests[split] = write_vqa_split(
            output_root,
            split=split,
            queries=queries[split],
            references=references[split],
            metadata={
                "dataset": "VinDr-CXR-VQA",
                "source": str(source),
                "image_root": str(image_root),
                "image_manifest": str(image_manifest) if image_manifest is not None else None,
                "image_index_source": image_index_source,
                "semantic_only": semantic_only,
                "allowed_question_types": sorted(allowed),
                "missing_split_policy": missing_split_policy,
                "split_provenance_counts": {
                    key.removeprefix("split_provenance:"): value
                    for key, value in sorted(counters.items())
                    if key.startswith("split_provenance:")
                },
                "partition_warning": (
                    "The public source lacked split metadata; this output split "
                    "is the full released VinDr-CXR-VQA corpus prospectively "
                    "frozen as an external evaluation partition, not the "
                    "dataset paper's official test split."
                    if counters.get(
                        "split_provenance:forced_full_release_external_evaluation"
                    )
                    else None
                ),
            },
        )
    return {
        "ok": True,
        "dataset": "VinDr-CXR-VQA",
        "output_root": str(output_root),
        "semantic_only": semantic_only,
        "missing_split_policy": missing_split_policy,
        "image_index_source": image_index_source,
        "indexed_image_keys": len(image_lookup),
        "splits": manifests,
        "counters": dict(sorted(counters.items())),
    }


__all__ = ["prepare_vindr_cxr_vqa"]
