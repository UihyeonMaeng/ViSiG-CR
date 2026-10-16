"""Preprocess the PhysioNet CXR resources used by ViSiG-CR.

The core ViSiG-CR memory remains a *single visual case graph*.  These adapters
therefore produce two different classes of artifacts:

1. ``mimic-cxr/cases.jsonl``: the study-level ``CaseRecord`` manifest consumed
   by ontology extraction, MedSigLIP indexing, and graph construction.
2. audit/reference layers from MIMIC labels, RadGraph, Chest ImaGenome, and
   MS-CXR.  They are kept outside the retrieval graph and can be used for
   extraction validation, anatomy/relation audit, and phrase-grounding audit.

The module is intentionally streaming where the source format permits it.  In
particular, Chest ImaGenome scene graphs are read directly from ``scene_graph.zip``
without extracting hundreds of thousands of JSON files.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..mimic import prepare_manifest
from .common import (
    JSONLWriter,
    clean_text,
    iter_json_records,
    json_safe,
    read_jsonl,
    write_json,
)


MIMIC_ID_PATTERN = re.compile(
    r"(?:^|[/_\\-])p(?P<subject>\d{3,})(?:[/_\\-].*?)?s(?P<study>\d{3,})(?:\D|$)",
    re.IGNORECASE,
)
STUDY_PATTERN = re.compile(r"(?:^|\D)s(?P<study>\d{3,})(?:\D|$)", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"(?:^|\D)p(?P<subject>\d{3,})(?:\D|$)", re.IGNORECASE)


def _path(cfg: Mapping[str, Any], key: str) -> Path:
    value = cfg.get("paths", {}).get(key)
    if not value:
        raise KeyError(f"config.paths.{key} is required")
    return Path(str(value))


def _source_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _id_text(value: Any) -> str:
    """Normalize integer-like identifiers without introducing a trailing .0."""

    text = clean_text(value)
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def _parse_mimic_ids(*values: Any) -> tuple[str, str, str]:
    """Return ``(subject_id, study_id, case_id)`` when IDs are discoverable."""

    text = " ".join(_id_text(value).replace("\\", "/") for value in values)
    match = MIMIC_ID_PATTERN.search(text)
    subject_id = match.group("subject") if match else ""
    study_id = match.group("study") if match else ""
    if not subject_id:
        subject = SUBJECT_PATTERN.search(text)
        subject_id = subject.group("subject") if subject else ""
    if not study_id:
        study = STUDY_PATTERN.search(text)
        study_id = study.group("study") if study else ""
    return subject_id, study_id, f"s{study_id}" if study_id else ""


def _normalize_relation(raw: Any) -> dict[str, str] | None:
    if isinstance(raw, Mapping):
        relation_type = clean_text(
            raw.get("type")
            or raw.get("relation")
            or raw.get("label")
            or raw.get("name")
        )
        target = clean_text(
            raw.get("target_entity_id")
            or raw.get("target")
            or raw.get("to")
            or raw.get("object")
        )
    elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
        relation_type = clean_text(raw[0])
        target = clean_text(raw[1])
    else:
        return None
    if not relation_type and not target:
        return None
    return {"type": relation_type, "target_entity_id": target}


def _normalize_radgraph_entities(raw_entities: Any) -> tuple[list[dict[str, Any]], int]:
    if isinstance(raw_entities, Mapping):
        items: Iterable[tuple[str, Any]] = (
            (clean_text(key), value) for key, value in raw_entities.items()
        )
    elif isinstance(raw_entities, list):
        items = ((clean_text(index), value) for index, value in enumerate(raw_entities))
    else:
        return [], 0

    entities: list[dict[str, Any]] = []
    relation_count = 0
    for entity_id, raw in items:
        if not isinstance(raw, Mapping):
            continue
        relations = [
            relation
            for relation in (
                _normalize_relation(item) for item in raw.get("relations", []) or []
            )
            if relation is not None
        ]
        relation_count += len(relations)
        entities.append(
            {
                "entity_id": entity_id,
                "label": clean_text(raw.get("label") or raw.get("type")),
                "tokens": clean_text(
                    raw.get("tokens") or raw.get("text") or raw.get("span")
                ),
                "start_ix": raw.get("start_ix", raw.get("start")),
                "end_ix": raw.get("end_ix", raw.get("end")),
                "relations": relations,
            }
        )
    return entities, relation_count


def prepare_mimic_label_tables(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    chunksize: int = 100_000,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert CheXpert/NegBio label tables to case-keyed JSONL streams."""

    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    output_dir = root / "mimic-cxr" / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "chexpert": _path(cfg, "mimic_chexpert_csv"),
        "negbio": _path(cfg, "mimic_negbio_csv"),
    }
    results: dict[str, Any] = {}
    for source_name, source_path in sources.items():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output_path = output_dir / f"{source_name}.jsonl"
        written = 0
        label_columns: list[str] = []
        with JSONLWriter(output_path) as writer:
            for frame in pd.read_csv(source_path, chunksize=chunksize):
                required = {"subject_id", "study_id"}
                missing = required - set(frame.columns)
                if missing:
                    raise ValueError(
                        f"{source_path} lacks required columns: {sorted(missing)}"
                    )
                if not label_columns:
                    label_columns = [
                        column
                        for column in frame.columns
                        if column not in {"subject_id", "study_id", "dicom_id"}
                    ]
                for row in frame.to_dict(orient="records"):
                    subject_id = _id_text(row.get("subject_id"))
                    study_id = _id_text(row.get("study_id"))
                    labels: dict[str, Any] = {}
                    for column in label_columns:
                        value = row.get(column)
                        if pd.isna(value):
                            labels[column] = None
                        elif isinstance(value, (int, float)):
                            labels[column] = float(value)
                        else:
                            labels[column] = clean_text(value)
                    writer.write(
                        {
                            "case_id": f"s{study_id}",
                            "subject_id": subject_id,
                            "study_id": study_id,
                            "source": source_name,
                            "labels": labels,
                        }
                    )
                    written += 1
                    if limit is not None and written >= limit:
                        break
                if limit is not None and written >= limit:
                    break
        results[source_name] = {
            "source": str(source_path),
            "output": str(output_path),
            "records_written": written,
            "label_columns": label_columns,
        }
    return results


def prepare_radgraph(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    include_report_text: bool = False,
    limit_per_file: int | None = None,
) -> dict[str, Any]:
    """Normalize RadGraph report graphs while retaining exact entity provenance."""

    source_root = _path(cfg, "radgraph_root")
    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    output_path = root / "radgraph" / "report_graphs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_specs = [
        ("mimic-cxr", "all", source_root / "MIMIC-CXR_graphs.json"),
        ("chexpert", "all", source_root / "CheXpert_graphs.json"),
        ("radgraph", "train", source_root / "train.json"),
        ("radgraph", "validate", source_root / "dev.json"),
        ("radgraph", "test", source_root / "test.json"),
    ]
    missing = [str(path) for _, _, path in source_specs if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing RadGraph source files: " + ", ".join(missing))

    file_stats: dict[str, Any] = {}
    total_records = 0
    total_entities = 0
    total_relations = 0
    mapped_cases = 0
    with JSONLWriter(output_path) as writer:
        for dataset, split, source_path in source_specs:
            records = 0
            entities_seen = 0
            relations_seen = 0
            for source_key, raw in iter_json_records(source_path):
                text = clean_text(
                    raw.get("text")
                    or raw.get("report")
                    or raw.get("sentence")
                    or raw.get("raw_text")
                )
                entities, relation_count = _normalize_radgraph_entities(
                    raw.get("entities") or raw.get("entity") or raw.get("annotations")
                )
                subject_id, study_id, case_id = _parse_mimic_ids(
                    source_key,
                    raw.get("path"),
                )
                subject_id = _id_text(raw.get("subject_id")) or subject_id
                study_id = _id_text(raw.get("study_id")) or study_id
                case_id = f"s{study_id}" if study_id else case_id
                output: dict[str, Any] = {
                    "record_id": f"{source_path.stem}:{source_key}",
                    "source_key": source_key,
                    "dataset": dataset,
                    "split": split,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "case_id": case_id,
                    "text_sha256": _sha256_text(text) if text else "",
                    "text_length": len(text),
                    "entities": entities,
                    "source_file": str(source_path),
                }
                if include_report_text:
                    output["report_text"] = text
                writer.write(output)
                records += 1
                total_records += 1
                entities_seen += len(entities)
                total_entities += len(entities)
                relations_seen += relation_count
                total_relations += relation_count
                mapped_cases += int(bool(case_id))
                if limit_per_file is not None and records >= limit_per_file:
                    break
            file_stats[source_path.name] = {
                "dataset": dataset,
                "split": split,
                "records_written": records,
                "entities": entities_seen,
                "relations": relations_seen,
            }

    return {
        "source_root": str(source_root),
        "output": str(output_path),
        "records_written": total_records,
        "entities": total_entities,
        "relations": total_relations,
        "records_with_mimic_case_id": mapped_cases,
        "include_report_text": include_report_text,
        "files": file_stats,
    }


def _scene_graph_identity(raw: Mapping[str, Any], member_name: str) -> dict[str, str]:
    image_id = clean_text(
        raw.get("image_id")
        or raw.get("dicom_id")
        or raw.get("id")
        or Path(member_name).stem
    )
    subject_id, study_id, case_id = _parse_mimic_ids(
        raw.get("path"), member_name
    )
    subject_id = _id_text(raw.get("subject_id")) or subject_id
    study_id = _id_text(raw.get("study_id")) or study_id
    case_id = f"s{study_id}" if study_id else case_id
    return {
        "image_id": image_id,
        "dicom_id": clean_text(raw.get("dicom_id")) or image_id,
        "subject_id": subject_id,
        "study_id": study_id,
        "case_id": case_id,
    }


def prepare_chest_imagenome(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Stream Chest ImaGenome scene graphs directly from the source ZIP."""

    zip_path = _path(cfg, "chest_imagenome_scene_graph_zip")
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    output_path = root / "chest-imagenome" / "scene_graphs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    object_count = 0
    attribute_count = 0
    relationship_count = 0
    malformed: list[str] = []
    with zipfile.ZipFile(zip_path) as archive, JSONLWriter(output_path) as writer:
        members = sorted(
            (
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.lower().endswith(".json")
            ),
            key=lambda item: item.filename,
        )
        for member in members:
            try:
                with archive.open(member) as handle:
                    raw = json.load(handle)
            except Exception as exc:
                malformed.append(f"{member.filename}: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(raw, Mapping):
                malformed.append(f"{member.filename}: top-level value is not an object")
                continue
            identity = _scene_graph_identity(raw, member.filename)
            objects = raw.get("objects") or []
            attributes = raw.get("attributes") or []
            relationships = raw.get("relationships") or raw.get("relations") or []
            writer.write(
                {
                    **identity,
                    "objects": json_safe(objects),
                    "attributes": json_safe(attributes),
                    "relationships": json_safe(relationships),
                    "source_member": member.filename,
                    "source_zip": str(zip_path),
                }
            )
            records += 1
            object_count += len(objects) if isinstance(objects, list) else 0
            attribute_count += len(attributes) if isinstance(attributes, list) else 0
            relationship_count += (
                len(relationships) if isinstance(relationships, list) else 0
            )
            if limit is not None and records >= limit:
                break

    return {
        "source_zip": str(zip_path),
        "output": str(output_path),
        "records_written": records,
        "objects": object_count,
        "attributes": attribute_count,
        "relationships": relationship_count,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:20],
    }


def _candidate_dicom_id(image: Mapping[str, Any]) -> str:
    for key in ("dicom_id", "DICOM_ID", "image_id", "id"):
        value = clean_text(image.get(key))
        if value:
            return Path(value).stem
    filename = clean_text(
        image.get("file_name") or image.get("filename") or image.get("path")
    )
    return Path(filename.replace("\\", "/")).stem if filename else ""


def _mimic_image_paths(
    manifest_path: Path,
    jpg_root: Path,
    needed_dicom_ids: set[str],
) -> dict[str, str]:
    if not manifest_path.is_file() or not needed_dicom_ids:
        return {}
    found: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative = line.strip()
            if not relative:
                continue
            dicom_id = Path(relative).stem
            if dicom_id in needed_dicom_ids:
                found[dicom_id] = str((jpg_root / relative).resolve())
                if len(found) == len(needed_dicom_ids):
                    break
    return found


def prepare_ms_cxr(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert MS-CXR COCO annotations to one grounding record per annotation."""

    source_path = _path(cfg, "ms_cxr_json")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"MS-CXR JSON must be an object: {source_path}")

    images_raw = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories_raw = payload.get("categories") or []
    if not isinstance(images_raw, list) or not isinstance(annotations, list):
        raise ValueError("MS-CXR JSON must contain list-valued images and annotations")

    images = {
        clean_text(image.get("id")): image
        for image in images_raw
        if isinstance(image, Mapping)
    }
    categories = {
        clean_text(category.get("id")): clean_text(
            category.get("name") or category.get("label")
        )
        for category in categories_raw
        if isinstance(category, Mapping)
    }
    needed_dicom_ids = {
        _candidate_dicom_id(image)
        for image in images.values()
        if _candidate_dicom_id(image)
    }
    image_paths = _mimic_image_paths(
        _path(cfg, "mimic_image_filenames"),
        _path(cfg, "mimic_jpg_root"),
        needed_dicom_ids,
    )

    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    output_path = root / "ms-cxr" / "grounding_annotations.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_image_metadata = 0
    missing_local_image = 0
    with JSONLWriter(output_path) as writer:
        for annotation in annotations:
            if not isinstance(annotation, Mapping):
                continue
            image_id = clean_text(annotation.get("image_id"))
            image = images.get(image_id)
            if not isinstance(image, Mapping):
                missing_image_metadata += 1
                image = {}
            dicom_id = _candidate_dicom_id(image)
            local_image_path = image_paths.get(dicom_id, "")
            if not local_image_path or not Path(local_image_path).is_file():
                missing_local_image += 1
            filename = clean_text(
                image.get("file_name") or image.get("filename") or image.get("path")
            )
            subject_id, study_id, case_id = _parse_mimic_ids(filename, local_image_path)
            category_id = clean_text(annotation.get("category_id"))
            writer.write(
                {
                    "annotation_id": clean_text(annotation.get("id")) or str(written),
                    "image_id": image_id,
                    "dicom_id": dicom_id,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "case_id": case_id,
                    "image_path": local_image_path,
                    "source_image_path": filename,
                    "category_id": category_id,
                    "category": categories.get(category_id, ""),
                    "bbox_xywh": json_safe(annotation.get("bbox") or []),
                    "area": json_safe(annotation.get("area")),
                    "iscrowd": json_safe(annotation.get("iscrowd")),
                    "phrase": clean_text(
                        annotation.get("phrase")
                        or annotation.get("label_text")
                        or annotation.get("text")
                        or annotation.get("caption")
                    ),
                    "source_file": str(source_path),
                }
            )
            written += 1
            if limit is not None and written >= limit:
                break

    return {
        "source": str(source_path),
        "output": str(output_path),
        "records_written": written,
        "image_records": len(images),
        "category_records": len(categories),
        "missing_image_metadata": missing_image_metadata,
        "missing_local_image": missing_local_image,
    }


def prepare_mimic(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    split: str = "all",
    limit: int | None = None,
    case_limit: int | None = None,
    label_limit: int | None = None,
    skip_labels: bool = False,
    allow_missing_images: bool = False,
) -> dict[str, Any]:
    """Prepare the MIMIC case manifest and optional label-audit streams.

    ``limit`` is retained as a compatibility alias.  New workflows should use
    ``case_limit`` and ``label_limit`` independently so a small case-manifest
    smoke run does not silently truncate the full CheXpert/NegBio audit tables.
    """

    if limit is not None:
        if case_limit is None:
            case_limit = limit
        if label_limit is None:
            label_limit = limit
    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    cases_path = root / "mimic-cxr" / "cases.jsonl"
    manifest = prepare_manifest(
        dict(cfg),
        output_path=cases_path,
        split=split,
        limit=case_limit,
        allow_missing=allow_missing_images,
    )
    labels: dict[str, Any]
    if skip_labels:
        labels = {"skipped": True, "reason": "skip_labels requested"}
    else:
        labels = prepare_mimic_label_tables(
            cfg,
            output_root=root,
            limit=label_limit,
        )
    return {
        "cases": manifest,
        "labels": labels,
        "case_limit": case_limit,
        "label_limit": label_limit,
        "skip_labels": skip_labels,
    }


def prepare_physionet_all(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    split: str = "all",
    limit: int | None = None,
    case_limit: int | None = None,
    label_limit: int | None = None,
    auxiliary_limit: int | None = None,
    skip_labels: bool = False,
    allow_missing_images: bool = False,
    include_radgraph_text: bool = False,
) -> dict[str, Any]:
    """Prepare every dataset downloaded by ``download_physionet_benchmarks.sh``.

    ``limit`` remains a compatibility alias for all three explicit limits.
    """

    if limit is not None:
        case_limit = limit if case_limit is None else case_limit
        label_limit = limit if label_limit is None else label_limit
        auxiliary_limit = limit if auxiliary_limit is None else auxiliary_limit
    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    root.mkdir(parents=True, exist_ok=True)
    result = {
        "output_root": str(root),
        "mimic": prepare_mimic(
            cfg,
            output_root=root,
            split=split,
            case_limit=case_limit,
            label_limit=label_limit,
            skip_labels=skip_labels,
            allow_missing_images=allow_missing_images,
        ),
        "radgraph": prepare_radgraph(
            cfg,
            output_root=root,
            include_report_text=include_radgraph_text,
            limit_per_file=auxiliary_limit,
        ),
        "chest_imagenome": prepare_chest_imagenome(
            cfg,
            output_root=root,
            limit=auxiliary_limit,
        ),
        "ms_cxr": prepare_ms_cxr(cfg, output_root=root, limit=auxiliary_limit),
    }
    result["config_hint"] = {
        "paths": {
            "cases_jsonl": str(root / "mimic-cxr" / "cases.jsonl"),
            "mimic_chexpert_labels_jsonl": str(
                root / "mimic-cxr" / "labels" / "chexpert.jsonl"
            ),
            "mimic_negbio_labels_jsonl": str(
                root / "mimic-cxr" / "labels" / "negbio.jsonl"
            ),
            "radgraph_preprocessed_jsonl": str(root / "radgraph" / "report_graphs.jsonl"),
            "chest_imagenome_preprocessed_jsonl": str(
                root / "chest-imagenome" / "scene_graphs.jsonl"
            ),
            "ms_cxr_preprocessed_jsonl": str(
                root / "ms-cxr" / "grounding_annotations.jsonl"
            ),
        }
    }
    write_json(root / "manifest.json", result)
    return result


def _gzip_readable(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            handle.read(1)
        return True
    except Exception:
        return False


def _validate_jsonl(
    path: Path,
    required_fields: Sequence[str],
    *,
    max_records: int,
) -> tuple[int, list[str]]:
    if not path.is_file():
        return 0, [f"Missing output: {path}"]
    errors: list[str] = []
    count = 0
    for row in read_jsonl(path):
        count += 1
        missing = [field for field in required_fields if field not in row]
        if missing:
            errors.append(f"{path}:{count} missing fields {missing}")
        if count >= max_records:
            break
    return count, errors


def validate_physionet(
    cfg: Mapping[str, Any],
    *,
    output_root: str | Path | None = None,
    max_records: int = 100,
    full_image_check: bool = False,
    require_outputs: bool = True,
) -> dict[str, Any]:
    """Validate downloaded sources and, optionally, normalized artifacts.

    Set ``require_outputs=False`` immediately after source download.  The
    default validates both the source bundle and every normalized JSONL output.
    """

    root = Path(output_root) if output_root else _path(cfg, "physionet_preprocessed_root")
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    required_files = {
        "mimic_image_filenames": _path(cfg, "mimic_image_filenames"),
        "mimic_metadata_csv": _path(cfg, "mimic_metadata_csv"),
        "mimic_split_csv": _path(cfg, "mimic_split_csv"),
        "mimic_chexpert_csv": _path(cfg, "mimic_chexpert_csv"),
        "mimic_negbio_csv": _path(cfg, "mimic_negbio_csv"),
        "mimic_test_labels_csv": _path(cfg, "mimic_test_labels_csv"),
        "mimic_record_list_csv": _path(cfg, "mimic_record_list_csv"),
        "mimic_study_list_csv": _path(cfg, "mimic_study_list_csv"),
        "mimic_provider_list_csv": _path(cfg, "mimic_provider_list_csv"),
        "chest_imagenome_scene_graph_zip": _path(
            cfg, "chest_imagenome_scene_graph_zip"
        ),
        "ms_cxr_json": _path(cfg, "ms_cxr_json"),
        "ms_cxr_csv": _path(cfg, "ms_cxr_csv"),
        "ms_cxr_converter": _path(cfg, "ms_cxr_converter"),
    }
    for name, path in required_files.items():
        checks[name] = _source_metadata(path)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Missing or empty source file: {path}")

    for name in (
        "mimic_metadata_csv",
        "mimic_split_csv",
        "mimic_chexpert_csv",
        "mimic_negbio_csv",
        "mimic_record_list_csv",
        "mimic_study_list_csv",
        "mimic_provider_list_csv",
    ):
        path = required_files[name]
        if path.is_file() and not _gzip_readable(path):
            errors.append(f"Unreadable gzip file: {path}")

    scene_zip = required_files["chest_imagenome_scene_graph_zip"]
    if scene_zip.is_file():
        try:
            with zipfile.ZipFile(scene_zip) as archive:
                bad = archive.testzip()
                checks["chest_imagenome_zip_first_bad_member"] = bad
                if bad:
                    errors.append(f"Corrupt ZIP member: {bad}")
        except Exception as exc:
            errors.append(f"Could not read {scene_zip}: {exc}")

    radgraph_root = _path(cfg, "radgraph_root")
    radgraph_required = [
        radgraph_root / "train.json",
        radgraph_root / "dev.json",
        radgraph_root / "test.json",
        radgraph_root / "MIMIC-CXR_graphs.json",
        radgraph_root / "CheXpert_graphs.json",
        radgraph_root / "models" / "model_checkpoint" / "model.tar.gz",
    ]
    checks["radgraph"] = [_source_metadata(path) for path in radgraph_required]
    errors.extend(
        f"Missing or empty source file: {path}"
        for path in radgraph_required
        if not path.is_file() or path.stat().st_size == 0
    )

    reports_root = _path(cfg, "mimic_report_root") / "files"
    first_report = next(reports_root.rglob("*.txt"), None) if reports_root.is_dir() else None
    checks["mimic_report_root"] = {
        "path": str(reports_root),
        "exists": reports_root.is_dir(),
        "sample_report": str(first_report) if first_report else "",
    }
    if first_report is None:
        errors.append(f"No MIMIC report TXT found under {reports_root}")

    image_manifest = required_files["mimic_image_filenames"]
    image_manifest_count = 0
    image_present_count = 0
    image_missing_examples: list[str] = []
    if image_manifest.is_file():
        with image_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                relative = line.strip()
                if not relative:
                    continue
                image_manifest_count += 1
                should_check = full_image_check or image_manifest_count <= max_records
                if should_check:
                    path = _path(cfg, "mimic_jpg_root") / relative
                    if path.is_file() and path.stat().st_size > 0:
                        image_present_count += 1
                    elif len(image_missing_examples) < 20:
                        image_missing_examples.append(relative)
        if image_manifest_count != 377_110:
            warnings.append(
                f"IMAGE_FILENAMES contains {image_manifest_count} rows; expected 377110 for the configured release"
            )
    checks["mimic_jpg"] = {
        "manifest_count": image_manifest_count,
        "checked_count": image_manifest_count if full_image_check else min(max_records, image_manifest_count),
        "present_count": image_present_count,
        "missing_examples": image_missing_examples,
        "full_image_check": full_image_check,
    }
    if full_image_check and image_present_count != image_manifest_count:
        errors.append(
            f"Full MIMIC-CXR-JPG check found {image_manifest_count - image_present_count} missing/empty files"
        )
    elif image_missing_examples:
        warnings.append(
            "The sampled MIMIC-CXR-JPG files are not all present. Do not build the visual index until the full image tree is available."
        )

    output_checks: dict[str, Any] = {}
    if require_outputs:
        output_specs = [
            (
                root / "mimic-cxr" / "cases.jsonl",
                (
                    "case_id",
                    "subject_id",
                    "study_id",
                    "image_path",
                    "report_path",
                    "split",
                ),
            ),
            (
                root / "mimic-cxr" / "labels" / "chexpert.jsonl",
                ("case_id", "source", "labels"),
            ),
            (
                root / "mimic-cxr" / "labels" / "negbio.jsonl",
                ("case_id", "source", "labels"),
            ),
            (
                root / "radgraph" / "report_graphs.jsonl",
                ("record_id", "entities", "source_file"),
            ),
            (
                root / "chest-imagenome" / "scene_graphs.jsonl",
                ("image_id", "objects", "attributes", "relationships"),
            ),
            (
                root / "ms-cxr" / "grounding_annotations.jsonl",
                ("annotation_id", "image_id", "bbox_xywh", "source_file"),
            ),
        ]
        for path, fields in output_specs:
            count, file_errors = _validate_jsonl(
                path, fields, max_records=max_records
            )
            output_checks[str(path)] = {
                "records_checked": count,
                "required_fields": list(fields),
            }
            errors.extend(file_errors)
    checks["preprocessed_outputs"] = {
        "required": require_outputs,
        "files": output_checks,
    }

    result = {
        "ok": not errors,
        "source_only": not require_outputs,
        "output_root": str(root),
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "validation.json", result)
    return result


__all__ = [
    "prepare_chest_imagenome",
    "prepare_mimic",
    "prepare_mimic_label_tables",
    "prepare_ms_cxr",
    "prepare_physionet_all",
    "prepare_radgraph",
    "validate_physionet",
]
