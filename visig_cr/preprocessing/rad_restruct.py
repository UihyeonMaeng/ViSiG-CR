"""Rad-ReStruct preprocessing for a separate hierarchical VQA protocol.

Rad-ReStruct is not flattened into the primary independent-question score.  The
adapter emits one query per hierarchy node, explicit parent/query order, and
reference answers in a separate stream.  A downstream session runner must feed
*model-predicted* parent answers; gold history is never copied into the query.
The official repository/evaluator remains the source of truth for the final
hierarchy-aware report metrics.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .common import canonical_split, iter_json_records
from .vqa_common import canonical_id, clean_text, write_vqa_split


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _field(row: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _source_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Rad-ReStruct source not found: {path}")
    files = [
        item for item in path.rglob("*")
        if item.is_file()
        and item.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        and not any(token in item.name.lower() for token in ("readme", "license", "config", "manifest"))
    ]
    if not files:
        raise FileNotFoundError(f"No supported Rad-ReStruct JSON found beneath: {path}")
    return sorted(files)


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        yield dict(value)
        return
    for outer_id, row in iter_json_records(path):
        record = dict(row)
        record.setdefault("_outer_id", outer_id)
        yield record


def _image_index(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"IU-Xray image root not found: {root}")
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    return index


def _resolve_images(record: Mapping[str, Any], root: Path, index: Mapping[str, Path]) -> list[Path]:
    raw = _field(record, "image_paths", "images", "image_path", "image", "image_ids", "image_id")
    values: list[Any]
    if isinstance(raw, (list, tuple)):
        values = list(raw)
    elif raw is None:
        values = []
    else:
        values = [raw]
    resolved: list[Path] = []
    for value in values:
        if isinstance(value, Mapping):
            value = _field(value, "path", "image_path", "id", "image_id", "filename")
        text = clean_text(value)
        if not text:
            continue
        candidate = Path(text)
        for path in (candidate, root / candidate):
            if path.is_file():
                resolved.append(path.resolve())
                break
        else:
            for key in (candidate.name, candidate.stem, text):
                if key in index:
                    resolved.append(index[key].resolve())
                    break
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(resolved))


def _children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for key in ("children", "subquestions", "sub_questions", "follow_ups", "questions"):
        raw = node.get(key)
        if isinstance(raw, list):
            values.extend(item for item in raw if isinstance(item, Mapping))
    return values


def _question(node: Mapping[str, Any]) -> str:
    return clean_text(_field(node, "question", "question_text", "query", "name", "label"))


def _answer(node: Mapping[str, Any]) -> Any:
    return _field(node, "answer", "answers", "reference_answer", "gold_answer", "target", "value")


def _walk_questions(
    nodes: Sequence[Mapping[str, Any]],
    *,
    parent_local_id: str | None = None,
    depth: int = 0,
    path: tuple[int, ...] = (),
) -> Iterator[tuple[Mapping[str, Any], str | None, int, tuple[int, ...], str]]:
    for index, node in enumerate(nodes):
        current_path = path + (index,)
        local_id = clean_text(_field(node, "question_id", "query_id", "id", "node_id")) or ".".join(map(str, current_path))
        if _question(node):
            yield node, parent_local_id, depth, current_path, local_id
            child_parent = local_id
        else:
            child_parent = parent_local_id
        children = _children(node)
        if children:
            yield from _walk_questions(
                children,
                parent_local_id=child_parent,
                depth=depth + 1,
                path=current_path,
            )


def _root_questions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("questions", "question_tree", "tree", "nodes", "qa", "qas"):
        value = record.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        if isinstance(value, Mapping):
            return [value]
    if _question(record):
        return [record]
    return []


def prepare_rad_restruct(
    cfg: dict[str, Any],
    *,
    source: str | Path | None = None,
    image_root: str | Path | None = None,
    output_root: str | Path | None = None,
    max_reports: int | None = None,
) -> dict[str, Any]:
    source = source or cfg["paths"]["rad_restruct_root"]
    image_root = image_root or cfg["paths"]["iu_xray_image_root"]
    output_root = output_root or Path(cfg["paths"]["work_root"]) / "datasets" / "vindr" / "rad-restruct"
    task_cfg = cfg.get("dataset_protocol", {}).get("external_queries", {}).get(
        "rad_restruct", {}
    )
    if max_reports is None and task_cfg.get("max_reports") is not None:
        max_reports = int(task_cfg["max_reports"])

    image_root_path = Path(image_root)
    images = _image_index(image_root_path)
    queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()
    report_count = 0
    seen: set[str] = set()

    stop = False
    for file_index, source_file in enumerate(_source_files(source)):
        for row_index, record in enumerate(_iter_records(source_file)):
            if max_reports is not None and report_count >= max_reports:
                stop = True
                break
            roots = _root_questions(record)
            if not roots:
                continue
            resolved_images = _resolve_images(record, image_root_path, images)
            if not resolved_images:
                counters["skipped_missing_image"] += 1
                continue
            report_id = clean_text(
                _field(record, "report_id", "study_id", "case_id", "id", "_outer_id")
            ) or f"{file_index:02d}:{row_index:07d}"
            split = canonical_split(clean_text(_field(record, "split", "partition", "source_split")) or "test")
            local_to_query: dict[str, str] = {}
            node_rows = list(_walk_questions(roots))
            # Allocate IDs before resolving parents.
            for node, _, _, path, local_id in node_rows:
                base = f"{canonical_id(report_id)}:{canonical_id(local_id) or '.'.join(map(str, path))}"
                query_id = f"rad_restruct:{base}"
                suffix = 1
                while query_id in seen:
                    suffix += 1
                    query_id = f"rad_restruct:{base}:{suffix}"
                seen.add(query_id)
                local_to_query[local_id] = query_id

            sequence_ids: list[str] = []
            for sequence_index, (node, parent_local, depth, path, local_id) in enumerate(node_rows):
                question = _question(node)
                answer = _answer(node)
                if not question or answer in (None, ""):
                    counters["skipped_missing_question_or_answer"] += 1
                    continue
                query_id = local_to_query[local_id]
                parent_query_id = local_to_query.get(parent_local or "")
                history_ids = list(sequence_ids)
                query = {
                    "query_id": query_id,
                    "source_query_id": local_id,
                    "dataset": "Rad-ReStruct",
                    "source_split": split,
                    "report_id": report_id,
                    "image_path": str(resolved_images[0]),
                    "additional_image_paths": [str(path) for path in resolved_images[1:]],
                    "question": question,
                    "reader_question": question,
                    "retrieval_question": question,
                    "options": [],
                    "question_family": clean_text(_field(node, "question_family", "category", "type")) or "hierarchical_report",
                    "task_type": "hierarchical_autoregressive",
                    "target_attribute": clean_text(_field(node, "target_attribute")) or "structured_report_node",
                    "reference_format": "scalar",
                    "hierarchy_depth": depth,
                    "hierarchy_path": list(path),
                    "sequence_index": sequence_index,
                    "parent_query_id": parent_query_id,
                    "history_query_ids": history_ids,
                    "history_policy": "model_predictions_only",
                    "official_evaluator_required": True,
                }
                reference = {
                    "query_id": query_id,
                    "dataset": "Rad-ReStruct",
                    "source_split": split,
                    "report_id": report_id,
                    "question_family": query["question_family"],
                    "task_type": "hierarchical_autoregressive",
                    "hierarchy_depth": depth,
                    "hierarchy_path": list(path),
                    "sequence_index": sequence_index,
                    "parent_query_id": parent_query_id,
                    "reference_answer": answer,
                }
                queries[split].append(query)
                references[split].append(reference)
                sequence_ids.append(query_id)
                counters["written_questions"] += 1
                counters[f"depth:{depth}"] += 1
            report_count += 1
            counters["written_reports"] += 1
        if stop:
            break

    if not counters["written_questions"]:
        raise ValueError(
            "No Rad-ReStruct questions were written. Verify the official JSON and IU-Xray image paths."
        )
    manifests: dict[str, Any] = {}
    for split in sorted(queries):
        manifests[split] = write_vqa_split(
            output_root,
            split=split,
            queries=queries[split],
            references=references[split],
            metadata={
                "dataset": "Rad-ReStruct",
                "source": str(source),
                "image_root": str(image_root),
                "evaluation_mode": "hierarchical_autoregressive",
                "official_evaluator_required": True,
            },
        )
    return {
        "ok": True,
        "dataset": "Rad-ReStruct",
        "output_root": str(output_root),
        "evaluation_mode": "hierarchical_autoregressive",
        "official_evaluator_required": True,
        "splits": manifests,
        "counters": dict(sorted(counters.items())),
    }


__all__ = ["prepare_rad_restruct"]
