"""Patient-disjoint MIMIC memory/query protocol construction."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..io_utils import file_sha256, read_jsonl, stable_json_sha256, write_json
from ..protocol import memory_counts, protocol_layout
from ..schemas import CaseRecord


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _rank_cases(
    cases: list[CaseRecord],
    *,
    seed: int,
    max_cases_per_subject: int | None,
) -> list[CaseRecord]:
    """Create one deterministic ranking while preserving subject diversity."""

    by_subject: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        by_subject[case.subject_id].append(case)
    rng = random.Random(seed)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    for subject in subjects:
        by_subject[subject].sort(key=lambda row: (row.study_id, row.case_id))
        rng.shuffle(by_subject[subject])
        if max_cases_per_subject is not None:
            by_subject[subject] = by_subject[subject][:max_cases_per_subject]

    ranked: list[CaseRecord] = []
    depth = 0
    while True:
        added = False
        for subject in subjects:
            rows = by_subject[subject]
            if depth < len(rows):
                ranked.append(rows[depth])
                added = True
        if not added:
            break
        depth += 1
    return ranked


def _select_query_cases(
    cases: list[CaseRecord],
    *,
    seed: int,
    max_cases: int | None,
) -> list[CaseRecord]:
    ranked = _rank_cases(
        cases,
        seed=seed,
        max_cases_per_subject=None,
    )
    return ranked[:max_cases] if max_cases else ranked


def _ids(cases: Iterable[CaseRecord], field: str) -> set[str]:
    return {str(getattr(case, field)) for case in cases}


def prepare_mimic_roles(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None = None,
    output_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create nested memory manifests and a disjoint MIMIC query pool.

    The default protocol uses MIMIC ``train`` for memory and ``validate/test``
    for queries.  Optional train holdout is supported only when explicitly
    enabled in ``dataset_protocol.train_holdout``.
    """

    protocol_cfg = cfg.get("dataset_protocol", {})
    layout = protocol_layout(
        cfg,
        requested_protocol=requested_protocol,
        requested_memory_size=None,
    )
    if output_root is not None:
        root = Path(output_root)
        # Preserve the protocol directory layout under a caller-selected root.
        layout = dict(layout)
        layout["root"] = root
        maximum = int(layout["maximum_memory_size"])
        size = int(layout["memory_size"])
        layout.update(
            {
                "memory_pool": root / f"memory_pool_{maximum}",
                "memory": root / f"memory_{size}",
                "memory_pool_cases_jsonl": root
                / f"memory_pool_{maximum}"
                / "cases.jsonl",
                "memory_cases_jsonl": root / f"memory_{size}" / "cases.jsonl",
                "query_cases_jsonl": root / "query" / "cases.jsonl",
                "role_assignments_jsonl": root / "role_assignments.jsonl",
                "split_manifest_json": root / "split_manifest.json",
            }
        )
    root = Path(layout["root"])
    if root.exists() and overwrite:
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Protocol output already exists: {root}. Use --overwrite to rebuild it."
        )
    root.mkdir(parents=True, exist_ok=True)

    master_path = Path(
        cfg.get("paths", {}).get(
            "master_cases_jsonl",
            cfg.get("paths", {}).get("cases_jsonl", ""),
        )
    )
    if not master_path.is_file():
        raise FileNotFoundError(f"Master MIMIC cases manifest not found: {master_path}")
    cases = [CaseRecord.model_validate(row) for row in read_jsonl(master_path)]
    if not cases:
        raise ValueError(f"Master MIMIC cases manifest is empty: {master_path}")

    seed = int(protocol_cfg.get("seed", cfg.get("project", {}).get("seed", 42)))
    memory_cfg = protocol_cfg.get("memory", {})
    memory_splits = {
        str(value) for value in memory_cfg.get("source_splits", ["train"])
    }
    query_cfg = protocol_cfg.get("in_domain_queries", {}).get(
        "mimic_case_pool", {}
    )
    query_splits = {
        str(value) for value in query_cfg.get("source_splits", ["validate", "test"])
    }

    memory_candidates = [case for case in cases if case.split in memory_splits]
    query_candidates = [case for case in cases if case.split in query_splits]

    holdout_cfg = protocol_cfg.get("train_holdout", {})
    holdout_subjects: set[str] = set()
    if bool(holdout_cfg.get("enabled", False)):
        train_subjects = sorted(_ids(memory_candidates, "subject_id"))
        rng = random.Random(seed + 1009)
        rng.shuffle(train_subjects)
        requested_count = holdout_cfg.get("subject_count")
        if requested_count is None:
            fraction = float(holdout_cfg.get("subject_fraction", 0.10))
            requested_count = max(1, round(len(train_subjects) * fraction))
        holdout_subjects = set(train_subjects[: int(requested_count)])
        query_candidates.extend(
            case for case in memory_candidates if case.subject_id in holdout_subjects
        )
        memory_candidates = [
            case for case in memory_candidates if case.subject_id not in holdout_subjects
        ]

    query_cases = _select_query_cases(
        query_candidates,
        seed=seed + 1,
        max_cases=(
            int(query_cfg["max_cases"])
            if query_cfg.get("max_cases") is not None
            else None
        ),
    )
    query_subjects = _ids(query_cases, "subject_id")
    memory_candidates = [
        case for case in memory_candidates if case.subject_id not in query_subjects
    ]

    per_subject_raw = memory_cfg.get("max_cases_per_subject", 1)
    max_cases_per_subject = (
        None if per_subject_raw is None else int(per_subject_raw)
    )
    ranked_memory = _rank_cases(
        memory_candidates,
        seed=seed,
        max_cases_per_subject=max_cases_per_subject,
    )
    counts = memory_counts(cfg)
    maximum = counts[-1]
    if len(ranked_memory) < maximum:
        raise ValueError(
            f"Only {len(ranked_memory)} eligible memory cases remain, but the "
            f"largest requested nested memory contains {maximum} cases"
        )
    memory_pool = ranked_memory[:maximum]

    pool_path = Path(layout["memory_pool_cases_jsonl"])
    _write_jsonl(pool_path, (case.model_dump() for case in memory_pool))
    memory_paths: dict[str, str] = {}
    for count in counts:
        target = root / f"memory_{count}" / "cases.jsonl"
        _write_jsonl(target, (case.model_dump() for case in memory_pool[:count]))
        memory_paths[str(count)] = str(target)

    query_path = Path(layout["query_cases_jsonl"])
    _write_jsonl(query_path, (case.model_dump() for case in query_cases))

    memory_rank = {case.case_id: index + 1 for index, case in enumerate(memory_pool)}
    assignments: list[dict[str, Any]] = []
    for case in memory_pool:
        assignments.append(
            {
                "case_id": case.case_id,
                "subject_id": case.subject_id,
                "study_id": case.study_id,
                "dicom_id": case.dicom_id,
                "source_split": case.split,
                "role": "memory",
                "memory_rank": memory_rank[case.case_id],
                "included_memory_sizes": [
                    count for count in counts if memory_rank[case.case_id] <= count
                ],
            }
        )
    for case in query_cases:
        assignments.append(
            {
                "case_id": case.case_id,
                "subject_id": case.subject_id,
                "study_id": case.study_id,
                "dicom_id": case.dicom_id,
                "source_split": case.split,
                "role": "query",
                "memory_rank": None,
                "included_memory_sizes": [],
            }
        )
    assignments.sort(key=lambda row: (row["role"], row["case_id"]))
    _write_jsonl(Path(layout["role_assignments_jsonl"]), assignments)

    memory_subjects = _ids(memory_pool, "subject_id")
    memory_studies = _ids(memory_pool, "study_id")
    memory_case_ids = _ids(memory_pool, "case_id")
    memory_images = _ids(memory_pool, "dicom_id")
    query_studies = _ids(query_cases, "study_id")
    query_case_ids = _ids(query_cases, "case_id")
    query_images = _ids(query_cases, "dicom_id")
    overlap = {
        "subject_overlap_count": len(memory_subjects & query_subjects),
        "study_overlap_count": len(memory_studies & query_studies),
        "case_overlap_count": len(memory_case_ids & query_case_ids),
        "image_overlap_count": len(memory_images & query_images),
    }
    if any(overlap.values()):
        raise AssertionError(f"Protocol leakage detected: {overlap}")

    manifest = {
        "protocol_id": str(layout["protocol_id"]),
        "seed": seed,
        "split_unit": str(protocol_cfg.get("split_unit", "subject")),
        "master_cases_jsonl": str(master_path),
        "master_cases_sha256": file_sha256(master_path),
        "memory_source_splits": sorted(memory_splits),
        "query_source_splits": sorted(query_splits),
        "train_holdout_enabled": bool(holdout_cfg.get("enabled", False)),
        "train_holdout_subject_count": len(holdout_subjects),
        "nested_memory_case_counts": counts,
        "memory_pool_case_count": len(memory_pool),
        "memory_pool_subject_count": len(memory_subjects),
        "query_case_count": len(query_cases),
        "query_subject_count": len(query_subjects),
        "memory_case_counts": {
            str(count): count for count in counts
        },
        "memory_case_paths": memory_paths,
        "memory_subject_sha256": stable_json_sha256(sorted(memory_subjects)),
        "query_subject_sha256": stable_json_sha256(sorted(query_subjects)),
        "memory_case_sha256": stable_json_sha256(
            [case.case_id for case in memory_pool]
        ),
        "query_case_sha256": stable_json_sha256(
            [case.case_id for case in query_cases]
        ),
        "source_split_counts": dict(sorted(Counter(case.split for case in cases).items())),
        **overlap,
    }
    write_json(Path(layout["split_manifest_json"]), manifest)
    return {
        "ok": True,
        "protocol_root": str(root),
        "split_manifest": str(layout["split_manifest_json"]),
        "memory_pool_cases": str(pool_path),
        "query_cases": str(query_path),
        "role_assignments": str(layout["role_assignments_jsonl"]),
        **manifest,
    }


__all__ = ["prepare_mimic_roles"]
