"""MIMIC-CXR study-level manifest construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm


def subject_prefix(subject_id: int | str) -> str:
    value = str(int(subject_id))
    return f"p{value[:2]}"


def build_image_path(
    jpg_root: Path,
    subject_id: int,
    study_id: int,
    dicom_id: str,
) -> Path:
    return (
        jpg_root
        / "files"
        / subject_prefix(subject_id)
        / f"p{int(subject_id)}"
        / f"s{int(study_id)}"
        / f"{dicom_id}.jpg"
    )


def build_report_path(report_root: Path, subject_id: int, study_id: int) -> Path:
    return (
        report_root
        / "files"
        / subject_prefix(subject_id)
        / f"p{int(subject_id)}"
        / f"s{int(study_id)}.txt"
    )


def normalize_view(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).strip().upper()


def choose_one_frontal(group: pd.DataFrame, priority: list[str]) -> pd.Series | None:
    ranked = group.copy()
    rank_map = {view: index for index, view in enumerate(priority)}
    ranked["_view_rank"] = (
        ranked["view"].map(rank_map).fillna(len(priority)).astype(int)
    )
    ranked = ranked.sort_values(["_view_rank", "dicom_id"], kind="stable")
    return None if ranked.empty else ranked.iloc[0]


def prepare_manifest(
    cfg: dict[str, Any],
    *,
    output_path: str | Path | None = None,
    split: str = "all",
    limit: int | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    paths = cfg["paths"]
    prep = cfg.get("mimic_prepare", {})
    metadata_path = Path(paths["mimic_metadata_csv"])
    split_path = Path(paths["mimic_split_csv"])
    jpg_root = Path(paths["mimic_jpg_root"])
    report_root = Path(paths["mimic_report_root"])
    target = Path(output_path) if output_path else Path(paths["cases_jsonl"])

    for required in (metadata_path, split_path):
        if not required.exists():
            raise FileNotFoundError(f"Required file does not exist: {required}")

    metadata = pd.read_csv(metadata_path)
    split_frame = pd.read_csv(split_path)
    required_metadata = {"dicom_id", "subject_id", "study_id", "ViewPosition"}
    required_split = {"dicom_id", "subject_id", "study_id", "split"}
    missing_metadata = required_metadata - set(metadata.columns)
    missing_split = required_split - set(split_frame.columns)
    if missing_metadata:
        raise ValueError(f"Metadata CSV lacks columns: {sorted(missing_metadata)}")
    if missing_split:
        raise ValueError(f"Split CSV lacks columns: {sorted(missing_split)}")

    merged = metadata.merge(
        split_frame[["dicom_id", "subject_id", "study_id", "split"]],
        on=["dicom_id", "subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    )
    merged["view"] = merged["ViewPosition"].map(normalize_view)
    frontal_views = [str(view).upper() for view in prep.get("frontal_views", ["PA", "AP"])]
    priority = [str(view).upper() for view in prep.get("view_priority", frontal_views)]
    merged = merged[merged["view"].isin(frontal_views)].copy()
    if split != "all":
        merged = merged[merged["split"] == split].copy()

    selected: list[pd.Series] = []
    for _, group in tqdm(
        merged.groupby(["subject_id", "study_id"], sort=True),
        desc="Selecting one frontal image per study",
    ):
        row = choose_one_frontal(group, priority)
        if row is not None:
            selected.append(row)
            if limit is not None and len(selected) >= limit:
                break

    target.parent.mkdir(parents=True, exist_ok=True)
    require_existing = bool(prep.get("require_existing_files", True)) and not allow_missing
    written = 0
    missing_images = 0
    missing_reports = 0
    with target.open("w", encoding="utf-8") as output:
        for row in tqdm(selected, desc="Writing manifest"):
            subject_id = int(row["subject_id"])
            study_id = int(row["study_id"])
            dicom_id = str(row["dicom_id"])
            image_path = build_image_path(jpg_root, subject_id, study_id, dicom_id)
            report_path = build_report_path(report_root, subject_id, study_id)
            if not image_path.exists():
                missing_images += 1
            if not report_path.exists():
                missing_reports += 1
            if require_existing and (not image_path.exists() or not report_path.exists()):
                continue
            record = {
                "case_id": f"s{study_id}",
                "dataset": "MIMIC-CXR",
                "subject_id": str(subject_id),
                "study_id": str(study_id),
                "dicom_id": dicom_id,
                "image_path": str(image_path.resolve()),
                "report_path": str(report_path.resolve()),
                "view": str(row["view"]),
                "split": str(row["split"]),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "output": str(target),
        "written": written,
        "missing_images_seen": missing_images,
        "missing_reports_seen": missing_reports,
        "frontal_views": frontal_views,
        "view_priority": priority,
    }
