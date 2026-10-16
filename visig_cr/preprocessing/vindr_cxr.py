"""Prepare a lightweight VinDr-CXR image registry from dataset-side PNGs.

DICOM-to-PNG conversion is intentionally kept beside the restricted raw data,
not inside the ViSiG-CR repository.  This module validates the converted PNG tree
and writes only small JSONL/JSON artifacts under ``work/datasets``.  Every row
keeps an absolute reference to the dataset-side PNG; image bytes are never
copied or symlinked into ``work``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from ..io_utils import stable_json_sha256, write_json


EXPECTED_SPLITS = {"train": 15_000, "test": 3_000}
IMAGE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SUPPORTED_SUFFIXES = {".png"}


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _configured_path(cfg: Mapping[str, Any], key: str) -> Path:
    value = cfg.get("paths", {}).get(key)
    if not value:
        raise KeyError(f"config.paths.{key} is required")
    return Path(str(value))


def _split_pngs(image_root: Path, split: str) -> list[Path]:
    split_root = image_root / split
    if not split_root.is_dir():
        return []
    return sorted(
        path
        for path in split_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(split_root).parts)
    )


def prepare_vindr_cxr(
    cfg: Mapping[str, Any],
    *,
    image_root: str | Path | None = None,
    output_root: str | Path | None = None,
    max_images: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Index and validate model-readable VinDr PNGs without copying them.

    Expected dataset-side layout::

        <image_root>/train/<image_id>.png
        <image_root>/test/<image_id>.png

    The canonical full-run output is a compact registry::

        <output_root>/images.jsonl
        <output_root>/issues.jsonl
        <output_root>/manifest.json
    """

    if max_images is not None and max_images <= 0:
        raise ValueError("max_images must be positive")

    image_root_path = Path(image_root) if image_root else _configured_path(
        cfg, "vindr_cxr_png_root"
    )
    output_root_path = Path(output_root) if output_root else _configured_path(
        cfg, "vindr_cxr_preprocessed_root"
    )
    raw_root = Path(
        str(
            cfg.get("paths", {}).get(
                "vindr_cxr_raw_root", image_root_path.parent
            )
        )
    )

    image_root_path = image_root_path.resolve()
    output_root_path = output_root_path.resolve()
    raw_root = raw_root.resolve()
    if not image_root_path.is_dir():
        raise FileNotFoundError(
            "VinDr-CXR PNG root not found: "
            f"{image_root_path}. Run /workspace/Datasets/06.CXR/"
            "prepare_vindr_cxr_png.py first."
        )

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    written_counts: Counter[str] = Counter()
    seen_ids: dict[str, Path] = {}
    stopped = False

    for split in EXPECTED_SPLITS:
        files = _split_pngs(image_root_path, split)
        source_counts[split] = len(files)
        for path in files:
            if max_images is not None and len(records) >= max_images:
                stopped = True
                break

            image_id = path.stem.lower()
            relative_path = path.relative_to(image_root_path)
            base_issue = {
                "dataset": "VinDr-CXR",
                "split": split,
                "image_id": image_id,
                "image_path": str(path),
            }
            if not IMAGE_ID_RE.fullmatch(image_id):
                issues.append(
                    {
                        **base_issue,
                        "issue_type": "invalid_image_id",
                        "error": "PNG filename stem must be a 32-character lowercase hexadecimal image_id",
                    }
                )
                continue
            previous = seen_ids.get(image_id)
            if previous is not None and previous != path:
                issues.append(
                    {
                        **base_issue,
                        "issue_type": "duplicate_image_id",
                        "error": f"same image_id already indexed from {previous}",
                    }
                )
                continue

            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    image_format = image.format
                if image_format != "PNG":
                    raise ValueError(f"expected PNG payload, found format={image_format!r}")
                if mode != "L":
                    raise ValueError(f"expected 8-bit grayscale mode 'L', found {mode!r}")
                if width <= 0 or height <= 0:
                    raise ValueError(f"invalid image geometry: {(width, height)}")
            except Exception as error:  # noqa: BLE001 - collect every bad image
                issues.append(
                    {
                        **base_issue,
                        "issue_type": "unreadable_png",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue

            source_dicom = raw_root / split / f"{image_id}.dicom"
            if not source_dicom.is_file():
                issues.append(
                    {
                        **base_issue,
                        "issue_type": "missing_source_dicom",
                        "error": f"expected source DICOM does not exist: {source_dicom}",
                    }
                )
                continue

            seen_ids[image_id] = path
            records.append(
                {
                    "dataset": "VinDr-CXR",
                    "image_id": image_id,
                    "split": split,
                    "image_path": str(path.resolve()),
                    "source_dicom_path": str(source_dicom.resolve()),
                    "relative_image_path": relative_path.as_posix(),
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "format": "PNG",
                }
            )
            written_counts[split] += 1
        if stopped:
            break

    records.sort(key=lambda row: (str(row["split"]), str(row["image_id"])))
    issues.sort(
        key=lambda row: (
            str(row.get("split", "")),
            str(row.get("image_id", "")),
            str(row.get("issue_type", "")),
        )
    )
    output_root_path.mkdir(parents=True, exist_ok=True)
    images_path = output_root_path / "images.jsonl"
    issues_path = output_root_path / "issues.jsonl"
    _write_jsonl(images_path, records)
    _write_jsonl(issues_path, issues)

    full_run = max_images is None
    count_mismatches = {
        split: {
            "expected": expected,
            "found": int(written_counts.get(split, 0)),
        }
        for split, expected in EXPECTED_SPLITS.items()
        if int(written_counts.get(split, 0)) != expected
    }
    manifest = {
        "dataset": "VinDr-CXR",
        "schema_id": "vindr_cxr_image_registry_v1",
        "image_root": str(image_root_path),
        "raw_dicom_root": str(raw_root),
        "output_root": str(output_root_path),
        "path_mode": "source_reference",
        "materialized_images_in_work": False,
        "source_counts": dict(sorted(source_counts.items())),
        "written_counts": dict(sorted(written_counts.items())),
        "records_written": len(records),
        "issue_count": len(issues),
        "expected_full_counts": EXPECTED_SPLITS,
        "count_mismatches": count_mismatches,
        "max_images": max_images,
        "require_complete": require_complete,
        "image_id_sha256": stable_json_sha256(
            [str(row["image_id"]) for row in records]
        ),
        "outputs": {
            "images_jsonl": str(images_path),
            "issues_jsonl": str(issues_path),
            "manifest_json": str(output_root_path / "manifest.json"),
        },
    }
    write_json(output_root_path / "manifest.json", manifest)

    if not records:
        raise ValueError(
            "No VinDr-CXR PNG rows were indexed. Check config.paths.vindr_cxr_png_root "
            "and run the dataset-side DICOM-to-PNG converter first."
        )
    if issues:
        raise ValueError(
            f"VinDr-CXR PNG indexing found {len(issues):,} issue(s); "
            f"inspect {issues_path}"
        )
    if full_run and require_complete and count_mismatches:
        raise ValueError(
            "VinDr-CXR PNG counts do not match the full 15,000 train / 3,000 test release; "
            f"inspect {output_root_path / 'manifest.json'} or use --allow-incomplete for a smoke subset"
        )

    return {"ok": True, **manifest}


__all__ = ["EXPECTED_SPLITS", "prepare_vindr_cxr"]
