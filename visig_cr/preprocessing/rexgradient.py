"""Prepare ReXGradient-160K for the ViSiG-CR study-level pipeline.

The compact layout intentionally keeps only five dataset artifacts:

* ``cases.jsonl``: runtime case manifest plus nested multi-view metadata.
* ``reports/``: sectioned reports referenced by ``cases.jsonl``.
* ``images/``: selected 8-bit runtime images, or a source-image symlink.
* ``issues.jsonl``: missing-image and skipped-study records.
* ``manifest.json``: provenance, preprocessing settings, and statistics.

The source dataset under ``/workspace/Datasets`` is never modified.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..image_utils import image_normalization_provenance, load_uint8_image
from .common import (
    JSONLWriter,
    as_list,
    assert_real_payload,
    canonical_split,
    clean_text,
    ensure_symlink,
    extract_series_uid,
    extract_sop_uid,
    extract_study_uid_from_path,
    iter_json_records,
    json_safe,
    normalize_view,
    relative_or_none,
    reset_output_dir,
    safe_filename,
    value_at,
    write_json,
)


SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = ("train", "validate", "test")


def _field(record: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    lowered = {str(key).lower(): key for key in record}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return record[key]
    return default


@dataclass(frozen=True)
class ViewInfo:
    by_sop: dict[str, str]
    by_index: tuple[str, ...]


class ViewLookup:
    """Compact lookup built from ``*_metadata_view_position.json``."""

    def __init__(self) -> None:
        self.by_study_key: dict[str, ViewInfo] = {}
        self.by_study_uid: dict[str, ViewInfo] = {}
        self.record_count = 0
        self.nonempty_view_count = 0

    @classmethod
    def from_file(cls, path: Path, *, limit: int | None = None) -> "ViewLookup":
        lookup = cls()
        for index, (study_key, record) in enumerate(iter_json_records(path)):
            if limit is not None and index >= limit:
                break
            image_paths = as_list(_field(record, "ImagePath", "image_paths", default=[]))
            raw_views = as_list(
                _field(
                    record,
                    "ImageViewPosition",
                    "ViewPosition",
                    "view_positions",
                    default=[],
                )
            )
            normalized = tuple(normalize_view(item) for item in raw_views)
            by_sop: dict[str, str] = {}
            for image_index, image_ref in enumerate(image_paths):
                view = value_at(normalized, image_index, "UNKNOWN")
                sop_uid = extract_sop_uid(image_ref)
                if sop_uid and view != "UNKNOWN":
                    by_sop[sop_uid] = view

            if not by_sop and all(view == "UNKNOWN" for view in normalized):
                continue

            info = ViewInfo(by_sop=by_sop, by_index=normalized)
            lookup.by_study_key[study_key] = info
            study_uid = clean_text(
                _field(record, "StudyInstanceUid", "StudyInstanceUID", "study_instance_uid")
            )
            if study_uid:
                lookup.by_study_uid[study_uid] = info
            lookup.record_count += 1
            lookup.nonempty_view_count += sum(view != "UNKNOWN" for view in normalized)
        return lookup

    def get(
        self,
        *,
        study_key: str,
        study_uid: str,
        sop_uid: str,
        image_index: int,
    ) -> str:
        info = self.by_study_key.get(study_key) or self.by_study_uid.get(study_uid)
        if info is None:
            return "UNKNOWN"
        by_sop = info.by_sop.get(sop_uid)
        if by_sop:
            return by_sop
        return value_at(info.by_index, image_index, "UNKNOWN")


class ImageResolver:
    """Resolve ReXGradient image references and expose a stable runtime root."""

    def __init__(
        self,
        source_dataset_root: Path,
        output_dataset_root: Path,
        *,
        path_mode: str = "symlink",
        scan_fallback: bool = True,
    ) -> None:
        if path_mode not in {"symlink", "source"}:
            raise ValueError("path_mode must be 'symlink' or 'source'")
        self.source_dataset_root = source_dataset_root.absolute()
        self.source_image_root = (
            self.source_dataset_root / "extracted_images" / "deid_png"
        ).absolute()
        self.output_dataset_root = output_dataset_root.absolute()
        self.output_image_root = self.output_dataset_root / "images"
        self.path_mode = path_mode
        self.scan_fallback = scan_fallback
        self._sop_index: dict[str, list[Path]] | None = None
        if not self.source_image_root.is_dir():
            raise FileNotFoundError(
                f"ReXGradient PNG root does not exist: {self.source_image_root}"
            )

    def prepare_runtime_root(self, *, materialize_8bit: bool) -> None:
        if materialize_8bit:
            self.output_image_root.mkdir(parents=True, exist_ok=True)
        elif self.path_mode == "symlink":
            ensure_symlink(self.output_image_root, self.source_image_root)

    def _candidate_paths(self, image_ref: Any) -> list[Path]:
        text = clean_text(image_ref).replace("\\", "/")
        if not text:
            return []
        candidates: list[Path] = []
        raw = Path(text)
        if raw.is_absolute():
            candidates.append(raw)

        marker = "deid_png/"
        marker_index = text.find(marker)
        if marker_index >= 0:
            suffix = text[marker_index + len(marker) :].lstrip("/")
            candidates.append(self.source_image_root / suffix)

        candidates.extend(
            [
                self.source_dataset_root / text,
                self.source_dataset_root / "extracted_images" / text,
                self.source_image_root / text.lstrip("./"),
            ]
        )

        deduplicated: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = Path(os.path.normpath(str(candidate)))
            key = str(normalized)
            if key not in seen:
                deduplicated.append(normalized)
                seen.add(key)
        return deduplicated

    def _build_sop_index(self) -> None:
        index: dict[str, list[Path]] = defaultdict(list)
        for dirpath, _, filenames in os.walk(self.source_image_root):
            base = Path(dirpath)
            for filename in filenames:
                if filename.lower().endswith(".png"):
                    index[Path(filename).stem].append(base / filename)
        self._sop_index = dict(index)

    def resolve(self, image_ref: Any, *, study_uid: str = "") -> Path | None:
        for candidate in self._candidate_paths(image_ref):
            if candidate.is_file():
                return candidate.absolute()

        if not self.scan_fallback:
            return None
        if self._sop_index is None:
            self._build_sop_index()
        assert self._sop_index is not None

        sop_uid = extract_sop_uid(image_ref)
        candidates = self._sop_index.get(sop_uid, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0].absolute()
        if study_uid:
            for candidate in candidates:
                if extract_study_uid_from_path(candidate) == study_uid:
                    return candidate.absolute()
        return sorted(candidates, key=str)[0].absolute()

    def relative_path(self, source_path: Path) -> Path | None:
        return relative_or_none(source_path, self.source_image_root)

    def linked_or_source_path(self, source_path: Path) -> Path:
        if self.path_mode == "source":
            return source_path.absolute()
        relative = self.relative_path(source_path)
        if relative is None:
            return source_path.absolute()
        candidate = (self.output_image_root / relative).absolute()
        # In materialized mode only selected images exist under images/.
        # Fall back to the source path for any non-materialized image.
        if self.output_image_root.is_symlink() or candidate.is_file():
            return candidate
        return source_path.absolute()

    def runtime_path(self, source_path: Path, *, materialize_8bit: bool) -> Path:
        if not materialize_8bit:
            return self.linked_or_source_path(source_path)
        relative = self.relative_path(source_path)
        if relative is None:
            relative = Path("unmatched") / source_path.name
        return _materialize_8bit(source_path, self.output_image_root / relative)


def _materialize_8bit(source_path: Path, destination: Path) -> Path:
    """Write a fixed-range 8-bit proxy without inversion or min-max stretching."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination.absolute()

    temp = destination.with_suffix(destination.suffix + ".tmp")
    with load_uint8_image(source_path) as output:
        output.save(temp, format="PNG", optimize=True)
    temp.replace(destination)
    return destination.absolute()


def _build_report(record: dict[str, Any]) -> tuple[str, bool]:
    indication = clean_text(_field(record, "Indication", "History"))
    comparison = clean_text(_field(record, "Comparison"))
    findings = clean_text(_field(record, "Findings"))
    impression = clean_text(_field(record, "Impression"))
    report = (
        f"INDICATION:\n{indication}\n\n"
        f"COMPARISON:\n{comparison}\n\n"
        f"FINDINGS:\n{findings}\n\n"
        f"IMPRESSION:\n{impression}\n"
    )
    return report, bool(findings or impression)


def _select_representative(
    image_rows: list[dict[str, Any]],
    *,
    view_priority: tuple[str, ...],
    unknown_view_policy: str,
) -> dict[str, Any] | None:
    existing = [row for row in image_rows if row["exists"]]
    if not existing:
        return None

    rank = {view: index for index, view in enumerate(view_priority)}
    frontal = [row for row in existing if row["view"] in rank]
    if frontal:
        return sorted(
            frontal,
            key=lambda row: (
                rank[row["view"]],
                clean_text(row.get("sop_instance_uid")),
                clean_text(row.get("source_path")),
            ),
        )[0]

    unknown = [row for row in existing if row["view"] == "UNKNOWN"]
    if unknown_view_policy == "single" and len(existing) == 1 and len(unknown) == 1:
        return unknown[0]
    if unknown_view_policy == "first" and unknown:
        return sorted(
            unknown,
            key=lambda row: clean_text(row.get("sop_instance_uid")),
        )[0]
    return None


def _metadata_path(metadata_root: Path, source_split: str, suffix: str) -> Path:
    return metadata_root / f"{source_split}_metadata{suffix}"


def _patient_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "patient_id": clean_text(_field(record, "PatientID", "patient_id")),
        "sex": clean_text(_field(record, "PatientSex", "patient_sex")),
        "age": clean_text(_field(record, "PatientAge", "patient_age")),
        "ethnic_group": json_safe(_field(record, "EthnicGroup", "ethnic_group")),
        "weight": json_safe(_field(record, "PatientWeight", "patient_weight")),
        "size": json_safe(_field(record, "PatientSize", "patient_size")),
    }


def _study_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "accession_number": clean_text(
            _field(record, "AccessionNumber", "accession_number")
        ),
        "study_instance_uid": clean_text(
            _field(record, "StudyInstanceUid", "StudyInstanceUID", "study_instance_uid")
        ),
        "institution_name": clean_text(
            _field(record, "InstitutionName", "institution_name")
        ),
        "manufacturer": json_safe(_field(record, "Manufacturer", "manufacturer")),
        "study_date": json_safe(_field(record, "StudyDate", "study_date")),
        "study_description": clean_text(
            _field(record, "StudyDescription", "study_description")
        ),
    }


def prepare_rexgradient(
    *,
    source_root: str | Path = "/workspace/Datasets/06.CXR/ReXGradient-160K",
    output_root: str | Path = "/workspace/visig_cr/work/datasets/rex/ReXGradient-160K",
    overwrite: bool = False,
    path_mode: str = "symlink",
    view_priority: tuple[str, ...] = ("PA", "AP"),
    unknown_view_policy: str = "skip",
    scan_fallback: bool = True,
    materialize_8bit: bool = False,
    limit_per_split: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Create the compact ReXGradient runtime derivative for ViSiG-CR."""

    if unknown_view_policy not in {"skip", "single", "first"}:
        raise ValueError("unknown_view_policy must be skip, single, or first")

    source_dataset_root = Path(source_root).absolute()
    metadata_root = source_dataset_root / "metadata"
    output_dataset_root = reset_output_dir(output_root, overwrite=overwrite)
    if not metadata_root.is_dir():
        raise FileNotFoundError(f"Metadata directory does not exist: {metadata_root}")

    resolver = ImageResolver(
        source_dataset_root,
        output_dataset_root,
        path_mode=path_mode,
        scan_fallback=scan_fallback,
    )
    resolver.prepare_runtime_root(materialize_8bit=materialize_8bit)
    (output_dataset_root / "reports").mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "dataset": "ReXGradient-160K",
        "schema_id": "rexgradient_case_manifest",
        "source": {
            "dataset_root": str(source_dataset_root),
            "metadata_root": str(metadata_root),
            "image_root": str(resolver.source_image_root),
            "interstitial_pattern_bbox": (
                str(metadata_root / "interstitial_pattern_bbox.json")
                if (metadata_root / "interstitial_pattern_bbox.json").is_file()
                else None
            ),
        },
        "output_root": str(output_dataset_root),
        "settings": {
            "path_mode": path_mode,
            "materialize_8bit": bool(materialize_8bit),
            "image_mode": "materialized_8bit"
            if materialize_8bit
            else ("symlink" if path_mode == "symlink" else "source"),
            "image_normalization": {
                **image_normalization_provenance(),
                "applied_during_preprocessing": bool(materialize_8bit),
                "runtime_loader_required": not bool(materialize_8bit),
            },
            "view_priority": list(view_priority),
            "unknown_view_policy": unknown_view_policy,
            "scan_fallback": scan_fallback,
            "limit_per_split": limit_per_split,
        },
        "splits": {},
    }

    patient_sets: dict[str, set[str]] = {split: set() for split in OUTPUT_SPLITS}
    all_case_ids: set[str] = set()
    cases_path = output_dataset_root / "cases.jsonl"
    issues_path = output_dataset_root / "issues.jsonl"

    with ExitStack() as stack:
        cases_writer = stack.enter_context(JSONLWriter(cases_path))
        issues_writer = stack.enter_context(JSONLWriter(issues_path))

        for source_split in SOURCE_SPLITS:
            output_split = canonical_split(source_split)
            metadata_path = assert_real_payload(
                _metadata_path(metadata_root, source_split, ".json")
            )
            view_path = _metadata_path(
                metadata_root, source_split, "_view_position.json"
            )
            view_lookup = (
                ViewLookup.from_file(
                    assert_real_payload(view_path), limit=limit_per_split
                )
                if view_path.is_file()
                else ViewLookup()
            )

            counters: Counter[str] = Counter()
            selected_views: Counter[str] = Counter()
            image_views: Counter[str] = Counter()
            view_sources: Counter[str] = Counter()
            skip_reasons: Counter[str] = Counter()

            for record_index, (study_key, record) in enumerate(
                iter_json_records(metadata_path)
            ):
                if limit_per_split is not None and record_index >= limit_per_split:
                    break

                counters["studies_seen"] += 1
                patient_meta = _patient_metadata(record)
                study_meta = _study_metadata(record)
                patient_id = clean_text(patient_meta["patient_id"])
                accession_number = clean_text(study_meta["accession_number"])
                study_uid = clean_text(study_meta["study_instance_uid"])
                if not study_key:
                    study_key = study_uid or f"{source_split}:{record_index:08d}"

                case_id = f"rexgradient:{study_key}"
                subject_id = f"rexgradient:{patient_id or 'unknown'}"
                patient_sets[output_split].add(patient_id or subject_id)

                image_refs = as_list(
                    _field(record, "ImagePath", "image_paths", default=[])
                )
                base_views = as_list(
                    _field(record, "ImageViewPosition", "ViewPosition", default=[])
                )
                modalities = as_list(_field(record, "ImageModality", default=[]))
                shapes = as_list(_field(record, "ImageShape", default=[]))
                body_parts = as_list(_field(record, "ImageBodyPart", default=[]))

                image_rows: list[dict[str, Any]] = []
                for image_index, image_ref in enumerate(image_refs):
                    sop_uid = extract_sop_uid(image_ref)
                    base_view = normalize_view(
                        value_at(base_views, image_index, "")
                    )
                    override_view = view_lookup.get(
                        study_key=study_key,
                        study_uid=study_uid,
                        sop_uid=sop_uid,
                        image_index=image_index,
                    )
                    if override_view != "UNKNOWN":
                        view = override_view
                        view_source = "metadata_view_position"
                    elif base_view != "UNKNOWN":
                        view = base_view
                        view_source = "metadata"
                    else:
                        view = "UNKNOWN"
                        view_source = "missing"

                    source_image = resolver.resolve(image_ref, study_uid=study_uid)
                    exists = bool(source_image and source_image.is_file())
                    linked_path = (
                        resolver.linked_or_source_path(source_image)
                        if source_image is not None and not materialize_8bit
                        else None
                    )
                    image_row = {
                        "image_index": image_index,
                        "source_ref": clean_text(image_ref),
                        "source_path": str(source_image) if source_image else "",
                        "runtime_path": str(linked_path) if linked_path else "",
                        "series_instance_uid": extract_series_uid(
                            source_image or clean_text(image_ref)
                        ),
                        "sop_instance_uid": sop_uid,
                        "view": view,
                        "view_source": view_source,
                        "modality": clean_text(
                            value_at(modalities, image_index, "")
                        ),
                        "body_part": clean_text(
                            value_at(body_parts, image_index, "")
                        ),
                        "shape": json_safe(value_at(shapes, image_index, [])),
                        "exists": exists,
                        "selected": False,
                    }
                    image_rows.append(image_row)
                    counters["images_seen"] += 1
                    image_views[view] += 1
                    view_sources[view_source] += 1

                    if not exists:
                        counters["missing_images"] += 1
                        issues_writer.write(
                            {
                                "dataset": "ReXGradient-160K",
                                "issue_type": "missing_image",
                                "split": output_split,
                                "study_id": study_key,
                                "case_id": case_id,
                                "study_instance_uid": study_uid,
                                "image": image_row,
                            }
                        )

                selected = _select_representative(
                    image_rows,
                    view_priority=tuple(normalize_view(item) for item in view_priority),
                    unknown_view_policy=unknown_view_policy,
                )
                if selected is not None:
                    selected["selected"] = True
                    selected_views[selected["view"]] += 1

                report_text, report_has_findings_or_impression = _build_report(record)
                reasons: list[str] = []
                if not image_refs:
                    reasons.append("no_image_metadata")
                if image_refs and not any(row["exists"] for row in image_rows):
                    reasons.append("all_images_missing")
                if selected is None:
                    reasons.append("no_eligible_frontal_image")
                if not report_has_findings_or_impression:
                    reasons.append("empty_findings_and_impression")
                if case_id in all_case_ids:
                    reasons.append("duplicate_case_id")

                if reasons:
                    counters["studies_skipped"] += 1
                    for reason in reasons:
                        skip_reasons[reason] += 1
                    issues_writer.write(
                        {
                            "dataset": "ReXGradient-160K",
                            "issue_type": "study_skipped",
                            "split": output_split,
                            "study_id": study_key,
                            "case_id": case_id,
                            "study_instance_uid": study_uid,
                            "reasons": reasons,
                            "image_count": len(image_rows),
                            "views": [row["view"] for row in image_rows],
                        }
                    )
                    continue

                assert selected is not None
                source_selected = Path(selected["source_path"])
                selected_runtime_path = resolver.runtime_path(
                    source_selected, materialize_8bit=materialize_8bit
                )
                selected["runtime_path"] = str(selected_runtime_path)

                report_path = (
                    output_dataset_root
                    / "reports"
                    / output_split
                    / f"{safe_filename(study_key)}.txt"
                )
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report_text, encoding="utf-8")

                case_record = {
                    # Exact fields consumed by visig_cr.schemas.CaseRecord.
                    "case_id": case_id,
                    "dataset": "ReXGradient-160K",
                    "subject_id": subject_id,
                    "study_id": study_key,
                    "dicom_id": selected["sop_instance_uid"],
                    "image_path": str(selected_runtime_path.absolute()),
                    "report_path": str(report_path.absolute()),
                    "view": selected["view"],
                    "split": output_split,
                    # Rich metadata retained in the same record.
                    "source_metadata_key": study_key,
                    "patient": patient_meta,
                    "study": study_meta,
                    "selected_image_index": selected["image_index"],
                    "images": image_rows,
                }
                cases_writer.write(case_record)
                all_case_ids.add(case_id)
                counters["cases_written"] += 1
                counters["reports_written"] += 1

            manifest["splits"][output_split] = {
                **dict(counters),
                "source_metadata": str(metadata_path),
                "source_view_metadata": str(view_path) if view_path.is_file() else None,
                "view_lookup_records": view_lookup.record_count,
                "view_lookup_nonempty_views": view_lookup.nonempty_view_count,
                "selected_views": dict(selected_views),
                "all_image_views": dict(image_views),
                "view_sources": dict(view_sources),
                "skip_reasons": dict(skip_reasons),
                "unique_patients_seen": len(patient_sets[output_split]),
            }

    manifest["patient_overlap"] = {
        "train_validate": len(patient_sets["train"] & patient_sets["validate"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "validate_test": len(patient_sets["validate"] & patient_sets["test"]),
        "all_three": len(
            patient_sets["train"]
            & patient_sets["validate"]
            & patient_sets["test"]
        ),
    }
    manifest["outputs"] = {
        "cases_jsonl": str(cases_path),
        "reports_root": str(output_dataset_root / "reports"),
        "images_root": (
            str(output_dataset_root / "images")
            if (output_dataset_root / "images").exists()
            or (output_dataset_root / "images").is_symlink()
            else None
        ),
        "issues_jsonl": str(issues_path),
        "manifest_json": str(output_dataset_root / "manifest.json"),
    }
    write_json(output_dataset_root / "manifest.json", manifest)

    total_cases = sum(
        int(manifest["splits"].get(split, {}).get("cases_written", 0))
        for split in OUTPUT_SPLITS
    )
    if strict and total_cases == 0:
        raise RuntimeError("No ReXGradient cases were generated")
    return manifest
