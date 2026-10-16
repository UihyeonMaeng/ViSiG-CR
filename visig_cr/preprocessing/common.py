"""Shared utilities for ViSiG-CR dataset preprocessing.

The source ReXGradient/ReXVQA JSON files are very large top-level JSON
objects and may contain non-standard ``NaN`` values.  This module therefore
uses Python's tolerant ``json.JSONDecoder`` in an incremental parser instead
of loading the complete file into memory or requiring ijson.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


JSON_CHUNK_SIZE = 1 << 20
LFS_POINTER_PREFIX = "version https://git-lfs.github.com/spec/v1"


def is_missing(value: Any) -> bool:
    """Return True for null/NaN/empty metadata values."""

    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, str):
        return value.strip().lower() in {"", "nan", "none", "null", "n/a", "na"}
    return False


def clean_text(value: Any) -> str:
    """Normalize a report/metadata scalar to a safe string."""

    if is_missing(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite numbers and unsupported values."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def as_list(value: Any) -> list[Any]:
    """Convert JSON/CSV-style array values to a Python list."""

    if is_missing(value):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[:1] in {"[", "("}:
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(text)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
            with contextlib.suppress(ValueError, SyntaxError):
                parsed = ast.literal_eval(text)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
        return [value]
    return [value]


def value_at(values: Sequence[Any], index: int, default: Any = None) -> Any:
    return values[index] if index < len(values) else default


def canonical_split(value: str) -> str:
    normalized = clean_text(value).lower()
    aliases = {
        "train": "train",
        "training": "train",
        "valid": "validate",
        "validation": "validate",
        "validate": "validate",
        "val": "validate",
        "test": "test",
        "public_test": "test",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported split name: {value!r}")
    return aliases[normalized]


def normalize_view(value: Any) -> str:
    """Normalize DICOM/radiography view-position strings.

    The function intentionally does not infer a view from study description or
    image shape.  Unknown is safer than incorrectly labelling a lateral image as
    a frontal image.
    """

    text = clean_text(value).upper()
    if not text:
        return "UNKNOWN"
    compact = re.sub(r"[^A-Z0-9]+", " ", text).strip()
    joined = compact.replace(" ", "")

    if "DECUB" in joined:
        return "DECUBITUS"
    if "OBLIQ" in joined:
        return "OBLIQUE"
    if "LORDOT" in joined:
        return "LORDOTIC"

    pa_tokens = {
        "PA",
        "POSTEROANTERIOR",
        "POSTERIORANTERIOR",
        "POSTEROANT",
        "CHESTPA",
        "PAERECT",
        "PAUPRIGHT",
    }
    ap_tokens = {
        "AP",
        "ANTEROPOSTERIOR",
        "ANTERIORPOSTERIOR",
        "ANTEROPOST",
        "CHESTAP",
        "APERECT",
        "APUPRIGHT",
        "APSUPINE",
        "PORTABLEAP",
        "APPORTABLE",
    }
    lateral_tokens = {
        "LATERAL",
        "LAT",
        "LEFTLATERAL",
        "RIGHTLATERAL",
        "LL",
        "RL",
    }

    if joined in pa_tokens or re.search(r"(?:^| )PA(?: |$)", compact):
        return "PA"
    if joined in ap_tokens or re.search(r"(?:^| )AP(?: |$)", compact):
        return "AP"
    if joined in lateral_tokens or "LATERAL" in joined:
        return "LATERAL"
    return compact.replace(" ", "_") or "UNKNOWN"


def safe_filename(value: str, *, max_length: int = 180) -> str:
    """Create a deterministic filesystem-safe name while retaining readability."""

    raw = clean_text(value)
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "record"
    if sanitized == raw and len(sanitized) <= max_length:
        return sanitized
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, max_length - len(digest) - 1)
    return f"{sanitized[:prefix_length]}_{digest}"


def extract_sop_uid(image_ref: Any) -> str:
    text = clean_text(image_ref).replace("\\", "/")
    return Path(text).stem if text else ""


def _component_after_marker(path: str | Path, marker: str) -> str:
    parts = Path(str(path).replace("\\", "/")).parts
    with contextlib.suppress(ValueError, IndexError):
        index = parts.index(marker)
        return parts[index + 1]
    return ""


def extract_series_uid(path: str | Path) -> str:
    return _component_after_marker(path, "series")


def extract_study_uid_from_path(path: str | Path) -> str:
    return _component_after_marker(path, "studies")


def assert_real_payload(path: str | Path) -> Path:
    """Reject missing files and unresolved Git-LFS pointer files."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required source file does not exist: {target}")
    with target.open("r", encoding="utf-8", errors="replace") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX) + 16)
    if prefix.startswith(LFS_POINTER_PREFIX):
        raise ValueError(f"Git-LFS pointer detected; actual payload is missing: {target}")
    return target


class IncrementalJSONReader:
    """Small incremental JSON decoder tolerant of Python-style NaN constants."""

    def __init__(self, handle: TextIO, *, chunk_size: int = JSON_CHUNK_SIZE):
        self.handle = handle
        self.chunk_size = chunk_size
        self.buffer = ""
        self.position = 0
        self.eof = False
        # Python's default decoder accepts NaN/Infinity, unlike strict JSON
        # stream parsers.  The values are converted to null at output time.
        self.decoder = json.JSONDecoder()

    def _read_more(self) -> None:
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def _ensure(self) -> None:
        if self.position >= len(self.buffer) and not self.eof:
            self._read_more()

    def skip_whitespace(self) -> None:
        while True:
            self._ensure()
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._read_more()

    def peek(self) -> str:
        self.skip_whitespace()
        self._ensure()
        if self.position >= len(self.buffer):
            raise EOFError("Unexpected end of JSON stream")
        return self.buffer[self.position]

    def consume(self, expected: str) -> None:
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in JSON stream, found {actual!r}")
        self.position += 1

    def decode_next(self) -> Any:
        self.skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof:
                    context = self.buffer[max(0, self.position - 80) : self.position + 200]
                    raise ValueError(
                        f"Invalid/incomplete JSON near character {self.position}: {context!r}"
                    ) from exc
                self._read_more()
                continue
            self.position = end
            return value


def iter_json_records(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield records from a top-level JSON object or list without full loading."""

    target = assert_real_payload(path)
    with target.open("r", encoding="utf-8-sig") as handle:
        reader = IncrementalJSONReader(handle)
        opening = reader.peek()
        if opening == "{":
            reader.consume("{")
            if reader.peek() == "}":
                reader.consume("}")
                return
            while True:
                key = reader.decode_next()
                if not isinstance(key, str):
                    raise ValueError(f"Expected string key in top-level object: {target}")
                reader.consume(":")
                value = reader.decode_next()
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object value for key {key!r} in {target}")
                yield key, value
                delimiter = reader.peek()
                if delimiter == ",":
                    reader.consume(",")
                    continue
                if delimiter == "}":
                    reader.consume("}")
                    break
                raise ValueError(f"Unexpected delimiter {delimiter!r} in {target}")
        elif opening == "[":
            reader.consume("[")
            if reader.peek() == "]":
                reader.consume("]")
                return
            index = 0
            while True:
                value = reader.decode_next()
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object at list index {index} in {target}")
                key = clean_text(
                    value.get("query_id")
                    or value.get("study_id")
                    or value.get("case_id")
                    or index
                )
                yield key, value
                index += 1
                delimiter = reader.peek()
                if delimiter == ",":
                    reader.consume(",")
                    continue
                if delimiter == "]":
                    reader.consume("]")
                    break
                raise ValueError(f"Unexpected delimiter {delimiter!r} in {target}")
        else:
            raise ValueError(f"Expected top-level object or list in {target}, found {opening!r}")


@dataclass
class JSONLWriter:
    path: Path
    handle: TextIO | None = None

    def __enter__(self) -> "JSONLWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        return self

    def write(self, record: Mapping[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JSONLWriter is not open")
        self.handle.write(
            json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temp.replace(target)
    return target


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield value


def reset_output_dir(path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {target}. Pass --overwrite to rebuild it."
            )
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def ensure_symlink(link: str | Path, target: str | Path) -> Path:
    link_path = Path(link)
    target_path = Path(target).absolute()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return link_path
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Cannot create symlink because a path already exists: {link_path}")
    link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())
    return link_path


def copy_or_link_file(link: str | Path, target: str | Path) -> Path:
    link_path = Path(link)
    target_path = Path(target).absolute()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target_path, target_is_directory=False)
    return link_path


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def count_jsonl(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def path_exists(path: Any) -> bool:
    text = clean_text(path)
    return bool(text) and Path(text).is_file()


def relative_or_none(path: Path, root: Path) -> Path | None:
    with contextlib.suppress(ValueError):
        return path.absolute().relative_to(root.absolute())
    return None


def remove_option_prefix(text: str) -> str:
    return re.sub(r"^\s*[A-Z]\s*[\.)\]:-]\s*", "", clean_text(text), count=1)
