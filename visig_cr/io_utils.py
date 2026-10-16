"""Small file/serialization helpers shared across ViSiG-CR modules."""

from __future__ import annotations

import contextlib
import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    return value or {}


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def open_text(path: str | Path, mode: str = "rt"):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode, encoding="utf-8")
    return p.open(mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path, "rt") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            yield value


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = ensure_parent(path)
    with open_text(p, "at") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: Any) -> None:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a plain, fenced, or embedded JSON object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    with contextlib.suppress(json.JSONDecodeError):
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object")
    return obj


def l2_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(denom, eps, None)


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a local file.

    The helper is intentionally streaming so it is safe for large manifests and
    model-visible image files.  Callers that operate on optional files should
    check existence before invoking it.
    """

    import hashlib

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    """Hash a JSON-serializable value with deterministic key ordering."""

    import hashlib

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
