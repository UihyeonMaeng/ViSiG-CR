"""Experiment-protocol path resolution and artifact identity helpers.

The raw MIMIC preprocessing output is immutable.  A protocol selects a
patient-disjoint memory pool and query pool, then materializes size-specific
visual/graph artifacts beneath ``paths.protocol_root``.  This prevents a 500-
case graph from silently reusing rows or embeddings from a 10,000-case run.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io_utils import file_sha256, stable_json_sha256
from .version import RELATION_SCHEMA_VERSION


def memory_counts(cfg: dict[str, Any]) -> list[int]:
    raw = cfg.get("dataset_protocol", {}).get("memory", {}).get(
        "nested_case_counts", []
    )
    counts = sorted({int(value) for value in raw if int(value) > 0})
    if not counts:
        raise ValueError(
            "dataset_protocol.memory.nested_case_counts must contain at least "
            "one positive integer"
        )
    return counts


def active_memory_size(cfg: dict[str, Any], requested: int | None = None) -> int:
    counts = memory_counts(cfg)
    if requested is None:
        requested = int(
            cfg.get("dataset_protocol", {})
            .get("memory", {})
            .get("active_case_count", counts[-1])
        )
    requested = int(requested)
    if requested not in counts:
        raise ValueError(
            f"memory size {requested} is not pre-registered; expected one of {counts}"
        )
    return requested


def protocol_id(cfg: dict[str, Any], requested: str | None = None) -> str:
    value = requested or cfg.get("dataset_protocol", {}).get("protocol_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("A non-empty dataset protocol ID is required")
    if any(part in value for part in ("/", "\\", "..")):
        raise ValueError(f"Unsafe protocol ID: {value!r}")
    return value


def protocol_layout(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None = None,
    requested_memory_size: int | None = None,
) -> dict[str, Path | int | str]:
    pid = protocol_id(cfg, requested_protocol)
    size = active_memory_size(cfg, requested_memory_size)
    maximum = memory_counts(cfg)[-1]
    root = Path(cfg["paths"]["protocol_root"]) / pid
    pool = root / f"memory_pool_{maximum}"
    memory = root / f"memory_{size}"
    return {
        "protocol_id": pid,
        "memory_size": size,
        "maximum_memory_size": maximum,
        "root": root,
        "memory_pool": pool,
        "memory": memory,
        "memory_pool_cases_jsonl": pool / "cases.jsonl",
        "memory_cases_jsonl": memory / "cases.jsonl",
        "query_cases_jsonl": root / "query" / "cases.jsonl",
        "role_assignments_jsonl": root / "role_assignments.jsonl",
        "split_manifest_json": root / "split_manifest.json",
        "events_jsonl": pool / "ontology" / "events.jsonl",
        "extraction_status_jsonl": pool / "ontology" / "extraction_status.jsonl",
        "visual_dir": memory / "visual",
        "visual_manifest_json": memory / "visual" / "manifest.json",
        "graph_db": memory / "graph" / "visual_case_graph.sqlite",
        "results_dir": Path(
            cfg.get("paths", {}).get("results_dir", root / "results")
        ) / pid / f"memory_{size}",
    }


def apply_protocol_context(
    cfg: dict[str, Any],
    *,
    requested_protocol: str | None,
    requested_memory_size: int | None,
    command: str,
) -> dict[str, Any]:
    """Return a config copy with protocol-specific runtime paths.

    The context is applied when ``requested_protocol`` or
    ``requested_memory_size`` is provided, or when
    ``dataset_protocol.activate_by_default`` is true.  ``extract-ontology``
    operates on the maximum nested memory pool once; visual/graph/retrieval
    commands operate on the selected size-specific prefix.
    """

    activate = (
        requested_protocol is not None
        or requested_memory_size is not None
        or bool(cfg.get("dataset_protocol", {}).get("activate_by_default", False))
    )
    if not activate:
        return cfg

    resolved = deepcopy(cfg)
    layout = protocol_layout(
        resolved,
        requested_protocol=requested_protocol,
        requested_memory_size=requested_memory_size,
    )
    paths = resolved["paths"]
    paths.setdefault("master_cases_jsonl", paths.get("cases_jsonl", ""))
    paths["events_jsonl"] = str(layout["events_jsonl"])
    paths["extraction_status_jsonl"] = str(layout["extraction_status_jsonl"])
    paths["visual_dir"] = str(layout["visual_dir"])
    paths["graph_db"] = str(layout["graph_db"])
    paths["results_dir"] = str(layout["results_dir"])

    if command == "extract-ontology":
        paths["cases_jsonl"] = str(layout["memory_pool_cases_jsonl"])
    else:
        paths["cases_jsonl"] = str(layout["memory_cases_jsonl"])

    resolved["_protocol_context"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in layout.items()
    }
    return resolved


def artifact_identity(
    cfg: dict[str, Any],
    *,
    cases_jsonl: str | Path | None = None,
    events_jsonl: str | Path | None = None,
) -> dict[str, str]:
    """Build a compact identity map for graph/visual compatibility checks."""

    cases_path = Path(cases_jsonl or cfg["paths"]["cases_jsonl"])
    events_path = Path(events_jsonl or cfg["paths"]["events_jsonl"])
    context = cfg.get("_protocol_context", {})
    identity: dict[str, str] = {
        "project_version": str(cfg.get("project", {}).get("version", "")),
        "protocol_id": str(context.get("protocol_id", "legacy")),
        "memory_size": str(context.get("memory_size", "legacy")),
        "cases_jsonl": str(cases_path),
        "case_manifest_sha256": file_sha256(cases_path) if cases_path.is_file() else "missing",
        "events_jsonl": str(events_path),
        "events_sha256": file_sha256(events_path) if events_path.is_file() else "missing",
        "visual_index_config_sha256": stable_json_sha256(cfg.get("visual_index", {})),
        "visual_graph_config_sha256": stable_json_sha256(cfg.get("visual_graph", {})),
        "relation_graph_config_sha256": stable_json_sha256(cfg.get("relation_graph", {})),
        "normality_config_sha256": stable_json_sha256(cfg.get("ontology_extraction", {}).get("case_normality", {})),
        "relation_schema_version": RELATION_SCHEMA_VERSION,
    }
    ontology_path = Path(cfg["project"]["ontology_path"])
    identity["ontology_sha256"] = (
        file_sha256(ontology_path) if ontology_path.is_file() else "missing"
    )
    return identity


__all__ = [
    "active_memory_size",
    "apply_protocol_context",
    "artifact_identity",
    "memory_counts",
    "protocol_id",
    "protocol_layout",
]
