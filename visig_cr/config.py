"""Configuration loading and path resolution."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

from .io_utils import load_yaml


ONTOLOGY_BACKEND_PROFILES: dict[str, dict[str, str]] = {
    "qwen-vl8": {
        "model_family": "qwen",
        "checkpoint_id": "Qwen/Qwen3-VL-8B-Instruct",
        "runtime_mode": "text_only",
    },
    "qwen-vl30": {
        "model_family": "qwen",
        "checkpoint_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "runtime_mode": "text_only",
    },
    "llama-4bit": {
        "model_family": "llama",
        "checkpoint_id": "meta-llama/Llama-3.3-70B-Instruct",
        "runtime_mode": "bitsandbytes_4bit",
    },
    "llama-tp2": {
        "model_family": "llama",
        "checkpoint_id": "meta-llama/Llama-3.3-70B-Instruct",
        "runtime_mode": "bf16_tp2",
    },
}

_ONTOLOGY_BACKEND_ALIASES = {
    "qwen_vl8": "qwen-vl8",
    "qwen8": "qwen-vl8",
    "qwen_vl30": "qwen-vl30",
    "qwen30": "qwen-vl30",
    "llama_4bit": "llama-4bit",
    "llama": "llama-4bit",
    "llama_tp2": "llama-tp2",
}


def ontology_backend_choices() -> tuple[str, ...]:
    """Return the pre-registered ontology backends accepted by the main CLI."""

    return tuple(ONTOLOGY_BACKEND_PROFILES)


def resolve_ontology_backend(
    llm_cfg: dict[str, Any],
    requested_backend: str | None = None,
) -> dict[str, Any]:
    """Attach a frozen backend identity to an ontology client configuration.

    Every ontology server intentionally exposes the same OpenAI API model name
    (``ontology-extractor``).  The API name therefore cannot identify the
    checkpoint used for a long-running extraction.  This helper makes the
    server-side selection explicit in status/manifests and prevents Qwen-only
    chat-template arguments from being sent to a Llama backend.
    """

    configured = requested_backend or llm_cfg.get("backend", "qwen-vl8")
    token = str(configured).strip().lower().replace(" ", "-")
    backend = _ONTOLOGY_BACKEND_ALIASES.get(token, token)
    if backend not in ONTOLOGY_BACKEND_PROFILES:
        allowed = ", ".join(ontology_backend_choices())
        raise ValueError(
            f"Unknown ontology backend {configured!r}. Use one of: {allowed}"
        )

    profile = ONTOLOGY_BACKEND_PROFILES[backend]
    resolved = deepcopy(llm_cfg)
    resolved.update(
        {
            "backend": backend,
            "backend_id": backend,
            "model_family": profile["model_family"],
            "checkpoint_id": profile["checkpoint_id"],
            "runtime_mode": profile["runtime_mode"],
        }
    )
    if profile["model_family"] != "qwen":
        # ``enable_thinking`` is a Qwen chat-template argument.  Sending it to
        # Llama is unnecessary and can fail on strict server templates.
        resolved["disable_qwen_thinking"] = False
    return resolved


def resolve_path(value: str | Path, base: Path) -> str:
    value = str(value)
    if value.startswith(("http://", "https://")):
        return value
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)

    cfg["project"]["ontology_path"] = resolve_path(
        cfg["project"]["ontology_path"], config_path.parent
    )
    for key, value in list(cfg.get("paths", {}).items()):
        if isinstance(value, str):
            cfg["paths"][key] = resolve_path(value, config_path.parent)
    model_path = cfg.get("models", {}).get("medsiglip_path")
    if model_path:
        cfg["models"]["medsiglip_path"] = resolve_path(model_path, config_path.parent)
    return cfg
