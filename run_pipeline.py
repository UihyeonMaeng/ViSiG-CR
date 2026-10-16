#!/usr/bin/env python3
"""Command-line runner for ViSiG-CR memory construction, retrieval, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from tqdm import tqdm

from visig_cr import (
    AttributeSelectorAgent,
    CaseEvents,
    CaseRecord,
    EvidencePackage,
    Event,
    GraphRetriever,
    GraphStore,
    MedSigLIPEncoder,
    Ontology,
    OntologyExtractorAgent,
    VLMReader,
    VisualStore,
    append_jsonl,
    apply_protocol_context,
    artifact_identity,
    build_no_rag_package,
    evaluate_ontology_events,
    evaluate_predictions,
    evaluate_retrieval_packages,
    evidence_package_sha256,
    flatten_event_records,
    load_config,
    ontology_extraction_response_format,
    open_text,
    prompt_manifest,
    read_json,
    read_jsonl,
    safe_query_audit_metadata,
    self_test,
    write_json,
)
from visig_cr.io_utils import file_sha256, stable_json_sha256
from visig_cr.image_utils import image_normalization_provenance
from visig_cr.version import RELATION_SCHEMA_VERSION
from visig_cr.normality import derive_normality_metadata
from visig_cr.mimic import prepare_manifest
from visig_cr.config import ontology_backend_choices, resolve_ontology_backend
from visig_cr.retrieval.question_conditioned import (
    MEMORY_IDENTITY_KEYS, POLICY_VERSION, uses_question_conditioned,
)
from visig_cr.presence_index import build_presence_index, configured_identity as presence_identity, index_path as presence_index_path


CORE_METHODS = [
    "no_rag",
    "image_rag",
    "visual_boundary_pointwise",
    "visual_boundary_semantic_contrast",
    "visual_boundary_stored_contrast",
    "local_value_partition",
    "visual_pool",
    "direct_index",
    "visual_pool_semantic_contrast",
    "direct_index_semantic_contrast",
    "visig_cr",
]


ATTRIBUTE_SELECTOR_MODEL_CHOICES = ["qwen-vl8", "qwen-vl30"]


def _apply_attribute_selector_model_override(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    requested = getattr(args, "attribute_selector_model", None)
    if not requested:
        return
    model_cfg = cfg.setdefault("models", {}).setdefault("attribute_selector_llm", {})
    model_cfg["backend"] = str(requested)
    model_cfg["model_family"] = "qwen"


def _attribute_selector_provenance(cfg: Mapping[str, Any]) -> dict[str, Any]:
    selector_cfg = dict(cfg.get("attribute_selector", {}))
    model_cfg = dict(cfg.get("models", {}).get("attribute_selector_llm", {}))
    return {
        "backend": str(selector_cfg.get("backend", "llm")),
        "checkpoint_profile": str(model_cfg.get("backend", "qwen-vl8")),
        "api_model": str(model_cfg.get("model", "attribute-selector")),
        "model_family": str(model_cfg.get("model_family", "qwen")),
        "prompt_sha256": prompt_manifest().get("attribute_selector_system", ""),
        "selector_config_sha256": stable_json_sha256(selector_cfg),
    }


def _cases(path: str | Path, *, splits: set[str] | None = None) -> list[CaseRecord]:
    rows = [CaseRecord.model_validate(row) for row in read_jsonl(path)]
    return [row for row in rows if splits is None or row.split in splits]


def _events(path: str | Path) -> list[Event]:
    return flatten_event_records(read_jsonl(path))


def _case_normalities(
    path: str | Path,
    *,
    ontology: Ontology | None = None,
    abnormal_findings: Any = "clinical",
) -> dict[str, str]:
    """Read case normality, deriving legacy rows when requested.

    The optional derivation keeps graph rebuilding compatible with an existing
    pre-normality event store. Use ``scripts/backfill_case_normality.py`` when
    the durable JSONL itself should explicitly contain the field.
    """

    result: dict[str, str] = {}
    for row in read_jsonl(path):
        if not isinstance(row, Mapping) or not isinstance(row.get("events"), Mapping):
            continue
        record = CaseEvents.model_validate(row)
        if ontology is not None:
            value, _ = derive_normality_metadata(record.to_events(), ontology, abnormal_findings)
            if record.normality is not None and record.normality != value:
                raise ValueError(f"Stale normality for {record.case_id}; rerun normality backfill, not extraction")
            result[record.case_id] = value
        elif record.normality is not None:
            result[record.case_id] = record.normality
    return result


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _reset_jsonl(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")


def _prepare_identity_manifest(
    manifest_path: str | Path,
    identity: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    resume: bool,
    label: str,
) -> str:
    """Write or verify an artifact identity before an append/resume run."""

    manifest = Path(manifest_path)
    checkpoint = Path(checkpoint_path)
    normalized = dict(identity)
    if not resume:
        write_json(manifest, normalized)
        return "written"
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        write_json(manifest, normalized)
        return "written_empty_checkpoint"
    if not manifest.is_file():
        raise ValueError(
            f"Cannot resume {label} without the v2.0 identity manifest: "
            f"{manifest}. Use a fresh output path without --resume."
        )

    existing = read_json(manifest)
    mismatches = {
        key: {"expected": normalized.get(key), "actual": existing.get(key)}
        for key in normalized
        if existing.get(key) != normalized.get(key)
    }
    if mismatches:
        raise ValueError(
            f"{label.capitalize()} resume identity mismatch. Do not mix "
            "checkpoints, prompt/schema versions, extraction settings, or "
            "case pools in one JSONL file: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return "verified"


def _unique_ids(path: str | Path, *, label: str) -> set[str]:
    target = Path(path)
    if not target.is_file():
        return set()
    result: set[str] = set()
    duplicates: list[str] = []
    for row in read_jsonl(target):
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"{label} contains an empty query_id: {target}")
        if query_id in result:
            duplicates.append(query_id)
        result.add(query_id)
    if duplicates:
        raise ValueError(f"{label} contains duplicate query IDs: {sorted(set(duplicates))[:20]}")
    return result


def _unique_case_ids(path: str | Path, *, label: str) -> set[str]:
    """Return unique ``case_id`` values from a JSONL checkpoint file."""

    target = Path(path)
    if not target.is_file():
        return set()
    result: set[str] = set()
    duplicates: list[str] = []
    for row in read_jsonl(target):
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            raise ValueError(f"{label} contains an empty case_id: {target}")
        if case_id in result:
            duplicates.append(case_id)
        result.add(case_id)
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate case IDs: {sorted(set(duplicates))[:20]}"
        )
    return result


def _encoder(cfg: dict[str, Any]) -> MedSigLIPEncoder:
    return MedSigLIPEncoder(
        cfg["models"]["medsiglip_path"],
        device=cfg["visual_index"].get("device", "cuda"),
        dtype=cfg["visual_index"].get("dtype", "bfloat16"),
    )


def _validate_visual_manifest(cfg: dict[str, Any], expected: dict[str, str]) -> None:
    manifest_path = Path(cfg["paths"]["visual_dir"]) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Protocol visual manifest is missing: {manifest_path}. Rebuild the selected memory size."
        )
    manifest = read_json(manifest_path)
    mismatches = {
        key: {"expected": expected.get(key), "actual": manifest.get(key)}
        for key in (
            "protocol_id",
            "memory_size",
            "case_manifest_sha256",
            "visual_index_config_sha256",
        )
        if manifest.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            "Visual artifact does not match the active protocol: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def _retrieval_stack(cfg: dict[str, Any], method: str = "visig_cr"):
    ontology = Ontology(cfg["project"]["ontology_path"])
    graph = GraphStore(cfg["paths"]["graph_db"])
    try:
        is_qcr = uses_question_conditioned(cfg.get("retrieval", {}), method)
        if not is_qcr:
            graph.assert_artifact_metadata(
                {"relation_schema_version": RELATION_SCHEMA_VERSION, "relation_build_complete": True},
                keys=("relation_schema_version", "relation_build_complete"),
            )
        if cfg.get("_protocol_context"):
            expected = artifact_identity(cfg)
            graph.assert_artifact_metadata(
                expected,
                keys=MEMORY_IDENTITY_KEYS if is_qcr else (
                    "protocol_id",
                    "memory_size",
                    "case_manifest_sha256",
                    "events_sha256",
                    "ontology_sha256",
                    "visual_index_config_sha256",
                    "visual_graph_config_sha256",
                    "relation_graph_config_sha256",
                    "normality_config_sha256",
                    "relation_schema_version",
                ),
            )
            _validate_visual_manifest(cfg, expected)
        visual = VisualStore(cfg["paths"]["visual_dir"], cfg["visual_index"])
        encoder = _encoder(cfg)
        selector = AttributeSelectorAgent(
            ontology,
            cfg.get("attribute_selector", {}),
            cfg["models"].get("attribute_selector_llm"),
            encoder,
        )
        retriever = GraphRetriever(
            ontology, graph, visual, encoder, cfg["retrieval"], selector
        )
        return retriever, graph
    except Exception:
        graph.close()
        raise


def _query_fields(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    query_id = str(row.get("query_id", f"q{index:06d}")).strip()
    image = str(row.get("image_path", row.get("query_image_path", ""))).strip()
    retrieval_question = str(
        row.get(
            "retrieval_question",
            row.get("retrieval_query", row.get("question", "")),
        )
    ).strip()
    reader_question = str(row.get("reader_question", row.get("question", ""))).strip()
    options = row.get("options", [])
    if not isinstance(options, list):
        options = []
    if not query_id or not image or not retrieval_question:
        raise ValueError(
            f"Query row {index} requires query_id, image_path, and retrieval_question/question"
        )
    return {
        "query_id": query_id,
        "image": image,
        "retrieval_question": retrieval_question,
        "reader_question": reader_question or retrieval_question,
        "options": [str(option) for option in options],
        "audit_metadata": safe_query_audit_metadata(row),
    }


def cmd_self_test(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    return self_test(cfg["project"]["ontology_path"])


def cmd_ontology_summary(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    """Report the frozen ontology and model-facing extraction contract."""

    ontology = Ontology(cfg["project"]["ontology_path"])
    extraction_cfg = cfg.get("ontology_extraction", {})
    prompt_spec = ontology.extraction_prompt_spec(
        profile=str(extraction_cfg.get("prompt_vocabulary_mode", "minimal_v1")),
        max_finding_synonyms=int(
            extraction_cfg.get("max_finding_synonyms_per_concept", 6)
        ),
        max_anatomy_synonyms=int(
            extraction_cfg.get("max_anatomy_synonyms_per_concept", 4)
        ),
    )
    full_prompt_spec = ontology.extraction_prompt_spec(profile="full")
    max_events_per_case = max(1, int(extraction_cfg.get("max_events_per_case", 16)))
    response_format = ontology_extraction_response_format(
        ontology,
        prompt_spec,
        max_events_per_case=max_events_per_case,
    )
    response_schema = response_format["json_schema"]["schema"]

    def json_char_count(value: Any) -> int:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def prompt_group_statistics(value: Mapping[str, Any]) -> dict[str, Any]:
        grouped = value.get("finding_groups")
        if not isinstance(grouped, Mapping):
            flat = value.get("findings", {})
            return {
                "finding_count": len(flat) if isinstance(flat, Mapping) else 0,
                "family_count": 0,
                "role_family_group_count": 0,
                "role_counts": {},
            }
        role_counts: dict[str, int] = {}
        family_ids: set[str] = set()
        role_family_group_count = 0
        for role, role_groups in grouped.items():
            if not isinstance(role_groups, Mapping):
                continue
            family_ids.update(str(family_id) for family_id in role_groups)
            role_family_group_count += len(role_groups)
            role_counts[str(role)] = sum(
                len(family_findings)
                for family_findings in role_groups.values()
                if isinstance(family_findings, (Mapping, list))
            )
        return {
            "finding_count": sum(role_counts.values()),
            "family_count": len(family_ids),
            "role_family_group_count": role_family_group_count,
            "role_counts": role_counts,
        }

    prompt_chars = json_char_count(prompt_spec)
    full_prompt_chars = json_char_count(full_prompt_spec)
    prompt_stats = prompt_group_statistics(prompt_spec)
    reduction = (
        1.0 - (prompt_chars / full_prompt_chars) if full_prompt_chars else 0.0
    )
    return {
        "coverage": ontology.coverage_summary(),
        "prompt_sha256": prompt_manifest(),
        "extraction_contract": {
            "prompt_profile": prompt_spec.get("prompt_profile"),
            "model_facing_finding_count": prompt_stats["finding_count"],
            "model_facing_finding_family_count": prompt_stats["family_count"],
            "model_facing_role_family_group_count": prompt_stats[
                "role_family_group_count"
            ],
            "model_facing_finding_role_counts": prompt_stats["role_counts"],
            "model_facing_anatomy_count": len(prompt_spec.get("anatomy", {})),
            "prompt_spec_sha256": stable_json_sha256(prompt_spec),
            "prompt_json_char_count": prompt_chars,
            "full_prompt_json_char_count": full_prompt_chars,
            "prompt_json_reduction_fraction": round(reduction, 6),
            "response_schema_name": response_format["json_schema"]["name"],
            "response_schema_sha256": stable_json_sha256(response_schema),
            "response_schema_json_char_count": json_char_count(response_schema),
            "model_response_serialization": "sparse_event_array",
            "durable_event_serialization": "events[finding] -> list[event]",
            "xgrammar_array_cardinality_enforced_in_schema": False,
            "max_events_per_case": max_events_per_case,
            "event_cap_policy": str(
                extraction_cfg.get("event_cap_policy", "clinical_priority")
            ),
            "case_normality": dict(
                extraction_cfg.get("case_normality", {}) or {}
            ),
            "case_normality_derivation": "deterministic_post_validation",
            "prompt_current_sections_only": bool(
                extraction_cfg.get("prompt_current_sections_only", True)
            ),
            "selection_strategy": str(
                extraction_cfg.get("selection_strategy", "evidence_first")
            ),
            "context_only_direct_mention_only": bool(
                extraction_cfg.get("context_only_direct_mention_only", True)
            ),
            "forbid_family_expansion": bool(
                extraction_cfg.get("forbid_family_expansion", True)
            ),
            "emit_empty_finding_arrays": bool(
                extraction_cfg.get("emit_empty_finding_arrays", False)
            ),
            "semantic_validation": dict(
                extraction_cfg.get("semantic_validation", {}) or {}
            ),
            "generic_fallback_finding_ids": sorted(
                str(value)
                for value in extraction_cfg.get(
                    "generic_fallback_finding_ids", []
                )
            ),
            "default_ontology_backend": str(
                cfg.get("models", {})
                .get("ontology_extractor_llm", {})
                .get("backend", "qwen-vl8")
            ),
            "available_ontology_backends": list(ontology_backend_choices()),
        },
    }


def cmd_extract_ontology(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    ontology = Ontology(cfg["project"]["ontology_path"])
    extraction_cfg = cfg["ontology_extraction"]
    ontology_llm_cfg = resolve_ontology_backend(
        cfg["models"]["ontology_extractor_llm"],
        args.ontology_backend,
    )
    agent = OntologyExtractorAgent(
        ontology,
        ontology_llm_cfg,
        extraction_cfg,
    )
    normality_cfg = dict(extraction_cfg.get("case_normality", {}))
    normality_enabled = bool(normality_cfg.get("enabled", False))
    abnormal_findings = normality_cfg.get("abnormal_findings", "clinical")
    if normality_enabled:
        ontology.resolve_abnormality_findings(abnormal_findings)
    allowed = set(extraction_cfg.get("allowed_splits", ["train"]))
    cases = _cases(cfg["paths"]["cases_jsonl"], splits=allowed)
    if args.limit:
        cases = cases[: args.limit]

    events_path = Path(args.events_output or cfg["paths"]["events_jsonl"])
    status_path = Path(args.status_output or cfg["paths"]["extraction_status_jsonl"])
    manifest_path = events_path.with_name(events_path.stem + ".manifest.json")
    issues_path = (
        Path(args.issues)
        if args.issues
        else status_path.with_name(status_path.stem + ".issues.jsonl")
    )
    trace_path = Path(args.trace_output) if args.trace_output else None

    extraction_identity = {
        "schema_version": "visig_cr_ontology_extraction_manifest_v1_2",
        "project_version": str(cfg.get("project", {}).get("version", "unknown")),
        "ontology_backend": agent.ontology_backend,
        "ontology_model_family": agent.ontology_model_family,
        "ontology_checkpoint_id": agent.ontology_checkpoint_id,
        "ontology_runtime_mode": agent.ontology_runtime_mode,
        "ontology_api_model": agent.ontology_api_model,
        "ontology_sha256": ontology.ontology_hash,
        "ontology_prompt_profile": agent.prompt_spec.get(
            "prompt_profile", "unknown"
        ),
        "ontology_prompt_spec_sha256": agent.prompt_spec_sha256,
        "ontology_selection_strategy": agent.selection_strategy,
        "ontology_prompt_finding_role_counts": agent.prompt_group_statistics[
            "role_counts"
        ],
        "ontology_response_schema_name": agent.response_schema_name,
        "ontology_response_schema_sha256": agent.response_schema_sha256,
        "ontology_response_schema_char_count": agent.response_schema_char_count,
        "ontology_model_response_serialization": "sparse_event_array",
        "ontology_durable_event_serialization": "events[finding] -> list[event]",
        "ontology_system_prompt_sha256": prompt_manifest().get(
            "ontology_extractor_system"
        ),
        "ontology_extraction_config_sha256": stable_json_sha256(
            {
                key: value
                for key, value in extraction_cfg.items()
                if key != "case_normality"
            }
        ),
        "generation_config": {
            "max_tokens": int(ontology_llm_cfg.get("max_tokens", 1200)),
            "temperature": float(ontology_llm_cfg.get("temperature", 0.0)),
            "disable_qwen_thinking": bool(
                ontology_llm_cfg.get("disable_qwen_thinking", False)
            ),
            "allow_unconstrained_json_fallback": False,
        },
        "max_events_per_case": agent.max_events_per_case,
        "event_cap_policy": agent.event_cap_policy,
        "case_count": len(cases),
        "case_ids_sha256": stable_json_sha256([case.case_id for case in cases]),
        "protocol_id": cfg.get("_protocol_context", {}).get("protocol_id"),
    }

    if not args.resume:
        _reset_jsonl(events_path)
        _reset_jsonl(status_path)
        _reset_jsonl(issues_path)
        if trace_path is not None:
            _reset_jsonl(trace_path)
    manifest_action = _prepare_identity_manifest(
        manifest_path,
        extraction_identity,
        checkpoint_path=events_path,
        resume=bool(args.resume),
        label="ontology extraction",
    )

    completed_ids = (
        _unique_case_ids(events_path, label="ontology event output")
        if args.resume
        else set()
    )
    requested_ids = [case.case_id for case in cases]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("Ontology input manifest contains duplicate case IDs")

    new_event_count = 0
    new_event_group_count = 0
    newly_completed = 0
    failed = 0
    resumed_skip = 0
    validation_retry_count = 0
    trace_success_every = max(1, int(args.trace_success_every))

    progress = tqdm(cases, desc="Ontology Extractor Agent")
    for input_index, case in enumerate(progress, 1):
        if case.case_id in completed_ids:
            resumed_skip += 1
            progress.set_postfix(
                ok=newly_completed,
                fail=failed,
                skip=resumed_skip,
                retries=validation_retry_count,
            )
            continue

        started = time.perf_counter()
        try:
            with open_text(case.report_path, "rt") as handle:
                report = handle.read()
            events, status = agent.extract(case, report)
            elapsed = time.perf_counter() - started
            audit_record = status.pop("ontology_output_audit", None)
            retry_count = int(status.get("model_validation_retry_count", 0))
            validation_retry_count += retry_count

            # The event record is the durable completion checkpoint.  A case
            # with zero validated events still receives one CaseEvents row.
            normality = (
                ontology.case_normality(
                    events,
                    abnormal_findings=abnormal_findings,
                )
                if normality_enabled
                else None
            )
            append_jsonl(
                events_path,
                CaseEvents.from_events(
                    case.case_id,
                    events,
                    normality=normality,
                ).model_dump(),
            )
            completed_ids.add(case.case_id)

            status.update(
                {
                    "stage": "extract_ontology",
                    "input_index": input_index,
                    "elapsed_seconds": round(elapsed, 6),
                }
            )
            append_jsonl(status_path, status)

            newly_completed += 1
            new_event_count += len(events)
            new_event_group_count += len({event.finding for event in events})

            if (
                trace_path is not None
                and audit_record is not None
                and newly_completed % trace_success_every == 0
            ):
                append_jsonl(
                    trace_path,
                    {
                        "stage": "extract_ontology",
                        "case_id": case.case_id,
                        "input_index": input_index,
                        "ok": True,
                        "elapsed_seconds": round(elapsed, 6),
                        "ontology_output_audit": audit_record,
                    },
                )
        except Exception as exc:  # retain per-case failure provenance
            elapsed = time.perf_counter() - started
            failed += 1
            audit_record = getattr(exc, "audit_record", None)
            if isinstance(audit_record, dict):
                validation_retry_count += max(
                    0,
                    int(audit_record.get("attempt_count", 0)) - 1,
                )
            failure_status = {
                "stage": "extract_ontology",
                "case_id": case.case_id,
                "input_index": input_index,
                "ok": False,
                "elapsed_seconds": round(elapsed, 6),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ontology_sha256": ontology.ontology_hash,
                "ontology_backend": agent.ontology_backend,
                "ontology_model_family": agent.ontology_model_family,
                "ontology_checkpoint_id": agent.ontology_checkpoint_id,
                "ontology_runtime_mode": agent.ontology_runtime_mode,
                "ontology_selection_strategy": agent.selection_strategy,
                "ontology_prompt_profile": agent.prompt_spec.get(
                    "prompt_profile", "unknown"
                ),
                "ontology_response_schema_name": agent.response_schema_name,
                "ontology_response_schema_sha256": agent.response_schema_sha256,
                "ontology_response_schema_char_count": agent.response_schema_char_count,
            }
            if isinstance(audit_record, dict):
                failure_status["model_attempt_count"] = int(
                    audit_record.get("attempt_count", 0)
                )
                failure_status["model_validation_retry_count"] = max(
                    0,
                    int(audit_record.get("attempt_count", 0)) - 1,
                )
            append_jsonl(status_path, failure_status)

            issue_record = dict(failure_status)
            if isinstance(audit_record, dict):
                issue_record["ontology_output_audit"] = audit_record
            append_jsonl(issues_path, issue_record)
            if trace_path is not None and isinstance(audit_record, dict):
                append_jsonl(trace_path, issue_record)
            if args.fail_fast:
                raise

        progress.set_postfix(
            ok=newly_completed,
            fail=failed,
            skip=resumed_skip,
            retries=validation_retry_count,
            last_s=f"{time.perf_counter() - started:.2f}",
        )

    requested_set = set(requested_ids)
    completed_for_run = requested_set & completed_ids
    missing = sorted(requested_set - completed_ids)
    summary = {
        "case_count": len(cases),
        "completed_case_count": len(completed_for_run),
        "newly_completed_count": newly_completed,
        "failed_attempt_count": failed,
        "resume_skip_count": resumed_skip,
        "validation_retry_count": validation_retry_count,
        "new_event_count": new_event_count,
        "new_event_group_count": new_event_group_count,
        "missing_case_count": len(missing),
        "missing_case_ids": missing,
        "complete": not missing,
        "event_serialization": "case_id + normality + events[finding] -> list[event]",
        "events_output": str(events_path),
        "status_output": str(status_path),
        "issues_output": str(issues_path),
        "trace_output": str(trace_path) if trace_path is not None else None,
        "manifest_output": str(manifest_path),
        "manifest_action": manifest_action,
        "ontology_backend": agent.ontology_backend,
        "ontology_model_family": agent.ontology_model_family,
        "ontology_checkpoint_id": agent.ontology_checkpoint_id,
        "ontology_runtime_mode": agent.ontology_runtime_mode,
        "ontology_selection_strategy": agent.selection_strategy,
        "ontology_prompt_profile": agent.prompt_spec.get(
            "prompt_profile", "unknown"
        ),
        "ontology_response_schema_name": agent.response_schema_name,
        "ontology_response_schema_sha256": agent.response_schema_sha256,
        "ontology_response_schema_char_count": agent.response_schema_char_count,
    }
    if not summary["complete"] and not args.allow_incomplete:
        summary["exit_code"] = 2
    return summary


def cmd_build_visual_index(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    split = str(cfg["visual_graph"].get("memory_split", "train"))
    cases = _cases(cfg["paths"]["cases_jsonl"], splits={split})
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("No memory cases are available for visual indexing")
    store = VisualStore.build(
        cases,
        _encoder(cfg),
        cfg["paths"]["visual_dir"],
        cfg["visual_index"],
    )
    identity = artifact_identity(cfg)
    manifest = {
        **identity,
        "case_count": len(store.case_ids),
        "case_ids_sha256": stable_json_sha256(store.case_ids),
        "embedding_shape": list(store.embeddings.shape),
        "medsiglip_model_path": str(cfg["models"]["medsiglip_path"]),
        "image_normalization": image_normalization_provenance(),
    }
    write_json(Path(cfg["paths"]["visual_dir"]) / "manifest.json", manifest)
    return {"visual_dir": cfg["paths"]["visual_dir"], **manifest}


def cmd_build_graph(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    graph = GraphStore(cfg["paths"]["graph_db"])
    try:
        memory_split = str(cfg["visual_graph"].get("memory_split", "train"))
        cases = _cases(cfg["paths"]["cases_jsonl"], splits={memory_split})
        memory_case_ids = {case.case_id for case in cases}
        events = [
            event
            for event in _events(cfg["paths"]["events_jsonl"])
            if event.case_id in memory_case_ids
        ]
        ontology = Ontology(cfg["project"]["ontology_path"])
        normality_cfg = dict(
            cfg.get("ontology_extraction", {}).get("case_normality", {}) or {}
        )
        case_normalities = (
            {
                case_id: value
                for case_id, value in _case_normalities(
                    cfg["paths"]["events_jsonl"],
                    ontology=ontology,
                    abnormal_findings=normality_cfg.get(
                        "abnormal_findings", "clinical"
                    ),
                ).items()
                if case_id in memory_case_ids
            }
            if bool(normality_cfg.get("enabled", False))
            else {}
        )
        if cfg.get("_protocol_context"):
            _validate_visual_manifest(cfg, artifact_identity(cfg))
        visual = VisualStore(cfg["paths"]["visual_dir"], cfg["visual_index"])
        visual_ids = set(visual.case_ids)
        if visual_ids != memory_case_ids:
            raise ValueError(
                "Visual index/case manifest mismatch: "
                + json.dumps(
                    {
                        "manifest_only_count": len(memory_case_ids - visual_ids),
                        "visual_only_count": len(visual_ids - memory_case_ids),
                        "manifest_only_examples": sorted(memory_case_ids - visual_ids)[:20],
                        "visual_only_examples": sorted(visual_ids - memory_case_ids)[:20],
                    },
                    ensure_ascii=False,
                )
            )
        if not args.no_clean:
            graph.reset_memory()
        case_count = graph.import_cases(cases)
        normality_count = graph.import_case_normalities(case_normalities)
        normality_metadata = {}
        if normality_cfg.get("enabled", False):
            for raw in read_jsonl(cfg["paths"]["events_jsonl"]):
                if raw.get("case_id") not in memory_case_ids:
                    continue
                record = CaseEvents.model_validate(raw)
                _, provenance = derive_normality_metadata(
                    record.to_events(), ontology, normality_cfg.get("abnormal_findings", "clinical")
                )
                normality_metadata[record.case_id] = provenance
            graph.import_normality_provenance(normality_metadata)
        event_count = graph.import_events(events)
        visual_started = time.perf_counter()
        visual_stats = graph.build_visual_edges(visual, cfg["visual_graph"])
        visual_stats["visual_graph_build_seconds"] = time.perf_counter() - visual_started
        relation_stats = graph.build_relation_edges(ontology, cfg.get("relation_graph", {}))
        identity = artifact_identity(cfg)
        graph.set_artifact_metadata(
            {
                **identity,
                "case_count": case_count,
                "case_normality_count": normality_count,
                "event_count": event_count,
                "visual_case_ids_sha256": stable_json_sha256(visual.case_ids),
                "relation_schema_version": RELATION_SCHEMA_VERSION,
            }
        )
        return {
            "memory_split": memory_split,
            "clean_rebuild": not args.no_clean,
            "imported_case_count": case_count,
            "imported_case_normality_count": normality_count,
            "imported_event_count": event_count,
            "visual_graph": visual_stats,
            "relation_graph": relation_stats,
            "artifact_metadata": graph.artifact_metadata(),
            "clinical_pair_edges_materialized": True,
        }
    finally:
        graph.close()


def cmd_build_relations(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    graph = GraphStore(cfg["paths"]["graph_db"])
    try:
        if cfg.get("_protocol_context"):
            expected = artifact_identity(cfg)
            graph.assert_artifact_metadata(
                expected,
                keys=(
                    "protocol_id",
                    "memory_size",
                    "case_manifest_sha256",
                    "events_sha256",
                    "ontology_sha256",
                    "visual_index_config_sha256",
                    "visual_graph_config_sha256",
                    "normality_config_sha256",
                ),
            )
            _validate_visual_manifest(cfg, expected)
        ontology = Ontology(cfg["project"]["ontology_path"])
        result = graph.build_relation_edges(ontology, cfg.get("relation_graph", {}))
        graph.set_artifact_metadata(
            {**artifact_identity(cfg), "relation_schema_version": RELATION_SCHEMA_VERSION}
        )
        result["artifact_metadata"] = graph.artifact_metadata()
        return result
    finally:
        graph.close()


def cmd_graph_stats(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    graph = GraphStore(cfg["paths"]["graph_db"])
    try:
        result = graph.graph_stats(split=args.split)
        result["artifact_metadata"] = graph.artifact_metadata()
        return result
    finally:
        graph.close()


def cmd_retrieve(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    if args.method == "no_rag":
        package = build_no_rag_package(
            query_id=args.query_id,
            image_path=args.image,
            question=args.question,
            options=args.option or [],
        )
        result = package.model_dump()
        if args.output:
            write_json(args.output, result)
        return result

    retriever, graph = _retrieval_stack(cfg, args.method)
    try:
        package = retriever.retrieve(
            args.image,
            args.question,
            query_id=args.query_id,
            method=args.method,
            options=args.option or [],
        )
        package.audit_metadata = {
            **package.audit_metadata,
            "attribute_selector": _attribute_selector_provenance(cfg),
        }
        fallback_audit = package.audit_metadata.get("selector_fallback")
        if isinstance(fallback_audit, dict) and fallback_audit.get("used") is True:
            fallback_audit["retrieval_config_sha256"] = stable_json_sha256(
                cfg["retrieval"]
            )
        contrast_fallback_audit = package.audit_metadata.get("contrast_fallback")
        if (
            isinstance(contrast_fallback_audit, dict)
            and contrast_fallback_audit.get("used") is True
        ):
            contrast_fallback_audit["retrieval_config_sha256"] = (
                stable_json_sha256(cfg["retrieval"])
            )
        result = package.model_dump()
        if args.output:
            write_json(args.output, result)
        return result
    finally:
        graph.close()


def cmd_build_presence_index(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    output = args.output or presence_index_path(cfg["paths"]["graph_db"], cfg["retrieval"])
    result = build_presence_index(
        cfg["paths"]["graph_db"], output, Ontology(cfg["project"]["ontology_path"]),
        cfg["retrieval"], overwrite=args.overwrite,
    )
    write_json(Path(output).with_suffix(".summary.json"), result)
    return result


def cmd_batch_retrieve(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    output = Path(args.output)
    issues = Path(args.issues) if args.issues else output.with_name(output.stem + ".issues.jsonl")
    fallback_arg = getattr(args, "fallbacks", None)
    fallbacks = (
        Path(fallback_arg)
        if fallback_arg
        else output.with_name(output.stem + ".fallbacks.jsonl")
    )
    if fallbacks.resolve() in {output.resolve(), issues.resolve()}:
        raise ValueError("Fallback audit path must differ from package and issue outputs")
    is_qcr = uses_question_conditioned(cfg.get("retrieval", {}), args.method)
    if is_qcr:
        # A method name alone cannot distinguish the old graph packages from
        # the new pool-based method. Check before resetting/appending outputs.
        memory_identity = (
            artifact_identity(cfg)
            if cfg.get("paths", {}).get("cases_jsonl") and cfg.get("project", {}).get("ontology_path")
            else {}
        )
        _prepare_identity_manifest(
            output.with_suffix(output.suffix + ".manifest.json"),
            {
                "retrieval_policy": POLICY_VERSION,
                "presence_index": presence_identity(cfg),
                "queries_sha256": file_sha256(args.queries),
                "memory_identity": {key: memory_identity.get(key) for key in MEMORY_IDENTITY_KEYS},
                "runtime_config_sha256": stable_json_sha256({
                    key: cfg.get(key, {}) for key in (
                        "paths", "_protocol_context", "retrieval", "visual_index",
                        "attribute_selector", "models", "project",
                    )
                }),
                "attribute_selector_prompt": prompt_manifest()["attribute_selector_system"],
            },
            checkpoint_path=output, resume=bool(args.resume), label="retrieval packages",
        )
    if not args.resume:
        _reset_jsonl(output)
        _reset_jsonl(issues)
        _reset_jsonl(fallbacks)
    completed_ids = _unique_ids(output, label="package output") if args.resume else set()
    input_ids: set[str] = set()
    requested_ids: list[str] = []
    success = 0
    failed = 0
    resumed_skip = 0
    retriever = None
    graph = None
    if args.method != "no_rag":
        retriever, graph = _retrieval_stack(cfg, args.method)
    try:
        for index, row in enumerate(tqdm(read_jsonl(args.queries), desc="Retrieval"), 1):
            if args.limit is not None and len(requested_ids) >= args.limit:
                break
            fields: dict[str, Any] | None = None
            try:
                fields = _query_fields(row, index)
                query_id = fields["query_id"]
                if query_id in input_ids:
                    raise ValueError(f"Duplicate query_id in input: {query_id}")
                input_ids.add(query_id)
                requested_ids.append(query_id)
                if query_id in completed_ids:
                    resumed_skip += 1
                    continue
                if args.method == "no_rag":
                    package = build_no_rag_package(
                        query_id=query_id,
                        image_path=fields["image"],
                        question=fields["retrieval_question"],
                        options=fields["options"],
                        reader_question=fields["reader_question"],
                        audit_metadata=fields["audit_metadata"],
                    )
                else:
                    assert retriever is not None
                    package = retriever.retrieve(
                        fields["image"],
                        fields["retrieval_question"],
                        query_id=query_id,
                        method=args.method,
                        options=fields["options"],
                        reader_question=fields["reader_question"],
                    )
                    package.audit_metadata = {
                        **package.audit_metadata,
                        **fields["audit_metadata"],
                        "attribute_selector": _attribute_selector_provenance(cfg),
                    }
                    fallback_audit = package.audit_metadata.get("selector_fallback")
                    if (
                        isinstance(fallback_audit, dict)
                        and fallback_audit.get("used") is True
                    ):
                        fallback_audit["retrieval_config_sha256"] = stable_json_sha256(
                            cfg["retrieval"]
                        )
                    contrast_fallback_audit = package.audit_metadata.get(
                        "contrast_fallback"
                    )
                    if (
                        isinstance(contrast_fallback_audit, dict)
                        and contrast_fallback_audit.get("used") is True
                    ):
                        contrast_fallback_audit["retrieval_config_sha256"] = (
                            stable_json_sha256(cfg["retrieval"])
                        )
                append_jsonl(output, package.model_dump())
                completed_ids.add(query_id)
                success += 1
            except Exception as exc:
                failed += 1
                append_jsonl(
                    issues,
                    {
                        "stage": "batch_retrieve",
                        "query_id": (fields or {}).get("query_id", str(row.get("query_id", ""))),
                        "input_index": index,
                        "method": args.method,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if args.fail_fast:
                    raise
    finally:
        if graph is not None:
            graph.close()

    requested_set = set(requested_ids)
    completed_for_run = requested_set & completed_ids
    missing = sorted(requested_set - completed_ids)
    fallback_records: list[dict[str, Any]] = []
    fallback_reason_counts: dict[str, int] = {}
    fallback_dataset_counts: dict[str, int] = {}
    fallback_question_family_counts: dict[str, int] = {}
    fallback_for_run = 0
    contrast_fallback_for_run = 0
    contrast_records: list[dict[str, Any]] = []
    qcr_pool_sizes: dict[str, int] = {}
    qcr_expanded_count = 0
    contrast_fallback_reason_counts: dict[str, int] = {}
    contrast_fallback_dataset_counts: dict[str, int] = {}
    contrast_fallback_question_family_counts: dict[str, int] = {}
    candidate_pair_fallback_for_run = 0
    candidate_pair_fallback_attempted_for_run = 0
    for row in read_jsonl(output):
        package_audit = row.get("audit_metadata", {})
        query_id = str(row.get("query_id", ""))
        dataset = str(
            package_audit.get("dataset", package_audit.get("source_dataset", "unknown"))
        )
        question_family = str(package_audit.get("question_family", "unknown"))
        if query_id in requested_set and package_audit.get("retrieval_policy") == POLICY_VERSION:
            stats = row.get("boundary_stats", {})
            size = str(stats.get("boundary_case_count", 0))
            qcr_pool_sizes[size] = qcr_pool_sizes.get(size, 0) + 1
            qcr_expanded_count += int(stats.get("pool_expansion_count", 0) > 0)
        contrast_audit = package_audit.get("contrast_fallback", {})
        if (
            isinstance(contrast_audit, Mapping)
            and contrast_audit.get("used") is True
            and query_id in requested_set
        ):
            contrast_reason = str(
                contrast_audit.get("reason_code", "unknown")
            )
            contrast_fallback_for_run += 1
            contrast_records.append({
                "query_id": query_id, "dataset": dataset,
                "question_family": question_family, **dict(contrast_audit),
                "boundary_stats": row.get("boundary_stats", {}),
            })
            contrast_fallback_reason_counts[contrast_reason] = (
                contrast_fallback_reason_counts.get(contrast_reason, 0) + 1
            )
            contrast_fallback_dataset_counts[dataset] = (
                contrast_fallback_dataset_counts.get(dataset, 0) + 1
            )
            contrast_fallback_question_family_counts[question_family] = (
                contrast_fallback_question_family_counts.get(question_family, 0)
                + 1
            )
        pair_construction = package_audit.get("pair_construction", {})
        pair_fallback = (
            pair_construction.get("candidate_pair_fallback", {})
            if isinstance(pair_construction, Mapping)
            else {}
        )
        if isinstance(pair_fallback, Mapping) and query_id in requested_set:
            if pair_fallback.get("attempted") is True:
                candidate_pair_fallback_attempted_for_run += 1
            if pair_fallback.get("used") is True:
                candidate_pair_fallback_for_run += 1
        audit = package_audit.get("selector_fallback", {})
        if not isinstance(audit, Mapping) or audit.get("used") is not True:
            continue
        reason = str(audit.get("reason_code", "unknown"))
        fallback_records.append(
            {
                "query_id": query_id,
                "method": row.get("method", args.method),
                "dataset": dataset,
                "question_family": question_family,
                "reason_code": reason,
                "retriever": audit.get("retriever", "i2i"),
                "evidence_k": audit.get("evidence_k"),
                "selector_error_type": audit.get("selector_error_type", ""),
                "selector_error": audit.get("selector_error", ""),
                "attribute_selector": package_audit.get("attribute_selector", {}),
                "retrieval_config_sha256": audit.get(
                    "retrieval_config_sha256", ""
                ),
            }
        )
        if query_id in requested_set:
            fallback_for_run += 1
            fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
            fallback_dataset_counts[dataset] = fallback_dataset_counts.get(dataset, 0) + 1
            fallback_question_family_counts[question_family] = (
                fallback_question_family_counts.get(question_family, 0) + 1
            )
    _reset_jsonl(fallbacks)
    for record in fallback_records:
        append_jsonl(fallbacks, record)
    contrast_path = output.with_name(output.stem + ".contrast_fallbacks.jsonl")
    if is_qcr:
        _reset_jsonl(contrast_path)
        for record in contrast_records:
            append_jsonl(contrast_path, record)
    summary = {
        "method": args.method,
        "input_query_count": len(requested_ids),
        "completed_query_count": len(completed_for_run),
        "newly_completed_count": success,
        "failed_count": failed,
        "resume_skip_count": resumed_skip,
        "missing_package_count": len(missing),
        "missing_package_ids": missing,
        "selector_fallback_count": fallback_for_run,
        "selector_fallback_rate": (
            fallback_for_run / len(requested_ids) if requested_ids else 0.0
        ),
        "selector_fallback_rate_among_completed": (
            fallback_for_run / len(completed_for_run) if completed_for_run else 0.0
        ),
        "selector_fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "selector_fallback_dataset_counts": dict(sorted(fallback_dataset_counts.items())),
        "selector_fallback_question_family_counts": dict(
            sorted(fallback_question_family_counts.items())
        ),
        "contrast_fallback_count": contrast_fallback_for_run,
        "contrast_fallback_rate": (
            contrast_fallback_for_run / len(requested_ids)
            if requested_ids
            else 0.0
        ),
        "contrast_fallback_rate_among_completed": (
            contrast_fallback_for_run / len(completed_for_run)
            if completed_for_run
            else 0.0
        ),
        "contrast_fallback_reason_counts": dict(
            sorted(contrast_fallback_reason_counts.items())
        ),
        "contrast_fallback_dataset_counts": dict(
            sorted(contrast_fallback_dataset_counts.items())
        ),
        "contrast_fallback_question_family_counts": dict(
            sorted(contrast_fallback_question_family_counts.items())
        ),
        "candidate_pair_fallback_attempted_count": (
            candidate_pair_fallback_attempted_for_run
        ),
        "candidate_pair_fallback_count": candidate_pair_fallback_for_run,
        "candidate_pair_fallback_rate": (
            candidate_pair_fallback_for_run / len(requested_ids)
            if requested_ids
            else 0.0
        ),
        "candidate_pair_fallback_rate_among_completed": (
            candidate_pair_fallback_for_run / len(completed_for_run)
            if completed_for_run
            else 0.0
        ),
        "native_package_count": (
            len(completed_for_run)
            - fallback_for_run
            - contrast_fallback_for_run
        ),
        "complete": not missing,
        "output": str(output),
        "issues": str(issues),
        "fallbacks": str(fallbacks),
    }
    if not summary["complete"] and not args.allow_incomplete:
        summary["exit_code"] = 2
    if is_qcr:
        presence_counts: dict[str, int] = {}
        inferred_queries = restored_queries = 0
        for row in read_jsonl(output):
            if str(row.get("query_id", "")) not in requested_set:
                continue
            counts = row.get("audit_metadata", {}).get("presence_index", {}).get("selected_basis_counts", {})
            for basis, count in counts.items():
                presence_counts[basis] = presence_counts.get(basis, 0) + count
            inferred_queries += bool(counts.get("inferred_absent", 0))
            restored_queries += bool(counts.get("restored_explicit_absent", 0))
        summary.update({
            "retrieval_policy": POLICY_VERSION,
            "presence_evidence_basis_counts": presence_counts,
            "inferred_absent_query_count": inferred_queries,
            "inferred_absent_query_rate": inferred_queries / len(completed_for_run) if completed_for_run else 0.0,
            "restored_explicit_absent_query_count": restored_queries,
            "contrast_fallbacks": str(contrast_path),
            "pool_expanded_query_count": qcr_expanded_count,
            "pool_scanned_case_count_distribution": qcr_pool_sizes,
        })
    return summary


def cmd_reader(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    package = EvidencePackage.model_validate(read_json(args.package))
    reader_cfg = dict(cfg["reader"])
    if args.reader_family:
        reader_cfg["model_family"] = args.reader_family
    reader = VLMReader(
        cfg["models"]["reader_vlm"],
        reader_cfg,
    )
    result = reader.answer(package)
    if args.output:
        write_json(args.output, result)
    return result


def cmd_batch_reader(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    # Validate frozen evidence contracts before opening models or touching outputs.
    # A v1 package cannot be repaired by applying a v2 prompt to stale labels.
    for index, row in enumerate(read_jsonl(args.packages)):
        if args.limit is not None and index >= args.limit:
            break
        VLMReader._effective_reader_package(EvidencePackage.model_validate(row))
    output = Path(args.output)
    issues = Path(args.issues) if args.issues else output.with_name(output.stem + ".issues.jsonl")
    if not args.resume:
        _reset_jsonl(output)
        _reset_jsonl(issues)
    completed_ids = _unique_ids(output, label="prediction output") if args.resume else set()
    completed_identity: dict[str, tuple[str, str]] = {}
    if args.resume:
        for row in read_jsonl(output):
            query_id = str(row.get("query_id", "")).strip()
            audit = row.get("reader_output_audit", {})
            if not isinstance(audit, Mapping):
                audit = {}
            completed_identity[query_id] = (
                str(row.get("evidence_package_sha256", "")).strip(),
                str(audit.get("reader_protocol_sha256", "")).strip(),
            )
    reader_cfg = dict(cfg["reader"])
    if args.reader_family:
        reader_cfg["model_family"] = args.reader_family
    reader = VLMReader(
        cfg["models"]["reader_vlm"],
        reader_cfg,
    )
    package_ids: list[str] = []
    seen_packages: set[str] = set()
    success = 0
    failed = 0
    resumed_skip = 0

    for index, row in enumerate(tqdm(read_jsonl(args.packages), desc="Frozen reader"), 1):
        if args.limit is not None and len(package_ids) >= args.limit:
            break
        query_id = str(row.get("query_id", "")).strip()
        if args.resume and query_id in completed_ids:
            if not query_id:
                raise ValueError("Package lacks query_id")
            if query_id in seen_packages:
                raise ValueError(f"Duplicate package query_id: {query_id}")
            seen_packages.add(query_id)
            package_ids.append(query_id)
            package = EvidencePackage.model_validate(row)
            expected = (
                evidence_package_sha256(package),
                reader.reader_protocol_sha256(package),
            )
            actual = completed_identity.get(query_id, ("", ""))
            if actual != expected:
                raise ValueError(
                    "Reader resume identity mismatch for query_id="
                    f"{query_id!r}: saved package/prompt hashes={actual}, "
                    f"current={expected}. Use a fresh prediction output without "
                    "--resume after changing packages, prompt, layout, or reader "
                    "presentation settings."
                )
            resumed_skip += 1
            continue
        try:
            if not query_id:
                raise ValueError("Package lacks query_id")
            if query_id in seen_packages:
                raise ValueError(f"Duplicate package query_id: {query_id}")
            seen_packages.add(query_id)
            package_ids.append(query_id)
            package = EvidencePackage.model_validate(row)
            prediction = reader.answer(package)
            append_jsonl(output, prediction)
            completed_ids.add(query_id)
            success += 1
        except Exception as exc:
            failed += 1
            issue_record = {
                "stage": "batch_reader",
                "query_id": query_id,
                "input_index": index,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            reader_audit = getattr(exc, "audit_record", None)
            if isinstance(reader_audit, dict):
                issue_record["reader_output_audit"] = reader_audit
            append_jsonl(
                issues,
                issue_record,
            )
            if args.fail_fast:
                raise

    expected_ids = set(package_ids)
    reference_ids: set[str] | None = None
    if args.references:
        reference_ids = _unique_ids(args.references, label="reference")
        expected_ids |= reference_ids
    missing = sorted(expected_ids - completed_ids)
    extra = sorted(completed_ids - expected_ids)
    reference_package_mismatch = (
        {
            "package_only_ids": sorted(set(package_ids) - reference_ids)[:50],
            "reference_only_ids": sorted(reference_ids - set(package_ids))[:50],
        }
        if reference_ids is not None and set(package_ids) != reference_ids
        else None
    )
    complete = not missing and not extra and reference_package_mismatch is None
    summary = {
        "package_count": len(package_ids),
        "newly_completed_count": success,
        "failed_count": failed,
        "resume_skip_count": resumed_skip,
        "prediction_count": len(completed_ids),
        "missing_prediction_count": len(missing),
        "missing_prediction_ids": missing,
        "extra_prediction_count": len(extra),
        "extra_prediction_ids": extra,
        "reference_package_mismatch": reference_package_mismatch,
        "complete": complete,
        "output": str(output),
        "issues": str(issues),
    }
    if not complete and not args.allow_incomplete:
        summary["exit_code"] = 2
    return summary


def cmd_evaluate_ontology(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    return evaluate_ontology_events(
        list(read_jsonl(args.predicted)),
        list(read_jsonl(args.references)),
    )


def cmd_build_retrieval_annotations(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    from visig_cr.retrieval_evaluation import build_shared_annotations
    import os
    import tempfile
    specs = {}
    qcr_specs = True
    spec_row_count = 0
    for row in read_jsonl(args.question_specs):
        spec_row_count += 1
        qcr_specs = qcr_specs and row.get("audit_metadata", {}).get("retrieval_policy") == POLICY_VERSION
        qid = str(row.get("query_id", "")).strip()
        fallback_audit = row.get("audit_metadata", {}).get("selector_fallback", {})
        is_selector_fallback = (
            isinstance(fallback_audit, Mapping)
            and fallback_audit.get("used") is True
        )
        if is_selector_fallback and row.get("question_spec") is None:
            continue
        if not qid or qid in specs or row.get("question_spec") is None:
            raise ValueError("Shared specs need unique query_id and a question_spec object")
        specs[qid] = row["question_spec"]
    for path in args.packages:
        _unique_ids(path, label="retrieval packages")
    graph = GraphStore(cfg["paths"]["graph_db"])
    temporary = None
    try:
        if not (qcr_specs and spec_row_count):
            graph.assert_artifact_metadata({"relation_schema_version": RELATION_SCHEMA_VERSION, "relation_build_complete": True},
                                           keys=("relation_schema_version", "relation_build_complete"))
        graph.assert_artifact_metadata(
            artifact_identity(cfg), keys=MEMORY_IDENTITY_KEYS if qcr_specs and spec_row_count else
            ("events_sha256", "ontology_sha256", "case_manifest_sha256", "normality_config_sha256", "relation_schema_version")
        )
        ontology = Ontology(cfg["project"]["ontology_path"])
        _validate_visual_manifest(cfg, artifact_identity(cfg))
        visual = VisualStore(cfg["paths"]["visual_dir"], cfg["visual_index"])
        encoder = _encoder(cfg)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        protected = {Path(path).resolve() for path in [*args.packages, args.question_specs, Path(cfg["paths"]["events_jsonl"])]}
        if args.output.resolve() in protected:
            raise ValueError("Annotation output must not overwrite packages/specifications/events")
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        count = _write_jsonl(temporary, build_shared_annotations(
            (row for path in args.packages for row in read_jsonl(path)), specs,
            graph=graph, ontology=ontology, visual=visual, encoder=encoder,
            retrieval_cfg=cfg["retrieval"], k=int(cfg["evaluation"]["retrieval"].get("k", 6)),
        ))
        os.replace(temporary, args.output)
        temporary = None
        return {"query_count": count, "output": str(args.output),
                "sha256": file_sha256(args.output), "reader_inputs_modified": False,
                "provenance": "shared_case_ontology_proxy"}
    finally:
        graph.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cmd_evaluate_retrieval(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    packages = list(read_jsonl(args.packages))
    references = list(read_jsonl(args.references)) if args.references else None
    annotation_path = getattr(args, "case_annotations", None)
    if annotation_path is None and not getattr(args, "legacy_package_metrics", False):
        raise ValueError("Use --case-annotations from build-retrieval-annotations for cross-method evaluation; "
                         "--legacy-package-metrics is diagnostic only")
    eval_cfg = cfg["evaluation"]["retrieval"]
    result = evaluate_retrieval_packages(
        packages, references,
        case_annotations=list(read_jsonl(annotation_path)) if annotation_path else None,
        ontology=Ontology(cfg["project"]["ontology_path"]) if annotation_path else None,
        k=int(eval_cfg.get("k", 6)), tau_query=float(eval_cfg.get("tau_query", 0.20)),
        tau_non_target=float(eval_cfg.get("tau_non_target", 0.60)),
        tau_group_visual=float(eval_cfg.get("tau_group_visual", 0.20)),
    )
    if getattr(args, "output", None):
        write_json(args.output, result)
    return result


def cmd_evaluate_reader(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    expected_ids = None
    expected_options_by_query_id = None
    if args.packages:
        expected_ids = sorted(_unique_ids(args.packages, label="package"))
        expected_options_by_query_id = {}
        for row in read_jsonl(args.packages):
            query_id = str(row.get("query_id", "")).strip()
            options = row.get("options", [])
            if options is None:
                options = []
            if not isinstance(options, list):
                raise ValueError(
                    f"Package {query_id} options must be a JSON array"
                )
            expected_options_by_query_id[query_id] = options
    result = evaluate_predictions(
        list(read_jsonl(args.predictions)),
        list(read_jsonl(args.references)),
        strict_ids=not bool(args.allow_incomplete),
        expected_query_ids=expected_ids,
        expected_options_by_query_id=expected_options_by_query_id,
        group_by=tuple(args.group_by or ("dataset", "question_family")),
    )
    if args.output:
        write_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--protocol", help="Activate a named dataset protocol")
    parser.add_argument(
        "--memory-size",
        type=int,
        help="Select one pre-registered nested memory size from config.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("self-test", help="Run the deterministic relation-graph smoke test")
    sub.add_parser("ontology-summary", help="Validate and summarize the ontology")

    p = sub.add_parser("prepare-mimic", help="Deprecated compatibility wrapper; use prepare_datasets.py mimic")
    p.add_argument("--output", type=Path)
    p.add_argument("--split", choices=["train", "validate", "test", "all"], default="all")
    p.add_argument("--limit", type=int)
    p.add_argument("--allow-missing", action="store_true")

    p = sub.add_parser("extract-ontology", help="Run the Ontology Extractor Agent on the active memory pool")
    p.add_argument(
        "--ontology-backend",
        choices=ontology_backend_choices(),
        help=(
            "Record the checkpoint/runtime selected in model.sh. Defaults to "
            "models.ontology_extractor_llm.backend from config.yaml."
        ),
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--events-output", type=Path)
    p.add_argument("--status-output", type=Path)
    p.add_argument("--issues", type=Path)
    p.add_argument(
        "--trace-output",
        type=Path,
        help="Optional append-only model-output audit JSONL",
    )
    p.add_argument(
        "--trace-success-every",
        type=int,
        default=1,
        help="Write every Nth successful model response to --trace-output; failures are always traced",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")

    p = sub.add_parser("build-visual-index", help="Build frozen image embeddings and an exact FAISS index")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("build-graph", help="Clean-build cases, events, visual edges, and offline relations")
    p.add_argument(
        "--no-clean",
        action="store_true",
        help="Retain existing corpus rows; discouraged for memory-size experiments",
    )
    sub.add_parser("build-relations", help="Rebuild offline event relations on the active visual graph")
    p = sub.add_parser("build-presence-index", help="Build offline ontology-only negative evidence sidecar; no graph/LLM rebuild")
    p.add_argument("--output", type=Path)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("graph-stats", help="Report graph, event-store, and artifact metadata")
    p.add_argument("--split", default="train")

    p = sub.add_parser("retrieve", help="Build one frozen evidence package")
    p.add_argument("--image", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--query-id", default="query")
    p.add_argument("--method", choices=CORE_METHODS, default="visig_cr")
    p.add_argument(
        "--attribute-selector-model",
        choices=ATTRIBUTE_SELECTOR_MODEL_CHOICES,
        help=(
            "Declare the Qwen checkpoint served as attribute-selector. "
            "Default comes from config.yaml (qwen-vl8); use qwen-vl30 only "
            "when model.sh is serving the 30B selector."
        ),
    )
    p.add_argument("--option", action="append")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("batch-retrieve", help="Stream frozen packages for a JSONL query file")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--method", choices=CORE_METHODS, default="visig_cr")
    p.add_argument(
        "--attribute-selector-model",
        choices=ATTRIBUTE_SELECTOR_MODEL_CHOICES,
        help=(
            "Declare the Qwen checkpoint served as attribute-selector. "
            "Default comes from config.yaml (qwen-vl8)."
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--issues", type=Path)
    p.add_argument(
        "--fallbacks",
        type=Path,
        help="Write selector fallback audit rows (default: <output-stem>.fallbacks.jsonl)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")

    p = sub.add_parser("reader", help="Run the frozen reader on one package")
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--reader-family",
        choices=["qwen", "llava-med"],
        help="Select model-family-specific prompt and decoding safeguards",
    )

    p = sub.add_parser("batch-reader", help="Stream packages through one initialized VLM reader")
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--references", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--issues", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument(
        "--reader-family",
        choices=["qwen", "llava-med"],
        help="Select model-family-specific prompt and decoding safeguards",
    )

    p = sub.add_parser("evaluate-ontology", help="Evaluate predicted ontology events")
    p.add_argument("--predicted", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)

    p = sub.add_parser("build-retrieval-annotations", help="Build evaluator-only common case annotations")
    p.add_argument("--packages", type=Path, nargs="+", required=True)
    p.add_argument("--question-specs", type=Path, required=True,
                   help="Frozen canonical packages or query_id/question_spec JSONL, shared across methods")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("evaluate-retrieval", help="Evaluate delivered RAG evidence packages")
    p.add_argument("--packages", type=Path, required=True)
    p.add_argument("--references", type=Path)
    p.add_argument("--case-annotations", type=Path)
    p.add_argument("--legacy-package-metrics", action="store_true")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("evaluate-reader", help="Strictly evaluate reader answers and calibration")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--packages", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--group-by",
        action="append",
        choices=[
            "dataset",
            "question_family",
            "question_type",
            "task_type",
            "target_attribute",
            "source_split",
            "split",
            "retrieval_path",
            "selector_fallback_reason",
            "contrast_fallback_reason",
        ],
    )
    p.add_argument("--allow-incomplete", action="store_true")
    return parser


COMMANDS = {
    "self-test": cmd_self_test,
    "ontology-summary": cmd_ontology_summary,
    "extract-ontology": cmd_extract_ontology,
    "build-visual-index": cmd_build_visual_index,
    "build-graph": cmd_build_graph,
    "build-relations": cmd_build_relations,
    "build-presence-index": cmd_build_presence_index,
    "graph-stats": cmd_graph_stats,
    "retrieve": cmd_retrieve,
    "batch-retrieve": cmd_batch_retrieve,
    "reader": cmd_reader,
    "batch-reader": cmd_batch_reader,
    "evaluate-ontology": cmd_evaluate_ontology,
    "evaluate-retrieval": cmd_evaluate_retrieval,
    "build-retrieval-annotations": cmd_build_retrieval_annotations,
    "evaluate-reader": cmd_evaluate_reader,
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = apply_protocol_context(
        cfg,
        requested_protocol=args.protocol,
        requested_memory_size=args.memory_size,
        command=args.command,
    )
    _apply_attribute_selector_model_override(cfg, args)
    result = COMMANDS[args.command](args, cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result.get("exit_code", 0)) if isinstance(result, dict) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
