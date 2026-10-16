"""Offline, ontology-only absence augmentation. Never reads radiology reports.

The original event store is immutable. A compressed byte mask records observed
states; sparse pointers reference the existing events supporting added negatives.
Unmentioned is always unknown, never an absence assertion.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from .io_utils import file_sha256, stable_json_sha256
from .schemas import Event

POLICY_VERSION = "ontology_presence_index_v1"
MEMORY_KEYS = ("protocol_id", "memory_size", "case_manifest_sha256", "events_sha256", "ontology_sha256")
OBSERVED, ACTIVE_POSITIVE, CURRENT_ABSENT, UNCERTAIN, INFERRED, REOPENED = 1, 2, 4, 8, 16, 32
NOTES = frozenset({
    "section_corrected", "unsectioned_report", "evidence_whitespace_aligned",
    "laterality_corrected_from_evidence", "uncertainty_filled_from_evidence",
    "uncertainty_corrected_from_evidence", "severity_corrected_from_evidence",
    "unsupported_severity_removed", "inapplicable_severity_corrected",
})
NORMAL_RULES = {
    "normal_lung_transparency": ["atelectasis", "consolidation", "pulmonary_edema", "lung_opacity"],
    "normal_cardiac_silhouette": ["cardiomegaly"],
    "intact_osseous_structures": ["fracture"],
}
NORMAL_SOURCE_ANATOMIES = {
    "normal_lung_transparency": ["lung"],
    "normal_cardiac_silhouette": ["heart", "cardiac_silhouette"],
    "intact_osseous_structures": ["bone"],
}


def settings(retrieval_cfg):
    cfg = retrieval_cfg.get("presence_index", {})
    return {
        "policy_version": POLICY_VERSION,
        "restore_pilot_absences": bool(cfg.get("restore_pilot_absences", True)),
        "infer_from_normal_descriptors": bool(cfg.get("infer_from_normal_descriptors", True)),
        "minimum_confidence": float(retrieval_cfg.get("minimum_event_confidence", .70)),
        "normal_rules": NORMAL_RULES,
        "normal_source_anatomies": NORMAL_SOURCE_ANATOMIES,
    }


def index_path(graph_path, retrieval_cfg):
    custom = retrieval_cfg.get("presence_index", {}).get("path")
    return Path(custom) if custom else Path(graph_path).with_name("presence_index.npz")


def configured_identity(cfg):
    if cfg.get("retrieval", {}).get("visig_cr_strategy") == "legacy_relation_graph":
        return {"enabled": False}
    if not cfg.get("retrieval", {}).get("presence_index", {}).get("enabled", False):
        return {"enabled": False}
    path = index_path(cfg["paths"]["graph_db"], cfg["retrieval"])
    if not path.is_file():
        raise FileNotFoundError(f"Build the offline presence index first: {path} (build-presence-index)")
    return {"enabled": True, "policy_version": POLICY_VERSION, "sha256": file_sha256(path)}


def _verified(event):
    marker = event.characteristics.get("_semantic_support", {})
    return isinstance(marker, dict) and marker.get("verified") is True and marker.get("schema_version") == "visig_cr_semantic_support_v1"


def _quality(event, ontology, cfg, tolerated):
    return (
        event.temporal_status == "current"
        and event.confidence >= cfg["minimum_confidence"]
        and bool(event.evidence_span.strip()) and _verified(event)
        and ontology.anatomy_allowed(event.finding, event.anatomy)
        and not (set(event.validation_errors) - NOTES - set(tolerated))
    )


def _metadata(conn):
    result = {}
    for key, value in conn.execute("SELECT key,value FROM artifact_metadata"):
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return {k: result.get(k) for k in MEMORY_KEYS}


def build_presence_index(graph_path, output, ontology, retrieval_cfg, *, overwrite=False):
    """One offline pass over frozen events. No graph edges, LLM, images or reports."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Presence index exists; use --overwrite deliberately: {output}")
    if output.resolve() == Path(graph_path).resolve():
        raise ValueError("Presence index must not overwrite the event database")
    cfg = settings(retrieval_cfg)
    conn = sqlite3.connect("file:" + Path(graph_path).absolute().as_posix() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("BEGIN")
    try:
        cases = list(conn.execute("SELECT case_id,split FROM cases ORDER BY case_id"))
        if any(r["split"] != "train" for r in cases):
            raise ValueError("Presence index requires train-only memory")
        case_ids = [r["case_id"] for r in cases]
        findings = sorted(ontology.extractable_findings)
        ci, fi = ({v: i for i, v in enumerate(ids)} for ids in (case_ids, findings))
        states = np.zeros((len(ci), len(fi)), dtype=np.uint8)
        sources = []
        additions = []
        restored_counts, blocked, stats = Counter(), Counter(), Counter()
        memory_identity = _metadata(conn)
        if memory_identity.get("ontology_sha256") not in {None, ontology.ontology_hash}:
            raise ValueError("Presence index ontology differs from the frozen memory")
        for row in conn.execute("SELECT * FROM events"):
            raw = dict(row)
            raw["characteristics"] = json.loads(raw.pop("characteristics_json") or "{}")
            raw["validation_errors"] = json.loads(raw.pop("validation_errors_json") or "[]")
            event = Event.model_validate(raw)
            stats["events_scanned"] += 1
            if event.finding not in fi:
                continue
            c, f = ci[event.case_id], fi[event.finding]
            states[c, f] |= OBSERVED
            if event.presence == "present" and event.temporal_status not in {"prior_only", "resolved"}:
                states[c, f] |= ACTIVE_POSITIVE
                if event.uncertainty != "definite":
                    states[c, f] |= UNCERTAIN
            if event.presence == "absent" and event.temporal_status == "current":
                states[c, f] |= CURRENT_ABSENT
                if not event.retrieval_eligible and ontology.retrieval_policy(event.finding) == "clinical":
                    stats["current_ineligible_clinical_absent"] += 1
            if (not event.retrieval_eligible and event.presence == "absent"
                    and "absence_not_in_pilot_allowlist" in event.validation_errors
                    and ontology.retrieval_policy(event.finding) == "clinical"):
                if cfg["restore_pilot_absences"] and _quality(event, ontology, cfg, {"absence_not_in_pilot_allowlist"}):
                    additions.append((c, f, event.event_id, "restored_explicit_absent"))
                    states[c, f] |= REOPENED
                    restored_counts[event.finding] += 1
                else:
                    blocked.update(set(event.validation_errors) - NOTES - {"absence_not_in_pilot_allowlist"} or {"quality_or_policy"})
            if (cfg["infer_from_normal_descriptors"] and event.finding in NORMAL_RULES
                    and event.presence == "present" and event.uncertainty == "definite"
                    and event.laterality in {"unspecified", "bilateral", "not_applicable"}
                    and event.anatomy in NORMAL_SOURCE_ANATOMIES[event.finding]
                    and _quality(event, ontology, cfg, {"context_only_finding"})):
                sources.append((c, event))
        # Block inference when a regional normal descriptor coexists with an
        # active positive (including uncertain/ineligible positives) in its region.
        regions = {"normal_lung_transparency": "lung", "normal_cardiac_silhouette": "cardiac_silhouette", "intact_osseous_structures": "bone"}
        conflicts = {}
        for source, region in regions.items():
            targets = NORMAL_RULES[source]
            conflicts[source] = [fi[f] for f in ontology.clinical_findings if f in fi and (
                f in targets or ontology.default_anatomy(f) == region
                or ontology.is_ancestor(region, ontology.default_anatomy(f), "anatomy")
                or any(ontology.finding_family(f) == ontology.finding_family(t) for t in targets)
            )]
        inferred_counts = Counter()
        used = set()
        for c, source in sorted(sources, key=lambda x: (-x[1].confidence, x[1].event_id)):
            if any(states[c, j] & ACTIVE_POSITIVE for j in conflicts[source.finding]):
                stats["normal_sources_blocked_by_positive_or_uncertain"] += 1
                continue
            for target in NORMAL_RULES[source.finding]:
                if target not in fi:
                    continue
                f = fi[target]
                # Never override a recorded assertion, including an invalid,
                # historical, uncertain, or explicitly not-mentioned assertion.
                if states[c, f] & OBSERVED or (c, f) in used:
                    continue
                additions.append((c, f, source.event_id, "inferred_absent"))
                states[c, f] |= INFERRED
                inferred_counts[target] += 1
                used.add((c, f))
        additions.sort()
        summary = {
            **dict(stats), "case_count": len(ci), "finding_count": len(fi),
            "state_bytes": int(states.nbytes), "pointer_count": len(additions),
            "restored_explicit_absent_count": sum(restored_counts.values()),
            "current_clinical_absent_kept_blocked": stats["current_ineligible_clinical_absent"] - sum(restored_counts.values()),
            "restored_by_finding": dict(sorted(restored_counts.items())),
            "inferred_absent_count": sum(inferred_counts.values()),
            "inferred_by_finding": dict(sorted(inferred_counts.items())),
            "retained_block_reasons": dict(sorted(blocked.items())),
            "unmentioned_promoted_to_absent": 0,
        }
        meta = {"settings": cfg, "memory_identity": memory_identity,
                "ontology_sha256": ontology.ontology_hash, "summary": summary,
                "case_ids_sha256": stable_json_sha256(case_ids)}
    finally:
        conn.rollback()
        conn.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, states=states, case_ids=np.asarray(case_ids, dtype=str),
                                finding_ids=np.asarray(findings, dtype=str),
                                additions=np.asarray(additions, dtype=str).reshape(-1, 4),
                                metadata=np.asarray(json.dumps(meta, sort_keys=True)))
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return {**summary, "output": str(output), "file_bytes": output.stat().st_size, "sha256": file_sha256(output)}


class PresenceIndex:
    def __init__(self, path, graph, ontology, retrieval_cfg):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Build the offline presence index first: {path} (build-presence-index)")
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.states = data["states"].copy()
            cases, findings = data["case_ids"].tolist(), data["finding_ids"].tolist()
            rows = data["additions"].tolist()
        memory_meta = graph.artifact_metadata()
        if (self.metadata["settings"] != settings(retrieval_cfg)
                or self.metadata["ontology_sha256"] != ontology.ontology_hash
                or self.metadata["memory_identity"] != {k: memory_meta.get(k) for k in MEMORY_KEYS}
                or self.metadata["case_ids_sha256"] != stable_json_sha256(graph.case_ids())
                or self.metadata["summary"]["events_scanned"] != graph.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]):
            raise ValueError("Stale presence index: rebuild for the active memory/ontology/policy")
        self.sha256 = file_sha256(path)
        self._rows = defaultdict(list)
        self._virtual = {}
        for c, f, source, basis in rows:
            case_id, finding = cases[int(c)], findings[int(f)]
            self._rows[case_id].append((finding, source, basis))
            if basis == "inferred_absent":
                self._virtual[f"presence_index:{case_id}:{finding}"] = (finding, source)

    def event(self, event_id, graph, ontology):
        if event_id not in self._virtual:
            return graph.get_event(event_id)
        finding, source_id = self._virtual[event_id]
        source = graph.get_event(source_id)
        return source.model_copy(update={
            "event_id": event_id, "finding": finding, "anatomy": ontology.default_anatomy(finding),
            "laterality": "unspecified", "presence": "absent", "severity": "not_applicable",
            "uncertainty": "unspecified", "confidence": min(source.confidence, .85),
            "retrieval_eligible": True, "validation_errors": [],
            "characteristics": {"_semantic_support": source.characteristics.get("_semantic_support", {}),
                "_presence_support": {"basis": "inferred_absent", "source_event_id": source_id,
                    "source_finding": source.finding, "policy_version": POLICY_VERSION}},
        })

    def additional_events(self, case_ids, findings, spec, graph, ontology):
        if spec.target_scope != "finding" or spec.target_attribute != "presence" or spec.comparison_mode != "attribute_axis":
            return []
        wanted = set(findings)
        result = []
        for case_id in case_ids:
            for finding, source_id, basis in self._rows.get(case_id, []):
                if finding not in wanted:
                    continue
                if basis == "inferred_absent":
                    if finding != spec.target_finding:
                        continue
                    result.append(self.event(f"presence_index:{case_id}:{finding}", graph, ontology))
                else:
                    source = graph.get_event(source_id)
                    chars = dict(source.characteristics)
                    chars["_presence_support"] = {"basis": basis, "source_event_id": source_id,
                                                   "source_finding": finding, "policy_version": POLICY_VERSION}
                    result.append(source.model_copy(update={"retrieval_eligible": True, "characteristics": chars}))
        return result
