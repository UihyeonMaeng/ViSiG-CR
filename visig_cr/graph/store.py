"""SQLite storage for cases, events, visual edges, and offline relation edges."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from tqdm import tqdm

from ..io_utils import ensure_parent
from ..ontology import Ontology
from ..schemas import CaseRecord, CaseRelationEdge, Event, VisualEdge, VisualPath
from .relations import RELATION_SCHEMA_VERSION, build_case_relation_edge
from time import perf_counter
from ..visual import VisualStore


class VisualCaseGraphStore:
    """Persistent train-memory store with relation-annotated visual edges.

    ``visual_edges`` defines the bounded topology. ``relation_edges`` attaches a
    question-independent event-relation bundle to each visual neighbor pair.
    Online retrieval prioritizes these atomic relations and performs final exact
    validation against the event store.
    """

    def __init__(self, db_path: str | Path):
        self.path = ensure_parent(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def close(self) -> None:
        self.conn.close()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                dataset TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                study_id TEXT NOT NULL,
                dicom_id TEXT NOT NULL,
                image_path TEXT NOT NULL,
                report_path TEXT NOT NULL,
                view TEXT,
                split TEXT NOT NULL,
                normality TEXT
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                finding TEXT NOT NULL,
                anatomy TEXT NOT NULL,
                laterality TEXT NOT NULL,
                presence TEXT NOT NULL,
                severity TEXT NOT NULL,
                uncertainty TEXT NOT NULL,
                confidence REAL NOT NULL,
                characteristics_json TEXT NOT NULL,
                evidence_span TEXT,
                section TEXT,
                start_char INTEGER,
                end_char INTEGER,
                temporal_status TEXT,
                retrieval_eligible INTEGER NOT NULL,
                validation_errors_json TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            );
            CREATE TABLE IF NOT EXISTS visual_edges (
                source_case_id TEXT NOT NULL,
                target_case_id TEXT NOT NULL,
                similarity REAL NOT NULL,
                source_rank INTEGER NOT NULL,
                is_mutual INTEGER NOT NULL,
                is_backfill INTEGER NOT NULL,
                PRIMARY KEY(source_case_id, target_case_id)
            );
            CREATE TABLE IF NOT EXISTS relation_edges (
                source_case_id TEXT NOT NULL,
                target_case_id TEXT NOT NULL,
                visual_similarity REAL NOT NULL,
                source_rank INTEGER NOT NULL,
                is_mutual INTEGER NOT NULL,
                is_backfill INTEGER NOT NULL,
                relation_score REAL NOT NULL,
                relation_types_json TEXT NOT NULL,
                matched_events_json TEXT NOT NULL,
                no_matched_events_json TEXT NOT NULL,
                matched_event_count INTEGER NOT NULL,
                source_unmatched_count INTEGER NOT NULL,
                target_unmatched_count INTEGER NOT NULL,
                ontology_sha256 TEXT NOT NULL,
                relation_schema_version TEXT NOT NULL,
                PRIMARY KEY(source_case_id, target_case_id),
                FOREIGN KEY(source_case_id, target_case_id)
                    REFERENCES visual_edges(source_case_id, target_case_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_case ON events(case_id);
            CREATE INDEX IF NOT EXISTS idx_events_lookup
                ON events(finding, anatomy, laterality, presence, severity);
            CREATE INDEX IF NOT EXISTS idx_visual_edges_source
                ON visual_edges(source_case_id, source_rank);
            CREATE INDEX IF NOT EXISTS idx_visual_edges_target
                ON visual_edges(target_case_id);
            CREATE INDEX IF NOT EXISTS idx_relation_edges_source
                ON relation_edges(source_case_id, relation_score DESC, source_rank);
            CREATE INDEX IF NOT EXISTS idx_relation_edges_target
                ON relation_edges(target_case_id);
            CREATE TABLE IF NOT EXISTS artifact_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        # Backward-compatible migration for graph databases created before the
        # case-level normality fallback was introduced.
        case_columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(cases)").fetchall()
        }
        if "normality" not in case_columns:
            self.conn.execute("ALTER TABLE cases ADD COLUMN normality TEXT")
        if "normality_provenance_json" not in case_columns:
            self.conn.execute("ALTER TABLE cases ADD COLUMN normality_provenance_json TEXT NOT NULL DEFAULT '{}'")
        relation_columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(relation_edges)")
        }
        if "normality_relation_json" not in relation_columns:
            self.conn.execute("ALTER TABLE relation_edges ADD COLUMN normality_relation_json TEXT NOT NULL DEFAULT '{}'")
        self.conn.commit()

    def reset_memory(self) -> None:
        """Remove every corpus-dependent row before a clean graph rebuild."""

        with self.conn:
            self.conn.execute("DELETE FROM relation_edges")
            self.conn.execute("DELETE FROM visual_edges")
            self.conn.execute("DELETE FROM events")
            self.conn.execute("DELETE FROM cases")
            self.conn.execute("DELETE FROM artifact_metadata")
        self.get_case.cache_clear()
        self.get_events.cache_clear()
        self.get_event.cache_clear()
        self.visual_in_degree.cache_clear()
        self.visual_out_degree.cache_clear()
        self.get_relation_edge.cache_clear()

    def set_artifact_metadata(self, values: dict[str, Any]) -> None:
        rows = [(str(key), json.dumps(value, ensure_ascii=False)) for key, value in values.items()]
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO artifact_metadata(key,value) VALUES (?,?)",
                rows,
            )

    def artifact_metadata(self) -> dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key,value FROM artifact_metadata ORDER BY key"
        ).fetchall()
        result: dict[str, Any] = {}
        for row in rows:
            try:
                result[str(row[0])] = json.loads(str(row[1]))
            except json.JSONDecodeError:
                result[str(row[0])] = str(row[1])
        return result

    def assert_artifact_metadata(
        self,
        expected: dict[str, Any],
        *,
        keys: Sequence[str] | None = None,
    ) -> None:
        actual = self.artifact_metadata()
        check_keys = list(keys or expected.keys())
        mismatches = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in check_keys
            if actual.get(key) != expected.get(key)
        }
        if mismatches:
            raise ValueError(
                "Graph artifact metadata does not match the active corpus: "
                + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
            )

    # ------------------------------------------------------------------
    # Cases and ontology events
    # ------------------------------------------------------------------

    def import_cases(self, cases: Iterable[CaseRecord]) -> int:
        rows = [
            (
                case.case_id,
                case.dataset,
                case.subject_id,
                case.study_id,
                case.dicom_id,
                case.image_path,
                case.report_path,
                case.view,
                case.split,
            )
            for case in cases
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO cases
            (case_id,dataset,subject_id,study_id,dicom_id,image_path,report_path,view,split)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        self.get_case.cache_clear()
        return len(rows)

    def import_case_normalities(self, normalities: dict[str, str]) -> int:
        """Attach deterministic case-level normality labels to imported cases."""

        rows = [
            (str(value), str(case_id))
            for case_id, value in normalities.items()
            if str(value) in {"normal", "abnormal"}
        ]
        if not rows:
            return 0
        self.conn.executemany(
            "UPDATE cases SET normality=? WHERE case_id=?",
            rows,
        )
        self.conn.commit()
        self.get_case.cache_clear()
        return len(rows)

    def import_normality_provenance(self, metadata: dict[str, dict[str, Any]]) -> int:
        rows = [(json.dumps(value, ensure_ascii=False), str(case_id))
                for case_id, value in metadata.items()]
        with self.conn:
            self.conn.executemany(
                "UPDATE cases SET normality_provenance_json=? WHERE case_id=?", rows
            )
        return len(rows)

    def normality_provenance(self, case_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT normality_provenance_json FROM cases WHERE case_id=?", (case_id,)
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return json.loads(row[0] or "{}")

    def import_events(self, events: Iterable[Event]) -> int:
        rows = [
            (
                event.event_id,
                event.case_id,
                event.finding,
                event.anatomy,
                event.laterality,
                event.presence,
                event.severity,
                event.uncertainty,
                event.confidence,
                json.dumps(event.characteristics, ensure_ascii=False),
                event.evidence_span,
                event.section,
                event.start_char,
                event.end_char,
                event.temporal_status,
                int(event.retrieval_eligible),
                json.dumps(event.validation_errors, ensure_ascii=False),
            )
            for event in events
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO events
            (event_id,case_id,finding,anatomy,laterality,presence,severity,uncertainty,
             confidence,characteristics_json,evidence_span,section,start_char,end_char,
             temporal_status,retrieval_eligible,validation_errors_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        self.conn.commit()
        self.get_events.cache_clear()
        self.get_event.cache_clear()
        return len(rows)

    def case_ids(self, split: str | None = None) -> list[str]:
        if split:
            rows = self.conn.execute(
                "SELECT case_id FROM cases WHERE split=? ORDER BY case_id",
                (split,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT case_id FROM cases ORDER BY case_id"
            ).fetchall()
        return [str(row[0]) for row in rows]

    @lru_cache(maxsize=32768)
    def get_case(self, case_id: str) -> CaseRecord:
        row = self.conn.execute(
            "SELECT * FROM cases WHERE case_id=?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        return CaseRecord.model_validate(dict(row))

    def case_normalities_in_cases(self, case_ids: Sequence[str]) -> dict[str, str]:
        """Return known normality labels for the requested case IDs."""

        if not case_ids:
            return {}
        placeholders = ",".join("?" for _ in case_ids)
        rows = self.conn.execute(
            f"SELECT case_id,normality FROM cases WHERE case_id IN ({placeholders}) "
            "AND normality IN ('normal','abnormal')",
            [str(case_id) for case_id in case_ids],
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def all_case_normalities(self, *, split: str = "train") -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT case_id,normality FROM cases WHERE split=? "
            "AND normality IN ('normal','abnormal') ORDER BY case_id",
            (split,),
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    @lru_cache(maxsize=8192)
    def get_events(self, case_id: str) -> tuple[Event, ...]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE case_id=? ORDER BY event_id",
            (case_id,),
        ).fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    @lru_cache(maxsize=32768)
    def get_event(self, event_id: str) -> Event:
        row = self.conn.execute(
            "SELECT * FROM events WHERE event_id=?",
            (event_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._row_to_event(row)

    def events_in_cases(
        self,
        case_ids: Sequence[str],
        *,
        minimum_confidence: float = 0.0,
        retrieval_eligible_only: bool = True,
        findings: Sequence[str] | None = None,
        event_ids: Sequence[str] | None = None,
    ) -> list[Event]:
        if not case_ids:
            return []
        placeholders = ",".join("?" for _ in case_ids)
        clauses = [f"case_id IN ({placeholders})", "confidence>=?"]
        params: list[Any] = [*case_ids, float(minimum_confidence)]
        if retrieval_eligible_only:
            clauses.append("retrieval_eligible=1")
        if findings:
            finding_placeholders = ",".join("?" for _ in findings)
            clauses.append(f"finding IN ({finding_placeholders})")
            params.extend(str(value) for value in findings)
        if event_ids:
            event_placeholders = ",".join("?" for _ in event_ids)
            clauses.append(f"event_id IN ({event_placeholders})")
            params.extend(str(value) for value in event_ids)
        query = "SELECT * FROM events WHERE " + " AND ".join(clauses)
        return [self._row_to_event(row) for row in self.conn.execute(query, params)]

    def all_events(
        self,
        *,
        split: str = "train",
        minimum_confidence: float = 0.0,
        retrieval_eligible_only: bool = True,
    ) -> list[Event]:
        clauses = ["c.split=?", "e.confidence>=?"]
        params: list[Any] = [split, float(minimum_confidence)]
        if retrieval_eligible_only:
            clauses.append("e.retrieval_eligible=1")
        query = (
            "SELECT e.* FROM events e JOIN cases c ON c.case_id=e.case_id WHERE "
            + " AND ".join(clauses)
        )
        return [self._row_to_event(row) for row in self.conn.execute(query, params)]

    def lookup_events(
        self,
        *,
        finding: str,
        anatomy: str = "unspecified",
        laterality: str = "unspecified",
        minimum_confidence: float = 0.0,
        split: str = "train",
    ) -> list[Event]:
        clauses = [
            "e.finding=?",
            "e.confidence>=?",
            "e.retrieval_eligible=1",
            "c.split=?",
        ]
        params: list[Any] = [finding, float(minimum_confidence), split]
        if anatomy not in {"", "unspecified", "not_applicable"}:
            clauses.append("e.anatomy=?")
            params.append(anatomy)
        if laterality not in {"", "unspecified", "not_applicable"}:
            clauses.append("e.laterality IN (?, 'bilateral', 'unspecified')")
            params.append(laterality)
        query = (
            "SELECT e.* FROM events e JOIN cases c ON c.case_id=e.case_id WHERE "
            + " AND ".join(clauses)
        )
        return [self._row_to_event(row) for row in self.conn.execute(query, params)]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        data = dict(row)
        data["characteristics"] = json.loads(data.pop("characteristics_json") or "{}")
        data["validation_errors"] = json.loads(
            data.pop("validation_errors_json") or "[]"
        )
        data["retrieval_eligible"] = bool(data["retrieval_eligible"])
        return Event.model_validate(data)

    def subject_id(self, case_id: str) -> str:
        row = self.conn.execute(
            "SELECT subject_id FROM cases WHERE case_id=?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise KeyError(case_id)
        return str(row[0])

    # ------------------------------------------------------------------
    # Visual case graph
    # ------------------------------------------------------------------

    def clear_visual_edges(self) -> None:
        self.set_artifact_metadata({"relation_build_complete": False})
        self.conn.execute("DELETE FROM relation_edges")
        self.conn.execute("DELETE FROM visual_edges")
        self.conn.commit()
        self.visual_in_degree.cache_clear()
        self.visual_out_degree.cache_clear()
        self.get_relation_edge.cache_clear()

    def add_visual_edges(self, edges: Iterable[VisualEdge]) -> int:
        rows = [
            (
                edge.source_case_id,
                edge.target_case_id,
                edge.similarity,
                edge.source_rank,
                int(edge.is_mutual),
                int(edge.is_backfill),
            )
            for edge in edges
        ]
        self.conn.executemany(
            """INSERT OR REPLACE INTO visual_edges
            (source_case_id,target_case_id,similarity,source_rank,is_mutual,is_backfill)
            VALUES (?,?,?,?,?,?)""",
            rows,
        )
        if rows:
            self.visual_in_degree.cache_clear()
            self.visual_out_degree.cache_clear()
        return len(rows)

    def build_visual_edges(
        self,
        visual: VisualStore,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the directed bounded-outdegree visual-similarity graph."""

        split = str(cfg.get("memory_split", "train"))
        candidate_pool = max(1, int(cfg.get("candidate_pool", 96)))
        k_min = max(1, int(cfg.get("k_min", 4)))
        k_max = max(k_min, int(cfg.get("k_max", 16)))
        candidate_pool = max(candidate_pool, k_max)
        min_similarity = float(cfg.get("min_similarity", 0.20))
        outlier_audit_similarity = float(cfg.get("outlier_audit_similarity", 0.10))
        exclude_same_subject = bool(cfg.get("exclude_same_subject", True))
        commit_every = max(1, int(cfg.get("commit_every_cases", 500)))

        self.clear_visual_edges()
        case_ids = [
            case_id
            for case_id in self.case_ids(split)
            if case_id in visual.id_to_row
        ]
        eligible = set(case_ids)
        selected_by_source: dict[str, list[tuple[str, float, int, bool]]] = {}
        weak_nodes: list[str] = []
        outlier_nodes: list[str] = []
        isolated_nodes: list[str] = []
        shortage_nodes: list[str] = []
        total_indexed = len(getattr(visual, "id_to_row", {}))

        for source_case in tqdm(case_ids, desc="Building visual case graph"):
            source_subject = self.subject_id(source_case)
            candidates: list[tuple[str, float]] = []
            search_k = min(total_indexed, max(candidate_pool + 1, k_max + 1))
            while search_k > 0:
                candidates = []
                seen: set[str] = set()
                for target_case, similarity in visual.neighbors_for_case(
                    source_case,
                    search_k,
                ):
                    if target_case == source_case or target_case in seen:
                        continue
                    if target_case not in eligible:
                        continue
                    if (
                        exclude_same_subject
                        and self.subject_id(target_case) == source_subject
                    ):
                        continue
                    seen.add(target_case)
                    candidates.append((target_case, float(similarity)))
                    if len(candidates) >= candidate_pool:
                        break
                if len(candidates) >= candidate_pool or search_k >= total_indexed:
                    break
                search_k = min(total_indexed, max(search_k + 1, search_k * 2))

            if not candidates:
                selected_by_source[source_case] = []
                isolated_nodes.append(source_case)
                shortage_nodes.append(source_case)
                continue

            if candidates[0][1] < outlier_audit_similarity:
                outlier_nodes.append(source_case)

            selected: list[tuple[str, float, int, bool]] = []
            for rank, (target_case, similarity) in enumerate(candidates, 1):
                mandatory = rank <= k_min
                if mandatory or (rank <= k_max and similarity >= min_similarity):
                    selected.append(
                        (
                            target_case,
                            similarity,
                            rank,
                            mandatory and similarity < min_similarity,
                        )
                    )
                if rank >= k_max:
                    break
            if any(item[3] for item in selected):
                weak_nodes.append(source_case)
            if len(selected) < k_min:
                shortage_nodes.append(source_case)
            selected_by_source[source_case] = selected

        pair_set = {
            (source, target)
            for source, items in selected_by_source.items()
            for target, _, _, _ in items
        }
        inserted = 0
        for index, source_case in enumerate(case_ids, 1):
            rows = [
                VisualEdge(
                    source_case_id=source_case,
                    target_case_id=target,
                    similarity=similarity,
                    source_rank=rank,
                    is_mutual=(target, source_case) in pair_set,
                    is_backfill=is_backfill,
                )
                for target, similarity, rank, is_backfill in selected_by_source[
                    source_case
                ]
            ]
            inserted += self.add_visual_edges(rows)
            if index % commit_every == 0:
                self.conn.commit()
        self.conn.commit()

        stats = self.graph_stats(split=split)
        stats.update(
            {
                "candidate_pool": candidate_pool,
                "k_min": k_min,
                "k_max": k_max,
                "min_similarity": min_similarity,
                "weak_backfill_node_count": len(weak_nodes),
                "outlier_audit_node_count": len(outlier_nodes),
                "isolated_node_count": len(isolated_nodes),
                "candidate_shortage_node_count": len(shortage_nodes),
                "visual_edges_inserted": inserted,
                "outlier_audit_examples": outlier_nodes[:20],
                "isolated_examples": isolated_nodes[:20],
            }
        )
        return stats

    # ------------------------------------------------------------------
    # Offline event-relation annotations on visual edges
    # ------------------------------------------------------------------

    def clear_relation_edges(self) -> None:
        self.set_artifact_metadata({"relation_build_complete": False})
        self.conn.execute("DELETE FROM relation_edges")
        self.conn.commit()
        self.get_relation_edge.cache_clear()

    def add_relation_edges(self, edges: Iterable[CaseRelationEdge]) -> int:
        rows = []
        for edge in edges:
            rows.append(
                (
                    edge.source_case_id,
                    edge.target_case_id,
                    edge.visual_similarity,
                    edge.source_rank,
                    int(edge.is_mutual),
                    int(edge.is_backfill),
                    edge.relation_score,
                    json.dumps(edge.relation_types, ensure_ascii=False),
                    json.dumps(
                        [item.model_dump() for item in edge.matched_events],
                        ensure_ascii=False,
                    ),
                    json.dumps(edge.no_matched_events, ensure_ascii=False),
                    len(edge.matched_events),
                    len(edge.source_unmatched_event_ids()),
                    len(edge.target_unmatched_event_ids()),
                    edge.ontology_sha256,
                    edge.relation_schema_version,
                    json.dumps(edge.normality_relation.model_dump(), ensure_ascii=False),
                )
            )
        self.conn.executemany(
            """INSERT OR REPLACE INTO relation_edges
            (source_case_id,target_case_id,visual_similarity,source_rank,is_mutual,
             is_backfill,relation_score,relation_types_json,matched_events_json,
             no_matched_events_json,matched_event_count,source_unmatched_count,
             target_unmatched_count,ontology_sha256,relation_schema_version,normality_relation_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        if rows:
            self.get_relation_edge.cache_clear()
        return len(rows)

    def build_relation_edges(
        self,
        ontology: Ontology,
        cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Materialize event-level relation bundles for every visual edge."""

        started = perf_counter()
        self.set_artifact_metadata({"relation_build_complete": False})
        self.clear_relation_edges()
        normalities = self.all_case_normalities()
        edge_count = int(
            self.conn.execute("SELECT COUNT(*) FROM visual_edges").fetchone()[0]
        )
        cursor = self.conn.execute(
            """SELECT * FROM visual_edges
            ORDER BY source_case_id, source_rank, target_case_id"""
        )
        commit_every = max(1, int(cfg.get("commit_every_edges", 5000)))
        relation_type_counts: dict[str, int] = {}
        matched_event_relations = 0
        source_unmatched = 0
        target_unmatched = 0
        batch: list[CaseRelationEdge] = []
        inserted = 0
        for index, row in enumerate(
            tqdm(
                cursor,
                total=edge_count,
                desc="Building event relation edges",
            ),
            1,
        ):
            visual_edge = VisualEdge(
                source_case_id=str(row["source_case_id"]),
                target_case_id=str(row["target_case_id"]),
                similarity=float(row["similarity"]),
                source_rank=int(row["source_rank"]),
                is_mutual=bool(row["is_mutual"]),
                is_backfill=bool(row["is_backfill"]),
            )
            relation_edge = build_case_relation_edge(
                visual_edge,
                self.get_events(visual_edge.source_case_id),
                self.get_events(visual_edge.target_case_id),
                ontology,
                cfg,
                source_normality=normalities.get(visual_edge.source_case_id),
                target_normality=normalities.get(visual_edge.target_case_id),
            )
            batch.append(relation_edge)
            matched_event_relations += len(relation_edge.matched_events)
            source_unmatched += len(relation_edge.source_unmatched_event_ids())
            target_unmatched += len(relation_edge.target_unmatched_event_ids())
            for relation_type in relation_edge.relation_types:
                relation_type_counts[relation_type] = (
                    relation_type_counts.get(relation_type, 0) + 1
                )
            if len(batch) >= commit_every:
                inserted += self.add_relation_edges(batch)
                self.conn.commit()
                batch = []
        if batch:
            inserted += self.add_relation_edges(batch)
        self.conn.commit()
        self.set_artifact_metadata({"relation_schema_version": RELATION_SCHEMA_VERSION, "relation_build_complete": True})
        return {
            "visual_edges_considered": edge_count,
            "relation_edges_inserted": inserted,
            "matched_event_relations": matched_event_relations,
            "source_no_matched_event_references": source_unmatched,
            "target_no_matched_event_references": target_unmatched,
            "relation_type_edge_counts": dict(sorted(relation_type_counts.items())),
            "ontology_sha256": ontology.ontology_hash,
            "relation_schema_version": RELATION_SCHEMA_VERSION,
            "relation_build_seconds": perf_counter() - started,
        }

    @lru_cache(maxsize=65536)
    def get_relation_edge(
        self,
        source_case_id: str,
        target_case_id: str,
    ) -> CaseRelationEdge | None:
        row = self.conn.execute(
            """SELECT * FROM relation_edges
            WHERE source_case_id=? AND target_case_id=?""",
            (source_case_id, target_case_id),
        ).fetchone()
        return None if row is None else self._row_to_relation_edge(row)

    def relation_between(
        self,
        case_a: str,
        case_b: str,
    ) -> CaseRelationEdge | None:
        return self.get_relation_edge(case_a, case_b) or self.get_relation_edge(
            case_b, case_a
        )

    def relation_edges_in_cases(self, case_ids: Sequence[str]) -> Iterable[CaseRelationEdge]:
        """Yield only stored directed edges with both endpoints in a bounded set.

        Source lookup uses the primary-key prefix; work is proportional to the
        stored out-degree, not the number of possible candidate-case pairs.
        """
        ids = sorted(set(case_ids))
        allowed = set(ids)
        for offset in range(0, len(ids), 400):
            chunk = ids[offset:offset + 400]
            placeholders = ",".join("?" for _ in chunk)
            cursor = self.conn.execute(
                f"SELECT * FROM relation_edges WHERE source_case_id IN ({placeholders}) "
                "ORDER BY source_case_id, source_rank, target_case_id", chunk
            )
            for row in cursor:
                if row["target_case_id"] in allowed:
                    yield self._row_to_relation_edge(row)

    def relation_neighbors(
        self,
        source_case_id: str,
        limit: int,
        *,
        minimum_visual_similarity: float = -1.0,
    ) -> list[CaseRelationEdge]:
        rows = self.conn.execute(
            """SELECT * FROM relation_edges
            WHERE source_case_id=? AND visual_similarity>=?
            ORDER BY relation_score DESC, visual_similarity DESC, source_rank ASC
            LIMIT ?""",
            (source_case_id, float(minimum_visual_similarity), int(limit)),
        ).fetchall()
        return [self._row_to_relation_edge(row) for row in rows]

    @staticmethod
    def _row_to_relation_edge(row: sqlite3.Row) -> CaseRelationEdge:
        data = dict(row)
        return CaseRelationEdge.model_validate(
            {
                "source_case_id": data["source_case_id"],
                "target_case_id": data["target_case_id"],
                "visual_similarity": data["visual_similarity"],
                "source_rank": data["source_rank"],
                "is_mutual": bool(data["is_mutual"]),
                "is_backfill": bool(data["is_backfill"]),
                "relation_score": data["relation_score"],
                "relation_types": json.loads(data["relation_types_json"] or "[]"),
                "matched_events": json.loads(data["matched_events_json"] or "[]"),
                "no_matched_events": json.loads(
                    data["no_matched_events_json"] or "{}"
                ),
                "ontology_sha256": data["ontology_sha256"],
                "relation_schema_version": data["relation_schema_version"],
                "normality_relation": json.loads(data.get("normality_relation_json") or "{}"),
            }
        )

    def visual_neighbors(
        self,
        source_case_id: str,
        limit: int,
        *,
        minimum_similarity: float = -1.0,
        mutual_first: bool = True,
    ) -> list[VisualEdge]:
        order = (
            "is_mutual DESC, similarity DESC, source_rank ASC"
            if mutual_first
            else "similarity DESC, source_rank ASC"
        )
        rows = self.conn.execute(
            f"""SELECT * FROM visual_edges
            WHERE source_case_id=? AND similarity>=?
            ORDER BY {order}
            LIMIT ?""",
            (source_case_id, float(minimum_similarity), int(limit)),
        ).fetchall()
        return [
            VisualEdge(
                source_case_id=str(row["source_case_id"]),
                target_case_id=str(row["target_case_id"]),
                similarity=float(row["similarity"]),
                source_rank=int(row["source_rank"]),
                is_mutual=bool(row["is_mutual"]),
                is_backfill=bool(row["is_backfill"]),
            )
            for row in rows
        ]

    def visual_neighborhood(
        self,
        seed_case_id: str,
        *,
        max_hops: int = 1,
        neighbors_per_hop: int = 12,
        max_candidates: int = 128,
        minimum_edge_similarity: float = -1.0,
        hop_decay: float = 0.85,
        mutual_first: bool = True,
    ) -> list[VisualPath]:
        if max_hops < 1 or max_hops > 3:
            raise ValueError("max_hops must be between 1 and 3")

        best: dict[str, tuple[list[str], list[float], float]] = {}
        frontier: deque[tuple[str, list[str], list[float], float]] = deque(
            [(seed_case_id, [seed_case_id], [], 1.0)]
        )

        for hop in range(1, max_hops + 1):
            next_frontier: list[tuple[str, list[str], list[float], float]] = []
            while frontier:
                current, nodes, edge_scores, _ = frontier.popleft()
                for edge in self.visual_neighbors(
                    current,
                    neighbors_per_hop,
                    minimum_similarity=minimum_edge_similarity,
                    mutual_first=mutual_first,
                ):
                    target = edge.target_case_id
                    if target in nodes:
                        continue
                    new_nodes = [*nodes, target]
                    new_edges = [*edge_scores, edge.similarity]
                    bottleneck = min(new_edges)
                    path_score = bottleneck * (hop_decay ** (hop - 1))
                    previous = best.get(target)
                    if previous is None or path_score > previous[2]:
                        best[target] = (new_nodes, new_edges, path_score)
                        next_frontier.append((target, new_nodes, new_edges, path_score))
            next_frontier.sort(key=lambda item: item[3], reverse=True)
            frontier = deque(next_frontier[:max_candidates])
            if not frontier:
                break

        paths = [
            VisualPath(
                seed_case_id=seed_case_id,
                target_case_id=target,
                nodes=nodes,
                hop_count=len(edge_scores),
                bottleneck_similarity=min(edge_scores),
                mean_similarity=float(sum(edge_scores) / len(edge_scores)),
                path_score=float(score),
            )
            for target, (nodes, edge_scores, score) in best.items()
        ]
        paths.sort(key=lambda item: (item.path_score, -item.hop_count), reverse=True)
        return paths[:max_candidates]

    @lru_cache(maxsize=65536)
    def visual_in_degree(self, case_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM visual_edges WHERE target_case_id=?",
            (case_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    @lru_cache(maxsize=65536)
    def visual_out_degree(self, case_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM visual_edges WHERE source_case_id=?",
            (case_id,),
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def graph_stats(self, *, split: str = "train") -> dict[str, Any]:
        case_ids = self.case_ids(split)
        if not case_ids:
            return {
                "visual_nodes": 0,
                "visual_edges": 0,
                "cases": 0,
                "events": 0,
                "retrieval_eligible_events": 0,
                "relation_edges": 0,
                "matched_event_relations": 0,
                "no_matched_event_references": 0,
            }

        out_rows = self.conn.execute(
            "SELECT source_case_id, COUNT(*) FROM visual_edges GROUP BY source_case_id"
        ).fetchall()
        in_rows = self.conn.execute(
            "SELECT target_case_id, COUNT(*) FROM visual_edges GROUP BY target_case_id"
        ).fetchall()
        out_degree = {str(row[0]): int(row[1]) for row in out_rows}
        in_degree = {str(row[0]): int(row[1]) for row in in_rows}
        out_values = np.asarray(
            [out_degree.get(case_id, 0) for case_id in case_ids],
            dtype=np.float32,
        )
        in_values = np.asarray(
            [in_degree.get(case_id, 0) for case_id in case_ids],
            dtype=np.float32,
        )
        edge_count = int(
            self.conn.execute("SELECT COUNT(*) FROM visual_edges").fetchone()[0]
        )
        mutual_count = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM visual_edges WHERE is_mutual=1"
            ).fetchone()[0]
        )
        backfill_count = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM visual_edges WHERE is_backfill=1"
            ).fetchone()[0]
        )
        relation_row = self.conn.execute(
            """SELECT COUNT(*), COALESCE(SUM(matched_event_count),0),
                      COALESCE(SUM(source_unmatched_count + target_unmatched_count),0),
                      COALESCE(AVG(relation_score),0.0)
               FROM relation_edges"""
        ).fetchone()
        relation_edge_count = int(relation_row[0]) if relation_row else 0
        matched_event_relation_count = int(relation_row[1]) if relation_row else 0
        no_matched_reference_count = int(relation_row[2]) if relation_row else 0
        mean_relation_score = float(relation_row[3]) if relation_row else 0.0
        return {
            "cases": int(self.conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]),
            "events": int(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
            "retrieval_eligible_events": int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM events WHERE retrieval_eligible=1"
                ).fetchone()[0]
            ),
            "visual_nodes": len(case_ids),
            "visual_edges": edge_count,
            "relation_edges": relation_edge_count,
            "matched_event_relations": matched_event_relation_count,
            "no_matched_event_references": no_matched_reference_count,
            "mean_relation_score": mean_relation_score,
            "visual_mutual_edge_fraction": mutual_count / max(1, edge_count),
            "visual_backfill_edge_fraction": backfill_count / max(1, edge_count),
            "visual_out_degree_min": int(out_values.min()),
            "visual_out_degree_mean": float(out_values.mean()),
            "visual_out_degree_p95": float(np.percentile(out_values, 95)),
            "visual_out_degree_max": int(out_values.max()),
            "visual_in_degree_mean": float(in_values.mean()),
            "visual_in_degree_p95": float(np.percentile(in_values, 95)),
            "visual_in_degree_max": int(in_values.max()),
            "visual_isolated_out_degree": int(np.sum(out_values == 0)),
            "visual_isolated_in_degree": int(np.sum(in_values == 0)),
        }


# Short public name used throughout the runtime.
GraphStore = VisualCaseGraphStore
