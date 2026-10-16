"""Deterministic smoke for current pool retrieval and legacy graph storage."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .extraction import find_exact_span, section_for_offset
from .graph.store import GraphStore
from .ontology import Ontology
from .retrieval.attribute_selector import AttributeSelectorAgent
from .retrieval.engine import GraphRetriever
from .schemas import CaseRecord, Event, VisualEdge


class _FakeEncoder:
    def encode_one(self, _: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class _FakeVisualStore:
    def __init__(self, vectors: dict[str, tuple[float, float]]):
        self.case_ids = list(vectors)
        self.id_to_row = {case_id: index for index, case_id in enumerate(self.case_ids)}
        matrix = np.asarray([vectors[case_id] for case_id in self.case_ids], dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix

    @staticmethod
    def _norm(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def search(self, query_embedding: np.ndarray, k: int) -> list[tuple[str, float]]:
        query = self._norm(query_embedding)
        scores = self.embeddings @ query
        order = np.argsort(-scores)[:k]
        return [(self.case_ids[int(i)], float(scores[int(i)])) for i in order]

    def similarity_to_query(self, query_embedding: np.ndarray, case_id: str) -> float:
        query = self._norm(query_embedding)
        return float(self.embeddings[self.id_to_row[case_id]] @ query)

    def similarity_between(self, case_a: str, case_b: str) -> float:
        return float(
            self.embeddings[self.id_to_row[case_a]]
            @ self.embeddings[self.id_to_row[case_b]]
        )


def _case(case_id: str, subject_id: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        dataset="fixture",
        subject_id=subject_id,
        study_id=case_id,
        dicom_id=f"{case_id}-d",
        image_path=f"/{case_id}.png",
        report_path=f"/{case_id}.txt",
        view="PA",
        split="train",
    )


def _event(case_id: str, presence: str, text: str) -> Event:
    return Event(
        case_id=case_id,
        event_id=f"{case_id}:effusion",
        finding="pleural_effusion",
        anatomy="pleural_space",
        laterality="left",
        presence=presence,  # type: ignore[arg-type]
        severity="mild" if presence == "present" else "not_applicable",
        uncertainty="definite",
        confidence=0.95,
        evidence_span=text,
        section="findings",
        retrieval_eligible=True,
    )


def self_test(ontology_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    ontology = Ontology(ontology_path or root / "ontology.yaml")

    report = "FINDINGS: Small left pleural effusion. IMPRESSION: No pneumothorax."
    span = "Small left pleural effusion."
    offsets = find_exact_span(report, span)
    assert offsets is not None
    assert section_for_offset(report, offsets[0]) == "findings"

    selector = AttributeSelectorAgent(ontology, {"backend": "rules", "top_k_ontology": 3})
    spec = selector.select("Is there a left pleural effusion?")
    assert spec.target_attribute == "presence"
    assert set(spec.required_attribute_values) == {"present", "absent"}

    with tempfile.TemporaryDirectory(prefix="visig_cr-selftest-") as directory:
        graph = GraphStore(Path(directory) / "memory.sqlite")
        cases = [_case("c1", "p1"), _case("c2", "p2"), _case("c3", "p3")]
        graph.import_cases(cases)
        graph.import_events(
            [
                _event("c1", "present", "Small left pleural effusion."),
                _event("c2", "absent", "No left pleural effusion."),
                _event("c3", "present", "Mild left pleural effusion."),
                _event("c2", "absent", "No pneumothorax.").model_copy(
                    update={
                        "event_id": "c2:pneumothorax",
                        "finding": "pneumothorax",
                    }
                ),
            ]
        )
        graph.add_visual_edges(
            [
                VisualEdge(
                    source_case_id="c1",
                    target_case_id="c2",
                    similarity=0.95,
                    source_rank=1,
                    is_mutual=True,
                ),
                VisualEdge(
                    source_case_id="c2",
                    target_case_id="c1",
                    similarity=0.95,
                    source_rank=1,
                    is_mutual=True,
                ),
                VisualEdge(
                    source_case_id="c1",
                    target_case_id="c3",
                    similarity=0.92,
                    source_rank=2,
                    is_mutual=True,
                ),
                VisualEdge(
                    source_case_id="c3",
                    target_case_id="c1",
                    similarity=0.92,
                    source_rank=1,
                    is_mutual=True,
                ),
            ]
        )

        relation_stats = graph.build_relation_edges(
            ontology,
            {
                "minimum_event_confidence": 0.5,
                "include_conditional_events": True,
                "allow_same_anatomy_different_finding": True,
                "minimum_event_relation_score": 0.0,
                "max_matches_per_event": 3,
                "max_event_relations_per_edge": 32,
                "commit_every_edges": 100,
            },
        )
        relation_edge = graph.get_relation_edge("c1", "c2")
        assert relation_edge is not None
        assert any(
            "presence_contrast" in relation.relation_types
            for relation in relation_edge.matched_events
        )
        assert "c2:pneumothorax" in relation_edge.no_matched_events["c2"]

        visual = _FakeVisualStore(
            {"c1": (1.0, 0.0), "c2": (0.96, 0.04), "c3": (0.93, 0.07)}
        )
        retriever = GraphRetriever(
            ontology,
            graph,
            visual,  # type: ignore[arg-type]
            _FakeEncoder(),  # type: ignore[arg-type]
            {
                "k_seed": 1,
                "evidence_k": 4,
                "boundary_hops": 1,
                "boundary_neighbors_per_hop": 4,
                "relation_candidate_pool": 4,
                "relation_neighbors_per_hop": 4,
                "minimum_relation_route_score": 0.0,
                "allow_visual_fallback_when_relations_missing": True,
                "boundary_max_candidates_per_seed": 8,
                "boundary_max_cases": 8,
                "boundary_minimum_edge_similarity": 0.0,
                "minimum_event_confidence": 0.5,
                "minimum_target_match": 0.5,
                "tau_query": 0.0,
                "tau_boundary": 0.0,
                "group_candidates_per_value": 8,
                "minimum_group_non_target_match": 0.5,
                "minimum_group_visual_similarity": 0.0,
                "minimum_group_quality": 0.0,
                "candidate_weights": {},
                "set_weights": {},
            },
            selector,
        )
        package = retriever.retrieve(
            "/query.png",
            "Is there a left pleural effusion?",
            query_id="selftest",
            method="visig_cr",
        )
        values = {item.attribute_value for item in package.evidence}
        assert {"present", "absent"}.issubset(values)
        assert package.semantic_contrast_groups
        assert package.coverage_status["semantic_contrast_complete"] is True
        assert package.boundary_stats["graph_used"] is False
        assert all(group.pair_source == "question_conditioned_pool" for group in package.semantic_contrast_groups)

        tables = {
            row[0]
            for row in graph.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "visual_edges" in tables
        assert "relation_edges" in tables
        assert "typed_edges" not in tables
        graph.close()

    return {
        "ok": True,
        "ontology_sha256": ontology.ontology_hash,
        "selected_target_attribute": spec.target_attribute,
        "required_attribute_values": spec.required_attribute_values,
        "delivered_attribute_values": sorted(values),
        "semantic_contrast_group_count": len(package.semantic_contrast_groups),
        "relation_edges": relation_stats["relation_edges_inserted"],
        "relation_graph_schema": True,
        "main_retrieval_policy": package.audit_metadata["retrieval_policy"],
        "main_uses_graph": False,
    }
