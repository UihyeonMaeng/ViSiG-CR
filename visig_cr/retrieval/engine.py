"""Question-conditioned contrast retrieval with retained graph controls.

The main method filters existing events in an adaptive FAISS pool and builds
contrasts online. Visual/relation graph traversal remains available only to
explicit legacy experiments and the unchanged internal comparison controls.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from ..graph.store import GraphStore
from ..normality import ACTIVE_CLINICAL_TEMPORAL_STATUSES, derive_clinical_normality_reference
from ..ontology import Ontology
from ..schemas import (
    ClinicalTarget,
    Evidence,
    EvidencePackage,
    Event,
    QuestionSpec,
    SemanticContrastGroup,
)
from ..visual import MedSigLIPEncoder, VisualStore
from .attribute_selector import (
    AttributeSelectorAbstainError,
    AttributeSelectorAgent,
    AttributeSelectorResolutionError,
)
from .comparison import (
    attribute_value,
    best_target_match,
    comparison_role,
    pair_non_target_context_score,
)
from .relation_router import RelationRouter
from .question_conditioned import (
    IMPLEMENTATION_VERSION, POLICY_VERSION, retrieve_question_conditioned,
    uses_question_conditioned,
)
from ..io_utils import stable_json_sha256


@dataclass(frozen=True)
class BoundaryEntry:
    case_id: str
    seed_case_id: str
    query_similarity: float
    direct_seed_similarity: float
    path_score: float
    hop_count: int
    path_nodes: tuple[str, ...]
    relation_score: float = 0.0
    relation_types: tuple[str, ...] = ()
    relation_event_ids: tuple[str, ...] = ()
    no_matched_event_ids: tuple[str, ...] = ()
    relation_path_tokens: tuple[str, ...] = ()


SEMANTIC_SUPPORT_MARKER = "_semantic_support"


def _has_verified_semantic_support(event: Event) -> bool:
    if not isinstance(event.characteristics, dict):
        return False
    marker = event.characteristics.get(SEMANTIC_SUPPORT_MARKER)
    return (
        isinstance(marker, dict)
        and marker.get("verified") is True
        and marker.get("schema_version") == "visig_cr_semantic_support_v1"
    )


def _event_payload(event: Event) -> dict[str, Any]:
    payload = event.model_dump(
        include={
            "finding",
            "anatomy",
            "laterality",
            "presence",
            "severity",
            "uncertainty",
            "temporal_status",
            "confidence",
            "characteristics",
        }
    )
    # Internal validation provenance remains in the event store/graph but does
    # not need to consume reader prompt tokens.
    characteristics = payload.get("characteristics")
    if isinstance(characteristics, dict):
        payload["characteristics"] = {
            key: value
            for key, value in characteristics.items()
            if not str(key).startswith("_")
        }
    support = event.characteristics.get("_presence_support")
    if isinstance(support, dict):
        payload["presence_basis"] = support["basis"]
        payload["presence_source_finding"] = support["source_finding"]
        payload["presence_source_event_id"] = support["source_event_id"]
        payload["presence_policy_version"] = support["policy_version"]
    return payload


class ViSiGCRRetriever:
    def __init__(
        self,
        ontology: Ontology,
        graph: GraphStore,
        visual: VisualStore,
        encoder: MedSigLIPEncoder,
        retrieval_cfg: dict[str, Any],
        attribute_selector: AttributeSelectorAgent,
    ):
        self.ontology = ontology
        self.graph = graph
        self.visual = visual
        self.encoder = encoder
        self.cfg = retrieval_cfg
        self.attribute_selector = attribute_selector
        self.relation_router = RelationRouter(graph, ontology)
        self.presence_index = None

    def prepare_presence_index(self):
        if not self.cfg.get("presence_index", {}).get("enabled", False):
            self.presence_index = None
            return
        if self.cfg.get("presence_index", {}).get("enabled", False) and self.presence_index is None:
            from ..presence_index import PresenceIndex, index_path
            self.presence_index = PresenceIndex(
                index_path(self.graph.path, self.cfg), self.graph, self.ontology, self.cfg,
            )

    def retrieve(self, query_image_path: str, question: str, **kwargs: Any) -> EvidencePackage:
        started = perf_counter()
        package = self._retrieve(query_image_path, question, **kwargs)
        # Runtime diagnostics must not consume reader-visible coverage tokens.
        for key in (
            "group_pair_mode",
            "group_retained_candidate_count",
            "stored_case_edges_examined",
            "stored_event_relations_examined",
            "candidate_pair_checks",
            "stored_candidate_pair_checks",
            "candidate_pair_fallback_enabled",
            "candidate_pair_fallback_attempted",
            "candidate_pair_fallback_used",
            "candidate_pair_fallback_checks",
            "candidate_pair_fallback_group_count",
            "stored_edge_group_count",
            "complete_candidate_group_count",
            "incomplete_candidate_group_count",
            "incomplete_candidate_groups_rejected",
            "group_enumeration_seconds",
        ):
            if key in package.coverage_status:
                package.boundary_stats[key] = package.coverage_status.pop(key)
        package.boundary_stats["retrieval_total_seconds"] = perf_counter() - started
        is_qcr = uses_question_conditioned(self.cfg, package.method)
        package.boundary_stats["implementation_version"] = IMPLEMENTATION_VERSION if is_qcr else "2.1"
        if is_qcr:
            package.audit_metadata["retrieval_policy"] = POLICY_VERSION
            package.audit_metadata["retrieval_config_sha256"] = stable_json_sha256(self.cfg)
            if self.presence_index is not None:
                from collections import Counter
                package.boundary_stats["implementation_version"] = IMPLEMENTATION_VERSION
                package.audit_metadata["presence_index"] = {
                    "policy_version": self.presence_index.metadata["settings"]["policy_version"],
                    "sha256": self.presence_index.sha256,
                    "selected_basis_counts": dict(Counter(
                        e.event.get("presence_basis", "stored_event") for e in package.evidence
                    )),
                }
        package.boundary_stats.setdefault("selector_fallback_used", False)
        package.audit_metadata.setdefault(
            "selector_fallback",
            {"used": False},
        )
        return package

    def _retrieve(
        self,
        query_image_path: str,
        question: str,
        *,
        query_id: str = "query",
        method: str = "visig_cr",
        options: Sequence[str] | None = None,
        reader_question: str | None = None,
    ) -> EvidencePackage:
        if method == "no_rag":
            raise ValueError(
                "no_rag must be built with visig_cr.no_rag.build_no_rag_package; "
                "GraphRetriever intentionally has no no-RAG dependency path"
            )
        is_qcr = uses_question_conditioned(self.cfg, method)
        if is_qcr:
            self.prepare_presence_index()
        if method == "visig_cr" and not is_qcr and str(self.cfg.get("group_pair_mode", "stored_edges")) != "stored_edges":
            raise ValueError("ViSiG-CR 2.0 requires stored_edges; use an explicitly named all-pairs control")
        option_list = list(options or [])
        try:
            spec = self.attribute_selector.select(question, options=option_list)
        except (AttributeSelectorAbstainError, AttributeSelectorResolutionError) as exc:
            fallback_cfg = self.cfg.get("selector_fallback", {})
            fallback_enabled = bool(
                isinstance(fallback_cfg, dict)
                and fallback_cfg.get("enabled", False)
            )
            if method != "visig_cr" or not fallback_enabled:
                raise
            strategy = str(fallback_cfg.get("strategy", "i2i")).strip().lower()
            if strategy != "i2i":
                raise ValueError(
                    "retrieval.selector_fallback.strategy must be 'i2i'; "
                    f"received {strategy!r}"
                ) from exc
            delivered_question = reader_question or self._reader_question(
                question,
                option_list,
            )
            query_embedding = self.encoder.encode_one(query_image_path)
            reason_code = (
                "selector_abstain"
                if isinstance(exc, AttributeSelectorAbstainError)
                else "selector_unresolved_target"
            )
            return self._retrieve_selector_fallback_i2i(
                query_id=query_id,
                query_image=query_image_path,
                retrieval_question=question,
                delivered_question=delivered_question,
                options=option_list,
                query_embedding=query_embedding,
                reason_code=reason_code,
                selector_error=exc,
            )
        delivered_question = reader_question or self._reader_question(
            question,
            spec.options,
        )
        query_embedding = self.encoder.encode_one(query_image_path)

        if is_qcr:
            return retrieve_question_conditioned(
                self, query_id, query_image_path, delivered_question, spec, query_embedding,
            )

        if method == "image_rag":
            return self._retrieve_image_rag(
                query_id,
                query_image_path,
                delivered_question,
                spec,
                query_embedding,
            )
        if method in {
            "visig_cr",
            "visual_boundary_pointwise",
            "local_value_partition",
            "visual_boundary_semantic_contrast",
            "visual_boundary_stored_contrast",
        }:
            return self._retrieve_from_visual_boundary(
                query_id,
                query_image_path,
                delivered_question,
                spec,
                query_embedding,
                method=method,
            )
        if method in {"visual_pool", "visual_pool_semantic_contrast"}:
            return self._retrieve_visual_pool(
                query_id,
                query_image_path,
                delivered_question,
                spec,
                query_embedding,
                semantic_contrast=(method == "visual_pool_semantic_contrast"),
            )
        if method in {"direct_index", "direct_index_semantic_contrast"}:
            return self._retrieve_direct_index(
                query_id,
                query_image_path,
                delivered_question,
                spec,
                query_embedding,
                semantic_contrast=(method == "direct_index_semantic_contrast"),
            )
        raise ValueError(f"Unknown retrieval method: {method}")

    # ------------------------------------------------------------------
    # Visual-only entry and boundary construction
    # ------------------------------------------------------------------

    @staticmethod
    def _reader_question(question: str, options: Sequence[str]) -> str:
        if not options:
            return question
        option_block = "\n".join(options)
        return f"{question}\n\nOPTIONS\n{option_block}"

    def _seed_list(self, query_embedding: np.ndarray) -> list[tuple[str, float]]:
        return self.visual.search(query_embedding, int(self.cfg.get("k_seed", 8)))

    def _seed_records(self, seeds: Sequence[tuple[str, float]]) -> list[dict[str, Any]]:
        return [
            {
                "case_id": case_id,
                "visual_similarity": float(score),
                "image_path": self.graph.get_case(case_id).image_path,
            }
            for case_id, score in seeds
        ]

    def _visual_boundary(
        self,
        seeds: Sequence[tuple[str, float]],
        query_embedding: np.ndarray,
    ) -> tuple[list[BoundaryEntry], dict[str, Any]]:
        max_hops = int(self.cfg.get("boundary_hops", 1))
        if max_hops < 0 or max_hops > 3:
            raise ValueError("retrieval.boundary_hops must be between 0 and 3")
        neighbors_per_hop = int(self.cfg.get("boundary_neighbors_per_hop", 12))
        max_per_seed = int(self.cfg.get("boundary_max_candidates_per_seed", 128))
        max_cases = int(self.cfg.get("boundary_max_cases", 96))
        minimum_edge_similarity = float(
            self.cfg.get("boundary_minimum_edge_similarity", -1.0)
        )
        hop_decay = float(self.cfg.get("hop_decay", 0.85))
        mutual_first = bool(self.cfg.get("mutual_visual_first", True))

        best: dict[str, BoundaryEntry] = {}
        for seed_case_id, query_similarity in seeds:
            seed_entry = BoundaryEntry(
                case_id=seed_case_id,
                seed_case_id=seed_case_id,
                query_similarity=float(query_similarity),
                direct_seed_similarity=1.0,
                path_score=1.0,
                hop_count=0,
                path_nodes=(seed_case_id,),
            )
            best[seed_case_id] = self._better_boundary(best.get(seed_case_id), seed_entry)

            if max_hops == 0:
                continue
            for path in self.graph.visual_neighborhood(
                seed_case_id,
                max_hops=max_hops,
                neighbors_per_hop=neighbors_per_hop,
                max_candidates=max_per_seed,
                minimum_edge_similarity=minimum_edge_similarity,
                hop_decay=hop_decay,
                mutual_first=mutual_first,
            ):
                case_id = path.target_case_id
                if case_id not in self.visual.id_to_row:
                    continue
                entry = BoundaryEntry(
                    case_id=case_id,
                    seed_case_id=seed_case_id,
                    query_similarity=float(
                        self.visual.similarity_to_query(query_embedding, case_id)
                    ),
                    direct_seed_similarity=float(
                        self.visual.similarity_between(seed_case_id, case_id)
                    ),
                    path_score=float(path.path_score),
                    hop_count=int(path.hop_count),
                    path_nodes=tuple(path.nodes),
                )
                best[case_id] = self._better_boundary(best.get(case_id), entry)

        ranked = sorted(
            best.values(),
            key=lambda item: (
                item.query_similarity,
                item.path_score,
                item.direct_seed_similarity,
                -item.hop_count,
            ),
            reverse=True,
        )[:max_cases]
        return ranked, {
            "seed_count": len(seeds),
            "boundary_case_count": len(ranked),
            "boundary_hops": max_hops,
            "boundary_max_cases": max_cases,
            "boundary_mean_query_similarity": (
                float(np.mean([item.query_similarity for item in ranked]))
                if ranked
                else 0.0
            ),
            "boundary_mean_path_score": (
                float(np.mean([item.path_score for item in ranked])) if ranked else 0.0
            ),
        }

    def _relation_boundary(
        self,
        seeds: Sequence[tuple[str, float]],
        query_embedding: np.ndarray,
        spec: QuestionSpec,
    ) -> tuple[list[BoundaryEntry], dict[str, Any]]:
        """Traverse relation-annotated visual edges in question-priority order."""

        max_hops = int(self.cfg.get("boundary_hops", 1))
        if max_hops < 0 or max_hops > 3:
            raise ValueError("retrieval.boundary_hops must be between 0 and 3")
        neighbors_per_hop = int(
            self.cfg.get(
                "relation_neighbors_per_hop",
                self.cfg.get("boundary_neighbors_per_hop", 12),
            )
        )
        relation_candidate_pool = max(
            neighbors_per_hop,
            int(self.cfg.get("relation_candidate_pool", 32)),
        )
        max_per_seed = int(self.cfg.get("boundary_max_candidates_per_seed", 128))
        max_cases = int(self.cfg.get("boundary_max_cases", 96))
        minimum_visual = float(
            self.cfg.get("boundary_minimum_edge_similarity", -1.0)
        )
        minimum_route = float(self.cfg.get("minimum_relation_route_score", 0.05))
        hop_decay = float(self.cfg.get("hop_decay", 0.85))

        best: dict[str, BoundaryEntry] = {}
        traversed_relation_edges = 0
        question_matched_edges = 0
        examined_relation_edges = 0
        no_matched_event_edges = 0

        for seed_case_id, query_similarity in seeds:
            seed_entry = BoundaryEntry(
                case_id=seed_case_id,
                seed_case_id=seed_case_id,
                query_similarity=float(query_similarity),
                direct_seed_similarity=1.0,
                path_score=1.0,
                hop_count=0,
                path_nodes=(seed_case_id,),
            )
            best[seed_case_id] = self._better_boundary(
                best.get(seed_case_id),
                seed_entry,
            )
            if max_hops == 0:
                continue

            frontier: deque[tuple[str, tuple[str, ...], float, tuple[str, ...]]] = deque(
                [(seed_case_id, (seed_case_id,), 1.0, ())]
            )
            discovered_for_seed = 0
            for hop in range(1, max_hops + 1):
                next_frontier: list[
                    tuple[str, tuple[str, ...], float, tuple[str, ...]]
                ] = []
                while frontier:
                    current, path_nodes, previous_score, path_tokens = frontier.popleft()
                    routed = []
                    for edge in self.graph.relation_neighbors(
                        current,
                        relation_candidate_pool,
                        minimum_visual_similarity=minimum_visual,
                    ):
                        examined_relation_edges += 1
                        route = self.relation_router.score_edge(edge, spec)
                        routed.append((route.score, edge.visual_similarity, edge, route))
                    routed.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    for _, _, edge, route in routed[:neighbors_per_hop]:
                        if route.score <= 0.0 or route.score < minimum_route:
                            continue
                        target = edge.target_case_id
                        if target in path_nodes or target not in self.visual.id_to_row:
                            continue
                        traversed_relation_edges += 1
                        if route.relation_types:
                            question_matched_edges += 1
                        if route.no_matched_target_event_ids:
                            no_matched_event_edges += 1
                        edge_label = ",".join(route.relation_types) or "visual_fallback"
                        token = f"relation:{current}->{target}:{edge_label}"
                        new_tokens = (*path_tokens, token)
                        new_nodes = (*path_nodes, target)
                        route_path_score = min(previous_score, route.score)
                        if hop > 1:
                            route_path_score *= hop_decay
                        entry = BoundaryEntry(
                            case_id=target,
                            seed_case_id=seed_case_id,
                            query_similarity=float(
                                self.visual.similarity_to_query(
                                    query_embedding,
                                    target,
                                )
                            ),
                            direct_seed_similarity=float(
                                self.visual.similarity_between(seed_case_id, target)
                            ),
                            path_score=max(0.0, min(1.0, route_path_score)),
                            hop_count=hop,
                            path_nodes=new_nodes,
                            relation_score=route.score,
                            relation_types=route.relation_types,
                            relation_event_ids=route.target_event_ids,
                            no_matched_event_ids=route.no_matched_target_event_ids,
                            relation_path_tokens=new_tokens,
                        )
                        best[target] = self._better_boundary(best.get(target), entry)
                        next_frontier.append(
                            (target, new_nodes, route_path_score, new_tokens)
                        )
                        discovered_for_seed += 1
                        if discovered_for_seed >= max_per_seed:
                            break
                    if discovered_for_seed >= max_per_seed:
                        break
                if discovered_for_seed >= max_per_seed or not next_frontier:
                    break
                next_frontier.sort(key=lambda item: item[2], reverse=True)
                frontier = deque(next_frontier[:max_per_seed])

        relation_expanded = any(item.hop_count > 0 for item in best.values())
        if (
            not relation_expanded
            and bool(self.cfg.get("allow_visual_fallback_when_relations_missing", True))
        ):
            visual_boundary, visual_stats = self._visual_boundary(seeds, query_embedding)
            visual_stats.update(
                {
                    "relation_routing": True,
                    "relation_fallback_used": True,
                    "relation_priority": list(
                        self.relation_router.priorities(spec)
                    ),
                    "traversed_relation_edges": 0,
                    "examined_relation_edges": examined_relation_edges,
                    "visual_fallback_case_count": sum(item.hop_count > 0 for item in visual_boundary),
                }
            )
            return visual_boundary, visual_stats

        seed_ids = {case_id for case_id, _ in seeds[:max_cases]}
        ranked = sorted(
            best.values(),
            key=lambda item: (
                item.case_id in seed_ids,
                item.relation_score,
                item.query_similarity,
                item.path_score,
                item.direct_seed_similarity,
                -item.hop_count,
            ),
            reverse=True,
        )[:max_cases]
        return ranked, {
            "seed_count": len(seeds),
            "boundary_case_count": len(ranked),
            "boundary_hops": max_hops,
            "boundary_max_cases": max_cases,
            "relation_routing": True,
            "relation_fallback_used": False,
            "relation_priority": list(self.relation_router.priorities(spec)),
            "traversed_relation_edges": traversed_relation_edges,
            "examined_relation_edges": examined_relation_edges,
            "visual_fallback_case_count": 0,
            "question_matched_relation_edges": question_matched_edges,
            "no_matched_event_edges": no_matched_event_edges,
            "boundary_mean_query_similarity": (
                float(np.mean([item.query_similarity for item in ranked]))
                if ranked
                else 0.0
            ),
            "boundary_mean_relation_score": (
                float(np.mean([item.relation_score for item in ranked]))
                if ranked
                else 0.0
            ),
            "boundary_mean_path_score": (
                float(np.mean([item.path_score for item in ranked])) if ranked else 0.0
            ),
        }

    @staticmethod
    def _better_boundary(
        current: BoundaryEntry | None,
        candidate: BoundaryEntry,
    ) -> BoundaryEntry:
        if current is None:
            return candidate
        current_key = (
            current.relation_score,
            current.path_score,
            current.query_similarity,
            current.direct_seed_similarity,
            -current.hop_count,
        )
        candidate_key = (
            candidate.relation_score,
            candidate.path_score,
            candidate.query_similarity,
            candidate.direct_seed_similarity,
            -candidate.hop_count,
        )
        return candidate if candidate_key > current_key else current

    # ------------------------------------------------------------------
    # Online event scan and candidate scoring
    # ------------------------------------------------------------------

    def _relevant_findings(self, spec: QuestionSpec) -> list[str]:
        findings = [
            target.finding
            for target in spec.effective_targets()
            if target.finding not in {"", "unspecified", "not_applicable"}
        ]
        if spec.target_attribute == "finding_identity":
            findings.extend(
                value for value in spec.answer_space if value in self.ontology.findings
            )
        return list(dict.fromkeys(findings))

    def _scan_boundary(
        self,
        boundary: Sequence[BoundaryEntry],
        spec: QuestionSpec,
        *,
        query_conditioned: bool = False,
    ) -> list[Evidence]:
        if spec.target_scope == "case_normality":
            return self._scan_case_normality_boundary(boundary, spec)
        minimum_confidence = float(self.cfg.get("minimum_event_confidence", 0.70))
        minimum_target_match = float(self.cfg.get("minimum_target_match", 0.60))
        tau_query = float(self.cfg.get("tau_query", 0.20))
        tau_boundary = float(self.cfg.get("tau_boundary", 0.20))
        boundary_by_case = {entry.case_id: entry for entry in boundary}
        events = self.graph.events_in_cases(
            list(boundary_by_case),
            minimum_confidence=minimum_confidence,
            retrieval_eligible_only=(spec.comparison_mode != "candidate_predicate"),
            findings=self._relevant_findings(spec) or None,
        )
        if query_conditioned and self.presence_index is not None:
            events.extend(self.presence_index.additional_events(
                list(boundary_by_case), self._relevant_findings(spec), spec, self.graph, self.ontology,
            ))

        candidates: list[Evidence] = []
        for event in events:
            if query_conditioned:
                # Stable/new/changing positive findings still exist. Negative
                # exemplars need a current absence assertion; history is not one.
                if event.presence == "not_mentioned":
                    continue
                allowed_statuses = (
                    ACTIVE_CLINICAL_TEMPORAL_STATUSES
                    if event.presence == "present" else {"current"}
                )
                if event.temporal_status not in allowed_statuses:
                    continue
            if not self._event_allowed_for_spec(event, spec):
                continue
            if not self._event_compatible_with_visible_target(event, spec):
                continue
            entry = boundary_by_case[event.case_id]
            if entry.query_similarity < tau_query:
                continue
            if max(entry.direct_seed_similarity, entry.path_score) < tau_boundary:
                continue
            value = attribute_value(event, spec.target_attribute)
            if value is None:
                continue
            if (
                spec.comparison_mode == "candidate_predicate"
                and value not in set(spec.answer_space)
            ):
                continue
            target, target_match = best_target_match(event, spec, self.ontology)
            if target is None or target_match <= 0.0 or target_match < minimum_target_match:
                continue
            candidate_score = (
                entry.query_similarity if query_conditioned
                else self._candidate_score(entry, event, target_match)
            )
            event_relation_score = (
                entry.relation_score
                if event.event_id in entry.relation_event_ids
                else 0.90 * entry.relation_score
                if event.event_id in entry.no_matched_event_ids
                else 0.25 * entry.relation_score
            )
            case = self.graph.get_case(event.case_id)
            candidates.append(
                Evidence(
                    case_id=event.case_id,
                    event_id=event.event_id,
                    image_path=case.image_path,
                    seed_case_id=entry.seed_case_id,
                    visual_similarity_query=entry.query_similarity,
                    visual_similarity_seed=entry.direct_seed_similarity,
                    boundary_path=self._path_tokens(entry),
                    boundary_hop_count=entry.hop_count,
                    boundary_path_score=entry.path_score,
                    relation_score=max(0.0, min(1.0, event_relation_score)),
                    relation_types=list(entry.relation_types),
                    relation_event_ids=list(entry.relation_event_ids),
                    used_no_matched_event_ids=(
                        [event.event_id]
                        if event.event_id in entry.no_matched_event_ids
                        else []
                    ),
                    target_match=target_match,
                    non_target_match=target_match,
                    event_confidence=event.confidence,
                    candidate_score=candidate_score,
                    target_attribute=spec.target_attribute,
                    attribute_value=value,
                    comparison_role=comparison_role(value, spec),
                    event=_event_payload(event),
                    evidence_span=event.evidence_span,
                    matched_option_ids=spec.option_ids_for_value(value),
                )
            )
        return self._deduplicate_candidates(candidates)

    def _event_compatible_with_visible_target(
        self,
        event: Event,
        spec: QuestionSpec,
    ) -> bool:
        """Reject a small set of known ontology-collapse false exemplars.

        Existing extraction artifacts occasionally map a fractured sternotomy
        wire to the broad ``fracture`` finding. That event is valid report
        extraction but is not a useful osseous-fracture appearance exemplar
        unless the visible question itself asks about a wire/device. Keep this
        correction at retrieval time so no ontology re-extraction is required.
        """

        if not bool(self.cfg.get("filter_device_fracture_events", True)):
            return True
        if event.finding != "fracture":
            return True
        span = event.evidence_span.casefold()
        if "wire" not in span:
            return True
        question = spec.question_text.casefold()
        return any(
            token in question
            for token in ("wire", "sternotomy", "cerclage", "device")
        )

    def _scan_case_normality_boundary(
        self,
        boundary: Sequence[BoundaryEntry],
        spec: QuestionSpec,
    ) -> list[Evidence]:
        """Convert visually local cases into global abnormality exemplars.

        The case label is deliberately represented on the existing presence
        axis: ``abnormal -> present`` and ``normal -> absent``.  No synthetic
        ontology finding is created and no event-level relation is claimed.
        """

        boundary_by_case = {entry.case_id: entry for entry in boundary}
        policy = str(
            self.cfg.get(
                "case_normality_reference_policy",
                "explicit_event_supported",
            )
        ).strip().lower()
        if policy not in {"explicit_event_supported", "closed_world_proxy"}:
            raise ValueError(
                "retrieval.case_normality_reference_policy must be "
                "'explicit_event_supported' or 'closed_world_proxy'"
            )
        stored_normalities = self.graph.case_normalities_in_cases(
            list(boundary_by_case)
        )
        tau_query = float(self.cfg.get("tau_query", 0.20))
        tau_boundary = float(self.cfg.get("tau_boundary", 0.20))
        weights = self.cfg.get("candidate_weights", {})
        candidates: list[Evidence] = []
        for case_id in boundary_by_case:
            if policy == "explicit_event_supported":
                normality, provenance = derive_clinical_normality_reference(
                    self.graph.get_events(case_id),
                    self.ontology,
                    self.cfg.get("case_normality_abnormal_findings", "clinical"),
                    explicit_normal_findings=self.cfg.get(
                        "case_normality_explicit_normal_findings",
                        [],
                    ),
                )
                if normality is None:
                    continue
                derivation = "explicit_existing_ontology_events"
                is_clinical_reference = True
                evidence_spans = provenance.get("supporting_evidence_spans", [])
                evidence_span = "\n".join(dict.fromkeys(
                    str(span).strip() for span in evidence_spans if str(span).strip()
                ))
            else:
                normality = stored_normalities.get(case_id)
                if normality is None:
                    continue
                provenance = self.graph.normality_provenance(case_id)
                derivation = "closed_world_event_proxy"
                is_clinical_reference = False
                evidence_span = ""
            entry = boundary_by_case[case_id]
            if entry.query_similarity < tau_query:
                continue
            if max(entry.direct_seed_similarity, entry.path_score) < tau_boundary:
                continue
            value = "present" if normality == "abnormal" else "absent"
            case = self.graph.get_case(case_id)
            candidate_score = (
                float(weights.get("lambda_query", 0.35)) * entry.query_similarity
                + float(weights.get("lambda_boundary", 0.20)) * entry.path_score
                + float(weights.get("lambda_target", 0.25))
                + float(weights.get("lambda_relation", 0.20)) * entry.relation_score
                - float(weights.get("lambda_hop", 0.05)) * max(0, entry.hop_count - 1)
            )
            candidates.append(
                Evidence(
                    case_id=case_id,
                    event_id=f"case_normality:{case_id}",
                    image_path=case.image_path,
                    seed_case_id=entry.seed_case_id,
                    visual_similarity_query=entry.query_similarity,
                    visual_similarity_seed=entry.direct_seed_similarity,
                    boundary_path=self._path_tokens(entry),
                    boundary_hop_count=entry.hop_count,
                    boundary_path_score=entry.path_score,
                    relation_score=entry.relation_score,
                    relation_types=list(entry.relation_types),
                    relation_event_ids=[],
                    used_no_matched_event_ids=[],
                    target_match=1.0,
                    non_target_match=None,
                    event_confidence=None,
                    candidate_score=candidate_score,
                    target_attribute="presence",
                    attribute_value=value,
                    comparison_role=comparison_role(value, spec),
                    event={
                        "case_level_attribute": "normality",
                        "normality": normality,
                        "abnormality_presence": value,
                        "derivation": derivation,
                        "not_a_clinical_normality_reference": (
                            not is_clinical_reference
                        ),
                        "is_clinical_normality_reference": is_clinical_reference,
                        "normality_provenance": provenance,
                    },
                    evidence_span=evidence_span,
                    matched_option_ids=spec.option_ids_for_value(value),
                )
            )
        return self._deduplicate_candidates(candidates)

    def _event_allowed_for_spec(self, event: Event, spec: QuestionSpec) -> bool:
        if bool(self.cfg.get("require_semantic_support", False)) and not (
            _has_verified_semantic_support(event)
        ):
            return False
        if event.retrieval_eligible:
            return True
        if spec.comparison_mode != "candidate_predicate":
            return False
        candidate_by_finding = {
            candidate.finding: candidate
            for candidate in spec.answer_candidates
            if candidate.finding
        }
        candidate = candidate_by_finding.get(event.finding)
        if candidate is None:
            return False

        tolerated = {"section_corrected", "unsectioned_report"}
        policy = self.ontology.retrieval_policy(event.finding)
        if policy == "context_only":
            # Explicit normal/context statements may act as options only for a
            # candidate-predicate question (for example, "Is the left lung
            # normal?" with Normal vs finding options).  Their normal
            # context-only routing flag remains visible in the package.
            tolerated.add("context_only_finding")
        elif candidate.presence == "absent":
            # The memory stores explicit absence assertions even when a finding
            # is outside the pilot absence allowlist.  A question that exposes
            # that exact finding as an absent candidate may use it; unmentioned
            # findings still never qualify.
            tolerated.add("absence_not_in_pilot_allowlist")
        else:
            return False
        return not (set(event.validation_errors) - tolerated)

    def _candidate_score(
        self,
        entry: BoundaryEntry,
        event: Event,
        target_match: float,
    ) -> float:
        weights = self.cfg.get("candidate_weights", {})
        hub_scale = max(2.0, float(self.cfg.get("hub_degree_scale", 32)))
        in_degree = self.graph.visual_in_degree(entry.case_id)
        hub_penalty = min(1.0, math.log1p(in_degree) / math.log1p(hub_scale))
        return (
            float(weights.get("lambda_query", 0.35)) * entry.query_similarity
            + float(weights.get("lambda_boundary", 0.20)) * entry.path_score
            + float(weights.get("lambda_target", 0.25)) * target_match
            + float(weights.get("lambda_event", 0.20)) * event.confidence
            + float(weights.get("lambda_relation", 0.20))
            * (
                entry.relation_score
                if event.event_id in entry.relation_event_ids
                else 0.90 * entry.relation_score
                if event.event_id in entry.no_matched_event_ids
                else 0.25 * entry.relation_score
            )
            - float(weights.get("lambda_hop", 0.05)) * max(0, entry.hop_count - 1)
            - float(weights.get("lambda_hub", 0.03)) * hub_penalty
        )

    @staticmethod
    def _path_tokens(entry: BoundaryEntry) -> list[str]:
        tokens = [f"seed:{entry.seed_case_id}"]
        if entry.relation_path_tokens:
            tokens.extend(entry.relation_path_tokens)
        else:
            for source, target in zip(entry.path_nodes[:-1], entry.path_nodes[1:]):
                tokens.append(f"visual:{source}->{target}")
        for event_id in entry.no_matched_event_ids:
            tokens.append(f"no_matched_event:{event_id}")
        tokens.append(f"case:{entry.case_id}")
        return tokens

    @staticmethod
    def _deduplicate_candidates(candidates: Iterable[Evidence]) -> list[Evidence]:
        best: dict[tuple[str, str, str], Evidence] = {}
        for candidate in candidates:
            key = (candidate.case_id, candidate.event_id, candidate.attribute_value)
            current = best.get(key)
            if current is None or (
                candidate.candidate_score,
                candidate.target_match,
                candidate.event_confidence or 0.0,
            ) > (
                current.candidate_score,
                current.target_match,
                current.event_confidence or 0.0,
            ):
                best[key] = candidate
        return list(best.values())

    # ------------------------------------------------------------------
    # Semantic contrast group construction and set selection
    # ------------------------------------------------------------------

    def _candidate_redundancy(
        self,
        candidate: Evidence,
        selected: Sequence[Evidence],
    ) -> float:
        similarities = [
            self.visual.similarity_between(candidate.case_id, previous.case_id)
            for previous in selected
            if candidate.case_id in self.visual.id_to_row
            and previous.case_id in self.visual.id_to_row
            and candidate.case_id != previous.case_id
        ]
        return max(similarities, default=0.0)

    @staticmethod
    def _is_direct_seed_evidence(candidate: Evidence) -> bool:
        """Return whether the displayed case is itself an initial I2I seed."""

        return (
            candidate.boundary_hop_count == 0
            and candidate.case_id == candidate.seed_case_id
        )

    def _offline_pair_relation(
        self,
        left: Evidence,
        right: Evidence,
    ) -> tuple[Any | None, bool]:
        """Return a stored event-pair relation and whether its orientation is direct."""

        # Directed edges may retain different capped event-pair subsets. Search
        # both orientations even when the forward case edge already exists.
        for source, target, direct in (
            (left, right, True), (right, left, False)
        ):
            edge = self.graph.get_relation_edge(source.case_id, target.case_id)
            if edge is None:
                continue
            for relation in edge.matched_events:
                if (relation.source_event_id == source.event_id
                        and relation.target_event_id == target.event_id):
                    return relation, direct
        return None, True

    def _pair_quality(
        self,
        left: Evidence,
        right: Evidence,
        spec: QuestionSpec,
        *,
        use_offline_relation: bool = True,
    ) -> tuple[float, float | None, float, list[str], float]:
        """Validate a candidate pair against both stored and live event semantics.

        The offline relation is used as a routing/quality prior. The live
        comparator is always recomputed, so ontology or event changes cannot
        silently turn a stale edge into a valid reader-visible group.
        """

        if left.case_id == right.case_id:
            return 0.0, 0.0, -1.0, [], 0.0
        if left.attribute_value == right.attribute_value:
            return 0.0, 0.0, -1.0, [], 0.0
        if spec.target_scope == "case_normality":
            pair_visual = self.visual.similarity_between(left.case_id, right.case_id)
            q_left = max(0.0, min(1.0, left.visual_similarity_query))
            q_right = max(0.0, min(1.0, right.visual_similarity_query))
            local = max(
                0.0,
                min(1.0, min(left.boundary_path_score, right.boundary_path_score)),
            )
            pair_visual_clamped = max(0.0, min(1.0, pair_visual))
            quality = math.sqrt(q_left * q_right) * math.sqrt(
                max(0.0, pair_visual_clamped * local)
            )
            # No non-target event dimensions are defined for this coarse
            # fallback.  The group is validated by visual locality plus opposite
            # deterministic normality values, not by a fabricated event match.
            edge = self.graph.relation_between(left.case_id, right.case_id) if use_offline_relation else None
            labels = ["normality_contrast"] if edge is not None and edge.normality_relation.relation == "different" else []
            return max(0.0, min(1.0, quality)), None, pair_visual, labels, 0.0
        left_event = self.graph.get_event(left.event_id)
        right_event = self.graph.get_event(right.event_id)
        online_context = pair_non_target_context_score(
            left_event,
            right_event,
            spec.target_attribute,
            self.ontology,
        )
        relation, _ = self._offline_pair_relation(left, right) if use_offline_relation else (None, True)
        relation_types: list[str] = []
        relation_score = 0.0
        context = online_context
        if relation is not None:
            relation_types = list(relation.relation_types)
            relation_score = float(relation.relation_score)
            offline_context = relation.non_target_scores.get(spec.target_attribute)
            if offline_context is not None:
                # Conservative final validation: a stale or over-broad stored
                # relation cannot improve the freshly recomputed context score.
                context = min(online_context, float(offline_context))
        # No direct event relation => no direct relation score. Path priors
        # remain solely in boundary_path_score/relation routing provenance.

        pair_visual = self.visual.similarity_between(left.case_id, right.case_id)
        q_left = max(0.0, min(1.0, left.visual_similarity_query))
        q_right = max(0.0, min(1.0, right.visual_similarity_query))
        local = max(
            0.0,
            min(1.0, min(left.boundary_path_score, right.boundary_path_score)),
        )
        pair_visual_clamped = max(0.0, min(1.0, pair_visual))
        visual_term = math.sqrt(q_left * q_right) * math.sqrt(
            max(0.0, pair_visual_clamped * local)
        )
        relation_term = 0.85 + 0.15 * max(0.0, min(1.0, relation_score))
        quality = max(0.0, min(1.0, context * visual_term * relation_term))
        return quality, context, pair_visual, relation_types, relation_score

    def _candidate_groups(
        self,
        candidates: Sequence[Evidence],
        spec: QuestionSpec,
        *,
        pair_mode: str = "stored_edges",
        directional_ranking: bool = False,
    ) -> list[tuple[Evidence, Evidence, SemanticContrastGroup]]:
        """Enumerate and validate candidate contrast pairs.

        ``all_pairs`` is an explicit graph-free control only. The main path
        starts from stored relations. When enabled for directional ViSiG-CR, a
        bounded fallback may compare only the top target-matched candidates in
        the already discovered visual boundary. Every fallback pair is
        revalidated with live event semantics and explicitly audited; it is
        never represented as a stored graph relation.
        """
        if pair_mode not in {"stored_edges", "all_pairs"}:
            raise ValueError("group_pair_mode must be 'stored_edges' or 'all_pairs'")
        started = perf_counter()
        value_buckets: dict[str, list[Evidence]] = defaultdict(list)
        per_value = max(1, int(self.cfg.get("group_candidates_per_value", 16)))
        for candidate in candidates:
            if candidate.attribute_value not in {"", "unknown"}:
                value_buckets[candidate.attribute_value].append(candidate)
        for value in value_buckets:
            value_buckets[value] = sorted(
                value_buckets[value],
                key=lambda item: (-item.candidate_score, item.case_id, item.event_id),
            )[:per_value]
        retained = [item for bucket in value_buckets.values() for item in bucket]
        by_event = {item.event_id: item for item in retained}
        by_case = {item.case_id: item for item in retained}
        minimum_context = float(self.cfg.get("minimum_group_non_target_match", 0.60))
        minimum_visual = float(self.cfg.get("minimum_group_visual_similarity", 0.20))
        minimum_quality = float(self.cfg.get("minimum_group_quality", 0.05))
        stats = {
            "group_pair_mode": pair_mode,
            "group_retained_candidate_count": len(retained),
            "stored_case_edges_examined": 0,
            "stored_event_relations_examined": 0,
            "candidate_pair_checks": 0,
            "stored_candidate_pair_checks": 0,
            "candidate_pair_fallback_enabled": False,
            "candidate_pair_fallback_attempted": False,
            "candidate_pair_fallback_used": False,
            "candidate_pair_fallback_checks": 0,
            "candidate_pair_fallback_group_count": 0,
            "stored_edge_group_count": 0,
        }
        seen: set[tuple[str, str]] = set()

        def pairs():
            if pair_mode == "all_pairs":
                for value_a, value_b in combinations(sorted(value_buckets), 2):
                    for left in value_buckets[value_a]:
                        for right in value_buckets[value_b]:
                            yield left, right
                return
            for edge in self.graph.relation_edges_in_cases(list(by_case)):
                stats["stored_case_edges_examined"] += 1
                if spec.target_scope == "case_normality":
                    if edge.normality_relation.relation != "different":
                        continue
                    edge_pairs = [(by_case[edge.source_case_id], by_case[edge.target_case_id])]
                else:
                    edge_pairs = []
                    for relation in edge.matched_events:
                        stats["stored_event_relations_examined"] += 1
                        left = by_event.get(relation.source_event_id)
                        right = by_event.get(relation.target_event_id)
                        if left is None or right is None:
                            continue
                        # Reject the offline-incompatible pair before loading
                        # live events. Unknown is not fabricated as a match.
                        context = relation.non_target_scores.get(spec.target_attribute)
                        if context is None or context <= 0.0 or context < minimum_context:
                            continue
                        edge_pairs.append((left, right))
                for left, right in edge_pairs:
                    key = tuple(sorted((left.event_id, right.event_id)))
                    if key not in seen:
                        yield left, right

        groups: list[tuple[Evidence, Evidence, SemanticContrastGroup]] = []

        def consider_pair(
            left: Evidence,
            right: Evidence,
            *,
            pair_source: str,
            use_offline_relation: bool,
        ) -> bool:
            if left.case_id == right.case_id or left.attribute_value == right.attribute_value:
                return False
            key = tuple(sorted((left.event_id, right.event_id)))
            if key in seen:
                return False
            seen.add(key)
            stats["candidate_pair_checks"] += 1
            if pair_source == "candidate_pair_fallback":
                stats["candidate_pair_fallback_checks"] += 1
            else:
                stats["stored_candidate_pair_checks"] += 1
            quality, context, pair_visual, relation_types, relation_score = self._pair_quality(
                left,
                right,
                spec,
                use_offline_relation=use_offline_relation,
            )
            if (
                (context is not None and (context <= 0.0 or context < minimum_context))
                or pair_visual < minimum_visual
                or quality <= 0.0
                or quality < minimum_quality
            ):
                return False
            values = [left.attribute_value, right.attribute_value]
            query_endpoint_margin = max(
                0.0,
                min(
                    1.0,
                    abs(
                        float(left.visual_similarity_query)
                        - float(right.visual_similarity_query)
                    ),
                ),
            )
            minimum_query_margin = float(
                self.cfg.get("minimum_group_query_endpoint_margin", 0.0)
            )
            if directional_ranking and query_endpoint_margin < minimum_query_margin:
                return False
            query_preferred_value = (
                left.attribute_value
                if left.visual_similarity_query >= right.visual_similarity_query
                else right.attribute_value
            )
            # A large endpoint margin is useful only when the pair is itself a
            # valid semantic contrast. Keep this score retrieval-only; it is
            # never rendered into the reader prompt as a query-label hint.
            directional_score = max(
                0.0,
                min(1.0, query_endpoint_margin * quality),
            )
            group = SemanticContrastGroup(
                group_id=f"scg{len(groups) + 1:03d}",
                target_attribute=spec.target_attribute,
                member_case_ids=[left.case_id, right.case_id],
                member_event_ids=[left.event_id, right.event_id],
                member_attribute_values=values,
                member_comparison_roles=[left.comparison_role, right.comparison_role],
                required_values_covered=[
                    value for value in values
                    if value in spec.required_attribute_values
                ],
                group_quality=quality,
                visual_locality=max(-1.0, min(1.0, pair_visual)),
                non_target_match=context,
                relation_types=relation_types,
                relation_score=max(0.0, min(1.0, relation_score)),
                pair_source=pair_source,
                query_endpoint_margin=query_endpoint_margin,
                directional_score=directional_score,
                query_preferred_attribute_value=query_preferred_value,
            )
            groups.append((left, right, group))
            return True

        initial_pair_source = (
            "all_pairs_control" if pair_mode == "all_pairs" else "stored_edge"
        )
        for left, right in pairs():
            consider_pair(
                left,
                right,
                pair_source=initial_pair_source,
                use_offline_relation=(pair_mode == "stored_edges"),
            )

        stats["stored_edge_group_count"] = sum(
            group.pair_source == "stored_edge" for _, _, group in groups
        )
        fallback_cfg = self.cfg.get("candidate_pair_fallback", {})
        fallback_enabled = bool(
            directional_ranking
            and pair_mode == "stored_edges"
            and isinstance(fallback_cfg, dict)
            and fallback_cfg.get("enabled", False)
        )
        stats["candidate_pair_fallback_enabled"] = fallback_enabled
        required_values = list(dict.fromkeys(spec.required_attribute_values))
        complete_group_exists = any(
            self._group_covers_required_values(group, spec)
            for _, _, group in groups
        )
        # One pair can completely cover exactly two required values. Restricting
        # this path to that case prevents an implicit all-pairs expansion and
        # avoids promoting optional/uncertain values over a missing binary pole.
        fallback_values_available = (
            len(required_values) == 2
            and all(value_buckets.get(value) for value in required_values)
        )
        if fallback_enabled and not complete_group_exists and fallback_values_available:
            stats["candidate_pair_fallback_attempted"] = True
            fallback_per_value = max(
                1,
                int(fallback_cfg.get("candidates_per_value", 8)),
            )
            max_fallback_checks = max(
                1,
                int(fallback_cfg.get("max_pair_checks", 64)),
            )
            left_value, right_value = required_values
            fallback_checks_before = stats["candidate_pair_fallback_checks"]
            stop = False
            for left in value_buckets[left_value][:fallback_per_value]:
                for right in value_buckets[right_value][:fallback_per_value]:
                    if (
                        stats["candidate_pair_fallback_checks"]
                        - fallback_checks_before
                        >= max_fallback_checks
                    ):
                        stop = True
                        break
                    consider_pair(
                        left,
                        right,
                        pair_source="candidate_pair_fallback",
                        use_offline_relation=False,
                    )
                if stop:
                    break
            fallback_groups = sum(
                group.pair_source == "candidate_pair_fallback"
                for _, _, group in groups
            )
            stats["candidate_pair_fallback_group_count"] = fallback_groups
            stats["candidate_pair_fallback_used"] = fallback_groups > 0

        stats["complete_candidate_group_count"] = sum(
            self._group_covers_required_values(group, spec)
            for _, _, group in groups
        )
        stats["incomplete_candidate_group_count"] = (
            len(groups) - stats["complete_candidate_group_count"]
        )
        groups.sort(
            key=lambda item: (
                len(item[2].required_values_covered),
                item[2].directional_score if directional_ranking else 0.0,
                int(self._is_direct_seed_evidence(item[0]))
                + int(self._is_direct_seed_evidence(item[1])),
                item[2].group_quality,
                (item[0].candidate_score + item[1].candidate_score) / 2,
            ), reverse=True,
        )
        stats["group_enumeration_seconds"] = perf_counter() - started
        self._last_group_stats = stats
        return groups

    @staticmethod
    def _group_covers_required_values(
        group: SemanticContrastGroup,
        spec: QuestionSpec,
    ) -> bool:
        values = set(group.member_attribute_values)
        required = set(spec.required_attribute_values)
        if required:
            return required.issubset(values)
        return len(values) >= spec.minimum_distinct_attribute_values

    def _semantic_contrast_select(
        self,
        candidates: Sequence[Evidence],
        spec: QuestionSpec,
        *,
        pair_mode: str = "stored_edges",
        adaptive_directional: bool = False,
    ) -> tuple[list[Evidence], list[SemanticContrastGroup], dict[str, Any]]:
        evidence_k = int(self.cfg.get("evidence_k", 6))
        if evidence_k <= 0:
            return [], [], self._coverage_status([], spec)
        groups = self._candidate_groups(
            candidates,
            spec,
            pair_mode=pair_mode,
            directional_ranking=adaptive_directional,
        )
        enumerated_group_count = len(groups)
        # The strict policy is intentionally scoped to two-pole questions. A
        # single pair cannot cover a larger answer space; silently enforcing
        # pair-level completeness there would turn every multi-class query into
        # query-only fallback.
        require_complete_groups = bool(
            adaptive_directional
            and len(set(spec.required_attribute_values)) == 2
            and self.cfg.get("require_complete_contrast_groups", True)
        )
        incomplete_group_count = sum(
            not self._group_covers_required_values(group, spec)
            for _, _, group in groups
        )
        if require_complete_groups:
            groups = [
                item
                for item in groups
                if self._group_covers_required_values(item[2], spec)
            ]
        self._last_group_stats = {
            **getattr(self, "_last_group_stats", {}),
            "incomplete_candidate_groups_rejected": (
                incomplete_group_count if require_complete_groups else 0
            ),
        }
        weights = self.cfg.get("set_weights", {})
        lambda_coverage = float(weights.get("lambda_coverage", 1.0))
        lambda_group = float(weights.get("lambda_group", 0.6))
        lambda_candidate = float(weights.get("lambda_candidate", 0.25))
        lambda_redundancy = float(weights.get("lambda_redundancy", 0.15))
        lambda_optional = float(weights.get("lambda_optional", 0.2))
        lambda_directional = (
            float(weights.get("lambda_directional", 0.0))
            if adaptive_directional
            else 0.0
        )
        # Direct seeds carry visual-locality evidence, not answer-label
        # authority. Reward a contrast *group* for retaining a seed endpoint;
        # never reward the seed endpoint's present/absent value itself.
        lambda_seed_anchor = float(weights.get("lambda_seed_anchor", 0.20))
        configured_max_groups = (
            self.cfg.get("max_contrast_groups")
            if adaptive_directional
            else None
        )
        if configured_max_groups is None:
            max_contrast_groups = evidence_k // 2
        else:
            max_contrast_groups = max(0, int(configured_max_groups))
        max_contrast_groups = min(max_contrast_groups, evidence_k // 2)
        fill_ungrouped_evidence = (
            bool(self.cfg.get("fill_ungrouped_evidence", True))
            if adaptive_directional
            else True
        )
        require_directional_agreement = (
            bool(self.cfg.get("require_directional_group_agreement", False))
            if adaptive_directional
            else False
        )

        selected: list[Evidence] = []
        selected_ids: set[str] = set()
        selected_groups: list[SemanticContrastGroup] = []
        covered_values: set[str] = set()
        remaining_groups = list(groups)
        preferred_group_value = ""

        while (
            len(selected) + 2 <= evidence_k
            and len(selected_groups) < max_contrast_groups
            and remaining_groups
        ):
            best_index: int | None = None
            best_gain = float("-inf")
            for index, (left, right, group) in enumerate(remaining_groups):
                if left.case_id in selected_ids or right.case_id in selected_ids:
                    continue
                if (
                    require_directional_agreement
                    and preferred_group_value
                    and group.query_preferred_attribute_value
                    and group.query_preferred_attribute_value != preferred_group_value
                ):
                    continue
                values = set(group.member_attribute_values)
                new_required = len(
                    (values & set(spec.required_attribute_values)) - covered_values
                )
                new_values = len(values - covered_values)
                optional_gain = len(
                    (values & set(spec.optional_attribute_values)) - covered_values
                )
                redundancy = max(
                    self._candidate_redundancy(left, selected),
                    self._candidate_redundancy(right, selected),
                )
                candidate_score = (
                    left.candidate_score + right.candidate_score
                ) / 2
                seed_anchor_fraction = (
                    int(self._is_direct_seed_evidence(left))
                    + int(self._is_direct_seed_evidence(right))
                ) / 2.0
                gain = (
                    lambda_coverage * (2.0 * new_required + 0.5 * new_values)
                    + lambda_optional * optional_gain
                    + lambda_group * group.group_quality
                    + lambda_directional * group.directional_score
                    + lambda_candidate * candidate_score
                    + lambda_seed_anchor * seed_anchor_fraction
                    - lambda_redundancy * redundancy
                )
                if gain > best_gain:
                    best_gain = gain
                    best_index = index
            if best_index is None:
                break
            left, right, group = remaining_groups.pop(best_index)
            if best_gain <= 0 and selected:
                break
            updates = {
                "semantic_contrast_group_id": group.group_id,
                "group_quality": group.group_quality,
                "non_target_match": group.non_target_match,
                "selection_gain": best_gain,
            }
            selected.extend(
                [
                    left.model_copy(update=updates),
                    right.model_copy(update=updates),
                ]
            )
            selected_ids.update({left.case_id, right.case_id})
            covered_values.update(group.member_attribute_values)
            selected_groups.append(group)
            if not preferred_group_value:
                preferred_group_value = group.query_preferred_attribute_value
            remaining_groups = [
                item
                for item in remaining_groups
                if item[0].case_id not in selected_ids
                and item[1].case_id not in selected_ids
            ]

        remaining = [
            candidate for candidate in candidates if candidate.case_id not in selected_ids
        ]
        while fill_ungrouped_evidence and len(selected) < evidence_k and remaining:
            best_candidate: Evidence | None = None
            best_gain = float("-inf")
            for candidate in remaining:
                value = candidate.attribute_value
                new_required = int(
                    value in spec.required_attribute_values
                    and value not in covered_values
                )
                new_value = int(value not in covered_values)
                optional_gain = int(
                    value in spec.optional_attribute_values
                    and value not in covered_values
                )
                role_bonus = (
                    lambda_optional
                    if candidate.comparison_role == "UNCERTAINTY_CALIBRATOR"
                    else 0.0
                )
                redundancy = self._candidate_redundancy(candidate, selected)
                gain = (
                    lambda_coverage * (2.0 * new_required + 0.5 * new_value)
                    + lambda_optional * optional_gain
                    + role_bonus
                    + lambda_candidate * candidate.candidate_score
                    - lambda_redundancy * redundancy
                )
                if gain > best_gain:
                    best_candidate = candidate
                    best_gain = gain
            if best_candidate is None:
                break
            selected.append(
                best_candidate.model_copy(update={"selection_gain": best_gain})
            )
            selected_ids.add(best_candidate.case_id)
            covered_values.add(best_candidate.attribute_value)
            remaining = [
                candidate
                for candidate in remaining
                if candidate.case_id != best_candidate.case_id
            ]

        coverage = self._coverage_status(selected, spec)
        grouped_values = {
            value for group in selected_groups for value in group.member_attribute_values
        }
        required = set(spec.required_attribute_values)
        group_complete = bool(selected_groups) and (
            required.issubset(grouped_values) if required
            else len(grouped_values) >= spec.minimum_distinct_attribute_values
        )
        coverage.update(
            {
                "semantic_contrast_complete": group_complete,
                "grouped_attribute_values": sorted(grouped_values),
                "missing_grouped_required_attribute_values": sorted(required - grouped_values),
                **getattr(self, "_last_group_stats", {}),
                "selection_mode": "semantic_contrast_group",
                "candidate_count": len(candidates),
                "candidate_attribute_values": sorted({
                    item.attribute_value for item in candidates
                    if item.attribute_value not in {"", "unknown"}
                }),
                "candidate_group_count": len(groups),
                "enumerated_candidate_group_count": enumerated_group_count,
                "selected_group_count": len(selected_groups),
                "group_co_presentation_count": len(selected_groups),
                "selected_seed_anchored_group_count": sum(
                    any(
                        self._is_direct_seed_evidence(item)
                        for item in selected
                        if item.case_id in set(group.member_case_ids)
                    )
                    for group in selected_groups
                ),
                "selection_policy_version": (
                    "directional_complete_contrast_v2"
                    if adaptive_directional
                    else "semantic_contrast_legacy_v1"
                ),
                "configured_evidence_budget": evidence_k,
                "max_contrast_groups": max_contrast_groups,
                "fill_ungrouped_evidence": fill_ungrouped_evidence,
                "require_complete_contrast_groups": require_complete_groups,
                "require_directional_group_agreement": require_directional_agreement,
                "minimum_group_query_endpoint_margin": (
                    float(self.cfg.get("minimum_group_query_endpoint_margin", 0.0))
                    if adaptive_directional
                    else 0.0
                ),
                "selected_query_endpoint_margins": [
                    group.query_endpoint_margin for group in selected_groups
                ],
                "selected_directional_scores": [
                    group.directional_score for group in selected_groups
                ],
                "selected_query_preferred_attribute_values": [
                    group.query_preferred_attribute_value
                    for group in selected_groups
                ],
                "selective_query_only_fallback": (
                    not selected_groups and not fill_ungrouped_evidence
                ),
            }
        )
        return selected[:evidence_k], selected_groups, coverage

    def _value_partition_select(
        self,
        candidates: Sequence[Evidence],
        spec: QuestionSpec,
    ) -> tuple[list[Evidence], dict[str, Any]]:
        evidence_k = int(self.cfg.get("evidence_k", 6))
        buckets: dict[str, list[Evidence]] = defaultdict(list)
        for candidate in candidates:
            buckets[candidate.attribute_value].append(candidate)
        for value in buckets:
            buckets[value].sort(
                key=lambda item: item.candidate_score,
                reverse=True,
            )
        order = [
            *spec.required_attribute_values,
            *spec.optional_attribute_values,
            *sorted(
                value
                for value in buckets
                if value
                not in {
                    *spec.required_attribute_values,
                    *spec.optional_attribute_values,
                }
            ),
        ]
        selected: list[Evidence] = []
        selected_ids: set[str] = set()
        while len(selected) < evidence_k and order:
            next_order: list[str] = []
            for value in order:
                while buckets.get(value) and buckets[value][0].case_id in selected_ids:
                    buckets[value].pop(0)
                if buckets.get(value) and len(selected) < evidence_k:
                    item = buckets[value].pop(0)
                    selected.append(item)
                    selected_ids.add(item.case_id)
                if buckets.get(value):
                    next_order.append(value)
            order = next_order
        coverage = self._coverage_status(selected, spec)
        coverage["selection_mode"] = "value_partition"
        coverage["candidate_count"] = len(candidates)
        coverage["candidate_attribute_values"] = sorted({
            item.attribute_value for item in candidates
            if item.attribute_value not in {"", "unknown"}
        })
        return selected, coverage

    def _pointwise_select(
        self,
        candidates: Sequence[Evidence],
        spec: QuestionSpec,
    ) -> tuple[list[Evidence], dict[str, Any]]:
        selected = []
        seen: set[str] = set()
        for item in sorted(candidates, key=lambda item: item.candidate_score, reverse=True):
            if item.case_id not in seen and len(selected) < int(self.cfg.get("evidence_k", 6)):
                selected.append(item)
                seen.add(item.case_id)
        coverage = self._coverage_status(selected, spec)
        coverage["selection_mode"] = "pointwise_candidate_score"
        coverage["candidate_count"] = len(candidates)
        coverage["candidate_attribute_values"] = sorted({
            item.attribute_value for item in candidates
            if item.attribute_value not in {"", "unknown"}
        })
        return selected, coverage

    # ------------------------------------------------------------------
    # End-to-end conditions
    # ------------------------------------------------------------------

    def _retrieve_from_visual_boundary(
        self,
        query_id: str,
        query_image: str,
        question: str,
        spec: QuestionSpec,
        query_embedding: np.ndarray,
        *,
        method: str,
    ) -> EvidencePackage:
        started = perf_counter()
        seeds = self._seed_list(query_embedding)
        if method == "visig_cr":
            boundary, boundary_stats = self._relation_boundary(seeds, query_embedding, spec)
            boundary_stats["case_normality_fallback"] = spec.target_scope == "case_normality"
        else:
            boundary, boundary_stats = self._visual_boundary(seeds, query_embedding)
            boundary_stats.update(
                {
                    "relation_routing": False,
                    "relation_fallback_used": False,
                }
            )
        routed_at = perf_counter()
        candidates = self._scan_boundary(boundary, spec)
        if method in {"visig_cr", "visual_boundary_semantic_contrast", "visual_boundary_stored_contrast"}:
            evidence, groups, coverage = self._semantic_contrast_select(
                candidates,
                spec,
                pair_mode=(str(self.cfg.get("group_pair_mode", "stored_edges")) if method == "visig_cr"
                           else "stored_edges" if method == "visual_boundary_stored_contrast"
                           else "all_pairs"),
                adaptive_directional=(method == "visig_cr"),
            )
        elif method == "local_value_partition":
            evidence, coverage = self._value_partition_select(candidates, spec)
            groups = []
        else:
            evidence, coverage = self._pointwise_select(candidates, spec)
            groups = []

        boundary_stats["routing_seconds"] = routed_at - started
        boundary_stats["event_scan_and_selection_seconds"] = perf_counter() - routed_at
        if spec.target_scope == "case_normality":
            normality_policy = str(
                self.cfg.get(
                    "case_normality_reference_policy",
                    "explicit_event_supported",
                )
            )
            coverage["normality_policy"] = normality_policy
            boundary_stats["case_normality_reference_policy"] = normality_policy
            coverage["non_target_context_applicable"] = False

        warnings: list[str] = []
        if not boundary:
            warnings.append("The visual boundary is empty.")
        if not candidates:
            warnings.append(
                "No boundary event passed target, confidence, and visual-preservation gates."
            )
        coverage.update(
            {
                "relation_routing": bool(boundary_stats.get("relation_routing", False)),
                "relation_routed_evidence_count": sum(
                    bool(item.relation_types) for item in evidence
                ),
                "no_matched_event_evidence_count": sum(
                    bool(item.used_no_matched_event_ids) for item in evidence
                ),
            }
        )
        if boundary_stats.get("relation_fallback_used"):
            warnings.append(
                "No question-compatible relation expansion was available; "
                "the configured visual-boundary fallback was used."
            )
        if spec.target_scope == "case_normality":
            warnings.append(
                "No explicit finding key was available; reader exemplars were "
                "derived from existing explicit global-normal descriptors and "
                "current positive clinical events over the visual boundary."
            )
        if (coverage.get("selection_mode") == "semantic_contrast_group"
                and coverage.get("required_value_complete")
                and not coverage.get("semantic_contrast_complete")):
            warnings.append(
                "Target values are present, but compatible complete comparison groups are missing."
            )
        missing = coverage.get("missing_required_attribute_values", [])
        if missing:
            warnings.append(
                "Required memory-case values are missing from the delivered evidence: "
                + ", ".join(missing)
            )
        contrast_fallback_used = bool(
            method == "visig_cr"
            and coverage.get("selective_query_only_fallback")
        )
        contrast_fallback_reason = (
            "no_valid_semantic_contrast"
            if contrast_fallback_used
            else ""
        )
        if contrast_fallback_used:
            warnings.append(
                "No valid semantic contrast group passed retrieval selection; "
                "the frozen reader receives the query without retrieved cases."
            )
        return EvidencePackage(
            query_id=query_id,
            query_image_path=query_image,
            question=question,
            retrieval_question=spec.question_text or question,
            options=list(spec.options),
            question_spec=spec,
            method=method,
            seed_cases=self._seed_records(seeds),
            boundary_stats=boundary_stats,
            evidence=evidence,
            semantic_contrast_groups=groups,
            coverage_status=coverage,
            warnings=warnings,
            audit_metadata={
                "contrast_fallback": {
                    "used": contrast_fallback_used,
                    "reason_code": contrast_fallback_reason,
                    "strategy": (
                        "query_only" if contrast_fallback_used else "not_applicable"
                    ),
                    "candidate_group_count": int(
                        coverage.get("candidate_group_count", 0)
                    ),
                    "selected_group_count": int(
                        coverage.get("selected_group_count", 0)
                    ),
                    "selection_policy_version": str(
                        coverage.get("selection_policy_version", "")
                    ),
                },
                "pair_construction": {
                    "complete_contrast_only": bool(
                        coverage.get("require_complete_contrast_groups", False)
                    ),
                    "candidate_pair_fallback": {
                        "enabled": bool(
                            coverage.get("candidate_pair_fallback_enabled", False)
                        ),
                        "attempted": bool(
                            coverage.get("candidate_pair_fallback_attempted", False)
                        ),
                        "used": bool(
                            coverage.get("candidate_pair_fallback_used", False)
                        ),
                        "pair_checks": int(
                            coverage.get("candidate_pair_fallback_checks", 0)
                        ),
                        "valid_group_count": int(
                            coverage.get("candidate_pair_fallback_group_count", 0)
                        ),
                    },
                    "selected_pair_sources": sorted(
                        {group.pair_source for group in groups}
                    ),
                },
            },
        )

    def _retrieve_image_rag(
        self,
        query_id: str,
        query_image: str,
        question: str,
        spec: QuestionSpec,
        query_embedding: np.ndarray,
    ) -> EvidencePackage:
        evidence_k = int(self.cfg.get("evidence_k", 6))
        neighbors = self.visual.search(
            query_embedding,
            max(evidence_k, int(self.cfg.get("k_seed", 8))),
        )
        evidence: list[Evidence] = []
        for case_id, score in neighbors[:evidence_k]:
            case = self.graph.get_case(case_id)
            evidence.append(
                Evidence(
                    case_id=case_id,
                    event_id="",
                    image_path=case.image_path,
                    seed_case_id=case_id,
                    visual_similarity_query=float(score),
                    visual_similarity_seed=1.0,
                    boundary_path=[f"visual:{case_id}"],
                    boundary_hop_count=0,
                    boundary_path_score=float(score),
                    target_match=0.0,
                    non_target_match=0.0,
                    event_confidence=0.0,
                    candidate_score=float(score),
                    target_attribute=spec.target_attribute,
                    attribute_value="unknown",
                    comparison_role="VALUE_EXEMPLAR",
                    event={},
                    evidence_span="",
                )
            )
        return EvidencePackage(
            query_id=query_id,
            query_image_path=query_image,
            question=question,
            retrieval_question=spec.question_text or question,
            options=list(spec.options),
            question_spec=spec,
            method="image_rag",
            seed_cases=self._seed_records(
                neighbors[: int(self.cfg.get("k_seed", 8))]
            ),
            boundary_stats={"boundary_case_count": len(evidence)},
            evidence=evidence,
            coverage_status=self._coverage_status(evidence, spec),
        )

    def _retrieve_selector_fallback_i2i(
        self,
        *,
        query_id: str,
        query_image: str,
        retrieval_question: str,
        delivered_question: str,
        options: Sequence[str],
        query_embedding: np.ndarray,
        reason_code: str,
        selector_error: Exception,
    ) -> EvidencePackage:
        """Build a selector-free visual package for semantic selector failures.

        This path deliberately uses the baseline-native reference presentation:
        no synthetic QuestionSpec, target attribute, event, or relation metadata
        is invented merely to satisfy the structured ViSiG-CR schema.
        """

        fallback_cfg = dict(self.cfg.get("selector_fallback", {}))
        evidence_k = int(fallback_cfg.get("evidence_k", self.cfg.get("evidence_k", 6)))
        if evidence_k <= 0:
            raise ValueError("retrieval.selector_fallback.evidence_k must be positive")
        neighbors = self.visual.search(query_embedding, evidence_k)
        evidence: list[Evidence] = []
        for case_id, score in neighbors:
            case = self.graph.get_case(case_id)
            visual_score = float(score)
            evidence.append(
                Evidence(
                    case_id=case_id,
                    event_id="",
                    image_path=case.image_path,
                    seed_case_id=case_id,
                    visual_similarity_query=visual_score,
                    visual_similarity_seed=visual_score,
                    boundary_path=["selector_fallback_i2i", f"case:{case_id}"],
                    boundary_hop_count=0,
                    boundary_path_score=visual_score,
                    target_match=0.0,
                    non_target_match=0.0,
                    event_confidence=0.0,
                    candidate_score=visual_score,
                    target_attribute=None,
                    attribute_value="unknown",
                    comparison_role="VALUE_EXEMPLAR",
                    event={},
                    evidence_span="",
                    retrieval_signals={"i2i": visual_score},
                    reader_visible_image=True,
                )
            )
        return EvidencePackage(
            reader_presentation="retrieved_references",
            query_id=query_id,
            query_image_path=query_image,
            question=delivered_question,
            retrieval_question=retrieval_question,
            options=list(options),
            question_spec=None,
            method="visig_cr",
            seed_cases=[],
            boundary_stats={
                "attribute_selector_bypassed": True,
                "selector_fallback_used": True,
                "selector_fallback_reason": reason_code,
                "selector_fallback_retriever": "i2i",
                "retrieval_signal": "i2i",
                "requested_reference_count": evidence_k,
                "delivered_reference_count": len(evidence),
                "candidate_count_before_truncation": len(neighbors),
            },
            evidence=evidence,
            semantic_contrast_groups=[],
            coverage_status={},
            warnings=[
                "Attribute selection did not yield a defensible structured target; "
                "the preconfigured selector-free I2I fallback was used."
            ],
            audit_metadata={
                "selector_fallback": {
                    "used": True,
                    "reason_code": reason_code,
                    "retriever": "i2i",
                    "evidence_k": evidence_k,
                    "selector_error_type": type(selector_error).__name__,
                    "selector_error": str(selector_error),
                }
            },
        )

    def _retrieve_visual_pool(
        self,
        query_id: str,
        query_image: str,
        question: str,
        spec: QuestionSpec,
        query_embedding: np.ndarray,
        *,
        semantic_contrast: bool,
    ) -> EvidencePackage:
        pool_size = int(self.cfg.get("visual_pool_size", 64))
        pool = self.visual.search(query_embedding, pool_size)
        boundary = [
            BoundaryEntry(
                case_id=case_id,
                seed_case_id=case_id,
                query_similarity=float(score),
                direct_seed_similarity=1.0,
                path_score=max(0.0, min(1.0, float(score))),
                hop_count=0,
                path_nodes=(case_id,),
            )
            for case_id, score in pool
        ]
        candidates = self._scan_boundary(boundary, spec)
        if semantic_contrast:
            evidence, groups, coverage = self._semantic_contrast_select(
                candidates,
                spec,
                pair_mode="all_pairs",
            )
            method = "visual_pool_semantic_contrast"
        else:
            evidence, coverage = self._value_partition_select(candidates, spec)
            groups = []
            method = "visual_pool"
        return EvidencePackage(
            query_id=query_id,
            query_image_path=query_image,
            question=question,
            retrieval_question=spec.question_text or question,
            options=list(spec.options),
            question_spec=spec,
            method=method,
            seed_cases=self._seed_records(pool[: int(self.cfg.get("k_seed", 8))]),
            boundary_stats={
                "seed_count": min(len(pool), int(self.cfg.get("k_seed", 8))),
                "boundary_case_count": len(boundary),
                "boundary_hops": 0,
            },
            evidence=evidence,
            semantic_contrast_groups=groups,
            coverage_status=coverage,
        )

    def _retrieve_direct_index(
        self,
        query_id: str,
        query_image: str,
        question: str,
        spec: QuestionSpec,
        query_embedding: np.ndarray,
        *,
        semantic_contrast: bool,
    ) -> EvidencePackage:
        if spec.target_scope == "case_normality":
            normalities = self.graph.all_case_normalities(split="train")
            boundary: list[BoundaryEntry] = []
            for case_id in normalities:
                if case_id not in self.visual.id_to_row:
                    continue
                similarity = float(
                    self.visual.similarity_to_query(query_embedding, case_id)
                )
                boundary.append(
                    BoundaryEntry(
                        case_id=case_id,
                        seed_case_id="GLOBAL_INDEX",
                        query_similarity=similarity,
                        direct_seed_similarity=similarity,
                        path_score=max(0.0, min(1.0, similarity)),
                        hop_count=0,
                        path_nodes=(case_id,),
                    )
                )
            candidates = self._scan_case_normality_boundary(boundary, spec)
            if semantic_contrast:
                evidence, groups, coverage = self._semantic_contrast_select(
                    candidates,
                    spec,
                    pair_mode="all_pairs",
                )
                method = "direct_index_semantic_contrast_case_normality"
            else:
                evidence, coverage = self._value_partition_select(candidates, spec)
                groups = []
                method = "direct_index_case_normality"
            seeds = self._seed_list(query_embedding)
            return EvidencePackage(
                query_id=query_id,
                query_image_path=query_image,
                question=question,
                retrieval_question=spec.question_text or question,
                options=list(spec.options),
                question_spec=spec,
                method=method,
                seed_cases=self._seed_records(seeds),
                boundary_stats={
                    "boundary_case_count": len(boundary),
                    "boundary_hops": None,
                    "global_index": True,
                    "case_normality_fallback": True,
                },
                evidence=evidence,
                semantic_contrast_groups=groups,
                coverage_status=coverage,
                warnings=[
                    "No explicit finding key was available; retrieval used the "
                    "deterministic case-level normality fallback."
                ],
            )
        minimum_confidence = float(self.cfg.get("minimum_event_confidence", 0.70))
        events = self.graph.all_events(
            split="train",
            minimum_confidence=minimum_confidence,
            retrieval_eligible_only=(spec.comparison_mode != "candidate_predicate"),
        )
        candidates: list[Evidence] = []
        for event in events:
            if not self._event_allowed_for_spec(event, spec):
                continue
            if event.case_id not in self.visual.id_to_row:
                continue
            value = attribute_value(event, spec.target_attribute)
            if value is None:
                continue
            if (
                spec.comparison_mode == "candidate_predicate"
                and value not in set(spec.answer_space)
            ):
                continue
            target, target_match = best_target_match(event, spec, self.ontology)
            if target is None or target_match < float(
                self.cfg.get("minimum_target_match", 0.60)
            ):
                continue
            similarity = float(
                self.visual.similarity_to_query(query_embedding, event.case_id)
            )
            case = self.graph.get_case(event.case_id)
            candidates.append(
                Evidence(
                    case_id=event.case_id,
                    event_id=event.event_id,
                    image_path=case.image_path,
                    seed_case_id="GLOBAL_INDEX",
                    visual_similarity_query=similarity,
                    visual_similarity_seed=similarity,
                    boundary_path=["global_event_index", f"case:{event.case_id}"],
                    boundary_hop_count=0,
                    boundary_path_score=max(0.0, min(1.0, similarity)),
                    target_match=target_match,
                    non_target_match=target_match,
                    event_confidence=event.confidence,
                    candidate_score=(
                        0.55 * similarity
                        + 0.25 * target_match
                        + 0.20 * event.confidence
                    ),
                    target_attribute=spec.target_attribute,
                    attribute_value=value,
                    comparison_role=comparison_role(value, spec),
                    event=_event_payload(event),
                    evidence_span=event.evidence_span,
                    matched_option_ids=spec.option_ids_for_value(value),
                )
            )
        candidates = self._deduplicate_candidates(candidates)
        if semantic_contrast:
            evidence, groups, coverage = self._semantic_contrast_select(
                candidates,
                spec,
                pair_mode="all_pairs",
            )
            method = "direct_index_semantic_contrast"
        else:
            evidence, coverage = self._value_partition_select(candidates, spec)
            groups = []
            method = "direct_index"
        seeds = self._seed_list(query_embedding)
        return EvidencePackage(
            query_id=query_id,
            query_image_path=query_image,
            question=question,
            retrieval_question=spec.question_text or question,
            options=list(spec.options),
            question_spec=spec,
            method=method,
            seed_cases=self._seed_records(seeds),
            boundary_stats={
                "boundary_case_count": len({item.case_id for item in candidates}),
                "boundary_hops": None,
                "global_index": True,
            },
            evidence=evidence,
            semantic_contrast_groups=groups,
            coverage_status=coverage,
        )

    # ------------------------------------------------------------------
    # Coverage reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _coverage_status(
        evidence: Sequence[Evidence],
        spec: QuestionSpec,
    ) -> dict[str, Any]:
        observed = sorted(
            {
                item.attribute_value
                for item in evidence
                if item.attribute_value not in {"", "unknown"}
            }
        )
        required = list(spec.required_attribute_values)
        covered = [value for value in required if value in observed]
        missing = [value for value in required if value not in observed]
        complete = (
            not missing
            if required
            else len(observed) >= spec.minimum_distinct_attribute_values
        )
        required_option_ids = sorted({
            option_id
            for value in required
            for option_id in spec.option_ids_for_value(value)
        })
        covered_option_ids = sorted({
            option_id
            for value in covered
            for option_id in spec.option_ids_for_value(value)
        })
        return {
            "target_attribute": spec.target_attribute,
            "comparison_mode": spec.comparison_mode,
            "required_attribute_values": required,
            "optional_attribute_values": list(spec.optional_attribute_values),
            "observed_attribute_values": observed,
            "covered_required_attribute_values": covered,
            "missing_required_attribute_values": missing,
            "required_option_ids": required_option_ids,
            "covered_option_ids": covered_option_ids,
            "missing_option_ids": [
                option_id
                for option_id in required_option_ids
                if option_id not in covered_option_ids
            ],
            "minimum_distinct_attribute_values": (
                spec.minimum_distinct_attribute_values
            ),
            "distinct_attribute_value_count": len(observed),
            "required_value_complete": complete,
            "semantic_contrast_complete": complete,
        }


# Public method name used by runners.
GraphRetriever = ViSiGCRRetriever
