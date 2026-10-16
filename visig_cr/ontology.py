"""Hierarchical clinical ontology utilities."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import re
from typing import Any, Literal

from .io_utils import load_yaml, normalize_token
from .schemas import CaseNormality, Event


OntologyKind = Literal["finding", "anatomy"]


class Ontology:
    """Load and validate the coverage-frozen CXR ontology.

    The YAML is a controlled extraction vocabulary and a question-time attribute
    specification. Structural validation is performed at startup so alias
    collisions or broken hierarchy links cannot silently contaminate retrieval.
    """

    def __init__(self, path: str | Path, *, validate: bool = True):
        self.path = Path(path)
        self.raw = load_yaml(self.path)
        self.ontology_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.metadata: dict[str, Any] = self.raw.get("metadata", {})
        self.findings: dict[str, dict[str, Any]] = self.raw.get("findings", {})
        self.anatomy: dict[str, dict[str, Any]] = self.raw.get("anatomy", {})
        self.allowed_values: dict[str, list[str]] = self.raw.get("allowed_values", {})
        self.attributes: dict[str, dict[str, Any]] = self.raw.get("attributes", {})
        self.target_attributes: dict[str, dict[str, Any]] = self.raw.get(
            "target_attributes", {}
        )
        self.uncertainty_strength: dict[str, float | None] = self.raw.get(
            "uncertainty_strength", {}
        )
        self.severity_rank: dict[str, int] = self.raw.get("severity_rank", {})
        self.question_patterns: dict[str, list[str]] = self.raw.get(
            "question_patterns", {}
        )
        self.coverage_sets: dict[str, list[str]] = self.raw.get("coverage_sets", {})
        self.dataset_mappings: dict[str, dict[str, str]] = self.raw.get(
            "dataset_mappings", {}
        )
        self.dataset_task_mappings: dict[str, Any] = self.raw.get(
            "dataset_task_mappings", {}
        )
        self.semantic_support: dict[str, Any] = self.raw.get(
            "semantic_support", {}
        )
        raw_families = self.raw.get("finding_families", {})
        self.finding_families: dict[str, list[str]] = {
            str(family_id): [str(member) for member in members]
            for family_id, members in raw_families.items()
            if isinstance(members, list)
        }
        self._finding_to_family: dict[str, str] = {}
        for family_id, members in self.finding_families.items():
            for finding_id in members:
                self._finding_to_family.setdefault(finding_id, family_id)

        self._alias_collisions: dict[str, list[str]] = {}
        self.finding_aliases = self._build_alias_map(self.findings, "finding")
        self.anatomy_aliases = self._build_alias_map(self.anatomy, "anatomy")

        self.differential_pairs: set[tuple[str, str]] = set()
        for pair in self.raw.get("differential_pairs", []):
            if isinstance(pair, list) and len(pair) == 2:
                self.differential_pairs.add(
                    tuple(sorted((str(pair[0]), str(pair[1]))))
                )

        self._validation_report = self._validate_structure()
        if validate and not self._validation_report["ok"]:
            joined = "\n  - ".join(self._validation_report["errors"])
            raise ValueError(f"Invalid ontology {self.path}:\n  - {joined}")

    def _build_alias_map(
        self,
        table: dict[str, dict[str, Any]],
        kind: OntologyKind,
    ) -> dict[str, str]:
        aliases: dict[str, str] = {}
        collisions: defaultdict[str, set[str]] = defaultdict(set)
        for node_id, spec in table.items():
            raw_aliases = [node_id, spec.get("label", ""), *spec.get("synonyms", [])]
            for raw in raw_aliases:
                alias = normalize_token(str(raw))
                if not alias:
                    continue
                previous = aliases.get(alias)
                if previous is not None and previous != node_id:
                    collisions[f"{kind}:{alias}"].update({previous, node_id})
                    continue
                aliases[alias] = node_id
        self._alias_collisions.update(
            {key: sorted(values) for key, values in collisions.items()}
        )
        return aliases

    @property
    def extractable_findings(self) -> list[str]:
        return [
            finding_id
            for finding_id, spec in self.findings.items()
            if bool(spec.get("extractable", not spec.get("abstract", False)))
        ]

    @property
    def selectable_findings(self) -> list[str]:
        return [
            finding_id
            for finding_id, spec in self.findings.items()
            if bool(spec.get("selectable", True))
        ]

    @property
    def clinical_findings(self) -> list[str]:
        return [
            finding_id
            for finding_id in self.extractable_findings
            if self.retrieval_policy(finding_id) == "clinical"
        ]

    @property
    def context_findings(self) -> list[str]:
        return [
            finding_id
            for finding_id in self.extractable_findings
            if self.retrieval_policy(finding_id) != "clinical"
        ]

    def resolve_abnormality_findings(
        self,
        configured: Any = "clinical",
    ) -> set[str]:
        """Resolve the finding set used by the deterministic case normality label.

        ``clinical`` intentionally means every extractable finding whose
        retrieval policy is ``clinical``.  A YAML list may be supplied when an
        experiment needs a narrower benchmark-specific abnormality definition.
        """

        if configured is None or (isinstance(configured, str) and configured in {"clinical", "all_clinical"}):
            return set(self.clinical_findings)
        if not isinstance(configured, list):
            raise ValueError(
                "ontology_extraction.case_normality.abnormal_findings must be "
                "'clinical' or a list of canonical finding IDs"
            )
        values = {normalize_token(str(value)) for value in configured if str(value).strip()}
        unknown = sorted(values - set(self.extractable_findings))
        if unknown:
            raise ValueError(
                "case normality references unknown or non-extractable findings: "
                f"{unknown}"
            )
        return values

    def case_normality(
        self,
        events: list[Event] | tuple[Event, ...],
        *,
        abnormal_findings: Any = "clinical",
    ) -> CaseNormality:
        """Derive one coarse case-level normal/abnormal label from validated events.

        This is deliberately deterministic and does not invoke the ontology LLM.
        A case is ``abnormal`` when at least one configured, retrieval-eligible
        current finding is asserted with ``presence=present``; otherwise it is
        ``normal`` under this configured closed-world retrieval definition.
        """

        abnormal_set = self.resolve_abnormality_findings(abnormal_findings)
        for event in events:
            if (
                event.finding in abnormal_set
                and event.presence == "present"
                and event.temporal_status == "current"
                and event.retrieval_eligible
            ):
                return "abnormal"
        return "normal"

    def canonical_finding(self, value: str) -> str | None:
        return self.finding_aliases.get(normalize_token(value))

    def canonical_anatomy(self, value: str) -> str | None:
        return self.anatomy_aliases.get(normalize_token(value))

    def finding_spec(self, finding_id: str) -> dict[str, Any]:
        return self.findings.get(finding_id, {})

    def default_anatomy(self, finding_id: str) -> str:
        return str(
            self.findings.get(finding_id, {}).get("default_anatomy", "unspecified")
        )

    def allowed_anatomies(self, finding_id: str) -> list[str]:
        values = self.findings.get(finding_id, {}).get("allowed_anatomy")
        if isinstance(values, list) and values:
            return [str(value) for value in values]
        return [self.default_anatomy(finding_id), "unspecified"]

    def anatomy_allowed(self, finding_id: str, anatomy_id: str) -> bool:
        if anatomy_id in {"", "unspecified", "not_applicable"}:
            return True
        allowed = set(self.allowed_anatomies(finding_id))
        if anatomy_id in allowed:
            return True
        # A more specific or broader location remains compatible when it lies
        # on the same explicitly allowed hierarchy path.
        generic_roots = {"thorax"}
        return any(
            self.is_ancestor(candidate, anatomy_id, "anatomy")
            or self.is_ancestor(anatomy_id, candidate, "anatomy")
            for candidate in allowed
            if candidate in self.anatomy
            and candidate not in generic_roots
            and anatomy_id not in generic_roots
        )

    def retrieval_policy(self, finding_id: str) -> str:
        value = str(
            self.findings.get(finding_id, {}).get("retrieval_policy", "clinical")
        )
        return value if value in {"clinical", "context_only", "excluded"} else "excluded"

    def semantic_present_only(self, finding_id: str) -> bool:
        values = self.semantic_support.get("present_only_findings", [])
        return finding_id in {str(value) for value in values}

    def curated_semantic_patterns(
        self,
        finding_id: str,
        presence: str,
    ) -> list[str]:
        mapping = self.semantic_support.get("curated_implications", {})
        if not isinstance(mapping, dict):
            return []
        finding_mapping = mapping.get(finding_id, {})
        if not isinstance(finding_mapping, dict):
            return []
        values = finding_mapping.get(presence, [])
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if str(value).strip()]

    def finding_family(self, finding_id: str) -> str:
        """Return the controlled extraction family for one finding.

        ``finding_families`` is the authoritative model-facing grouping.  The
        hierarchy parent is retained as a deterministic fallback for compatible
        older ontology files.
        """

        explicit = self._finding_to_family.get(finding_id)
        if explicit:
            return explicit
        parent = self.findings.get(finding_id, {}).get("parent")
        return str(parent) if parent else "unclassified"

    def question_attributes_for_finding(self, finding_id: str) -> list[str]:
        values = self.findings.get(finding_id, {}).get("question_attributes")
        if isinstance(values, list):
            return [str(value) for value in values]
        return list(self.target_attributes)

    def label(self, node: str, kind: OntologyKind) -> str:
        table = self.findings if kind == "finding" else self.anatomy
        spec = table.get(node, {})
        return str(spec.get("label", node.replace("_", " ")))

    def description(self, node: str, kind: OntologyKind) -> str:
        table = self.findings if kind == "finding" else self.anatomy
        spec = table.get(node, {})
        return str(spec.get("description", self.label(node, kind)))

    def allowed(self, field: str, value: str) -> bool:
        if field == "finding":
            return value in self.findings and value in self.extractable_findings
        if field == "anatomy":
            return value in self.anatomy
        return value in self.allowed_values.get(field, [])

    def ancestors(self, node: str, kind: OntologyKind) -> list[str]:
        table = self.findings if kind == "finding" else self.anatomy
        out: list[str] = []
        current = node
        seen: set[str] = set()
        while current in table and current not in seen:
            seen.add(current)
            parent = table[current].get("parent")
            if not parent:
                break
            parent = str(parent)
            out.append(parent)
            current = parent
        return out

    def hierarchy_path(self, node: str, kind: OntologyKind) -> list[str]:
        return [node, *self.ancestors(node, kind)]

    def descendants(
        self,
        node: str,
        kind: OntologyKind,
        *,
        extractable_only: bool = False,
    ) -> list[str]:
        """Return descendants in deterministic breadth-first order."""
        table = self.findings if kind == "finding" else self.anatomy
        children: dict[str, list[str]] = {}
        for child, spec in table.items():
            parent = spec.get("parent")
            if parent:
                children.setdefault(str(parent), []).append(child)
        for values in children.values():
            values.sort()

        out: list[str] = []
        queue = list(children.get(node, []))
        seen: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            if (
                not extractable_only
                or kind != "finding"
                or current in self.extractable_findings
            ):
                out.append(current)
            queue.extend(children.get(current, []))
        return out

    def is_ancestor(self, ancestor: str, node: str, kind: OntologyKind) -> bool:
        return ancestor in self.ancestors(node, kind)

    def finding_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        aa, bb = self.ancestors(a, "finding"), self.ancestors(b, "finding")
        if a in bb or b in aa:
            return 0.75
        if (set(aa) & set(bb)) - {"cxr_finding"}:
            return 0.40
        return 0.0

    def anatomy_similarity(self, a: str, b: str) -> float | None:
        masked = {"unspecified", "not_applicable", ""}
        if a in masked or b in masked:
            return None
        if a == b:
            return 1.0
        aa, bb = self.ancestors(a, "anatomy"), self.ancestors(b, "anatomy")
        if a in bb or b in aa:
            return 0.80
        common = set(aa) & set(bb)
        if common - {"thorax", "lung", "bone"}:
            return 0.50
        if common & {"lung", "bone"}:
            return 0.35
        if "thorax" in aa and "thorax" in bb:
            return 0.25
        return 0.0

    def laterality_similarity(self, a: str, b: str) -> float | None:
        masked = {"unspecified", "not_applicable", ""}
        if a in masked or b in masked:
            return None
        if a == b:
            return 1.0
        if {a, b} in ({"left", "bilateral"}, {"right", "bilateral"}):
            return 0.50
        return 0.0

    def severity_similarity(self, a: str, b: str) -> float | None:
        if a not in self.severity_rank or b not in self.severity_rank:
            return None
        max_rank = max(self.severity_rank.values())
        if max_rank <= 0:
            return 1.0
        return 1.0 - abs(self.severity_rank[a] - self.severity_rank[b]) / max_rank

    def uncertainty_similarity(self, a: Event, b: Event) -> float | None:
        if a.presence == "not_mentioned" or b.presence == "not_mentioned":
            return None
        if a.presence != b.presence:
            return 0.0
        qa = self.uncertainty_strength.get(a.uncertainty)
        qb = self.uncertainty_strength.get(b.uncertainty)
        if qa is None or qb is None:
            # ``unspecified`` is missing evidence, not perfect compatibility.
            # Returning ``None`` keeps it masked in downstream relation scoring,
            # matching anatomy/laterality/severity missing-value semantics.
            return None
        return 1.0 - abs(float(qa) - float(qb))

    def is_differential_pair(self, a: str, b: str) -> bool:
        return tuple(sorted((a, b))) in self.differential_pairs

    def answer_space(self, target_attribute: str) -> list[str]:
        spec = self.target_attributes.get(target_attribute, {})
        values = spec.get("answer_space")
        if isinstance(values, list) and values:
            return [str(x) for x in values]
        if target_attribute == "anatomy":
            return list(self.anatomy.keys())
        if target_attribute == "finding_identity":
            return self.extractable_findings
        return []

    def dataset_mapping(self, dataset: str) -> dict[str, str]:
        return dict(self.dataset_mappings.get(dataset, {}))

    def coverage_summary(self) -> dict[str, Any]:
        retrieval_policies: defaultdict[str, int] = defaultdict(int)
        tiers: defaultdict[str, int] = defaultdict(int)
        for finding_id in self.extractable_findings:
            spec = self.findings[finding_id]
            retrieval_policies[self.retrieval_policy(finding_id)] += 1
            tiers[str(spec.get("coverage_tier", "unspecified"))] += 1
        return {
            "ontology_sha256": self.ontology_hash,
            "finding_node_count": len(self.findings),
            "extractable_finding_count": len(self.extractable_findings),
            "selectable_finding_count": len(self.selectable_findings),
            "clinical_retrieval_finding_count": len(self.clinical_findings),
            "context_finding_count": len(self.context_findings),
            "anatomy_node_count": len(self.anatomy),
            "differential_pair_count": len(self.differential_pairs),
            "coverage_sets": {
                name: {"count": len(values), "members": list(values)}
                for name, values in self.coverage_sets.items()
            },
            "retrieval_policies": dict(sorted(retrieval_policies.items())),
            "coverage_tiers": dict(sorted(tiers.items())),
            "validation": self.validation_report(),
        }

    def validation_report(self) -> dict[str, Any]:
        return {
            "ok": bool(self._validation_report["ok"]),
            "errors": list(self._validation_report["errors"]),
            "warnings": list(self._validation_report["warnings"]),
        }

    def _validate_structure(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []

        if not self.findings:
            errors.append("findings table is empty")
        if not self.anatomy:
            errors.append("anatomy table is empty")

        for collision, nodes in sorted(self._alias_collisions.items()):
            errors.append(f"alias collision {collision!r}: {nodes}")

        for kind, table in (("finding", self.findings), ("anatomy", self.anatomy)):
            for node_id, spec in table.items():
                parent = spec.get("parent")
                if parent and str(parent) not in table:
                    errors.append(
                        f"{kind} node {node_id!r} references missing parent {parent!r}"
                    )
                if node_id in self.ancestors(node_id, kind):
                    errors.append(f"cycle detected in {kind} hierarchy at {node_id!r}")

        for finding_id, spec in self.findings.items():
            if bool(spec.get("abstract", False)) and bool(spec.get("extractable", False)):
                errors.append(
                    f"finding {finding_id!r} cannot be both abstract and extractable"
                )
            if bool(spec.get("extractable", False)):
                default = str(spec.get("default_anatomy", "unspecified"))
                if default not in self.anatomy:
                    errors.append(
                        f"finding {finding_id!r} has unknown default anatomy {default!r}"
                    )
                for anatomy_id in spec.get("allowed_anatomy", []):
                    if str(anatomy_id) not in self.anatomy:
                        errors.append(
                            f"finding {finding_id!r} allows unknown anatomy {anatomy_id!r}"
                        )
            policy = str(spec.get("retrieval_policy", "clinical"))
            if policy not in {"clinical", "context_only", "excluded"}:
                errors.append(
                    f"finding {finding_id!r} has invalid retrieval_policy {policy!r}"
                )

        family_membership: defaultdict[str, list[str]] = defaultdict(list)
        for family_id, members in self.finding_families.items():
            if family_id not in self.findings:
                errors.append(
                    f"finding family {family_id!r} is not a finding-hierarchy node"
                )
            for finding_id in members:
                family_membership[finding_id].append(family_id)
                if finding_id not in self.findings:
                    errors.append(
                        f"finding family {family_id!r} references unknown finding "
                        f"{finding_id!r}"
                    )
                    continue
                parent = self.findings[finding_id].get("parent")
                if parent and str(parent) != family_id:
                    errors.append(
                        f"finding family {family_id!r} conflicts with parent of "
                        f"{finding_id!r}: {parent!r}"
                    )
        for finding_id, families in sorted(family_membership.items()):
            if len(families) > 1:
                errors.append(
                    f"finding {finding_id!r} appears in multiple finding families: "
                    f"{sorted(families)}"
                )
        missing_family = sorted(
            set(self.extractable_findings) - set(family_membership)
        )
        if missing_family:
            errors.append(
                "extractable findings missing from finding_families: "
                f"{missing_family}"
            )

        for a, b in sorted(self.differential_pairs):
            if a == b:
                errors.append(f"differential pair contains identical concept {a!r}")
            if a not in self.findings or b not in self.findings:
                errors.append(f"differential pair references unknown finding: {(a, b)}")

        for set_name, members in self.coverage_sets.items():
            duplicates = len(members) - len(set(members))
            if duplicates:
                errors.append(f"coverage set {set_name!r} contains duplicate IDs")
            missing = sorted(set(members) - set(self.findings))
            if missing:
                errors.append(
                    f"coverage set {set_name!r} references unknown findings: {missing}"
                )
        if "conns_41" in self.coverage_sets and len(self.coverage_sets["conns_41"]) != 41:
            errors.append("coverage_sets.conns_41 must contain exactly 41 concepts")

        for dataset, mapping in self.dataset_mappings.items():
            for source_label, target in mapping.items():
                if target not in self.findings:
                    errors.append(
                        f"dataset mapping {dataset}:{source_label!r} -> unknown {target!r}"
                    )

        present_only = self.semantic_support.get("present_only_findings", [])
        if present_only is not None and not isinstance(present_only, list):
            errors.append("semantic_support.present_only_findings must be a list")
        elif isinstance(present_only, list):
            unknown = sorted(
                str(value)
                for value in present_only
                if str(value) not in self.extractable_findings
            )
            if unknown:
                errors.append(
                    "semantic_support.present_only_findings references unknown or "
                    f"non-extractable findings: {unknown}"
                )

        direct_anchor_patterns = self.semantic_support.get(
            "direct_anchor_patterns", {}
        )
        if (
            direct_anchor_patterns is not None
            and not isinstance(direct_anchor_patterns, dict)
        ):
            errors.append(
                "semantic_support.direct_anchor_patterns must be an object"
            )
        elif isinstance(direct_anchor_patterns, dict):
            for finding_id, patterns in direct_anchor_patterns.items():
                finding_id = str(finding_id)
                if finding_id not in self.extractable_findings:
                    errors.append(
                        "semantic_support.direct_anchor_patterns references "
                        f"unknown or non-extractable finding {finding_id!r}"
                    )
                    continue
                if not isinstance(patterns, list) or not patterns:
                    errors.append(
                        "semantic_support.direct_anchor_patterns."
                        f"{finding_id} must be a non-empty list"
                    )
                    continue
                for pattern in patterns:
                    try:
                        re.compile(str(pattern), flags=re.IGNORECASE)
                    except re.error as exc:
                        errors.append(
                            "invalid direct-anchor regex for "
                            f"{finding_id}: {pattern!r}: {exc}"
                        )

        implications = self.semantic_support.get("curated_implications", {})
        if implications is not None and not isinstance(implications, dict):
            errors.append("semantic_support.curated_implications must be an object")
        elif isinstance(implications, dict):
            for finding_id, by_presence in implications.items():
                if str(finding_id) not in self.extractable_findings:
                    errors.append(
                        "semantic_support.curated_implications references unknown or "
                        f"non-extractable finding {finding_id!r}"
                    )
                    continue
                if not isinstance(by_presence, dict):
                    errors.append(
                        "semantic_support.curated_implications."
                        f"{finding_id} must be an object"
                    )
                    continue
                for presence, patterns in by_presence.items():
                    if str(presence) not in {"present", "absent"}:
                        errors.append(
                            "semantic_support.curated_implications."
                            f"{finding_id} has invalid presence {presence!r}"
                        )
                        continue
                    if not isinstance(patterns, list) or not patterns:
                        errors.append(
                            "semantic_support.curated_implications."
                            f"{finding_id}.{presence} must be a non-empty list"
                        )
                        continue
                    for pattern in patterns:
                        try:
                            re.compile(str(pattern), flags=re.IGNORECASE)
                        except re.error as exc:
                            errors.append(
                                "invalid semantic-support regex for "
                                f"{finding_id}.{presence}: {pattern!r}: {exc}"
                            )

        if len(self.extractable_findings) < 41:
            warnings.append(
                "fewer than 41 extractable findings; this is below the intended coverage target"
            )
        if self.metadata.get("clinical_validation_required", True):
            warnings.append(
                "coverage is frozen, but clinical validity still requires external/manual audit"
            )

        return {"ok": not errors, "errors": errors, "warnings": warnings}

    def compact_prompt_spec(self) -> dict[str, Any]:
        return {
            "ontology_sha256": self.ontology_hash,
            "findings": {
                key: {
                    "label": value.get("label", key.replace("_", " ")),
                    "parent": value.get("parent"),
                    "synonyms": value.get("synonyms", []),
                    "default_anatomy": value.get("default_anatomy", "unspecified"),
                    "allowed_anatomy": value.get("allowed_anatomy", []),
                    "retrieval_policy": value.get("retrieval_policy", "clinical"),
                }
                for key, value in self.findings.items()
                if key in self.extractable_findings
            },
            "finding_hierarchy": {
                key: value.get("parent") for key, value in self.findings.items()
            },
            "anatomy": {
                key: {
                    "parent": value.get("parent"),
                    "synonyms": value.get("synonyms", []),
                }
                for key, value in self.anatomy.items()
            },
            "allowed_values": self.allowed_values,
            "attributes": self.attributes,
            "target_attributes": self.target_attributes,
        }

    def extraction_prompt_spec(
        self,
        *,
        profile: str = "minimal_v1",
        max_finding_synonyms: int = 6,
        max_anatomy_synonyms: int = 4,
    ) -> dict[str, Any]:
        """Return a bounded model-facing extraction vocabulary.

        The authoritative ontology, hierarchy, aliases, compatibility rules, and
        retrieval policies remain loaded in memory and are enforced after model
        generation. ``minimal_v2_keyed`` preserves the historical finding-keyed
        model vocabulary. ``minimal_v3_sparse`` exposes the same 57 canonical IDs
        as list items grouped by retrieval role and clinical family, which avoids
        presenting the ontology itself as a JSON completion checklist. The durable
        public representation remains ``events[finding] -> list[event]``.
        """

        normalized_profile = str(profile).strip().lower().replace("-", "_")
        if normalized_profile in {"full", "legacy", "compact_full"}:
            full = self.compact_prompt_spec()
            return {
                "prompt_profile": "full",
                "ontology_sha256": full["ontology_sha256"],
                "findings": full["findings"],
                "anatomy": full["anatomy"],
                "allowed_values": {
                    field: (
                        ["present", "absent"]
                        if field == "presence"
                        else list(full["allowed_values"].get(field, []))
                    )
                    for field in (
                        "laterality",
                        "presence",
                        "severity",
                        "uncertainty",
                    )
                },
            }
        minimal_v1_profiles = {"minimal", "minimal_v1", "bounded"}
        minimal_v2_profiles = {
            "minimal_v2",
            "minimal_v2_keyed",
            "keyed_v2",
            "report_first",
        }
        minimal_v3_profiles = {
            "minimal_v3",
            "minimal_v3_sparse",
            "sparse_v3",
            "report_first_sparse",
        }
        if normalized_profile not in (
            minimal_v1_profiles | minimal_v2_profiles | minimal_v3_profiles
        ):
            raise ValueError(
                "ontology_extraction.prompt_vocabulary_mode must be one of "
                "minimal_v1, minimal_v2_keyed, minimal_v3_sparse, or full"
            )

        max_finding_synonyms = max(0, int(max_finding_synonyms))
        max_anatomy_synonyms = max(0, int(max_anatomy_synonyms))

        def bounded_synonyms(values: Any, maximum: int) -> list[str]:
            if not isinstance(values, list) or maximum == 0:
                return []
            out: list[str] = []
            seen: set[str] = set()
            for value in values:
                text = str(value).strip()
                token = normalize_token(text)
                if not text or not token or token in seen:
                    continue
                seen.add(token)
                out.append(text)
                if len(out) >= maximum:
                    break
            return out

        findings: dict[str, dict[str, Any]] = {}
        used_anatomy: set[str] = {"unspecified", "not_applicable"}
        for finding_id in self.extractable_findings:
            spec = self.findings[finding_id]
            default_anatomy = str(spec.get("default_anatomy", "unspecified"))
            allowed_anatomy = [
                str(value) for value in spec.get("allowed_anatomy", [])
            ]
            if not allowed_anatomy:
                allowed_anatomy = [default_anatomy, "unspecified"]
            used_anatomy.update(allowed_anatomy)
            used_anatomy.add(default_anatomy)
            findings[finding_id] = {
                "label": spec.get("label", finding_id.replace("_", " ")),
                "synonyms": bounded_synonyms(
                    spec.get("synonyms", []),
                    max_finding_synonyms,
                ),
                "default_anatomy": default_anatomy,
                "allowed_anatomy": allowed_anatomy,
            }

        # Include ancestor labels only when an explicitly allowed location relies
        # on them for human-readable disambiguation; parent links themselves are
        # intentionally omitted from the model payload.
        expanded_anatomy = set(used_anatomy)
        for anatomy_id in list(used_anatomy):
            if anatomy_id in self.anatomy:
                expanded_anatomy.update(self.ancestors(anatomy_id, "anatomy"))

        anatomy = {
            anatomy_id: {
                "label": self.anatomy[anatomy_id].get(
                    "label", anatomy_id.replace("_", " ")
                ),
                "synonyms": bounded_synonyms(
                    self.anatomy[anatomy_id].get("synonyms", []),
                    max_anatomy_synonyms,
                ),
            }
            for anatomy_id in self.anatomy
            if anatomy_id in expanded_anatomy
        }
        allowed_values = {
            field: (
                ["present", "absent"]
                if field == "presence"
                else list(self.allowed_values.get(field, []))
            )
            for field in (
                "laterality",
                "presence",
                "severity",
                "uncertainty",
            )
        }

        if normalized_profile in minimal_v1_profiles:
            return {
                "prompt_profile": "minimal_v1",
                "ontology_sha256": self.ontology_hash,
                "findings": findings,
                "anatomy": anatomy,
                "allowed_values": allowed_values,
            }

        # Keep the model-facing vocabulary sparse and deterministic. Retrieval
        # role is represented by the outer key, and the ontology's controlled
        # finding family is represented by the inner key.
        grouped: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            "clinical": {},
            "context_only": {},
            "excluded": {},
        }
        for finding_id in self.extractable_findings:
            role = self.retrieval_policy(finding_id)
            family_id = self.finding_family(finding_id)
            grouped.setdefault(role, {}).setdefault(family_id, {})[finding_id] = findings[
                finding_id
            ]
        grouped = {
            role: families
            for role, families in grouped.items()
            if families
        }
        if normalized_profile in minimal_v3_profiles:
            sparse_groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for role, families in grouped.items():
                sparse_groups[role] = {}
                for family_id, family_findings in families.items():
                    sparse_groups[role][family_id] = [
                        {"id": finding_id, **spec}
                        for finding_id, spec in family_findings.items()
                    ]
            return {
                "prompt_profile": "minimal_v3_sparse",
                "ontology_sha256": self.ontology_hash,
                "finding_groups": sparse_groups,
                "anatomy": anatomy,
                "allowed_values": allowed_values,
            }
        return {
            "prompt_profile": "minimal_v2_keyed",
            "ontology_sha256": self.ontology_hash,
            "finding_groups": grouped,
            "anatomy": anatomy,
            "allowed_values": allowed_values,
        }
