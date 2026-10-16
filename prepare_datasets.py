#!/usr/bin/env python3
"""Prepare ViSiG-CR memory manifests, audit resources, and VQA query streams.

Raw normalization and experimental role assignment are deliberately separate:
``mimic``/``physionet-all`` create the immutable master artifacts, whereas
``mimic-roles`` creates a deterministic patient-disjoint protocol with nested
memory sizes.  Every VQA adapter writes model-visible queries separately from
reference answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from visig_cr import load_config
from visig_cr.preprocessing import (
    ClosedBinarySource,
    merge_closed_binary_vqa,
    filter_vqa_artifacts,
    prepare_chest_imagenome,
    prepare_closed_binary_vqa,
    prepare_medical_cxr_vqa,
    prepare_medical_cxr_vqa_binary_source,
    prepare_mimic,
    prepare_mimic_expert_vqa,
    prepare_mimic_roles,
    prepare_ms_cxr,
    prepare_physionet_all,
    prepare_rad_restruct,
    prepare_radgraph,
    prepare_rexgradient,
    prepare_rexvqa,
    prepare_vindr_cxr,
    prepare_vindr_cxr_vqa,
    subset_vqa_artifacts,
    validate_physionet,
    validate_vqa_artifacts,
)
from visig_cr.preprocessing.prepare_rex_datasets import validate_outputs as validate_rex


DEFAULT_REX_SOURCE = Path("/workspace/Datasets/06.CXR")
DEFAULT_REX_OUTPUT = Path("/workspace/visig_cr/work/datasets/rex")


def _write_rex_config_hint(output_base: Path) -> Path:
    target = output_base / "rex_visig_cr_paths.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "# Merge these entries into /workspace/visig_cr/config.yaml.",
                "paths:",
                f"  rexgradient_root: {output_base / 'ReXGradient-160K'}",
                f"  rexvqa_root: {output_base / 'ReXVQA'}",
                "",
                "# Select a ReX case manifest through a separately named protocol.",
                "# Do not overwrite the MIMIC master_cases_jsonl path.",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))


def _add_physionet_common(parser: argparse.ArgumentParser) -> None:
    _add_config(parser)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="Compatibility limit applied to all relevant streams; prefer explicit limits",
    )


def _add_protocol_common(parser: argparse.ArgumentParser) -> None:
    _add_config(parser)
    parser.add_argument("--protocol", help="Override dataset_protocol.protocol_id")


def _add_rex_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-base", type=Path, default=DEFAULT_REX_SOURCE)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_REX_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--path-mode",
        choices=("symlink", "source"),
        default="symlink",
        help="Use a runtime images symlink or direct source image paths",
    )
    parser.add_argument(
        "--view-priority",
        nargs="+",
        default=["PA", "AP"],
        help="Representative frontal-view priority, for example: PA AP",
    )
    parser.add_argument(
        "--unknown-view-policy",
        choices=("skip", "single", "first"),
        default="skip",
        help="Fallback when no PA/AP view is known",
    )
    parser.add_argument("--no-scan-fallback", action="store_true")
    parser.add_argument("--materialize-8bit", action="store_true")
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "physionet-all",
        help="Prepare MIMIC/JPG, label audits, RadGraph, Chest ImaGenome, and MS-CXR",
    )
    _add_physionet_common(p)
    p.add_argument("--split", choices=["train", "validate", "test", "all"], default="all")
    p.add_argument("--case-limit", type=int)
    p.add_argument("--label-limit", type=int)
    p.add_argument("--auxiliary-limit", type=int)
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Metadata-only smoke mode; never use its cases.jsonl for visual indexing",
    )
    p.add_argument("--include-radgraph-text", action="store_true")

    p = sub.add_parser("mimic", help="Prepare the MIMIC master case manifest and label audits")
    _add_physionet_common(p)
    p.add_argument("--split", choices=["train", "validate", "test", "all"], default="all")
    p.add_argument("--case-limit", type=int)
    p.add_argument("--label-limit", type=int)
    p.add_argument("--skip-labels", action="store_true")
    p.add_argument("--allow-missing-images", action="store_true")

    p = sub.add_parser("radgraph", help="Normalize RadGraph report graphs")
    _add_physionet_common(p)
    p.add_argument("--include-report-text", action="store_true")

    p = sub.add_parser("chest-imagenome", help="Stream Chest ImaGenome scene graphs to JSONL")
    _add_physionet_common(p)

    p = sub.add_parser("ms-cxr", help="Convert MS-CXR COCO grounding annotations")
    _add_physionet_common(p)

    p = sub.add_parser("validate-physionet", help="Validate raw and normalized PhysioNet artifacts")
    _add_config(p)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--max-records", type=int, default=100)
    p.add_argument("--full-image-check", action="store_true")
    p.add_argument("--source-only", action="store_true")

    p = sub.add_parser(
        "mimic-roles",
        help="Create patient-disjoint query cases and nested MIMIC memory manifests",
    )
    _add_protocol_common(p)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("mimic-vqa-eval", help="Prepare the expert-curated MIMIC-CXR VQA set")
    _add_protocol_common(p)
    p.add_argument("--source", type=Path)
    p.add_argument("--output-root", type=Path)

    p = sub.add_parser("medical-cxr-vqa", help="Prepare Medical-CXR-VQA against held-out MIMIC cases")
    _add_protocol_common(p)
    p.add_argument("--source", type=Path)
    p.add_argument("--output-root", type=Path)

    p = sub.add_parser(
        "medical-cxr-vqa-binary-source",
        help=(
            "Prepare the uncapped patient-disjoint Medical-CXR-VQA source view "
            "for strict binary conversion"
        ),
    )
    _add_protocol_common(p)
    p.add_argument("--source", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument(
        "--source-question-family",
        action="append",
        help=(
            "Raw Medical-CXR-VQA source family to admit before semantic "
            "presence mapping; repeat for presence plus abnormality"
        ),
    )

    p = sub.add_parser(
        "vindr-cxr",
        help="Index dataset-side VinDr PNGs without copying image bytes into work",
    )
    _add_config(p)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--max-images", type=int)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a smoke subset instead of requiring 15,000 train and 3,000 test PNGs",
    )

    p = sub.add_parser("vindr-cxr-vqa", help="Prepare the semantic-compatible VinDr-CXR-VQA subset")
    _add_config(p)
    p.add_argument("--source", type=Path)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--image-manifest", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--question-type", action="append")
    p.add_argument("--max-questions", type=int)
    p.add_argument(
        "--missing-split-policy",
        choices=("error", "full-release-test"),
        help=(
            "How to handle public VinDr-CXR-VQA rows without split metadata. "
            "full-release-test is an explicit full-release external-evaluation "
            "policy, not an official source test split."
        ),
    )
    p.add_argument(
        "--include-grounding-task",
        action="store_true",
        help="Keep semantic queries but mark the run as not semantic-only; boxes remain reference-only",
    )

    p = sub.add_parser("rad-restruct", help="Prepare Rad-ReStruct as a separate hierarchical VQA protocol")
    _add_config(p)
    p.add_argument("--source", type=Path)
    p.add_argument("--image-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--max-reports", type=int)

    p = sub.add_parser(
        "vqa-closed-binary",
        help="Create an auditable binary presence view from normalized VQA artifacts",
    )
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument(
        "--polarity-mode",
        choices=("exact", "leading", "sentence"),
        default="exact",
        help=(
            "exact: full-answer aliases only; leading: also explicit Yes/No prefixes; "
            "sentence: also frozen sentence-polarity patterns after audit"
        ),
    )
    p.add_argument(
        "--question-type",
        action="append",
        help="Retain only this normalized source question type; repeat as needed",
    )
    p.add_argument(
        "--source-question-family",
        action="append",
        help=(
            "Retain only this raw source family before semantic remapping; "
            "repeat as needed"
        ),
    )
    p.add_argument(
        "--allow-non-presence-family",
        action="store_true",
        help="Disable the default presence-family gate (not recommended for main results)",
    )
    p.add_argument(
        "--require-polar-question",
        action="store_true",
        help=(
            "Require an explicit yes/no question form beginning with a frozen "
            "auxiliary/modal prefix"
        ),
    )
    p.add_argument(
        "--allow-negated-question",
        action="store_true",
        help=(
            "Allow explicitly negated predicates; disabled by default because "
            "yes/no then reverses ordinary finding-presence semantics"
        ),
    )
    p.add_argument("--require-gt-finding", action="store_true")
    p.add_argument("--gold-provenance")
    p.add_argument("--audit-per-stratum", type=int, default=25)
    p.add_argument("--audit-per-exclusion-reason", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "vqa-merge-binary",
        help="Merge canonical binary views while preserving per-source provenance",
    )
    p.add_argument(
        "--source",
        nargs=3,
        action="append",
        required=True,
        metavar=("NAME", "QUERIES", "REFERENCES"),
        help="Repeat in priority order; the first source wins same-label duplicates",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--dataset-name", default="MIMIC-CXR Binary Suite")
    p.add_argument("--conflict-policy", choices=("exclude", "error"), default="exclude")
    p.add_argument("--audit-per-source-label", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser(
        "vqa-filter",
        help="Create an aligned VQA subset by a query metadata field",
    )
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--field", required=True)
    p.add_argument("--value", default="true")

    p = sub.add_parser("vqa-subset", help="Create an aligned deterministic query/reference smoke subset")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("validate-vqa", help="Validate answer isolation, image paths, and query/reference IDs")
    p.add_argument("--queries", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--allow-missing-images", action="store_true")

    p = sub.add_parser("rex-all", help="Prepare ReXGradient, then ReXVQA")
    _add_rex_common(p)
    p.add_argument("--allowed-validation-code", action="append", default=None)
    p.add_argument("--evaluator-answer-alias", action="store_true")

    p = sub.add_parser("rexgradient", help="Prepare ReXGradient only")
    _add_rex_common(p)

    p = sub.add_parser("rexvqa", help="Prepare ReXVQA only")
    _add_rex_common(p)
    p.add_argument("--allowed-validation-code", action="append", default=None)
    p.add_argument("--evaluator-answer-alias", action="store_true")

    p = sub.add_parser("validate-rex", help="Validate existing ReX derivatives")
    p.add_argument("--output-base", type=Path, default=DEFAULT_REX_OUTPUT)
    p.add_argument("--max-records", type=int)
    return parser


def _rex_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "overwrite": bool(args.overwrite),
        "path_mode": str(args.path_mode),
        "view_priority": tuple(str(item).upper() for item in args.view_priority),
        "unknown_view_policy": str(args.unknown_view_policy),
        "scan_fallback": not bool(args.no_scan_fallback),
        "materialize_8bit": bool(args.materialize_8bit),
        "limit_per_split": args.limit_per_split,
        "strict": bool(args.strict),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    command = args.command
    if command in {
        "physionet-all",
        "mimic",
        "radgraph",
        "chest-imagenome",
        "ms-cxr",
        "validate-physionet",
        "mimic-roles",
        "mimic-vqa-eval",
        "medical-cxr-vqa",
        "medical-cxr-vqa-binary-source",
        "vindr-cxr",
        "vindr-cxr-vqa",
        "rad-restruct",
    }:
        cfg = load_config(args.config)
        if command == "physionet-all":
            return prepare_physionet_all(
                cfg,
                output_root=args.output_root,
                split=args.split,
                limit=args.limit,
                case_limit=args.case_limit,
                label_limit=args.label_limit,
                auxiliary_limit=args.auxiliary_limit,
                skip_labels=bool(args.skip_labels),
                allow_missing_images=bool(args.allow_missing_images),
                include_radgraph_text=bool(args.include_radgraph_text),
            )
        if command == "mimic":
            return prepare_mimic(
                cfg,
                output_root=args.output_root,
                split=args.split,
                limit=args.limit,
                case_limit=args.case_limit,
                label_limit=args.label_limit,
                skip_labels=bool(args.skip_labels),
                allow_missing_images=bool(args.allow_missing_images),
            )
        if command == "radgraph":
            return prepare_radgraph(
                cfg,
                output_root=args.output_root,
                include_report_text=bool(args.include_report_text),
                limit_per_file=args.limit,
            )
        if command == "chest-imagenome":
            return prepare_chest_imagenome(cfg, output_root=args.output_root, limit=args.limit)
        if command == "ms-cxr":
            return prepare_ms_cxr(cfg, output_root=args.output_root, limit=args.limit)
        if command == "validate-physionet":
            return validate_physionet(
                cfg,
                output_root=args.output_root,
                max_records=args.max_records,
                full_image_check=bool(args.full_image_check),
                require_outputs=not bool(args.source_only),
            )
        if command == "mimic-roles":
            return prepare_mimic_roles(
                cfg,
                requested_protocol=args.protocol,
                output_root=args.output_root,
                overwrite=bool(args.overwrite),
            )
        if command == "mimic-vqa-eval":
            return prepare_mimic_expert_vqa(
                cfg,
                requested_protocol=args.protocol,
                source=args.source,
                output_root=args.output_root,
            )
        if command == "medical-cxr-vqa":
            return prepare_medical_cxr_vqa(
                cfg,
                requested_protocol=args.protocol,
                source=args.source,
                output_root=args.output_root,
            )
        if command == "medical-cxr-vqa-binary-source":
            return prepare_medical_cxr_vqa_binary_source(
                cfg,
                requested_protocol=args.protocol,
                source=args.source,
                output_root=args.output_root,
                source_question_families=args.source_question_family,
            )
        if command == "vindr-cxr":
            return prepare_vindr_cxr(
                cfg,
                image_root=args.image_root,
                output_root=args.output_root,
                max_images=args.max_images,
                require_complete=not bool(args.allow_incomplete),
            )
        if command == "vindr-cxr-vqa":
            return prepare_vindr_cxr_vqa(
                cfg,
                source=args.source,
                image_root=args.image_root,
                image_manifest=args.image_manifest,
                output_root=args.output_root,
                semantic_only=not bool(args.include_grounding_task),
                allowed_question_types=args.question_type,
                max_questions=args.max_questions,
                missing_split_policy=args.missing_split_policy,
            )
        if command == "rad-restruct":
            return prepare_rad_restruct(
                cfg,
                source=args.source,
                image_root=args.image_root,
                output_root=args.output_root,
                max_reports=args.max_reports,
            )

    if command == "vqa-closed-binary":
        return prepare_closed_binary_vqa(
            queries_path=args.queries,
            references_path=args.references,
            output_root=args.output_root,
            split=args.split,
            polarity_mode=args.polarity_mode,
            allowed_question_types=args.question_type,
            allowed_source_question_families=args.source_question_family,
            require_presence_family=not bool(args.allow_non_presence_family),
            require_polar_question=bool(args.require_polar_question),
            allow_negated_question=bool(args.allow_negated_question),
            require_gt_finding=bool(args.require_gt_finding),
            gold_provenance=args.gold_provenance,
            audit_per_stratum=args.audit_per_stratum,
            audit_per_exclusion_reason=args.audit_per_exclusion_reason,
            seed=args.seed,
        )
    if command == "vqa-merge-binary":
        return merge_closed_binary_vqa(
            sources=[
                ClosedBinarySource(str(name), Path(queries), Path(references))
                for name, queries, references in args.source
            ],
            output_root=args.output_root,
            split=args.split,
            dataset_name=args.dataset_name,
            conflict_policy=args.conflict_policy,
            audit_per_source_label=args.audit_per_source_label,
            seed=args.seed,
        )

    if command == "vqa-filter":
        return filter_vqa_artifacts(
            queries_path=args.queries,
            references_path=args.references,
            output_root=args.output_root,
            field=args.field,
            value=args.value,
        )
    if command == "vqa-subset":
        return subset_vqa_artifacts(
            queries_path=args.queries,
            references_path=args.references,
            output_root=args.output_root,
            limit=args.limit,
            seed=args.seed,
        )
    if command == "validate-vqa":
        return validate_vqa_artifacts(
            queries_path=args.queries,
            references_path=args.references,
            require_images=not bool(args.allow_missing_images),
        )
    if command == "validate-rex":
        return validate_rex(args.output_base, max_records=args.max_records)

    source_base = args.source_base.absolute()
    output_base = args.output_base.absolute()
    kwargs = _rex_kwargs(args)
    result: dict[str, Any] = {}
    if command in {"rex-all", "rexgradient"}:
        result["rexgradient"] = prepare_rexgradient(
            source_root=source_base / "ReXGradient-160K",
            output_root=output_base / "ReXGradient-160K",
            **kwargs,
        )
    if command in {"rex-all", "rexvqa"}:
        result["rexvqa"] = prepare_rexvqa(
            source_root=source_base / "ReXVQA",
            output_root=output_base / "ReXVQA",
            rexgradient_source_root=source_base / "ReXGradient-160K",
            rexgradient_output_root=output_base / "ReXGradient-160K",
            allowed_validation_codes=(
                set(args.allowed_validation_code)
                if getattr(args, "allowed_validation_code", None)
                else None
            ),
            evaluator_answer_alias=bool(getattr(args, "evaluator_answer_alias", False)),
            **kwargs,
        )
    result["config_hint"] = str(_write_rex_config_hint(output_base))
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"validate-physionet", "validate-rex", "validate-vqa"}:
        return 0 if result.get("ok", False) else 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
