#!/usr/bin/env python3
"""CLI for compact ReXGradient-160K and ReXVQA preparation."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

from .common import clean_text, read_jsonl, write_json
from .rexgradient import prepare_rexgradient
from .rexvqa import prepare_rexvqa

DEFAULT_SOURCE_BASE = Path("/workspace/Datasets/06.CXR")
DEFAULT_OUTPUT_BASE = Path("/workspace/visig_cr/work/datasets/rex")


def _add_common_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-base",
        type=Path,
        default=DEFAULT_SOURCE_BASE,
        help="Parent containing ReXGradient-160K/ and ReXVQA/",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="ViSiG-CR dataset output parent",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs")
    parser.add_argument(
        "--path-mode",
        choices=("symlink", "source"),
        default="symlink",
        help="Use datasets/.../images symlink paths or direct source paths",
    )
    parser.add_argument(
        "--view-priority",
        nargs="+",
        default=["PA", "AP"],
        help="Representative frontal-view priority",
    )
    parser.add_argument(
        "--unknown-view-policy",
        choices=("skip", "single", "first"),
        default="skip",
        help="Fallback when no PA/AP view is known",
    )
    parser.add_argument(
        "--no-scan-fallback",
        action="store_true",
        help="Do not build a recursive SOP-UID index for unresolved paths",
    )
    parser.add_argument(
        "--materialize-8bit",
        action="store_true",
        help="Write selected fixed-range 8-bit PNGs under ReXGradient-160K/images",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Debug only: process at most N records per source split",
    )
    parser.add_argument("--strict", action="store_true")


def _add_vqa_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allowed-validation-code",
        action="append",
        default=None,
        help="Optional ReXVQA validation_code allowlist; repeat for multiple codes",
    )
    parser.add_argument(
        "--evaluator-answer-alias",
        action="store_true",
        help=(
            "Also add an 'answer' field to reference records for evaluators; "
            "reference_answer remains the source field"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare compact ReXGradient-160K/ReXVQA derivatives for ViSiG-CR"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    all_parser = subparsers.add_parser("all", help="Prepare ReXGradient, then ReXVQA")
    _add_common_prepare_args(all_parser)
    _add_vqa_args(all_parser)

    gradient_parser = subparsers.add_parser(
        "rexgradient", help="Prepare ReXGradient-160K only"
    )
    _add_common_prepare_args(gradient_parser)

    vqa_parser = subparsers.add_parser("rexvqa", help="Prepare ReXVQA only")
    _add_common_prepare_args(vqa_parser)
    _add_vqa_args(vqa_parser)

    validate_parser = subparsers.add_parser("validate", help="Validate outputs")
    validate_parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    validate_parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Validate only the first N records per file",
    )
    return parser


def _common_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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


def _write_config_hint(output_base: Path) -> Path:
    target = output_base / "rex_visig_cr_paths.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(
            [
                "# Merge these entries into /workspace/visig_cr/config.yaml.",
                "paths:",
                f"  rexgradient_root: {output_base / 'ReXGradient-160K'}",
                f"  rexvqa_root: {output_base / 'ReXVQA'}",
                f"  cases_jsonl: {output_base / 'ReXGradient-160K' / 'cases.jsonl'}",
                "",
                "# Keep ontology_extraction.allowed_splits and visual_graph.memory_split as train.",
            ]
        ),
        encoding="utf-8",
    )
    return target


def _take(iterator: Iterator[dict[str, Any]], maximum: int | None) -> Iterator[dict[str, Any]]:
    return iterator if maximum is None else itertools.islice(iterator, maximum)


def validate_outputs(
    output_base: str | Path = DEFAULT_OUTPUT_BASE,
    *,
    max_records: int | None = None,
) -> dict[str, Any]:
    base = Path(output_base).absolute()
    gradient_root = base / "ReXGradient-160K"
    vqa_root = base / "ReXVQA"
    result: dict[str, Any] = {
        "output_base": str(base),
        "errors": [],
        "warnings": [],
    }

    cases_path = gradient_root / "cases.jsonl"
    required_case_fields = {
        "case_id",
        "dataset",
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "report_path",
        "view",
        "split",
    }
    case_count = 0
    case_ids: set[str] = set()
    if not cases_path.is_file():
        result["errors"].append(f"Missing {cases_path}")
    else:
        for row in _take(read_jsonl(cases_path), max_records):
            case_count += 1
            missing = required_case_fields - set(row)
            if missing:
                result["errors"].append(
                    f"Case {row.get('case_id')} missing fields: {sorted(missing)}"
                )
            case_id = clean_text(row.get("case_id"))
            if case_id in case_ids:
                result["errors"].append(f"Duplicate case_id: {case_id}")
            case_ids.add(case_id)
            if clean_text(row.get("split")) not in {"train", "validate", "test"}:
                result["errors"].append(
                    f"Invalid split for {case_id}: {row.get('split')}"
                )
            for field in ("image_path", "report_path"):
                path = Path(clean_text(row.get(field)))
                if not path.is_file():
                    result["errors"].append(f"Missing {field} for {case_id}: {path}")
            images = row.get("images")
            if not isinstance(images, list) or not images:
                result["warnings"].append(
                    f"Case {case_id} has no nested multi-view image metadata"
                )
            if len(result["errors"]) > 100:
                result["warnings"].append("Validation stopped after more than 100 errors")
                break
    result["case_count_checked"] = case_count

    vqa_split_results: dict[str, Any] = {}
    forbidden_query_fields = {
        "answer",
        "reference_answer",
        "correct_answer",
        "reference_explanation",
    }
    for split in ("train", "validate", "test"):
        query_path = vqa_root / "queries" / f"{split}.jsonl"
        retrieval_path = vqa_root / "retrieval_queries" / f"{split}.jsonl"
        reference_path = vqa_root / "reference_answers" / f"{split}.jsonl"
        missing_paths = [
            path
            for path in (query_path, retrieval_path, reference_path)
            if not path.is_file()
        ]
        if missing_paths:
            result["errors"].append(
                f"Missing ReXVQA files for split {split}: "
                + ", ".join(str(path) for path in missing_paths)
            )
            continue

        query_iter = _take(read_jsonl(query_path), max_records)
        retrieval_iter = _take(read_jsonl(retrieval_path), max_records)
        reference_iter = _take(read_jsonl(reference_path), max_records)
        count = 0
        alignment_mismatches = 0
        for query, retrieval, reference in itertools.zip_longest(
            query_iter,
            retrieval_iter,
            reference_iter,
        ):
            if query is None or retrieval is None or reference is None:
                alignment_mismatches += 1
                break
            count += 1
            ids = {
                clean_text(query.get("query_id")),
                clean_text(retrieval.get("query_id")),
                clean_text(reference.get("query_id")),
            }
            if len(ids) != 1:
                alignment_mismatches += 1
            leaked = forbidden_query_fields & set(query)
            if leaked:
                result["errors"].append(
                    f"Reference-only fields in query {query.get('query_id')}: {sorted(leaked)}"
                )
            image_path = Path(clean_text(query.get("image_path")))
            if not image_path.is_file():
                result["errors"].append(
                    f"Missing query image {query.get('query_id')}: {image_path}"
                )
            reference_answer = clean_text(
                reference.get("reference_answer") or reference.get("answer")
            )
            if not reference_answer:
                result["errors"].append(
                    f"Missing reference answer: {reference.get('query_id')}"
                )
            if len(result["errors"]) > 100:
                break

        vqa_split_results[split] = {
            "records_checked": count,
            "alignment_mismatches": alignment_mismatches,
        }
    result["vqa"] = vqa_split_results
    result["ok"] = not result["errors"]
    write_json(base / "rex_preprocessing_validation.json", result)
    return result


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "validate":
        result = validate_outputs(args.output_base, max_records=args.max_records)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ok"] else 1)

    source_base = Path(args.source_base).absolute()
    output_base = Path(args.output_base).absolute()
    common = _common_kwargs(args)
    result: dict[str, Any] = {}

    if args.command in {"all", "rexgradient"}:
        result["rexgradient"] = prepare_rexgradient(
            source_root=source_base / "ReXGradient-160K",
            output_root=output_base / "ReXGradient-160K",
            **common,
        )

    if args.command in {"all", "rexvqa"}:
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
            evaluator_answer_alias=bool(
                getattr(args, "evaluator_answer_alias", False)
            ),
            **common,
        )

    result["config_hint"] = str(_write_config_hint(output_base))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
