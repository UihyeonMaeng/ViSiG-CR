"""Pure query-only package construction for the no-RAG reader control."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schemas import EvidencePackage


_AUDIT_KEYS = {
    "dataset",
    "source_split",
    "split",
    "subject_id",
    "study_id",
    "dicom_id",
    "case_id",
    "question_family",
    "task_type",
    "target_attribute",
    "reference_format",
    "question_type",
    "source_query_id",
    "source_dataset",
    "source_suite_name",
    "answer_space",
    "closed_set_protocol",
    "conversion_rule_id",
    "answer_space_source",
}


def visible_question(
    question: str,
    options: Sequence[str] | None = None,
    *,
    reader_question: str | None = None,
) -> str:
    """Return exactly the question/options that are visible to the reader."""

    if reader_question and str(reader_question).strip():
        return str(reader_question).strip()
    stem = str(question).strip()
    visible_options = [str(item).strip() for item in (options or []) if str(item).strip()]
    if not visible_options:
        return stem
    return (
        stem
        + "\n\nOPTIONS\n"
        + "\n".join(visible_options)
        + "\n\nIn the required JSON object, set `answer` to exactly one "
        "displayed option letter; include every other required JSON field."
    )


def safe_query_audit_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy non-answer metadata that is useful for audits but never model-visible.

    Closed-binary artifacts created by ``vqa-closed-binary`` carry an explicit
    ``answer_space``.  Older normalized binary query files may contain only
    ``task_type=binary``.  Preserve backward compatibility by materializing the
    audited yes/no answer space for option-free binary rows, and record that the
    contract was inferred rather than silently treating the task as free text.
    """

    metadata = {key: row[key] for key in sorted(_AUDIT_KEYS) if key in row}
    explicit_answer_space = metadata.get("answer_space")
    if explicit_answer_space is not None:
        metadata.setdefault("answer_space_source", "explicit_query_metadata")
        return metadata

    task_type = str(row.get("task_type", "")).strip().casefold().replace("-", "_")
    options = row.get("options", [])
    has_visible_options = isinstance(options, Sequence) and not isinstance(
        options, (str, bytes)
    ) and any(str(item).strip() for item in options)
    if task_type in {"binary", "yes_no", "yes/no"} and not has_visible_options:
        metadata["answer_space"] = ["yes", "no"]
        metadata["answer_space_source"] = "inferred_from_binary_task_type"
    return metadata


def build_no_rag_package(
    *,
    query_id: str,
    image_path: str,
    question: str,
    options: Sequence[str] | None = None,
    reader_question: str | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> EvidencePackage:
    """Build a no-RAG package without ontology, graph, encoder, or selector use."""

    query_id = str(query_id).strip()
    image_path = str(image_path).strip()
    question = str(question).strip()
    if not query_id:
        raise ValueError("no-RAG query_id must be non-empty")
    if not image_path:
        raise ValueError(f"no-RAG query {query_id} lacks image_path")
    if not question and not (reader_question and str(reader_question).strip()):
        raise ValueError(f"no-RAG query {query_id} lacks a visible question")

    option_list = [str(item) for item in (options or [])]
    delivered = visible_question(
        question,
        option_list,
        reader_question=reader_question,
    )
    return EvidencePackage(
        input_mode="no_rag",
        query_id=query_id,
        query_image_path=image_path,
        question=delivered,
        retrieval_question=question,
        options=option_list,
        question_spec=None,
        method="no_rag",
        seed_cases=[],
        boundary_stats={},
        evidence=[],
        semantic_contrast_groups=[],
        coverage_status={},
        warnings=[],
        audit_metadata=dict(audit_metadata or {}),
    )


__all__ = [
    "build_no_rag_package",
    "safe_query_audit_metadata",
    "visible_question",
]
