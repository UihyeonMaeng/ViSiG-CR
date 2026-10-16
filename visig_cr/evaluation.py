"""Provenance-aware evaluation for ontology, retrieval groups, and frozen readers.

Reference annotations are loaded only by evaluation commands. Retrieval code never
imports them. Proxy metrics computed from the memory ontology are explicitly marked
as such and should not be reported as clinical validation without an independent
audit.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from .answer_utils import (
    normalize_binary_answer,
    normalize_option_answer,
    option_id_to_text,
)
from .io_utils import normalize_token
from .schemas import EvidencePackage, Event, QuestionSpec, events_from_record


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected a mapping or Pydantic model, got {type(value)!r}")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def canonical_answer(value: Any) -> str:
    """Normalize common Medical VQA answer variants for exact evaluation."""

    token = normalize_token(str(value or ""))
    aliases = {
        "y": "yes",
        "true": "yes",
        "present": "yes",
        "positive": "yes",
        "n": "no",
        "false": "no",
        "absent": "no",
        "negative": "no",
        "lt": "left",
        "l": "left",
        "rt": "right",
        "r": "right",
        "both": "bilateral",
        "b/l": "bilateral",
        "cannot_be_excluded": "cannot_exclude",
    }
    return aliases.get(token, token)


def _event_match_score(pred: Event, ref: dict[str, Any]) -> float:
    if pred.case_id != str(ref.get("case_id", pred.case_id)):
        return -1.0
    if pred.finding != normalize_token(str(ref.get("finding", ""))):
        return -1.0
    if pred.presence != normalize_token(str(ref.get("presence", ""))):
        return -1.0
    score = 2.0
    for field in ("anatomy", "laterality", "severity", "uncertainty"):
        expected = normalize_token(str(ref.get(field, "")))
        if expected and expected not in {"unspecified", "not_applicable"}:
            score += float(getattr(pred, field) == expected)
    return score


def evaluate_ontology_events(
    predicted_events: Sequence[Event | Mapping[str, Any]],
    reference_events: Sequence[Event | Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate report-grounded ontology events with one-to-one matching.

    Event identity is finding + presence within a case. Attribute accuracy and span
    precision are reported separately so a correct finding cannot hide a wrong
    location, side, or evidence span.
    """

    valid_predictions: list[Event] = []
    invalid_prediction_count = 0
    for row in predicted_events:
        try:
            valid_predictions.extend(events_from_record(row))
        except (ValidationError, TypeError, ValueError):
            invalid_prediction_count += 1

    references: list[dict[str, Any]] = []
    for row in reference_events:
        try:
            references.extend(event.model_dump() for event in events_from_record(row))
        except (ValidationError, TypeError, ValueError):
            references.append(_as_dict(row))
    refs_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ref in references:
        refs_by_case[str(ref.get("case_id", ""))].append(ref)

    used_ref_ids: set[tuple[str, int]] = set()
    matched: list[tuple[Event, dict[str, Any]]] = []
    for pred in sorted(valid_predictions, key=lambda item: (item.case_id, item.event_id)):
        candidates = refs_by_case.get(pred.case_id, [])
        best_index: int | None = None
        best_score = -1.0
        for index, ref in enumerate(candidates):
            key = (pred.case_id, index)
            if key in used_ref_ids:
                continue
            score = _event_match_score(pred, ref)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 2.0:
            used_ref_ids.add((pred.case_id, best_index))
            matched.append((pred, candidates[best_index]))

    tp = len(matched)
    precision = _safe_div(tp, len(valid_predictions))
    recall = _safe_div(tp, len(references))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    def attr_accuracy(field: str) -> float:
        checks: list[float] = []
        for pred, ref in matched:
            expected = normalize_token(str(ref.get(field, "")))
            if expected:
                checks.append(float(getattr(pred, field) == expected))
        return _mean(checks)

    span_checks: list[float] = []
    for pred, ref in matched:
        expected = str(ref.get("evidence_span", "")).strip()
        observed = pred.evidence_span.strip()
        if expected:
            span_checks.append(float(observed == expected))
        else:
            span_checks.append(float(bool(observed)))

    total_predictions = len(valid_predictions) + invalid_prediction_count
    return {
        "predicted_event_count": len(valid_predictions),
        "reference_event_count": len(references),
        "matched_event_count": tp,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "evidence_span_precision": _mean(span_checks),
        "anatomy_accuracy": attr_accuracy("anatomy"),
        "laterality_accuracy": attr_accuracy("laterality"),
        "severity_accuracy": attr_accuracy("severity"),
        "uncertainty_accuracy": attr_accuracy("uncertainty"),
        "schema_validity": _safe_div(len(valid_predictions), total_predictions),
        "invalid_prediction_count": invalid_prediction_count,
    }


def evaluate_attribute_selection(
    predicted_specs: Sequence[QuestionSpec | Mapping[str, Any]],
    reference_specs: Sequence[QuestionSpec | Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the text-only Attribute Selector Agent independently of images."""

    predicted = [
        row if isinstance(row, QuestionSpec) else QuestionSpec.model_validate(dict(row))
        for row in predicted_specs
    ]
    references = [
        row if isinstance(row, QuestionSpec) else QuestionSpec.model_validate(dict(row))
        for row in reference_specs
    ]
    n = min(len(predicted), len(references))
    if n == 0:
        return {
            "count": 0,
            "target_attribute_accuracy": 0.0,
            "target_finding_accuracy": 0.0,
            "required_value_exact_match": 0.0,
        }
    return {
        "count": n,
        "target_attribute_accuracy": _mean(
            float(predicted[i].target_attribute == references[i].target_attribute)
            for i in range(n)
        ),
        "target_finding_accuracy": _mean(
            float(predicted[i].target_finding == references[i].target_finding)
            for i in range(n)
        ),
        "target_anatomy_accuracy": _mean(
            float(predicted[i].target_anatomy == references[i].target_anatomy)
            for i in range(n)
        ),
        "target_laterality_accuracy": _mean(
            float(predicted[i].target_laterality == references[i].target_laterality)
            for i in range(n)
        ),
        "required_value_exact_match": _mean(
            float(
                set(predicted[i].required_attribute_values)
                == set(references[i].required_attribute_values)
            )
            for i in range(n)
        ),
    }


def _reference_map(
    annotations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if annotations is None:
        return {}
    if isinstance(annotations, Mapping):
        return {str(key): value for key, value in annotations.items()}
    out: dict[str, Any] = {}
    for row in annotations:
        query_id = str(row.get("query_id", ""))
        group_id = str(row.get("group_id", ""))
        if query_id and group_id:
            out[f"{query_id}:{group_id}"] = dict(row)
        elif query_id:
            out[query_id] = dict(row)
    return out


def evaluate_retrieval_packages(
    packages: Sequence[EvidencePackage | Mapping[str, Any]],
    reference_annotations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    case_annotations: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ontology: Any = None,
    k: int = 6,
    tau_query: float = 0.20,
    tau_non_target: float = 0.60,
    tau_group_visual: float = 0.20,
) -> dict[str, Any]:
    """Evaluate common selected-case annotations; keep reader inputs unchanged.

    Without common annotations, only explicitly marked package-metadata
    diagnostics are available. Baseline missing metadata is N/A, never zero.
    """
    from .retrieval_evaluation import evaluate_packages
    return evaluate_packages(
        packages, reference_annotations, case_annotations=case_annotations,
        ontology=ontology, k=k, tau_query=tau_query,
        tau_non_target=tau_non_target, tau_group_visual=tau_group_visual,
    )


def _macro_f1(
    labels: Sequence[str],
    predictions: Sequence[str],
    *,
    classes: Sequence[str] | None = None,
) -> float:
    evaluated_classes = (
        sorted(set(classes))
        if classes is not None
        else sorted(set(labels) | set(predictions))
    )
    scores: list[float] = []
    for label in evaluated_classes:
        tp = sum(
            reference_label == label and pred == label
            for reference_label, pred in zip(labels, predictions)
        )
        fp = sum(
            reference_label != label and pred == label
            for reference_label, pred in zip(labels, predictions)
        )
        fn = sum(
            reference_label == label and pred != label
            for reference_label, pred in zip(labels, predictions)
        )
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        scores.append(_safe_div(2 * precision * recall, precision + recall))
    return _mean(scores)


def _token_f1(reference: str, prediction: str) -> float:
    reference_tokens = [token for token in normalize_token(reference).split("_") if token]
    prediction_tokens = [token for token in normalize_token(prediction).split("_") if token]
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    ref_counts = Counter(reference_tokens)
    pred_counts = Counter(prediction_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    precision = _safe_div(overlap, len(prediction_tokens))
    recall = _safe_div(overlap, len(reference_tokens))
    return _safe_div(2 * precision * recall, precision + recall)


def _unique_query_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    empty = 0
    for raw in rows:
        row = dict(raw)
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            empty += 1
            continue
        if query_id in result:
            duplicates.append(query_id)
        else:
            result[query_id] = row
    if empty or duplicates:
        raise ValueError(
            f"Invalid {label} IDs: empty={empty}, duplicates={sorted(set(duplicates))[:20]}"
        )
    return result


def _reader_reference_map(
    references: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if isinstance(references, Mapping):
        rows = [
            {
                "query_id": str(key),
                **(
                    dict(value)
                    if isinstance(value, Mapping)
                    else {"reference_answer": value}
                ),
            }
            for key, value in references.items()
        ]
    else:
        rows = [dict(row) for row in references]
    return _unique_query_map(rows, label="reference")


def _as_answer_values(value: Any, *, split_text: bool) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    elif split_text and isinstance(value, str):
        text = value.strip()
        raw_values: list[Any]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            raw_values = parsed
        else:
            raw_values = [
                part for part in re.split(r"\s*(?:[,;|]|\band\b)\s*", text) if part
            ]
    else:
        raw_values = [value]
    values = [canonical_answer(item) for item in raw_values]
    return sorted({item for item in values if item})


def _reference_values(row: Mapping[str, Any]) -> tuple[list[str], bool]:
    for key in ("reference_answers", "answers"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            return _as_answer_values(value, split_text=False), True
    scalar = row.get(
        "answer",
        row.get("reference_answer", row.get("reference_answer_text", "")),
    )
    return _as_answer_values(scalar, split_text=False), False


def _prediction_values(row: Mapping[str, Any], *, multi_answer: bool) -> list[str]:
    value = row.get("answer", row.get("prediction", ""))
    return _as_answer_values(value, split_text=multi_answer)


def _raw_reference_answer(row: Mapping[str, Any]) -> Any:
    """Return the exact scalar answer field selected by the reader evaluator."""

    if "answer" in row:
        return row["answer"]
    if "reference_answer" in row:
        return row["reference_answer"]
    return row.get("reference_answer_text", "")


def _raw_prediction_answer(row: Mapping[str, Any]) -> tuple[Any, str]:
    """Return the raw answer and the field used, without stringifying JSON null."""

    if "answer" in row:
        return row["answer"], "answer"
    if "prediction" in row:
        return row["prediction"], "prediction"
    return None, "missing"


def _coerce_options(value: Any, *, query_id: str, source: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"{source} options for {query_id} must be a JSON array of strings"
        )
    return [str(item) for item in value]


def _visible_options(
    query_id: str,
    reference: Mapping[str, Any],
    expected_options_by_query_id: Mapping[str, Sequence[str]] | None,
) -> list[str]:
    """Resolve the model-visible options and reject package/reference drift."""

    reference_options = _coerce_options(
        reference.get("options"),
        query_id=query_id,
        source="Reference",
    )
    if expected_options_by_query_id is None or query_id not in expected_options_by_query_id:
        return reference_options

    package_options = _coerce_options(
        expected_options_by_query_id[query_id],
        query_id=query_id,
        source="Package",
    )
    if reference_options and package_options:
        reference_map = option_id_to_text(reference_options)
        package_map = option_id_to_text(package_options)
        if reference_map != package_map:
            raise ValueError(
                f"Visible option mismatch for {query_id}: "
                f"reference={reference_map!r}, package={package_map!r}"
            )
    return package_options or reference_options


def _invalid_answer_record(
    *,
    query_id: str,
    reason: str,
    message: str,
    raw_answer: Any,
    answer_field: str,
    allowed_option_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "reason": reason,
        "message": message,
        "answer_field": answer_field,
        "raw_answer": _safe_audit_value(raw_answer),
        "allowed_option_ids": list(allowed_option_ids),
    }


def _safe_audit_value(value: Any) -> Any:
    """Keep issue diagnostics JSON-serializable even for pathological values."""

    try:
        json.dumps(value)
    except (OverflowError, TypeError, ValueError):
        if isinstance(value, int):
            return {"type": "int", "bit_length": value.bit_length()}
        try:
            representation = repr(value)
        except (OverflowError, ValueError):
            representation = f"<{type(value).__name__}>"
        return {"type": type(value).__name__, "repr": representation[:500]}
    return value


def _binary_reference_answer_space(reference: Mapping[str, Any]) -> list[str]:
    value = reference.get("answer_space")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        value = []
    normalized = [canonical_answer(item) for item in value if str(item).strip()]
    if set(normalized) == {"yes", "no"}:
        return ["yes", "no"]
    if str(reference.get("task_type", "")).strip().casefold() == "binary":
        raw = canonical_answer(_raw_reference_answer(reference))
        if raw in {"yes", "no"}:
            return ["yes", "no"]
    return []


def _row_metadata(
    reference: Mapping[str, Any],
    prediction: Mapping[str, Any] | None,
) -> dict[str, str]:
    audit = prediction.get("audit_metadata", {}) if prediction else {}
    if not isinstance(audit, Mapping):
        audit = {}
    result: dict[str, str] = {}
    for key in (
        "dataset",
        "question_family",
        "question_type",
        "task_type",
        "target_attribute",
        "source_split",
        "split",
        "source_dataset",
        "source_suite_name",
        "closed_set_protocol",
    ):
        value = reference.get(key, audit.get(key, "unspecified"))
        result[key] = str(value or "unspecified")
    fallback = audit.get("selector_fallback", {})
    fallback_used = isinstance(fallback, Mapping) and fallback.get("used") is True
    contrast_fallback = audit.get("contrast_fallback", {})
    contrast_fallback_used = (
        isinstance(contrast_fallback, Mapping)
        and contrast_fallback.get("used") is True
    )
    if fallback_used:
        result["retrieval_path"] = "selector_fallback_i2i"
    elif contrast_fallback_used:
        result["retrieval_path"] = "contrast_fallback_query_only"
    else:
        result["retrieval_path"] = "native"
    result["selector_fallback_reason"] = (
        str(fallback.get("reason_code", "unknown"))
        if fallback_used
        else "not_applicable"
    )
    result["contrast_fallback_reason"] = (
        str(contrast_fallback.get("reason_code", "unknown"))
        if contrast_fallback_used
        else "not_applicable"
    )
    return result


def _calibration_metrics(
    confidences: Sequence[float],
    correctness: Sequence[float],
    *,
    calibration_bins: int,
) -> tuple[float, float]:
    ece = 0.0
    if calibration_bins > 0 and confidences:
        for index in range(calibration_bins):
            lo = index / calibration_bins
            hi = (index + 1) / calibration_bins
            members = [
                i
                for i, confidence in enumerate(confidences)
                if (lo <= confidence < hi)
                or (index == calibration_bins - 1 and confidence == 1.0)
            ]
            if not members:
                continue
            bin_conf = _mean(confidences[i] for i in members)
            bin_acc = _mean(correctness[i] for i in members)
            ece += len(members) / len(confidences) * abs(bin_acc - bin_conf)
    brier = _mean(
        (confidence - correct) ** 2
        for confidence, correct in zip(confidences, correctness)
    )
    return float(ece), float(brier)


def _group_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    calibration_bins: int,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unspecified"))].append(row)
    result: dict[str, dict[str, Any]] = {}
    for value, members in sorted(groups.items()):
        raw_predictions = [row for row in members if row.get("prediction_present")]
        valid_predictions = [row for row in members if row.get("prediction_valid")]
        confidences = [float(row["confidence"]) for row in valid_predictions]
        correctness = [float(row["correct"]) for row in valid_predictions]
        ece, brier = _calibration_metrics(
            confidences,
            correctness,
            calibration_bins=calibration_bins,
        )
        result[value] = {
            "reference_count": len(members),
            "prediction_count": len(raw_predictions),
            "valid_prediction_count": len(valid_predictions),
            "invalid_prediction_count": len(raw_predictions) - len(valid_predictions),
            "missing_prediction_count": len(members) - len(raw_predictions),
            "raw_completion_rate": _safe_div(len(raw_predictions), len(members)),
            "valid_completion_rate": _safe_div(len(valid_predictions), len(members)),
            "completion_rate": _safe_div(len(valid_predictions), len(members)),
            "accuracy": _mean(float(row["correct"]) for row in members),
            "answered_only_accuracy": _mean(correctness),
            "mean_token_f1": _mean(float(row.get("token_f1", 0.0)) for row in members),
            "calibration_prediction_count": len(valid_predictions),
            "ece": ece,
            "brier": brier,
        }
    return result


def evaluate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    calibration_bins: int = 10,
    strict_ids: bool = True,
    expected_query_ids: Sequence[str] | set[str] | None = None,
    expected_options_by_query_id: Mapping[str, Sequence[str]] | None = None,
    group_by: Sequence[str] = ("dataset", "question_family"),
) -> dict[str, Any]:
    """Evaluate answers with strict completion and dataset/family stratification.

    By default, reference, prediction, and optional package/query ID sets must be
    identical. Multiple-choice predictions are validated against the visible
    reference/package options. ``strict_ids=False`` is intended only for recovery
    diagnostics; missing and invalid predictions remain explicit failures in the
    all-reference accuracy and are excluded from calibration.
    """

    ref_map = _reader_reference_map(references)
    pred_map = _unique_query_map(predictions, label="prediction")
    reference_ids = set(ref_map)
    prediction_ids = set(pred_map)
    expected_ids = set(str(value) for value in expected_query_ids or reference_ids)

    missing_predictions = sorted(reference_ids - prediction_ids)
    extra_predictions = sorted(prediction_ids - reference_ids)
    missing_references_for_expected = sorted(expected_ids - reference_ids)
    extra_references_vs_expected = sorted(reference_ids - expected_ids)
    if strict_ids and (
        missing_predictions
        or extra_predictions
        or missing_references_for_expected
        or extra_references_vs_expected
    ):
        raise ValueError(
            "Reader evaluation ID mismatch: "
            + json.dumps(
                {
                    "missing_prediction_count": len(missing_predictions),
                    "missing_prediction_ids": missing_predictions[:20],
                    "extra_prediction_count": len(extra_predictions),
                    "extra_prediction_ids": extra_predictions[:20],
                    "missing_reference_for_expected_count": len(
                        missing_references_for_expected
                    ),
                    "extra_reference_vs_expected_count": len(
                        extra_references_vs_expected
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    per_query: list[dict[str, Any]] = []
    scalar_reference_labels: list[str] = []
    scalar_predictions: list[str] = []
    confidences: list[float] = []
    answered_correctness: list[float] = []
    multi_answer_rows: list[dict[str, float]] = []
    invalid_predictions: list[dict[str, Any]] = []

    for query_id in sorted(reference_ids):
        reference = ref_map[query_id]
        prediction = pred_map.get(query_id)
        reference_values, multi_answer = _reference_values(reference)
        if not reference_values:
            raise ValueError(f"Reference {query_id} lacks an answer")
        binary_answer_space = _binary_reference_answer_space(reference)
        if binary_answer_space:
            if multi_answer:
                raise ValueError(
                    f"Binary reference {query_id} must contain one scalar answer"
                )
            try:
                normalized_binary_reference = normalize_binary_answer(
                    _raw_reference_answer(reference),
                    mode="leading",
                )
            except ValueError as exc:
                raise ValueError(
                    f"Binary reference {query_id} has an invalid answer: {exc}"
                ) from exc
            reference_values = [normalized_binary_reference]

        visible_options = _visible_options(
            query_id,
            reference,
            expected_options_by_query_id,
        )
        option_map = option_id_to_text(visible_options) if visible_options else {}
        allowed_option_ids = sorted(option_map)
        if option_map:
            if multi_answer:
                raise ValueError(
                    f"Multiple-choice reference {query_id} must contain one scalar answer"
                )
            try:
                normalized_reference = normalize_option_answer(
                    _raw_reference_answer(reference),
                    visible_options,
                )
            except ValueError as exc:
                raise ValueError(
                    f"Multiple-choice reference {query_id} has an invalid answer: {exc}"
                ) from exc
            reference_values = [canonical_answer(normalized_reference)]

        metadata = _row_metadata(reference, prediction)
        prediction_valid = False
        invalid_answer: dict[str, Any] | None = None
        validation_reasons: list[str] = []
        if prediction is None:
            predicted_values: list[str] = []
            confidence: float | None = None
        else:
            answer_valid = False
            raw_answer, answer_field = _raw_prediction_answer(prediction)
            if raw_answer is None:
                reason = "missing_answer" if answer_field == "missing" else "null_answer"
                invalid_answer = _invalid_answer_record(
                    query_id=query_id,
                    reason=reason,
                    message=(
                        "Prediction does not contain an answer field"
                        if reason == "missing_answer"
                        else "Prediction answer is JSON null"
                    ),
                    raw_answer=raw_answer,
                    answer_field=answer_field,
                    allowed_option_ids=allowed_option_ids,
                )
                predicted_values = []
            elif option_map:
                try:
                    normalized_prediction = normalize_option_answer(
                        raw_answer,
                        visible_options,
                    )
                except ValueError as exc:
                    invalid_answer = _invalid_answer_record(
                        query_id=query_id,
                        reason="invalid_option_answer",
                        message=str(exc),
                        raw_answer=raw_answer,
                        answer_field=answer_field,
                        allowed_option_ids=allowed_option_ids,
                    )
                    predicted_values = []
                else:
                    predicted_values = [canonical_answer(normalized_prediction)]
                    answer_valid = True
            elif binary_answer_space:
                try:
                    normalized_prediction = normalize_binary_answer(
                        raw_answer,
                        mode="leading",
                    )
                except ValueError as exc:
                    invalid_answer = _invalid_answer_record(
                        query_id=query_id,
                        reason="invalid_binary_answer",
                        message=str(exc),
                        raw_answer=raw_answer,
                        answer_field=answer_field,
                        allowed_option_ids=[],
                    )
                    invalid_answer["allowed_answer_values"] = binary_answer_space
                    predicted_values = []
                else:
                    predicted_values = [normalized_prediction]
                    answer_valid = True
            else:
                predicted_values = _prediction_values(
                    prediction,
                    multi_answer=multi_answer,
                )
                if predicted_values:
                    answer_valid = True
                else:
                    invalid_answer = _invalid_answer_record(
                        query_id=query_id,
                        reason="empty_answer",
                        message="Prediction answer is empty after normalization",
                        raw_answer=raw_answer,
                        answer_field=answer_field,
                        allowed_option_ids=[],
                    )
            raw_confidence = prediction.get("confidence")
            confidence_error: tuple[str, str] | None = None
            confidence = None
            if isinstance(raw_confidence, bool):
                confidence_error = (
                    "boolean_confidence",
                    "Prediction confidence must be a JSON number within [0, 1], "
                    "not a boolean",
                )
            elif not isinstance(raw_confidence, (int, float)):
                confidence_error = (
                    "non_numeric_confidence",
                    "Prediction confidence must be a bare JSON number within "
                    "[0, 1]",
                )
            else:
                try:
                    confidence = float(raw_confidence)
                except (OverflowError, TypeError, ValueError):
                    confidence_error = (
                        "non_numeric_confidence",
                        "Prediction confidence must be a finite JSON number "
                        "within [0, 1]",
                    )
                else:
                    if not math.isfinite(confidence):
                        confidence_error = (
                            "non_finite_confidence",
                            "Prediction confidence must be finite and within [0, 1]",
                        )
                        confidence = None
                    elif not 0.0 <= confidence <= 1.0:
                        confidence_error = (
                            "confidence_out_of_range",
                            "Prediction confidence must be finite and within [0, 1]",
                        )
                        confidence = None

            if invalid_answer is not None:
                validation_reasons.append(str(invalid_answer["reason"]))
            if confidence_error is not None:
                validation_reasons.append(confidence_error[0])

            prediction_valid = answer_valid and confidence_error is None
            if validation_reasons:
                invalid_record = invalid_answer or _invalid_answer_record(
                    query_id=query_id,
                    reason=validation_reasons[0],
                    message=(
                        confidence_error[1]
                        if confidence_error is not None
                        else "Prediction validation failed"
                    ),
                    raw_answer=raw_answer,
                    answer_field=answer_field,
                    allowed_option_ids=allowed_option_ids,
                )
                invalid_record["validation_reasons"] = validation_reasons
                if confidence_error is not None:
                    invalid_record["raw_confidence"] = _safe_audit_value(
                        raw_confidence
                    )
                    invalid_record["confidence_error"] = confidence_error[1]
                invalid_predictions.append(invalid_record)

            # Diagnostic mode treats a schema-invalid prediction like an
            # unanswered item for every scored metric. The raw values remain in
            # ``invalid_predictions`` for audit.
            if not prediction_valid:
                predicted_values = []

        reference_set = set(reference_values)
        predicted_set = set(predicted_values)
        correct = float(reference_set == predicted_set) if prediction_valid else 0.0
        intersection = len(reference_set & predicted_set)
        set_precision = _safe_div(intersection, len(predicted_set))
        set_recall = _safe_div(intersection, len(reference_set))
        set_f1 = _safe_div(
            2 * set_precision * set_recall,
            set_precision + set_recall,
        )
        set_jaccard = _safe_div(
            intersection,
            len(reference_set | predicted_set),
        )

        token_f1 = (
            _token_f1(reference_values[0], predicted_values[0])
            if not multi_answer and predicted_values
            else (correct if multi_answer else 0.0)
        )
        row = {
            "query_id": query_id,
            **metadata,
            "prediction_present": prediction is not None,
            "prediction_valid": prediction_valid,
            "invalid_answer_reason": (
                invalid_answer["reason"] if invalid_answer is not None else None
            ),
            "validation_reasons": validation_reasons,
            "is_multiple_choice": bool(option_map),
            "is_binary_closed_set": bool(binary_answer_space),
            "answer_space": binary_answer_space or allowed_option_ids,
            "allowed_option_ids": allowed_option_ids,
            "reference_values": reference_values,
            "predicted_values": predicted_values,
            "multi_answer": multi_answer,
            "correct": correct,
            "confidence": confidence,
            "set_precision": set_precision,
            "set_recall": set_recall,
            "set_f1": set_f1,
            "set_jaccard": set_jaccard,
            "token_f1": token_f1,
        }
        per_query.append(row)

        if prediction_valid:
            assert confidence is not None
            confidences.append(confidence)
            answered_correctness.append(correct)
        if multi_answer:
            multi_answer_rows.append(
                {
                    "exact": correct,
                    "precision": set_precision,
                    "recall": set_recall,
                    "f1": set_f1,
                    "jaccard": set_jaccard,
                }
            )
        else:
            scalar_reference_labels.append(reference_values[0])
            scalar_predictions.append(predicted_values[0] if predicted_values else "")

    if strict_ids and invalid_predictions:
        reason_counts = Counter(
            reason
            for row in invalid_predictions
            for reason in row.get("validation_reasons", [row["reason"]])
        )
        raise ValueError(
            "Reader evaluation contains invalid answers: "
            + json.dumps(
                {
                    "invalid_prediction_count": len(invalid_predictions),
                    "reason_counts": dict(sorted(reason_counts.items())),
                    "invalid_prediction_ids": [
                        row["query_id"] for row in invalid_predictions[:20]
                    ],
                    "invalid_predictions": invalid_predictions[:20],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    ece, brier = _calibration_metrics(
        confidences,
        answered_correctness,
        calibration_bins=calibration_bins,
    )
    accuracy = _mean(float(row["correct"]) for row in per_query)
    option_letters = {"a", "b", "c", "d", "e"}
    is_option_letter_task = bool(scalar_reference_labels) and set(
        scalar_reference_labels
    ).issubset(option_letters)
    is_binary_task = bool(scalar_reference_labels) and set(
        scalar_reference_labels
    ).issubset({"yes", "no"})
    macro_classes: Sequence[str] | None
    if is_option_letter_task:
        macro_classes = sorted(set(scalar_reference_labels))
    elif is_binary_task:
        macro_classes = ["yes", "no"]
    else:
        macro_classes = None
    scalar_macro_f1 = (
        _macro_f1(
            scalar_reference_labels,
            scalar_predictions,
            classes=macro_classes,
        )
        if scalar_reference_labels
        else 0.0
    )

    grouped = {
        field: _group_summary(
            per_query,
            field=field,
            calibration_bins=calibration_bins,
        )
        for field in group_by
    }
    valid_prediction_count = sum(
        bool(row["prediction_valid"]) for row in per_query
    )
    raw_matched_prediction_count = len(reference_ids & prediction_ids)
    result: dict[str, Any] = {
        "reference_count": len(reference_ids),
        "prediction_count": len(prediction_ids),
        "matched_prediction_count": raw_matched_prediction_count,
        "valid_prediction_count": valid_prediction_count,
        "invalid_prediction_count": len(invalid_predictions),
        "invalid_answer_count": len(invalid_predictions),
        "invalid_prediction_ids": [row["query_id"] for row in invalid_predictions],
        "invalid_predictions": invalid_predictions,
        "missing_prediction_count": len(missing_predictions),
        "missing_prediction_ids": missing_predictions,
        "extra_prediction_count": len(extra_predictions),
        "extra_prediction_ids": extra_predictions,
        "raw_completion_rate": _safe_div(
            raw_matched_prediction_count,
            len(reference_ids),
        ),
        "valid_completion_rate": _safe_div(valid_prediction_count, len(reference_ids)),
        "completion_rate": _safe_div(valid_prediction_count, len(reference_ids)),
        "accuracy": accuracy,
        "all_reference_accuracy": accuracy,
        "answered_only_accuracy": _mean(answered_correctness),
        "mean_token_f1": _mean(float(row["token_f1"]) for row in per_query),
        "macro_f1": scalar_macro_f1,
        "scalar_answer_macro_f1": scalar_macro_f1,
        "option_letter_macro_f1": scalar_macro_f1 if is_option_letter_task else None,
        "semantic_class_macro_f1": scalar_macro_f1 if is_binary_task else None,
        "answer_format": (
            "multiple_choice"
            if is_option_letter_task
            else ("binary" if is_binary_task else "other")
        ),
        "calibration_definition": "top_label_confidence_vs_correctness",
        "brier_definition": "mean_squared_error_of_top_label_confidence_and_correctness",
        "calibration_prediction_count": len(confidences),
        "ece": ece,
        "brier": brier,
        "label_counts": dict(sorted(Counter(scalar_reference_labels).items())),
        "multi_answer_question_count": len(multi_answer_rows),
        "multi_answer_exact_match": _mean(row["exact"] for row in multi_answer_rows),
        "multi_answer_set_precision": _mean(
            row["precision"] for row in multi_answer_rows
        ),
        "multi_answer_set_recall": _mean(row["recall"] for row in multi_answer_rows),
        "multi_answer_set_f1": _mean(row["f1"] for row in multi_answer_rows),
        "multi_answer_jaccard": _mean(row["jaccard"] for row in multi_answer_rows),
        "grouped": grouped,
        "per_query": per_query,
    }
    return result
