"""Reference-free parsing and validation for visible answer spaces.

The multiple-choice helpers validate only model-visible options.  The binary
helpers implement a deliberately conservative yes/no canonicalizer used by the
closed-set VQA preprocessing and evaluation protocol.  They never inspect a
gold label when parsing a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Sequence


_OPTION_PREFIX_RE = re.compile(r"^\s*([A-Za-z])\s*[\.)\]:-]\s*(.*)$")
_OPTION_LETTER_RE = re.compile(
    r"^\s*(?:(?:option|choice)\s+)?\(?\s*([A-Za-z])\s*\)?\s*[\.)\]:-]?\s*$",
    flags=re.IGNORECASE,
)

_BINARY_MODES = {"exact", "leading", "sentence"}
_EXACT_YES = {"yes", "y", "true", "present", "positive"}
_EXACT_NO = {"no", "n", "false", "absent", "negative"}
_LEADING_POLARITY_RE = re.compile(
    r"^\s*(?:answer\s*[:=\-]\s*)?(yes|no)\b",
    flags=re.IGNORECASE,
)
_NEGATIVE_SENTENCE_RE = re.compile(
    r"^(?:"
    r"there\s+(?:is|are|was|were)\s+(?:no|not\s+any)\b|"
    r"no\s+(?:radiographic\s+)?(?:evidence|sign|signs|finding|findings)\b|"
    r"(?:the\s+)?(?:image|radiograph|x[\s-]?ray)\s+"
    r"(?:shows|demonstrates|reveals|depicts|contains)\s+(?:no|without)\b|"
    r"(?:absent|negative)\b"
    r")",
    flags=re.IGNORECASE,
)
_POSITIVE_SENTENCE_RE = re.compile(
    r"^(?:"
    r"there\s+(?:is|are|was|were)\b|"
    r"(?:the\s+)?(?:image|radiograph|x[\s-]?ray)\s+"
    r"(?:shows|demonstrates|reveals|depicts|contains)\b|"
    r"(?:present|positive)\b"
    r")",
    flags=re.IGNORECASE,
)
_AMBIGUITY_RE = re.compile(
    r"\b(?:"
    r"uncertain|indeterminate|equivocal|possible|possibly|perhaps|"
    r"may|might|could|probable|probably|likely|unlikely|"
    r"cannot\s+exclude|can't\s+exclude|can\s+not\s+exclude|"
    r"not\s+excluded|suggestive|suspicious|questionable"
    r")\b",
    flags=re.IGNORECASE,
)
_NEGATIVE_CUE_RE = re.compile(
    r"\b(?:no|not|without|absent|negative|free\s+of)\b",
    flags=re.IGNORECASE,
)
_POSITIVE_CUE_RE = re.compile(
    r"\b(?:present|positive|seen|identified|detected|demonstrated|visualized)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class BinaryAnswerParse:
    """Result of deterministic binary-answer canonicalization."""

    label: str | None
    rule: str
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.label in {"yes", "no"}


def _normalized_visible_text(value: str) -> str:
    """Normalize only typography for safe, exact option-text matching."""

    return " ".join(value.strip().casefold().split())


def _normalized_binary_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("’", "'").replace("`", "")
    text = re.sub(r"\s+", " ", text).strip()
    # Remove one layer of presentation-only quoting/parentheses.  We do not
    # remove arbitrary internal punctuation because it may carry negation.
    text = text.strip(' \t\r\n\"\'()[]{}')
    return text


def _first_clause(text: str, *, max_chars: int = 220) -> str:
    clipped = text[:max_chars]
    match = re.search(r"[.;!?]\s", clipped)
    return clipped[: match.start() if match else len(clipped)]


def _has_ambiguous_language(text: str) -> bool:
    return bool(_AMBIGUITY_RE.search(_first_clause(text)))


def _leading_contradiction(label: str, remainder: str) -> bool:
    clause = _first_clause(remainder)
    if not clause:
        return False
    if label == "yes":
        # Conservative by design: a leading "Yes" followed by a negative cue
        # can answer a negated question or contradict itself.  Such rows require
        # manual adjudication rather than an automatic binary label.
        return bool(_NEGATIVE_CUE_RE.search(clause))

    # "No, there is no ..." is internally consistent.  Reject only when the
    # explanation contains a positive assertion and no negative cue.
    return bool(_POSITIVE_CUE_RE.search(clause)) and not bool(
        _NEGATIVE_CUE_RE.search(clause)
    )


def parse_binary_answer(
    raw_answer: Any,
    *,
    mode: str = "leading",
) -> BinaryAnswerParse:
    """Parse a scalar binary answer with conservative, deterministic rules.

    Modes are cumulative:

    ``exact``
        Accept only a complete yes/no alias such as ``yes``, ``no``,
        ``present``, or ``absent``.
    ``leading``
        Also accept an explicit leading ``Yes, ...`` or ``No, ...`` explanation
        when the first clause is neither hedged nor contradictory.
    ``sentence``
        Also accept a small frozen set of unambiguous sentence-level polarity
        patterns such as ``There is no ...``.  This mode should be enabled only
        after a source-specific audit because it is less conservative.
    """

    normalized_mode = str(mode).strip().lower().replace("-", "_")
    if normalized_mode not in _BINARY_MODES:
        raise ValueError(
            f"Unknown binary polarity mode {mode!r}; expected one of "
            f"{sorted(_BINARY_MODES)}"
        )

    text = _normalized_binary_text(raw_answer)
    if not text:
        return BinaryAnswerParse(None, "none", "non_string_or_empty")

    exact_token = re.sub(r"[\s.!?,;:]+$", "", text.casefold()).strip()
    exact_token = re.sub(r"\s+", "_", exact_token)
    if exact_token in _EXACT_YES:
        return BinaryAnswerParse("yes", "exact_yes_alias")
    if exact_token in _EXACT_NO:
        return BinaryAnswerParse("no", "exact_no_alias")
    if normalized_mode == "exact":
        return BinaryAnswerParse(None, "none", "not_exact_binary_alias")

    leading = _LEADING_POLARITY_RE.match(text)
    if leading:
        label = leading.group(1).casefold()
        remainder = text[leading.end() :].lstrip(" ,:;-—")
        if _has_ambiguous_language(remainder):
            return BinaryAnswerParse(None, "none", "ambiguous_or_hedged_language")
        if _leading_contradiction(label, remainder):
            return BinaryAnswerParse(None, "none", "contradictory_leading_polarity")
        return BinaryAnswerParse(label, f"leading_{label}")

    if normalized_mode != "sentence":
        return BinaryAnswerParse(None, "none", "missing_explicit_leading_polarity")

    if _has_ambiguous_language(text):
        return BinaryAnswerParse(None, "none", "ambiguous_or_hedged_language")
    if _NEGATIVE_SENTENCE_RE.match(text):
        return BinaryAnswerParse("no", "negative_sentence_pattern")
    if _POSITIVE_SENTENCE_RE.match(text):
        return BinaryAnswerParse("yes", "positive_sentence_pattern")
    return BinaryAnswerParse(None, "none", "unmapped_sentence_polarity")


def normalize_binary_answer(raw_answer: Any, *, mode: str = "leading") -> str:
    """Return canonical ``yes`` or ``no``; raise on ambiguous/unmapped text."""

    parsed = parse_binary_answer(raw_answer, mode=mode)
    if not parsed.ok:
        raise ValueError(
            "Binary answer must map deterministically to yes/no; "
            f"reason={parsed.reason}, raw_answer={raw_answer!r}"
        )
    assert parsed.label is not None
    return parsed.label


def option_id_to_text(options: Sequence[str]) -> dict[str, str]:
    """Derive option IDs and bodies without consulting a reference label.

    Explicit labels such as ``A.`` are retained. Unlabeled options receive
    deterministic positional IDs beginning at ``A``.
    """

    mapped: dict[str, str] = {}
    for index, raw_option in enumerate(options):
        text = str(raw_option).strip()
        match = _OPTION_PREFIX_RE.match(text)
        if match:
            option_id = match.group(1).upper()
            body = match.group(2).strip()
        else:
            if index >= 26:
                raise ValueError("Reader supports at most 26 unlabeled options")
            option_id = chr(ord("A") + index)
            body = text
        if not option_id or not body:
            raise ValueError(f"Malformed visible option: {raw_option!r}")
        if option_id in mapped:
            raise ValueError(f"Duplicate visible option ID: {option_id}")
        mapped[option_id] = body
    return mapped


def normalize_option_answer(raw_answer: Any, options: Sequence[str]) -> str:
    """Return the canonical option ID after deterministic validation.

    Accepted repairs are deliberately reference-free: letter typography
    variants (``a``, ``(A)``, ``option A``) and an exact, uniquely matching
    displayed option body. Free-form values such as ``none`` are rejected
    unless that text uniquely identifies one of the visible options.
    """

    if not isinstance(raw_answer, str):
        raise ValueError(
            "Reader answer must be a JSON string; "
            f"got {type(raw_answer).__name__}"
        )
    answer = raw_answer.strip()
    if not answer:
        raise ValueError("Reader output lacks a non-empty answer")

    option_map = option_id_to_text(options)
    if not option_map:
        return answer

    allowed = set(option_map)
    direct = _OPTION_LETTER_RE.match(answer)
    if direct:
        option_id = direct.group(1).upper()
        if option_id in allowed:
            return option_id

    normalized_answer = _normalized_visible_text(answer)
    exact_matches = [
        option_id
        for option_id, body in option_map.items()
        if _normalized_visible_text(body) == normalized_answer
        or _normalized_visible_text(f"{option_id}. {body}") == normalized_answer
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError(
            f"Reader answer ambiguously matches multiple options: {answer!r}"
        )
    raise ValueError(
        f"Reader answer must be one of {sorted(allowed)}; got {answer!r}"
    )


__all__ = [
    "BinaryAnswerParse",
    "normalize_binary_answer",
    "normalize_option_answer",
    "option_id_to_text",
    "parse_binary_answer",
]
