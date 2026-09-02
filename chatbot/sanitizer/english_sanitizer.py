# chatbot/sanitizer/english_sanitizer.py
from __future__ import annotations

import re

# Conversational phrases that add little value to FAQ retrieval.
# Kept intentionally conservative to avoid removing domain-relevant terms.
_FILLER_PATTERN = re.compile(
    r"\b(?:"
    r"um+|uh+|hmm+|erm+|"
    r"please|"
    r"can you|could you|would you|"
    r"tell me|"
    r"i want to know|"
    r"i would like to know|"
    r"i'd like to know"
    r")\b",
    re.IGNORECASE,
)

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class EnglishSanitizationError(RuntimeError):
    """Raised when English query sanitization cannot be completed."""


def sanitize_english_query(raw_query: str) -> str:
    """
    Conservatively sanitize an English query for FAQ retrieval.

    The sanitizer removes common conversational filler, punctuation, and
    redundant whitespace. It does not perform linguistic normalization,
    translation, stemming, lemmatization, or keyword extraction.

    If sanitization would produce an empty string, the stripped original
    query is returned.

    Args:
        raw_query: Raw English query.

    Returns:
        Sanitized English query.

    Raises:
        TypeError: If ``raw_query`` is not a string.
        ValueError: If ``raw_query`` is empty or whitespace-only.
        EnglishSanitizationError: If an unexpected sanitization failure occurs.
    """
    if not isinstance(raw_query, str):
        raise TypeError("raw_query must be a string")

    normalized = raw_query.strip()

    if not normalized:
        raise ValueError("raw_query must not be empty")

    try:
        cleaned = _FILLER_PATTERN.sub(" ", normalized)
        cleaned = _PUNCTUATION_PATTERN.sub(" ", cleaned)
        cleaned = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()

        return cleaned if cleaned else normalized

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise EnglishSanitizationError(
            f"English query sanitization failed: {exc}"
        ) from exc
