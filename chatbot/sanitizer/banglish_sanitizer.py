# chatbot/sanitizer/banglish_sanitizer.py
from __future__ import annotations

import re

# Common Banglish conversational fillers.
# Kept intentionally conservative so domain-specific terms are preserved.
_FILLER_PATTERN = re.compile(
    r"\b(?:"
    r"accha|"
    r"arektu|"
    r"ektu|"
    r"hayre|"
    r"bhai|"
    r"apu|"
    r"please|"
    r"plz|"
    r"pls|"
    r"um+|"
    r"uh+|"
    r"hmm+"
    r")\b",
    re.IGNORECASE,
)

_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


class BanglishSanitizationError(RuntimeError):
    """Raised when Banglish query sanitization cannot be completed."""


def sanitize_banglish_query(raw_query: str) -> str:
    """
    Conservatively sanitize a Banglish query for FAQ retrieval.

    Removes common conversational filler, punctuation, and redundant
    whitespace. Banglish-to-English translation and linguistic normalization
    are handled by separate components.

    If sanitization would produce an empty string, the stripped original
    query is returned.

    Args:
        raw_query: Raw Banglish query.

    Returns:
        Sanitized Banglish query.

    Raises:
        TypeError: If ``raw_query`` is not a string.
        ValueError: If ``raw_query`` is empty or whitespace-only.
        BanglishSanitizationError: If an unexpected sanitization failure occurs.
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
        raise BanglishSanitizationError(
            f"Banglish query sanitization failed: {exc}"
        ) from exc
