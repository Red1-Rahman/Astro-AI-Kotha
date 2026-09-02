# chatbot/sanitizer/bangla_sanitizer.py
from __future__ import annotations

import re

# Common Bangla conversational fillers.
# Kept conservative so domain-specific astronomical terms are preserved.
_FILLER_PATTERN = re.compile(
    r"(?:"
    r"উম+|"
    r"আহ+|"
    r"হুম+|"
    r"আচ্ছা|"
    r"প্লিজ|"
    r"দয়া করে|"
    r"দয়া করে|"
    r"একটু|"
    r"বলুন তো|"
    r"বলবেন কি|"
    r"জানাবেন কি"
    r")",
    re.IGNORECASE,
)

# Bangla punctuation and common Unicode punctuation are replaced with spaces.
_PUNCTUATION_PATTERN = re.compile(
    r"[^\w\s\u0980-\u09FF]",
    re.UNICODE,
)

_WHITESPACE_PATTERN = re.compile(r"\s+")


class BanglaSanitizationError(RuntimeError):
    """Raised when Bangla query sanitization cannot be completed."""


def sanitize_bangla_query(raw_query: str) -> str:
    """
    Conservatively sanitize a Bangla query for FAQ retrieval.

    Removes common conversational filler, punctuation, and redundant
    whitespace. It does not translate, transliterate, lemmatize, or perform
    keyword extraction.

    If sanitization would produce an empty string, the stripped original
    query is returned.

    Args:
        raw_query: Raw Bangla query.

    Returns:
        Sanitized Bangla query.

    Raises:
        TypeError: If ``raw_query`` is not a string.
        ValueError: If ``raw_query`` is empty or whitespace-only.
        BanglaSanitizationError: If an unexpected sanitization failure occurs.
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
        raise BanglaSanitizationError(
            f"Bangla query sanitization failed: {exc}"
        ) from exc
