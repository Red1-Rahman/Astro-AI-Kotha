# tests/sanitizer/test_router.py
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from chatbot.language_detector import Language
from chatbot.sanitizer.router import (
    SanitizerRouterError,
    get_sanitizer,
    sanitize_query,
)


@pytest.mark.parametrize(
    ("language", "expected_name"),
    [
        (Language.ENGLISH, "sanitize_english_query"),
        (Language.BANGLA, "sanitize_bangla_query"),
        (Language.BANGLISH, "sanitize_banglish_query"),
    ],
)
def test_get_sanitizer_returns_correct_sanitizer(
    language: Language,
    expected_name: str,
) -> None:
    sanitizer = get_sanitizer(language)

    assert sanitizer.__name__ == expected_name


def test_get_sanitizer_rejects_invalid_language_type() -> None:
    with pytest.raises(TypeError, match="language must be a Language value"):
        get_sanitizer("english")  # type: ignore[arg-type]


def test_get_sanitizer_rejects_unknown_language() -> None:
    with pytest.raises(
        SanitizerRouterError,
        match="No sanitizer is available for language: unknown",
    ):
        get_sanitizer(Language.UNKNOWN)


@pytest.mark.parametrize(
    ("language", "query"),
    [
        (Language.ENGLISH, "What is Astro-AI?"),
        (Language.BANGLA, "অ্যাস্ট্রো এআই কী?"),
        (Language.BANGLISH, "Astro AI ki?"),
    ],
)
def test_sanitize_query_uses_language_specific_sanitizer(
    language: Language,
    query: str,
) -> None:
    sanitizer = Mock(return_value="sanitized query")

    with patch(
        "chatbot.sanitizer.router._SANITIZERS",
        {language: sanitizer},
    ):
        result = sanitize_query(query, language)

    assert result == "sanitized query"
    sanitizer.assert_called_once_with(query)


def test_sanitize_query_rejects_non_string_query() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        sanitize_query(123, Language.ENGLISH)  # type: ignore[arg-type]


def test_sanitize_query_propagates_value_error() -> None:
    sanitizer = Mock(side_effect=ValueError("query must not be empty"))

    with patch(
        "chatbot.sanitizer.router._SANITIZERS",
        {Language.ENGLISH: sanitizer},
    ):
        with pytest.raises(ValueError, match="query must not be empty"):
            sanitize_query("", Language.ENGLISH)


def test_sanitize_query_wraps_unexpected_sanitizer_error() -> None:
    sanitizer = Mock(side_effect=RuntimeError("unexpected failure"))

    with patch(
        "chatbot.sanitizer.router._SANITIZERS",
        {Language.ENGLISH: sanitizer},
    ):
        with pytest.raises(
            SanitizerRouterError,
            match="Sanitization failed for language 'english'",
        ):
            sanitize_query("test query", Language.ENGLISH)


def test_unknown_language_is_rejected_before_sanitization() -> None:
    with pytest.raises(
        SanitizerRouterError,
        match="No sanitizer is available for language: unknown",
    ):
        sanitize_query("test query", Language.UNKNOWN)
