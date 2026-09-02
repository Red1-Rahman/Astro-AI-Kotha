# tests/sanitizer/test_bangla_sanitizer.py
from __future__ import annotations

import pytest

from chatbot.sanitizer.bangla_sanitizer import (
    sanitize_bangla_query,
)


@pytest.mark.parametrize(
    ("raw_query", "expected"),
    [
        (
            "অ্যাস্ট্রো-এআই কী?",
            "অ্যাস্ট্রো এআই কী",
        ),
        (
            "আচ্ছা, অ্যাস্ট্রো-এআই কী?",
            "অ্যাস্ট্রো এআই কী",
        ),
        (
            "প্লিজ, গ্যালাক্সি ইভোলিউশন সম্পর্কে বলুন।",
            "গ্যালাক্সি ইভোলিউশন সম্পর্কে",
        ),
        (
            "দয়া করে গ্যালাক্সি বিশ্লেষণ কীভাবে কাজ করে বলুন।",
            "গ্যালাক্সি বিশ্লেষণ কীভাবে কাজ করে",
        ),
        (
            "একটু বলুন তো, Astro-AI কীভাবে কাজ করে?",
            "Astro AI কীভাবে কাজ করে",
        ),
        (
            "হুম... গ্যালাক্সি সম্পর্কে বলবেন কি?",
            "গ্যালাক্সি সম্পর্কে",
        ),
    ],
)
def test_sanitizes_bangla_query(
    raw_query: str,
    expected: str,
) -> None:
    assert sanitize_bangla_query(raw_query) == expected


def test_removes_punctuation() -> None:
    query = "গ্যালাক্সি ইভোলিউশন কীভাবে কাজ করে?!"

    assert sanitize_bangla_query(query) == (
        "গ্যালাক্সি ইভোলিউশন কীভাবে কাজ করে"
    )


def test_normalizes_whitespace() -> None:
    query = "  গ্যালাক্সি   ইভোলিউশন   কী?  "

    assert sanitize_bangla_query(query) == "গ্যালাক্সি ইভোলিউশন কী"


def test_preserves_domain_terms() -> None:
    query = "প্লিজ, Astro-AI দিয়ে গ্যালাক্সি ইভোলিউশন বিশ্লেষণ কীভাবে করা হয়?"

    result = sanitize_bangla_query(query)

    assert "Astro" in result
    assert "AI" in result
    assert "গ্যালাক্সি" in result
    assert "ইভোলিউশন" in result
    assert "বিশ্লেষণ" in result


@pytest.mark.parametrize(
    "query",
    [
        "প্লিজ",
        "আচ্ছা",
        "হুম",
        "একটু",
    ],
)
def test_preserves_query_when_only_filler_is_present(query: str) -> None:
    assert sanitize_bangla_query(query) == query


@pytest.mark.parametrize(
    "invalid_query",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_rejects_non_string_input(invalid_query: object) -> None:
    with pytest.raises(TypeError):
        sanitize_bangla_query(invalid_query)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "   ",
        "\n\t",
    ],
)
def test_rejects_empty_input(query: str) -> None:
    with pytest.raises(ValueError):
        sanitize_bangla_query(query)
