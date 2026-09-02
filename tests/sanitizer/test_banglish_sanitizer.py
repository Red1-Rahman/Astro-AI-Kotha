# tests/sanitizer/test_banglish_sanitizer.py
from __future__ import annotations

import pytest

from chatbot.sanitizer.banglish_sanitizer import (
    sanitize_banglish_query,
)


@pytest.mark.parametrize(
    ("raw_query", "expected"),
    [
        (
            "Accha Astro AI ki?",
            "Astro AI ki",
        ),
        (
            "Ektu bolen galaxy evolution ki?",
            "bolen galaxy evolution ki",
        ),
        (
            "Arektu explain koren Astro-AI ki kore?",
            "explain koren Astro AI ki kore",
        ),
        (
            "Bhai, Astro-AI kivabe kaj kore?",
            "Astro AI kivabe kaj kore",
        ),
        (
            "Ami ektu jante chai, galaxy evolution ki?",
            "Ami jante chai galaxy evolution ki",
        ),
        (
            "Hmmm... please bolen Astro-AI niye.",
            "bolen Astro AI niye",
        ),
    ],
)
def test_sanitizes_banglish_query(
    raw_query: str,
    expected: str,
) -> None:
    assert sanitize_banglish_query(raw_query) == expected


def test_removes_punctuation() -> None:
    query = "Astro-AI ki?!"

    assert sanitize_banglish_query(query) == "Astro AI ki"


def test_normalizes_whitespace() -> None:
    query = "  Astro   AI   ki   kore?  "

    assert sanitize_banglish_query(query) == "Astro AI ki kore"


def test_preserves_banglish_domain_terms() -> None:
    query = "Ami galaxy evolution niye jante chai"

    result = sanitize_banglish_query(query)

    assert "galaxy" in result
    assert "evolution" in result
    assert "Ami" in result
    assert "jante" in result


def test_preserves_query_when_only_filler_is_present() -> None:
    query = "Ektu"

    assert sanitize_banglish_query(query) == query


def test_preserves_query_when_sanitization_produces_empty_result() -> None:
    query = "hmm"

    assert sanitize_banglish_query(query) == query


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
        sanitize_banglish_query(invalid_query)  # type: ignore[arg-type]


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
        sanitize_banglish_query(query)
