# tests/sanitizer/test_english_sanitizer.py
from __future__ import annotations

import pytest

from chatbot.sanitizer.english_sanitizer import (
    sanitize_english_query,
)


@pytest.mark.parametrize(
    ("raw_query", "expected"),
    [
        (
            "What is Astro-AI?",
            "What is Astro AI",
        ),
        (
            "Can you tell me about galaxy evolution?",
            "about galaxy evolution",
        ),
        (
            "Please explain the galaxy analysis pipeline.",
            "explain the galaxy analysis pipeline",
        ),
        (
            "Um, how does the system work?",
            "how does the system work",
        ),
        (
            "Hmm... could you explain the project?",
            "explain the project",
        ),
        (
            "I would like to know about Astro-AI!",
            "about Astro AI",
        ),
    ],
)
def test_sanitizes_english_query(
    raw_query: str,
    expected: str,
) -> None:
    assert sanitize_english_query(raw_query) == expected


def test_removes_punctuation() -> None:
    query = "What is Astro-AI's purpose?!"

    assert sanitize_english_query(query) == "What is Astro AIs purpose"


def test_normalizes_whitespace() -> None:
    query = "  What   is   Astro-AI?   "

    assert sanitize_english_query(query) == "What is Astro AI"


def test_preserves_domain_terms() -> None:
    query = "Can you explain galaxy evolution using Astro-AI?"

    result = sanitize_english_query(query)

    assert "galaxy" in result
    assert "evolution" in result
    assert "Astro" in result
    assert "AI" in result


def test_preserves_query_when_only_filler_is_present() -> None:
    query = "Please"

    assert sanitize_english_query(query) == query


def test_preserves_query_when_sanitization_produces_empty_result() -> None:
    query = "um"

    assert sanitize_english_query(query) == query


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
        sanitize_english_query(invalid_query)  # type: ignore[arg-type]


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
        sanitize_english_query(query)
