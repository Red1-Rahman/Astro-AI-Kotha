# tests/test_matcher.py
from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest

from chatbot.faq_loader import FAQ
from chatbot.language_detector import Language
from chatbot.matcher import (
    FAQMatch,
    FAQMatcher,
    FAQMatcherError,
)


def create_faq(
    faq_id: int,
    question: str,
    *,
    intent: str = "test_intent",
    keywords: list[str] | None = None,
    related_ids: list[int] | None = None,
) -> FAQ:
    return FAQ(
        id=faq_id,
        category="test",
        intent=intent,
        question=question,
        answer=f"Answer for FAQ {faq_id}",
        keywords=keywords or [],
        entities=[],
        related_ids=related_ids or [],
    )


class StubNLPProcessor:
    def process_batch(self, texts: list[str]) -> list[str]:
        return [text.lower() for text in texts]

    def process(self, text: str) -> str:
        return text.lower()


def create_matcher_for_tests() -> FAQMatcher:
    matcher = object.__new__(FAQMatcher)

    matcher.faq_path = "test-faqs.json"
    matcher.database = Mock()
    matcher.faqs = [
        create_faq(
            1,
            "What is Astro AI?",
            keywords=["astro", "platform"],
        ),
        create_faq(
            2,
            "How does galaxy evolution analysis work?",
            keywords=["galaxy", "evolution", "analysis"],
        ),
        create_faq(
            3,
            "What data can Astro AI analyze?",
            keywords=["data", "analysis"],
        ),
    ]
    matcher.threshold = 0.4
    matcher.nlp_processor = StubNLPProcessor()
    matcher.vectorizer = Mock()
    matcher.faq_vectors = Mock()

    return matcher


def test_prepare_query_detects_and_sanitizes_english() -> None:
    with (
        patch(
            "chatbot.matcher.detect_language",
            return_value=Language.ENGLISH,
        ) as detect_mock,
        patch(
            "chatbot.matcher.sanitize_query",
            return_value="Astro AI",
        ) as sanitize_mock,
    ):
        result = FAQMatcher._prepare_query("Can you tell me about Astro-AI?")

    assert result == "Astro AI"
    detect_mock.assert_called_once_with(
        "Can you tell me about Astro-AI?"
    )
    sanitize_mock.assert_called_once_with(
        "Can you tell me about Astro-AI?",
        Language.ENGLISH,
    )


@pytest.mark.parametrize(
    ("language", "query"),
    [
        (Language.BANGLA, "অ্যাস্ট্রো এআই কী?"),
        (Language.BANGLISH, "Astro AI ki?"),
    ],
)
def test_prepare_query_rejects_non_english_until_translation_is_available(
    language: Language,
    query: str,
) -> None:
    with (
        patch(
            "chatbot.matcher.detect_language",
            return_value=language,
        ),
        patch(
            "chatbot.matcher.sanitize_query",
            return_value="sanitized query",
        ),
    ):
        with pytest.raises(
            FAQMatcherError,
            match=(
                "FAQ matching currently supports English queries only; "
                f"detected language: {language.value}"
            ),
        ):
            FAQMatcher._prepare_query(query)


def test_prepare_query_preserves_sanitizer_errors() -> None:
    with (
        patch(
            "chatbot.matcher.detect_language",
            return_value=Language.ENGLISH,
        ),
        patch(
            "chatbot.matcher.sanitize_query",
            side_effect=ValueError("invalid query"),
        ),
    ):
        with pytest.raises(ValueError, match="invalid query"):
            FAQMatcher._prepare_query("test query")


def test_match_rejects_non_string_query() -> None:
    matcher = create_matcher_for_tests()

    with pytest.raises(TypeError, match="query must be a string"):
        matcher.match(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_match_rejects_empty_query(query: str) -> None:
    matcher = create_matcher_for_tests()

    with pytest.raises(ValueError, match="query must not be empty"):
        matcher.match(query)


def test_match_returns_no_result_for_meaningless_processed_query() -> None:
    matcher = create_matcher_for_tests()

    matcher.nlp_processor.process = Mock(return_value="")

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        return_value="sanitized query",
    ):
        result = matcher.match("test query")

    assert result == FAQMatch(
        success=False,
        score=0.0,
        faq=None,
        message="The query did not contain meaningful terms.",
    )


def test_match_preserves_successful_matching_behavior() -> None:
    matcher = create_matcher_for_tests()

    query_vector = np.array([[1.0]])
    matcher.vectorizer.transform.return_value = query_vector
    matcher.faq_vectors = np.array(
        [
            [0.2],
            [0.9],
            [0.1],
        ]
    )

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        return_value="galaxy evolution analysis",
    ):
        result = matcher.match("How does galaxy evolution work?")

    assert result.success is True
    assert result.score == pytest.approx(1.0)
    assert result.faq is not None
    assert result.faq.id == 2


def test_match_returns_fallback_for_score_below_threshold() -> None:
    matcher = create_matcher_for_tests()

    matcher.vectorizer.transform.return_value = np.array([[1.0]])
    matcher.faq_vectors = np.array(
        [
            [0.2],
            [0.3],
            [0.1],
        ]
    )

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        return_value="unrelated query",
    ):
        result = matcher.match("unrelated query")

    assert result.success is False
    assert result.faq is None
    assert result.score == pytest.approx(0.3)
    assert result.message == (
        "No FAQ matched the query with sufficient confidence."
    )


def test_match_uses_sanitized_query_before_nlp_processing() -> None:
    matcher = create_matcher_for_tests()

    matcher.vectorizer.transform.return_value = np.array([[1.0]])
    matcher.faq_vectors = np.array([[1.0], [0.0], [0.0]])

    matcher.nlp_processor.process = Mock(return_value="processed query")

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        return_value="sanitized query",
    ) as prepare_mock:
        result = matcher.match("raw query")

    prepare_mock.assert_called_once_with("raw query")
    matcher.nlp_processor.process.assert_called_once_with(
        "sanitized query"
    )

    assert result.success is True


def test_match_dict_returns_serializable_result() -> None:
    matcher = create_matcher_for_tests()

    matcher.vectorizer.transform.return_value = np.array([[1.0]])
    matcher.faq_vectors = np.array([[1.0], [0.0], [0.0]])

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        return_value="Astro AI",
    ):
        result = matcher.match_dict("What is Astro AI?")

    assert result["success"] is True
    assert result["score"] == pytest.approx(1.0)

    faq = result["faq"]
    assert isinstance(faq, dict)
    assert faq["id"] == 1
    assert faq["question"] == "What is Astro AI?"


def test_get_faq_returns_faq_by_index() -> None:
    matcher = create_matcher_for_tests()

    assert matcher.get_faq(0).id == 1
    assert matcher.get_faq(1).id == 2


def test_get_faq_rejects_invalid_index() -> None:
    matcher = create_matcher_for_tests()

    with pytest.raises(IndexError, match="FAQ index out of range"):
        matcher.get_faq(3)


def test_len_returns_number_of_faqs() -> None:
    matcher = create_matcher_for_tests()

    assert len(matcher) == 3


def test_match_propagates_matcher_errors() -> None:
    matcher = create_matcher_for_tests()

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        side_effect=FAQMatcherError("unsupported language"),
    ):
        with pytest.raises(
            FAQMatcherError,
            match="unsupported language",
        ):
            matcher.match("Astro AI ki?")


def test_match_wraps_unexpected_errors() -> None:
    matcher = create_matcher_for_tests()

    with patch.object(
        FAQMatcher,
        "_prepare_query",
        side_effect=RuntimeError("unexpected failure"),
    ):
        with pytest.raises(
            FAQMatcherError,
            match="FAQ matching failed: unexpected failure",
        ):
            matcher.match("test query")
