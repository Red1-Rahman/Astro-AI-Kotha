# tests/test_matcher.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chatbot.faq_loader import FAQ
from chatbot.matcher import (
    DEFAULT_SIMILARITY_THRESHOLD,
    FAQMatch,
    FAQMatcher,
    FAQMatcherError,
)


def make_faq(
    faq_id: int,
    question: str,
    *,
    intent: str = "test_intent",
    keywords: list[str] | None = None,
    related_ids: list[int] | None = None,
) -> dict[str, object]:
    """Create a valid FAQ fixture."""
    return {
        "id": faq_id,
        "category": "Test",
        "intent": intent,
        "question": question,
        "answer": f"Answer for FAQ {faq_id}.",
        "keywords": keywords or [],
        "entities": [],
        "related_ids": related_ids or [],
    }


def make_database(
    faqs: list[dict[str, object]],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, object]:
    """Create a valid FAQ database fixture."""
    return {
        "metadata": {
            "version": "1.0",
            "description": "Test FAQ database",
            "nlp_processor": "spacy",
            "nlp_model": "en_core_web_sm",
            "preprocessing_steps": [
                "tokenization",
                "lemmatization",
                "stopword_removal",
                "lowercase_normalization",
            ],
            "matching_algorithm": "tfidf_cosine_similarity",
            "similarity_threshold": threshold,
            "total_faqs": len(faqs),
            "last_updated": "2026-03-25",
        },
        "faqs": faqs,
    }


def write_database(
    path: Path,
    faqs: list[dict[str, object]],
    *,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> None:
    """Write a test FAQ database."""
    path.write_text(
        json.dumps(
            make_database(faqs, threshold=threshold),
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def faq_data() -> list[dict[str, object]]:
    """Return a small deterministic FAQ dataset."""
    return [
        make_faq(
            1,
            "What are the system requirements for installing Astro-AI?",
            intent="install_dependencies",
            keywords=[
                "requirements",
                "python",
                "memory",
                "storage",
            ],
        ),
        make_faq(
            2,
            "How do I install Astro-AI?",
            intent="installation",
            keywords=[
                "install",
                "setup",
                "dependencies",
            ],
        ),
        make_faq(
            3,
            "How can I analyze galaxy evolution?",
            intent="galaxy_analysis",
            keywords=[
                "galaxy",
                "evolution",
                "analysis",
            ],
        ),
    ]


@pytest.fixture
def matcher(
    tmp_path: Path,
    faq_data: list[dict[str, object]],
) -> FAQMatcher:
    """Create a matcher using the fixture dataset."""
    path = tmp_path / "faqs.json"
    write_database(path, faq_data)

    return FAQMatcher(path)


class TestFAQMatcherInitialization:
    """Tests for matcher initialization."""

    def test_initializes_successfully(
        self,
        matcher: FAQMatcher,
    ) -> None:
        assert len(matcher) == 3
        assert matcher.threshold == DEFAULT_SIMILARITY_THRESHOLD

    def test_loads_faq_records(
        self,
        matcher: FAQMatcher,
    ) -> None:
        assert all(isinstance(faq, FAQ) for faq in matcher.faqs)

    def test_uses_metadata_threshold(
        self,
        tmp_path: Path,
        faq_data: list[dict[str, object]],
    ) -> None:
        path = tmp_path / "faqs.json"
        write_database(
            path,
            faq_data,
            threshold=0.7,
        )

        matcher = FAQMatcher(path)

        assert matcher.threshold == 0.7

    def test_custom_threshold_overrides_metadata(
        self,
        tmp_path: Path,
        faq_data: list[dict[str, object]],
    ) -> None:
        path = tmp_path / "faqs.json"
        write_database(
            path,
            faq_data,
            threshold=0.7,
        )

        matcher = FAQMatcher(path, threshold=0.25)

        assert matcher.threshold == 0.25

    @pytest.mark.parametrize(
        "threshold",
        [-0.01, 1.01, 2.0],
    )
    def test_rejects_invalid_threshold(
        self,
        tmp_path: Path,
        faq_data: list[dict[str, object]],
        threshold: float,
    ) -> None:
        path = tmp_path / "faqs.json"
        write_database(path, faq_data)

        with pytest.raises(
            ValueError,
            match="threshold must be between",
        ):
            FAQMatcher(path, threshold=threshold)

    def test_missing_faq_file_raises(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "missing.json"

        with pytest.raises(FAQMatcherError):
            FAQMatcher(path)


class TestFAQMatcherSearchText:
    """Tests for searchable FAQ text construction."""

    def test_includes_question(self) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
            )
        )

        result = FAQMatcher._build_search_text(faq)

        assert "What is Astro-AI?" in result

    def test_includes_intent(self) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                intent="project_overview",
            )
        )

        result = FAQMatcher._build_search_text(faq)

        assert "project_overview" in result

    def test_includes_keywords(self) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                keywords=["galaxy", "astronomy"],
            )
        )

        result = FAQMatcher._build_search_text(faq)

        assert "galaxy" in result
        assert "astronomy" in result


class TestFAQMatcher:
    """Tests for FAQ matching."""

    def test_returns_faq_match(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "What Python requirements are needed?"
        )

        assert isinstance(result, FAQMatch)

    def test_matches_relevant_faq(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "What Python requirements are needed "
            "to install Astro-AI?"
        )

        assert result.success is True
        assert result.faq is not None
        assert result.faq.id == 1

    def test_returns_score(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "How do I install Astro-AI?"
        )

        assert 0.0 <= result.score <= 1.0

    def test_score_is_float(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "How do I install Astro-AI?"
        )

        assert isinstance(result.score, float)

    def test_score_meets_threshold_on_success(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "How do I install Astro-AI?"
        )

        assert result.success is True
        assert result.score >= matcher.threshold

    def test_unrelated_query_fails_threshold(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match(
            "What is the weather forecast for tomorrow?"
        )

        assert result.success is False
        assert result.faq is None
        assert result.score < matcher.threshold
        assert result.message is not None

    def test_empty_query_raises(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="query must not be empty",
        ):
            matcher.match("")

    def test_whitespace_query_raises(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="query must not be empty",
        ):
            matcher.match("   ")

    def test_non_string_query_raises(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="query must be a string",
        ):
            matcher.match(123)  # type: ignore[arg-type]

    def test_stop_word_only_query(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match("the is are")

        assert result.success is False
        assert result.faq is None
        assert result.score == 0.0

    def test_case_insensitive_matching(
        self,
        matcher: FAQMatcher,
    ) -> None:
        lowercase = matcher.match(
            "how do i install astro ai"
        )
        uppercase = matcher.match(
            "HOW DO I INSTALL ASTRO AI"
        )

        assert lowercase.success is True
        assert uppercase.success is True
        assert lowercase.faq is not None
        assert uppercase.faq is not None
        assert lowercase.faq.id == uppercase.faq.id

    def test_match_dict_compatibility(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match_dict(
            "How do I install Astro-AI?"
        )

        assert result["success"] is True
        assert isinstance(result["score"], float)
        assert isinstance(result["faq"], dict)

    def test_match_dict_contains_faq_fields(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match_dict(
            "How do I install Astro-AI?"
        )

        faq = result["faq"]

        assert isinstance(faq, dict)
        assert faq["id"] == 2
        assert "question" in faq
        assert "answer" in faq

    def test_match_dict_for_no_match(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = matcher.match_dict(
            "Tell me about something unrelated."
        )

        assert result["success"] is False
        assert result["faq"] is None
        assert result["message"] is not None


class TestFAQMatcherHelpers:
    """Tests for matcher helper methods."""

    def test_get_faq(
        self,
        matcher: FAQMatcher,
    ) -> None:
        faq = matcher.get_faq(0)

        assert faq.id == 1

    def test_get_faq_preserves_index_order(
        self,
        matcher: FAQMatcher,
    ) -> None:
        assert matcher.get_faq(0).id == 1
        assert matcher.get_faq(1).id == 2
        assert matcher.get_faq(2).id == 3

    def test_get_faq_rejects_negative_index(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(IndexError, match="out of range"):
            matcher.get_faq(-1)

    def test_get_faq_rejects_too_large_index(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(IndexError, match="out of range"):
            matcher.get_faq(3)

    def test_len_returns_faq_count(
        self,
        matcher: FAQMatcher,
    ) -> None:
        assert len(matcher) == 3


class TestFAQMatcherFailureHandling:
    """Tests for defensive error handling."""

    def test_preprocessing_failure_is_wrapped(
        self,
        tmp_path: Path,
        faq_data: list[dict[str, object]],
    ) -> None:
        path = tmp_path / "faqs.json"
        write_database(path, faq_data)

        processor = MagicMock()
        processor.process_batch.side_effect = RuntimeError(
            "preprocessing failed"
        )

        with pytest.raises(
            FAQMatcherError,
            match="Unable to build FAQ search index",
        ):
            FAQMatcher(
                path,
                nlp_processor=processor,
            )

    def test_matching_failure_is_wrapped(
        self,
        matcher: FAQMatcher,
    ) -> None:
        matcher.nlp_processor.process = MagicMock(
            side_effect=RuntimeError("NLP failed")
        )

        with pytest.raises(
            FAQMatcherError,
            match="FAQ matching failed",
        ):
            matcher.match("How do I install Astro-AI?")
          
