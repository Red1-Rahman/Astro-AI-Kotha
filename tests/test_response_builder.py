# tests/test_response_builder.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot.faq_loader import FAQ
from chatbot.matcher import FAQMatch, FAQMatcher
from chatbot.response_builder import (
    DEFAULT_FALLBACK_MESSAGE,
    DEFAULT_ISSUE_URL,
    DEFAULT_SUPPORT_EMAIL,
    ChatResponse,
    RelatedQuestion,
    ResponseBuilder,
    ResponseBuilderError,
    format_response_from_dict,
)


def make_faq(
    faq_id: int,
    question: str,
    answer: str,
    *,
    related_ids: list[int] | None = None,
    intent: str = "test_intent",
    keywords: list[str] | None = None,
) -> dict[str, object]:
    """Create a valid FAQ fixture."""
    return {
        "id": faq_id,
        "category": "Test",
        "intent": intent,
        "question": question,
        "answer": answer,
        "keywords": keywords or ["test"],
        "entities": [],
        "related_ids": related_ids or [],
    }


def make_database(
    faqs: list[dict[str, object]],
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
            "similarity_threshold": 0.4,
            "total_faqs": len(faqs),
            "last_updated": "2026-03-25",
        },
        "faqs": faqs,
    }


def write_database(
    path: Path,
    faqs: list[dict[str, object]],
) -> None:
    """Write a test FAQ database."""
    path.write_text(
        json.dumps(
            make_database(faqs),
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def faq_data() -> list[dict[str, object]]:
    """Return a deterministic FAQ dataset."""
    return [
        make_faq(
            1,
            "What are the system requirements for Astro-AI?",
            "Astro-AI requires Python 3.10 or later.",
            related_ids=[2, 3],
            intent="system_requirements",
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
            "Install the required dependencies and run the application.",
            related_ids=[1],
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
            "Use the galaxy evolution analysis module.",
            related_ids=[1],
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
    """Create a matcher backed by the fixture dataset."""
    path = tmp_path / "faqs.json"
    write_database(path, faq_data)

    return FAQMatcher(path)


@pytest.fixture
def builder(matcher: FAQMatcher) -> ResponseBuilder:
    """Create a response builder."""
    return ResponseBuilder(matcher)


class TestResponseBuilderInitialization:
    """Tests for ResponseBuilder initialization."""

    def test_initializes_successfully(
        self,
        builder: ResponseBuilder,
    ) -> None:
        assert builder.support_email == DEFAULT_SUPPORT_EMAIL
        assert builder.issue_url == DEFAULT_ISSUE_URL

    def test_accepts_custom_support_email(
        self,
        matcher: FAQMatcher,
    ) -> None:
        builder = ResponseBuilder(
            matcher,
            support_email="help@example.com",
        )

        assert builder.support_email == "help@example.com"

    def test_accepts_custom_issue_url(
        self,
        matcher: FAQMatcher,
    ) -> None:
        builder = ResponseBuilder(
            matcher,
            issue_url="https://example.com/issues",
        )

        assert builder.issue_url == "https://example.com/issues"

    def test_rejects_invalid_matcher(self) -> None:
        with pytest.raises(
            TypeError,
            match="matcher must be an FAQMatcher",
        ):
            ResponseBuilder("not a matcher")  # type: ignore[arg-type]

    def test_rejects_empty_support_email(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="support_email must not be empty",
        ):
            ResponseBuilder(
                matcher,
                support_email="   ",
            )

    def test_rejects_empty_issue_url(
        self,
        matcher: FAQMatcher,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="issue_url must not be empty",
        ):
            ResponseBuilder(
                matcher,
                issue_url="   ",
            )


class TestChatResponse:
    """Tests for public response contracts."""

    def test_related_question_model(self) -> None:
        question = RelatedQuestion(
            id=1,
            question="What is Astro-AI?",
        )

        assert question.id == 1
        assert question.question == "What is Astro-AI?"

    def test_chat_response_model(self) -> None:
        response = ChatResponse(
            answer="Test answer.",
            score=0.85,
            related_questions=[],
        )

        assert response.answer == "Test answer."
        assert response.score == 0.85
        assert response.related_questions == []

    def test_score_must_be_between_zero_and_one(self) -> None:
        with pytest.raises(ValueError):
            ChatResponse(
                answer="Test answer.",
                score=1.5,
                related_questions=[],
            )

    def test_answer_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError):
            ChatResponse(
                answer="",
                score=0.5,
                related_questions=[],
            )


class TestSuccessfulResponse:
    """Tests for successful FAQ responses."""

    def test_build_returns_chat_response(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert isinstance(response, ChatResponse)

    def test_returns_faq_answer(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert response.answer == (
            "Astro-AI requires Python 3.10 or later."
        )

    def test_preserves_match_score(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert response.score == round(match.score, 6)

    def test_score_is_bounded(
        self,
        builder: ResponseBuilder,
    ) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Astro-AI is a galaxy evolution platform.",
            )
        )

        match = FAQMatch(
            success=True,
            score=0.87654321,
            faq=faq,
        )

        response = builder.build(match)

        assert 0.0 <= response.score <= 1.0
        assert response.score == 0.876543

    def test_resolves_related_questions(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert len(response.related_questions) == 2
        assert response.related_questions[0].id == 2
        assert response.related_questions[0].question == (
            "How do I install Astro-AI?"
        )
        assert response.related_questions[1].id == 3

    def test_related_questions_are_typed(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert all(
            isinstance(question, RelatedQuestion)
            for question in response.related_questions
        )

    def test_related_question_order_is_preserved(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What are the Python requirements for Astro-AI?"
        )

        response = builder.build(match)

        assert [q.id for q in response.related_questions] == [2, 3]


class TestFallbackResponse:
    """Tests for unsuccessful FAQ responses."""

    def test_builds_fallback_response(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What is the weather forecast?"
        )

        response = builder.build(match)

        assert isinstance(response, ChatResponse)
        assert response.answer
        assert response.score < matcher.threshold
        assert response.related_questions == []

    def test_fallback_contains_support_email(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What is the weather forecast?"
        )

        response = builder.build(match)

        assert builder.support_email in response.answer

    def test_fallback_contains_issue_url(
        self,
        builder: ResponseBuilder,
        matcher: FAQMatcher,
    ) -> None:
        match = matcher.match(
            "What is the weather forecast?"
        )

        response = builder.build(match)

        assert builder.issue_url in response.answer

    def test_fallback_does_not_expose_internal_diagnostic(
        self,
        builder: ResponseBuilder,
    ) -> None:
        match = FAQMatch(
            success=False,
            score=0.12,
            faq=None,
            message="Internal vector database failure details.",
        )

        response = builder.build(match)

        assert (
            "Internal vector database failure details."
            not in response.answer
        )

    def test_meaningless_query_gets_specific_message(
        self,
        builder: ResponseBuilder,
    ) -> None:
        match = FAQMatch(
            success=False,
            score=0.0,
            faq=None,
            message="The query did not contain meaningful terms.",
        )

        response = builder.build(match)

        assert "meaningful question" in response.answer
        assert builder.support_email not in response.answer


class TestResponseBuilderValidation:
    """Tests for invalid matcher results."""

    def test_rejects_non_match_result(
        self,
        builder: ResponseBuilder,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="result must be an FAQMatch",
        ):
            builder.build({"success": True})  # type: ignore[arg-type]

    def test_success_without_faq_raises(
        self,
        builder: ResponseBuilder,
    ) -> None:
        match = FAQMatch(
            success=True,
            score=0.9,
            faq=None,
        )

        with pytest.raises(
            ResponseBuilderError,
            match="does not contain an FAQ",
        ):
            builder.build(match)

    def test_empty_answer_raises(
        self,
        builder: ResponseBuilder,
    ) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Valid answer.",
            )
        )

        empty_faq = faq.model_copy(
            update={"answer": "   "}
        )

        match = FAQMatch(
            success=True,
            score=0.9,
            faq=empty_faq,
        )

        with pytest.raises(
            ResponseBuilderError,
            match="contains an empty answer",
        ):
            builder.build(match)

    @pytest.mark.parametrize(
        "score",
        [-0.01, 1.01, 2.0],
    )
    def test_rejects_invalid_score(
        self,
        builder: ResponseBuilder,
        score: float,
    ) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Astro-AI is a platform.",
            )
        )

        match = FAQMatch(
            success=True,
            score=score,
            faq=faq,
        )

        with pytest.raises(
            ResponseBuilderError,
            match="must be between 0.0 and 1.0",
        ):
            builder.build(match)

    def test_missing_related_faq_is_skipped(
        self,
        matcher: FAQMatcher,
    ) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Astro-AI is a platform.",
                related_ids=[],
            )
        )

        mutated_faq = faq.model_copy(
            update={"related_ids": [999]}
        )

        builder = ResponseBuilder(matcher)

        match = FAQMatch(
            success=True,
            score=0.9,
            faq=mutated_faq,
        )

        response = builder.build(match)

        assert response.related_questions == []


class TestBuildFromQuery:
    """Tests for query-to-response convenience behavior."""

    def test_build_from_query(
        self,
        builder: ResponseBuilder,
    ) -> None:
        response = builder.build_from_query(
            "How do I install Astro-AI?"
        )

        assert isinstance(response, ChatResponse)
        assert response.answer
        assert response.score >= 0.4

    def test_build_from_query_rejects_empty_query(
        self,
        builder: ResponseBuilder,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="query must not be empty",
        ):
            builder.build_from_query("")

    def test_build_from_query_handles_no_match(
        self,
        builder: ResponseBuilder,
    ) -> None:
        response = builder.build_from_query(
            "Tell me something completely unrelated."
        )

        assert isinstance(response, ChatResponse)
        assert response.related_questions == []


class TestLegacyCompatibility:
    """Tests for the temporary legacy dictionary adapter."""

    def test_format_response_from_dict(
        self,
        matcher: FAQMatcher,
    ) -> None:
        faq = matcher.faqs[0]

        result = {
            "success": True,
            "score": 0.85,
            "faq": faq.model_dump(),
            "message": None,
        }

        response = ResponseBuilder(matcher).build(
            FAQMatch(
                success=True,
                score=0.85,
                faq=faq,
            )
        )

        legacy_response = format_response_from_dict(result)

        assert legacy_response == response

    def test_legacy_no_match(
        self,
        matcher: FAQMatcher,
    ) -> None:
        result = {
            "success": False,
            "score": 0.1,
            "faq": None,
            "message": "No FAQ matched the query.",
        }

        response = format_response_from_dict(result)

        assert isinstance(response, ChatResponse)
        assert response.score == 0.1
        assert response.related_questions == []

    def test_legacy_missing_success_field(
        self,
    ) -> None:
        result = {
            "score": 0.5,
            "faq": None,
        }

        with pytest.raises(
            ResponseBuilderError,
            match="missing required field",
        ):
            format_response_from_dict(result)


class TestResponseSerialization:
    """Tests for API serialization behavior."""

    def test_model_dump(self, builder: ResponseBuilder) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Astro-AI is a platform.",
            )
        )

        response = builder.build(
            FAQMatch(
                success=True,
                score=0.87654321,
                faq=faq,
            )
        )

        data = response.model_dump()

        assert data == {
            "answer": "Astro-AI is a platform.",
            "score": 0.876543,
            "related_questions": [],
        }

    def test_model_dump_json(
        self,
        builder: ResponseBuilder,
    ) -> None:
        faq = FAQ.model_validate(
            make_faq(
                1,
                "What is Astro-AI?",
                "Astro-AI is a platform.",
            )
        )

        response = builder.build(
            FAQMatch(
                success=True,
                score=0.9,
                faq=faq,
            )
        )

        serialized = response.model_dump_json()

        assert '"answer":"Astro-AI is a platform."' in serialized
        assert '"score":0.9' in serialized
