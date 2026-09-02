# tests/test_response_builder.py
from __future__ import annotations

from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from chatbot.faq_loader import FAQ
from chatbot.matcher import FAQMatch
from chatbot.response_builder import (
    DEFAULT_FALLBACK_MESSAGE,
    DEFAULT_ISSUE_URL,
    DEFAULT_SUPPORT_EMAIL,
    ChatResponse,
    RelatedQuestion,
    ResponseBuilder,
    ResponseBuilderError,
    format_response,
    format_response_from_dict,
)


def create_faq(
    faq_id: int,
    question: str,
    answer: str = "Test answer.",
    *,
    related_ids: list[int] | None = None,
) -> FAQ:
    return FAQ(
        id=faq_id,
        category="test",
        intent="test_intent",
        question=question,
        answer=answer,
        keywords=[],
        entities=[],
        related_ids=related_ids or [],
    )


def create_matcher(
    faqs: list[FAQ] | None = None,
) -> Mock:
    matcher = Mock()
    matcher.faqs = faqs or [
        create_faq(
            1,
            "What is Astro-AI?",
            related_ids=[2],
        ),
        create_faq(
            2,
            "How does Astro-AI work?",
        ),
    ]
    return matcher


def test_related_question_contract() -> None:
    related = RelatedQuestion(
        id=2,
        question="How does Astro-AI work?",
    )

    assert related.id == 2
    assert related.question == "How does Astro-AI work?"


def test_related_question_rejects_invalid_id() -> None:
    with pytest.raises(ValidationError):
        RelatedQuestion(
            id=0,
            question="How does Astro-AI work?",
        )


def test_related_question_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        RelatedQuestion(
            id=2,
            question="",
        )


def test_chat_response_contract() -> None:
    response = ChatResponse(
        answer="Astro-AI analyzes galaxy evolution.",
        score=0.85,
        related_questions=[],
    )

    assert response.answer == "Astro-AI analyzes galaxy evolution."
    assert response.score == 0.85
    assert response.related_questions == []


def test_chat_response_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            answer="Test answer.",
            score=1.1,
            related_questions=[],
        )

    with pytest.raises(ValidationError):
        ChatResponse(
            answer="Test answer.",
            score=-0.1,
            related_questions=[],
        )


def test_chat_response_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(
            answer="Test answer.",
            score=0.8,
            related_questions=[],
            language="english",
        )


def test_response_builder_requires_matcher() -> None:
    with pytest.raises(TypeError, match="matcher must be an FAQMatcher"):
        ResponseBuilder(Mock())  # type: ignore[arg-type]


def test_response_builder_rejects_empty_support_email() -> None:
    matcher = create_matcher()

    with pytest.raises(
        ValueError,
        match="support_email must not be empty",
    ):
        ResponseBuilder(
            matcher,  # type: ignore[arg-type]
            support_email=" ",
        )


def test_response_builder_rejects_empty_issue_url() -> None:
    matcher = create_matcher()

    with pytest.raises(
        ValueError,
        match="issue_url must not be empty",
    ):
        ResponseBuilder(
            matcher,  # type: ignore[arg-type]
            issue_url=" ",
        )


def test_build_successful_response() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    faq = matcher.faqs[0]
    result = FAQMatch(
        success=True,
        score=0.87654321,
        faq=faq,
    )

    response = builder.build(result)

    assert response.answer == "Test answer."
    assert response.score == 0.876543
    assert response.related_questions == [
        RelatedQuestion(
            id=2,
            question="How does Astro-AI work?",
        )
    ]


def test_build_successful_response_without_related_questions() -> None:
    faq = create_faq(
        1,
        "What is Astro-AI?",
    )
    matcher = create_matcher([faq])
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=True,
        score=0.8,
        faq=faq,
    )

    response = builder.build(result)

    assert response.answer == "Test answer."
    assert response.score == 0.8
    assert response.related_questions == []


def test_build_successful_response_requires_faq() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=True,
        score=0.8,
        faq=None,
    )

    with pytest.raises(
        ResponseBuilderError,
        match="Successful FAQ match does not contain an FAQ",
    ):
        builder.build(result)


def test_build_successful_response_rejects_empty_answer() -> None:
    faq = create_faq(
        1,
        "What is Astro-AI?",
        answer="   ",
    )
    matcher = create_matcher([faq])
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=True,
        score=0.8,
        faq=faq,
    )

    with pytest.raises(
        ResponseBuilderError,
        match="FAQ 1 contains an empty answer",
    ):
        builder.build(result)


def test_build_fallback_response() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=False,
        score=0.25,
        faq=None,
        message="No FAQ matched the query with sufficient confidence.",
    )

    response = builder.build(result)

    assert response.score == 0.25
    assert response.related_questions == []
    assert DEFAULT_FALLBACK_MESSAGE in response.answer
    assert DEFAULT_SUPPORT_EMAIL in response.answer
    assert DEFAULT_ISSUE_URL in response.answer


def test_build_fallback_for_meaningless_query() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=False,
        score=0.0,
        faq=None,
        message="The query did not contain meaningful terms.",
    )

    response = builder.build(result)

    assert response.answer == (
        "I couldn't identify a meaningful question from your "
        "message. Please rephrase your question and try again."
    )
    assert response.score == 0.0
    assert response.related_questions == []


def test_build_rejects_invalid_match_type() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="result must be an FAQMatch"):
        builder.build(Mock())  # type: ignore[arg-type]


def test_build_rejects_out_of_range_score() -> None:
    matcher = create_matcher()
    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=True,
        score=1.5,
        faq=matcher.faqs[0],
    )

    with pytest.raises(
        ResponseBuilderError,
        match="FAQ match score must be between 0.0 and 1.0",
    ):
        builder.build(result)


def test_build_from_query_delegates_to_matcher() -> None:
    matcher = create_matcher()
    match_result = FAQMatch(
        success=True,
        score=0.9,
        faq=matcher.faqs[0],
    )
    matcher.match.return_value = match_result

    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    response = builder.build_from_query("What is Astro-AI?")

    matcher.match.assert_called_once_with("What is Astro-AI?")
    assert response.answer == "Test answer."
    assert response.score == 0.9


def test_build_from_query_wraps_unexpected_errors() -> None:
    matcher = create_matcher()
    matcher.match.side_effect = RuntimeError("matcher failure")

    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    with pytest.raises(
        ResponseBuilderError,
        match="Unable to build response for query: matcher failure",
    ):
        builder.build_from_query("test query")


def test_format_response_uses_legacy_match_contract() -> None:
    faq = create_faq(
        1,
        "What is Astro-AI?",
    )

    result = FAQMatch(
        success=True,
        score=0.75,
        faq=faq,
    )

    response = format_response(result)

    assert isinstance(response, ChatResponse)
    assert response.answer == "Test answer."
    assert response.score == 0.75


def test_format_response_from_dict_preserves_legacy_shape() -> None:
    faq = create_faq(
        1,
        "What is Astro-AI?",
    )

    result = {
        "success": True,
        "score": 0.75,
        "faq": faq.model_dump(),
        "message": None,
    }

    response = format_response_from_dict(result)

    assert isinstance(response, ChatResponse)
    assert response.answer == "Test answer."
    assert response.score == 0.75
    assert response.related_questions == []


def test_format_response_from_dict_rejects_missing_success() -> None:
    with pytest.raises(
        ResponseBuilderError,
        match="Legacy match result is missing required field",
    ):
        format_response_from_dict(
            {
                "score": 0.5,
                "faq": None,
            }
        )


def test_format_response_from_dict_rejects_invalid_input_type() -> None:
    with pytest.raises(TypeError, match="result must be a dictionary"):
        format_response_from_dict([])  # type: ignore[arg-type]


def test_related_questions_skip_missing_faqs() -> None:
    faq = create_faq(
        1,
        "What is Astro-AI?",
        related_ids=[2, 999],
    )

    matcher = create_matcher(
        [
            faq,
            create_faq(
                2,
                "How does Astro-AI work?",
            ),
        ]
    )

    builder = ResponseBuilder(matcher)  # type: ignore[arg-type]

    result = FAQMatch(
        success=True,
        score=0.8,
        faq=faq,
    )

    response = builder.build(result)

    assert response.related_questions == [
        RelatedQuestion(
            id=2,
            question="How does Astro-AI work?",
        )
    ]
