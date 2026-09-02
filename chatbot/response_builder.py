# chatbot/response_builder.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from chatbot.faq_loader import FAQ
from chatbot.matcher import FAQMatch, FAQMatcher


DEFAULT_SUPPORT_EMAIL = "support@astro-ai.com"
DEFAULT_ISSUE_URL = (
    "https://github.com/Red1-Rahman/CodeAlpha_astro-ai-chatbot/issues"
)

DEFAULT_FALLBACK_MESSAGE = (
    "I'm sorry, but I couldn't find a sufficiently relevant answer "
    "in the Astro-AI FAQ knowledge base."
)


class ResponseBuilderError(RuntimeError):
    """Raised when an FAQ response cannot be constructed."""


class RelatedQuestion(BaseModel):
    """A related FAQ question exposed to the client."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(gt=0)
    question: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Public response contract for the FAQ chatbot."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    related_questions: list[RelatedQuestion]


class ResponseBuilder:
    """
    Build user-facing responses from FAQ matching results.

    Responsibilities:
    - Convert FAQMatch results into API-safe response models.
    - Format successful FAQ answers.
    - Generate the fallback response.
    - Resolve related FAQ questions.
    - Keep presentation logic separate from matching logic.
    """

    def __init__(
        self,
        matcher: FAQMatcher,
        *,
        support_email: str = DEFAULT_SUPPORT_EMAIL,
        issue_url: str = DEFAULT_ISSUE_URL,
    ) -> None:
        if not isinstance(matcher, FAQMatcher):
            raise TypeError("matcher must be an FAQMatcher")

        if not support_email.strip():
            raise ValueError("support_email must not be empty")

        if not issue_url.strip():
            raise ValueError("issue_url must not be empty")

        self.matcher = matcher
        self.support_email = support_email
        self.issue_url = issue_url

        self._faq_by_id = {
            faq.id: faq
            for faq in matcher.faqs
        }

    def build(self, result: FAQMatch) -> ChatResponse:
        """
        Build a public response from an FAQMatch.

        Args:
            result: Typed FAQ matching result.

        Returns:
            A validated ChatResponse.

        Raises:
            TypeError: If result is not an FAQMatch.
            ResponseBuilderError: If the match result is inconsistent.
        """
        if not isinstance(result, FAQMatch):
            raise TypeError("result must be an FAQMatch")

        if result.success:
            return self._build_success_response(result)

        return self._build_fallback_response(result)

    def build_from_query(self, query: str) -> ChatResponse:
        """
        Match a query and immediately build its response.

        This is a convenience method for the API layer.
        """
        try:
            result = self.matcher.match(query)
            return self.build(result)
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            raise ResponseBuilderError(
                f"Unable to build response for query: {exc}"
            ) from exc

    def _build_success_response(
        self,
        result: FAQMatch,
    ) -> ChatResponse:
        """Build a response for a successful FAQ match."""
        if result.faq is None:
            raise ResponseBuilderError(
                "Successful FAQ match does not contain an FAQ"
            )

        answer = result.faq.answer.strip()

        if not answer:
            raise ResponseBuilderError(
                f"FAQ {result.faq.id} contains an empty answer"
            )

        related_questions = self._build_related_questions(
            result.faq
        )

        return ChatResponse(
            answer=answer,
            score=self._normalize_score(result.score),
            related_questions=related_questions,
        )

    def _build_fallback_response(
        self,
        result: FAQMatch,
    ) -> ChatResponse:
        """Build a response when no FAQ meets the threshold."""
        answer = self._build_fallback_message(result)

        return ChatResponse(
            answer=answer,
            score=self._normalize_score(result.score),
            related_questions=[],
        )

    def _build_fallback_message(
        self,
        result: FAQMatch,
    ) -> str:
        """
        Build the fallback support message.

        The matcher's diagnostic message is intentionally not exposed
        directly to users because it is an internal implementation detail.
        """
        diagnostic = (
            result.message.strip()
            if result.message
            else DEFAULT_FALLBACK_MESSAGE
        )

        if diagnostic.startswith("The query did not contain"):
            return (
                "I couldn't identify a meaningful question from your "
                "message. Please rephrase your question and try again."
            )

        return (
            f"{DEFAULT_FALLBACK_MESSAGE}\n\n"
            f"For further assistance, contact {self.support_email} "
            f"or open an issue at {self.issue_url}."
        )

    def _build_related_questions(
        self,
        faq: FAQ,
    ) -> list[RelatedQuestion]:
        """
        Resolve related FAQ IDs into public question objects.

        Missing IDs should normally be impossible because FAQDatabase
        validates relationships during loading. The defensive check here
        protects this layer if the underlying matcher is constructed from
        an unexpected/mutated object.
        """
        related: list[RelatedQuestion] = []

        for related_id in faq.related_ids:
            related_faq = self._faq_by_id.get(related_id)

            if related_faq is None:
                continue

            related.append(
                RelatedQuestion(
                    id=related_faq.id,
                    question=related_faq.question,
                )
            )

        return related

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Normalize a similarity score to a stable six-decimal float."""
        if not 0.0 <= score <= 1.0:
            raise ResponseBuilderError(
                f"FAQ match score must be between 0.0 and 1.0: {score}"
            )

        return round(float(score), 6)


def format_response(result: FAQMatch) -> ChatResponse:
    """
    Compatibility helper for the existing API layer.

    This function preserves the simple ``format_response(result)`` usage
    while keeping response construction inside ResponseBuilder.
    """
    from chatbot.matcher import matcher

    return ResponseBuilder(matcher).build(result)


def format_response_from_dict(
    result: dict[str, Any],
) -> ChatResponse:
    """
    Compatibility helper for legacy dictionary matcher results.

    New code should use ``FAQMatch`` directly.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dictionary")

    try:
        success = bool(result["success"])
        score = float(result["score"])
        faq_data = result.get("faq")
        message = result.get("message")

        faq = (
            FAQ.model_validate(faq_data)
            if faq_data is not None
            else None
        )

        match = FAQMatch(
            success=success,
            score=score,
            faq=faq,
            message=message,
        )

        return format_response(match)

    except KeyError as exc:
        raise ResponseBuilderError(
            f"Legacy match result is missing required field: {exc}"
        ) from exc
    except Exception as exc:
        raise ResponseBuilderError(
            f"Unable to convert legacy match result: {exc}"
        ) from exc
