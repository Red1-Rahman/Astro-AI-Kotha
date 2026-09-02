# chatbot/matcher.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot.faq_loader import (
    DEFAULT_FAQ_PATH,
    FAQ,
    FAQDatabase,
    load_faq_database,
)
from chatbot.language_detector import Language, detect_language
from chatbot.nlp_processor import NLPProcessor
from chatbot.sanitizer.router import sanitize_query

DEFAULT_SIMILARITY_THRESHOLD = 0.4


class FAQMatcherError(RuntimeError):
    """Raised when the FAQ matcher cannot be initialized or used."""


@dataclass(frozen=True)
class FAQMatch:
    success: bool
    score: float
    faq: FAQ | None
    message: str | None = None


class FAQMatcher:
    def __init__(
        self,
        faq_path: str | None = None,
        *,
        threshold: float | None = None,
        nlp_processor: NLPProcessor | None = None,
    ) -> None:
        self.faq_path = faq_path or str(DEFAULT_FAQ_PATH)
        self.database: FAQDatabase = load_faq_database(self.faq_path)
        self.faqs: list[FAQ] = self.database.faqs

        configured_threshold = (
            threshold
            if threshold is not None
            else self.database.metadata.similarity_threshold
        )

        if not 0.0 <= configured_threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.threshold = configured_threshold

        self.nlp_processor = nlp_processor or NLPProcessor(
            self.database.metadata.nlp_model
        )

        self.vectorizer = TfidfVectorizer()
        self.faq_vectors = self._build_index()

    def _build_index(self):
        questions = [self._build_search_text(faq) for faq in self.faqs]

        try:
            processed_questions = self.nlp_processor.process_batch(questions)
            return self.vectorizer.fit_transform(processed_questions)
        except Exception as exc:
            raise FAQMatcherError(
                f"Unable to build FAQ search index: {exc}"
            ) from exc

    @staticmethod
    def _build_search_text(faq: FAQ) -> str:
        return " ".join([faq.question, faq.intent, *faq.keywords])

    @staticmethod
    def _prepare_query(query: str) -> str:
        """
        Detect the query language and apply the corresponding sanitizer.

        The matcher currently supports English retrieval only. Bangla and
        Banglish sanitization is performed here, but translation into English
        belongs to the translation layer and is intentionally not performed
        by the matcher.
        """
        language = detect_language(query)
        sanitized_query = sanitize_query(query, language)

        if language is not Language.ENGLISH:
            raise FAQMatcherError(
                f"FAQ matching currently supports English queries only; "
                f"detected language: {language.value}"
            )

        return sanitized_query

    def match(self, query: str) -> FAQMatch:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        try:
            prepared_query = self._prepare_query(query)
            processed_query = self.nlp_processor.process(prepared_query)

            if not processed_query:
                return FAQMatch(
                    success=False,
                    score=0.0,
                    faq=None,
                    message="The query did not contain meaningful terms.",
                )

            query_vector = self.vectorizer.transform([processed_query])
            similarities = cosine_similarity(
                query_vector,
                self.faq_vectors,
            )[0]

            if similarities.size == 0:
                raise FAQMatcherError(
                    "FAQ search index contains no vectors"
                )

            best_index = int(np.argmax(similarities))
            best_score = float(similarities[best_index])
            best_faq = self.faqs[best_index]

            if best_score < self.threshold:
                return FAQMatch(
                    success=False,
                    score=best_score,
                    faq=None,
                    message=(
                        "No FAQ matched the query with sufficient confidence."
                    ),
                )

            return FAQMatch(
                success=True,
                score=best_score,
                faq=best_faq,
            )

        except (TypeError, ValueError):
            raise
        except FAQMatcherError:
            raise
        except Exception as exc:
            raise FAQMatcherError(
                f"FAQ matching failed: {exc}"
            ) from exc

    def match_dict(self, query: str) -> dict[str, object]:
        result = self.match(query)

        return {
            "success": result.success,
            "score": result.score,
            "faq": (
                result.faq.model_dump()
                if result.faq is not None
                else None
            ),
            "message": result.message,
        }

    def get_faq(self, index: int) -> FAQ:
        if not 0 <= index < len(self.faqs):
            raise IndexError(f"FAQ index out of range: {index}")

        return self.faqs[index]

    def __len__(self) -> int:
        return len(self.faqs)


def create_matcher(
    faq_path: str | None = None,
    *,
    threshold: float | None = None,
) -> FAQMatcher:
    return FAQMatcher(
        faq_path=faq_path,
        threshold=threshold,
    )


matcher = create_matcher()
