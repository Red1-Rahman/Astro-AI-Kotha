# chatbot/sanitizer/router.py
from __future__ import annotations

from collections.abc import Callable

from chatbot.language_detector import Language
from chatbot.sanitizer.bangla_sanitizer import sanitize_bangla_query
from chatbot.sanitizer.banglish_sanitizer import sanitize_banglish_query
from chatbot.sanitizer.english_sanitizer import sanitize_english_query

Sanitizer = Callable[[str], str]


class SanitizerRouterError(RuntimeError):
    """Raised when a sanitizer cannot be selected."""


_SANITIZERS: dict[Language, Sanitizer] = {
    Language.ENGLISH: sanitize_english_query,
    Language.BANGLA: sanitize_bangla_query,
    Language.BANGLISH: sanitize_banglish_query,
}


def get_sanitizer(language: Language) -> Sanitizer:
    """
    Return the sanitizer associated with a supported language.

    Args:
        language: Detected input language.

    Returns:
        The language-specific sanitizer function.

    Raises:
        TypeError: If ``language`` is not a ``Language`` value.
        SanitizerRouterError: If the language is unsupported.
    """
    if not isinstance(language, Language):
        raise TypeError("language must be a Language value")

    try:
        return _SANITIZERS[language]
    except KeyError as exc:
        raise SanitizerRouterError(
            f"No sanitizer is available for language: {language.value}"
        ) from exc


def sanitize_query(query: str, language: Language) -> str:
    """
    Sanitize a query using the sanitizer for the detected language.

    Args:
        query: Raw user query.
        language: Detected input language.

    Returns:
        Sanitized query.

    Raises:
        TypeError: If ``query`` or ``language`` has an invalid type.
        ValueError: If the selected sanitizer rejects the query.
        SanitizerRouterError: If no sanitizer exists for the language.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    sanitizer = get_sanitizer(language)

    try:
        return sanitizer(query)
    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise SanitizerRouterError(
            f"Sanitization failed for language '{language.value}': {exc}"
        ) from exc
