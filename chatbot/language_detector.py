# chatbot/language_detector.py
from __future__ import annotations

import re
from enum import StrEnum

_BANGLA_CHAR_PATTERN = re.compile(
    r"[\u0980-\u09FF]"
)

_WORD_PATTERN = re.compile(
    r"[^\W\d_]+",
    re.UNICODE,
)

_BANGLISH_MARKERS = frozenset(
    {
        "accha",
        "ami",
        "amar",
        "amader",
        "apni",
        "apnar",
        "apnader",
        "ache",
        "achen",
        "ase",
        "asen",
        "ki",
        "kivabe",
        "kibhabe",
        "kobe",
        "kothay",
        "keno",
        "ken",
        "kirokom",
        "koto",
        "kon",
        "konta",
        "korbo",
        "korben",
        "koren",
        "korte",
        "lage",
        "lagbe",
        "nai",
        "nei",
        "naki",
        "hobe",
        "hobey",
        "hoy",
        "hoye",
        "jante",
        "jantechi",
        "bolen",
        "bolben",
        "den",
        "diben",
        "eta",
        "eita",
        "ota",
        "oita",
        "ei",
        "oi",
        "ekhane",
        "okhane",
        "jonno",
        "jonne",
        "theke",
        "porjonto",
        "shomporke",
        "somproke",
        "tumi",
        "tumar",
        "tumake",
        "hayre",
        "bujhchi",
        "ektu",
        "arektu",
    }
)

_MIN_BANGLISH_MARKERS = 2
_MIN_LATIN_RATIO = 0.8


class Language(StrEnum):
    """Languages supported by the Kotha input pipeline."""

    ENGLISH = "english"
    BANGLA = "bangla"
    BANGLISH = "banglish"
    UNKNOWN = "unknown"


class LanguageDetectionError(RuntimeError):
    """Raised when language detection cannot be completed."""


def _extract_words(text: str) -> set[str]:
    """Extract normalized alphabetic words from text."""

    return {
        word.lower()
        for word in _WORD_PATTERN.findall(text)
    }


def _has_bangla_script(text: str) -> bool:
    """Return whether the text contains Bangla Unicode characters."""

    return bool(
        _BANGLA_CHAR_PATTERN.search(text)
    )


def _is_predominantly_latin(text: str) -> bool:
    """Return whether alphabetic characters are predominantly Latin."""

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:
        return False

    latin_count = sum(
        "a" <= char.lower() <= "z"
        for char in letters
    )

    return (
        latin_count / len(letters)
        >= _MIN_LATIN_RATIO
    )


def _detect_latin_language(
    text: str,
) -> Language:
    """Classify predominantly Latin-script text."""

    words = _extract_words(text)

    banglish_matches = (
        words & _BANGLISH_MARKERS
    )

    if (
        len(banglish_matches)
        >= _MIN_BANGLISH_MARKERS
    ):
        return Language.BANGLISH

    return Language.ENGLISH


def detect_language(
    text: object,
) -> Language:
    """
    Detect the language of a user query.

    Detection is deterministic and heuristic-based.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    normalized = text.strip()

    if not normalized:
        raise ValueError(
            "text must not be empty"
        )

    if not any(
        char.isalpha()
        for char in normalized
    ):
        raise ValueError(
            "text must contain at least one "
            "alphabetic character"
        )

    try:
        if _has_bangla_script(normalized):
            return Language.BANGLA

        if not _is_predominantly_latin(
            normalized,
        ):
            return Language.UNKNOWN

        return _detect_latin_language(
            normalized,
        )

    except (TypeError, ValueError):
        raise

    except Exception as exc:
        raise LanguageDetectionError(
            f"Language detection failed: {exc}"
        ) from exc
