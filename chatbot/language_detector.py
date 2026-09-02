# chatbot/language_detector.py
from __future__ import annotations

import re
from enum import StrEnum

_BANGLA_CHAR_PATTERN = re.compile(r"[\u0980-\u09FF]")

# Common Banglish/Romanized-Bangla markers.
# These are intentionally conservative to avoid classifying ordinary English
# sentences as Banglish.
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

_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


class Language(StrEnum):
    """Languages supported by the Kotha input pipeline."""

    ENGLISH = "english"
    BANGLA = "bangla"
    BANGLISH = "banglish"
    UNKNOWN = "unknown"


class LanguageDetectionError(RuntimeError):
    """Raised when language detection cannot be performed."""


def _extract_words(text: str) -> set[str]:
    return {word.lower() for word in _WORD_PATTERN.findall(text)}


def _has_bangla_script(text: str) -> bool:
    return bool(_BANGLA_CHAR_PATTERN.search(text))


def _is_predominantly_latin(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False

    latin_letters = sum("a" <= char.lower() <= "z" for char in letters)
    return latin_letters / len(letters) >= 0.8


def detect_language(text: str) -> Language:
    """
    Detect whether text is English, Bangla, or Banglish.

    Detection is heuristic and deterministic. Bangla Unicode text is classified
    as Bangla. Latin-script text containing multiple recognizable Banglish
    markers is classified as Banglish. Other predominantly Latin text is
    classified as English.

    Args:
        text: Input text to classify.

    Returns:
        The detected Language.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty or contains no meaningful characters.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be empty")

    if not any(char.isalpha() for char in normalized):
        raise ValueError("text must contain at least one alphabetic character")

    if _has_bangla_script(normalized):
        return Language.BANGLA

    if not _is_predominantly_latin(normalized):
        return Language.UNKNOWN

    words = _extract_words(normalized)
    banglish_matches = words & _BANGLISH_MARKERS

    # Require at least two markers so ordinary English containing one
    # overlapping/common token is not classified as Banglish.
    if len(banglish_matches) >= 2:
        return Language.BANGLISH

    return Language.ENGLISH
