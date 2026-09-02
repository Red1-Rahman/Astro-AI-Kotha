# tests/test_language_detector.py
import pytest

from chatbot.language_detector import (
    Language,
    detect_language,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is Astro-AI?", Language.ENGLISH),
        ("How does the galaxy analysis work?", Language.ENGLISH),
        ("Tell me about galaxy evolution.", Language.ENGLISH),
        ("Can you explain the project?", Language.ENGLISH),
    ],
)
def test_detects_english(text: str, expected: Language) -> None:
    assert detect_language(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "অ্যাস্ট্রো-এআই কী?",
        "গ্যালাক্সি ইভোলিউশন কীভাবে কাজ করে?",
        "এই প্রজেক্ট সম্পর্কে বলুন।",
        "গ্যালাক্সি বিশ্লেষণ কীভাবে করা হয়?",
    ],
)
def test_detects_bangla(text: str) -> None:
    assert detect_language(text) is Language.BANGLA


@pytest.mark.parametrize(
    "text",
    [
        "Astro AI ki?",
        "Galaxy evolution ki kore kaj kore?",
        "Ami Astro AI somporke jante chai",
        "Apni ki bolben galaxy analysis kivabe kore?",
        "Eta ki vabe kaj kore?",
    ],
)
def test_detects_banglish(text: str) -> None:
    assert detect_language(text) is Language.BANGLISH


@pytest.mark.parametrize(
    "text",
    [
        "12345",
        "!!! ???",
        "বাংলা English",
    ],
)
def test_rejects_or_handles_non_standard_input(text: str) -> None:
    if text == "বাংলা English":
        assert detect_language(text) is Language.BANGLA
    else:
        with pytest.raises(ValueError):
            detect_language(text)


@pytest.mark.parametrize(
    "invalid_input",
    [
        None,
        123,
        [],
        {},
    ],
)
def test_rejects_non_string_input(invalid_input: object) -> None:
    with pytest.raises(TypeError):
        detect_language(invalid_input)  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_rejects_empty_input(text: str) -> None:
    with pytest.raises(ValueError):
        detect_language(text)


def test_language_values_are_stable() -> None:
    assert Language.ENGLISH.value == "english"
    assert Language.BANGLA.value == "bangla"
    assert Language.BANGLISH.value == "banglish"
    assert Language.UNKNOWN.value == "unknown"


def test_ordinary_english_is_not_misclassified_as_banglish() -> None:
    text = "I want to know how the system works."
    assert detect_language(text) is Language.ENGLISH


def test_single_overlapping_marker_does_not_trigger_banglish() -> None:
    text = "I want to know more about the project."
    assert detect_language(text) is Language.ENGLISH
