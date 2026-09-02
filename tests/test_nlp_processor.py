# tests/test_nlp_processor.py
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from spacy.language import Language

from chatbot.nlp_processor import (
    DEFAULT_MODEL,
    NLPProcessor,
    NLPProcessorError,
)


@pytest.fixture
def processor() -> NLPProcessor:
    """Create an NLP processor using the installed spaCy model."""
    return NLPProcessor()


class TestNLPProcessorInitialization:
    """Tests for NLPProcessor initialization."""

    def test_default_model_name(self) -> None:
        assert DEFAULT_MODEL == "en_core_web_sm"

    def test_initializes_with_default_model(self) -> None:
        processor = NLPProcessor()

        assert processor.model_name == DEFAULT_MODEL
        assert isinstance(processor._nlp, Language)

    def test_initializes_with_custom_model(self) -> None:
        processor = NLPProcessor("en_core_web_sm")

        assert processor.model_name == "en_core_web_sm"

    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name must not be empty"):
            NLPProcessor("")

    def test_rejects_whitespace_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name must not be empty"):
            NLPProcessor("   ")

    def test_missing_model_raises_processor_error(self) -> None:
        with patch(
            "chatbot.nlp_processor.spacy.load",
            side_effect=OSError("model not found"),
        ):
            NLPProcessor._load_model.cache_clear()

            with pytest.raises(
                NLPProcessorError,
                match="Unable to load spaCy model",
            ):
                NLPProcessor("missing_model")

            NLPProcessor._load_model.cache_clear()


class TestNLPProcessor:
    """Tests for text preprocessing."""

    def test_process_returns_string(self, processor: NLPProcessor) -> None:
        result = processor.process("What are the system requirements?")

        assert isinstance(result, str)

    def test_lowercases_text(self, processor: NLPProcessor) -> None:
        result = processor.process("PYTHON REQUIREMENTS")

        assert result == result.lower()

    def test_removes_punctuation(self, processor: NLPProcessor) -> None:
        result = processor.process("What are the requirements?")

        assert "?" not in result

    def test_removes_stop_words(self, processor: NLPProcessor) -> None:
        result = processor.process(
            "What are the requirements for installing Astro-AI?"
        )

        assert "the" not in result.split()
        assert "are" not in result.split()
        assert "for" not in result.split()

    def test_lemmatizes_words(self, processor: NLPProcessor) -> None:
        result = processor.process("galaxies are evolving")

        assert "galaxy" in result
        assert "evolve" in result

    def test_removes_whitespace_tokens(
        self,
        processor: NLPProcessor,
    ) -> None:
        result = processor.process(
            "   Python     requirements   "
        )

        assert result
        assert "  " not in result

    def test_preserves_meaningful_keywords(
        self,
        processor: NLPProcessor,
    ) -> None:
        result = processor.process(
            "What Python version does Astro-AI require?"
        )

        assert "python" in result
        assert "version" in result
        assert "astro" in result
        assert "require" in result

    def test_empty_text_raises(self, processor: NLPProcessor) -> None:
        with pytest.raises(ValueError, match="text must not be empty"):
            processor.process("")

    def test_whitespace_text_raises(
        self,
        processor: NLPProcessor,
    ) -> None:
        with pytest.raises(ValueError, match="text must not be empty"):
            processor.process("   ")

    def test_non_string_text_raises(
        self,
        processor: NLPProcessor,
    ) -> None:
        with pytest.raises(TypeError, match="text must be a string"):
            processor.process(123)  # type: ignore[arg-type]

    def test_process_batch(self, processor: NLPProcessor) -> None:
        texts = [
            "What are the system requirements?",
            "How do I install Astro-AI?",
        ]

        results = processor.process_batch(texts)

        assert len(results) == 2
        assert all(isinstance(result, str) for result in results)
        assert "requirement" in results[0]
        assert "install" in results[1]

    def test_process_batch_preserves_order(
        self,
        processor: NLPProcessor,
    ) -> None:
        texts = [
            "Python requirements",
            "Install dependencies",
            "Galaxy evolution",
        ]

        results = processor.process_batch(texts)

        assert len(results) == len(texts)
        assert "python" in results[0]
        assert "install" in results[1]
        assert "galaxy" in results[2]

    def test_process_batch_empty_list(
        self,
        processor: NLPProcessor,
    ) -> None:
        assert processor.process_batch([]) == []

    def test_process_batch_rejects_non_list(
        self,
        processor: NLPProcessor,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="texts must be a list of strings",
        ):
            processor.process_batch("hello")  # type: ignore[arg-type]

    def test_process_batch_rejects_non_string_item(
        self,
        processor: NLPProcessor,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="every item in texts must be a string",
        ):
            processor.process_batch(
                ["valid text", 123]  # type: ignore[list-item]
            )

    def test_process_batch_rejects_empty_item(
        self,
        processor: NLPProcessor,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="must not contain empty strings",
        ):
            processor.process_batch(["valid text", ""])


class TestNLPProcessorCaching:
    """Tests for spaCy model caching."""

    def test_same_model_is_cached(self) -> None:
        NLPProcessor._load_model.cache_clear()

        with patch(
            "chatbot.nlp_processor.spacy.load",
            return_value=MagicMock(),
        ) as mock_load:
            first = NLPProcessor("test_model")
            second = NLPProcessor("test_model")

            assert first._nlp is second._nlp
            mock_load.assert_called_once_with("test_model")

        NLPProcessor._load_model.cache_clear()
