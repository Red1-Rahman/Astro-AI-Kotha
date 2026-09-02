# chatbot/nlp_processor.py
from __future__ import annotations

import logging
from functools import lru_cache

import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "en_core_web_sm"


class NLPProcessorError(RuntimeError):
    """Raised when the NLP processor cannot be initialized or used."""


class NLPProcessor:
    """
    English NLP preprocessing for FAQ matching.

    The processor uses spaCy for:
    - tokenization
    - stop-word removal
    - punctuation removal
    - whitespace removal
    - lemmatization
    - lowercase normalization

    The processor intentionally does not perform:
    - translation
    - language detection
    - semantic embedding
    - answer generation
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")

        self.model_name = model_name
        self._nlp = self._load_model(model_name)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_model(model_name: str) -> Language:
        """
        Load and cache a spaCy model.

        Caching prevents repeatedly loading the same model when multiple
        NLPProcessor instances are created.
        """
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise NLPProcessorError(
                f"Unable to load spaCy model '{model_name}'. "
                f"Install it with: python -m spacy download {model_name}"
            ) from exc

    def process(self, text: str) -> str:
        """
        Normalize text for FAQ matching.

        Args:
            text: Input English text.

        Returns:
            A whitespace-normalized string of lowercase lemmas.

        Raises:
            ValueError: If text is empty or contains only whitespace.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        doc = self._nlp(text)

        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
        ]

        return " ".join(tokens)

    def process_batch(self, texts: list[str]) -> list[str]:
        """
        Normalize multiple texts using spaCy's efficient pipe interface.

        Args:
            texts: List of English texts.

        Returns:
            A list of normalized strings.

        Raises:
            TypeError: If texts is not a list or contains non-string values.
            ValueError: If any text is empty or whitespace-only.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("every item in texts must be a string")

            if not text.strip():
                raise ValueError("texts must not contain empty strings")

        documents = self._nlp.pipe(texts)

        return [self._process_doc(doc) for doc in documents]

    @staticmethod
    def _process_doc(doc: Doc) -> str:
        """Normalize a spaCy document."""
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and not token.is_space
            and token.lemma_.strip()
        ]

        return " ".join(tokens)
