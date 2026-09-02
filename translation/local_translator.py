# translation/local_translator.py
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from translation.translator import (
    DEFAULT_BN_TO_EN_MODEL,
    DEFAULT_EN_TO_BN_MODEL,
    LocalTranslationError,
    TranslationDirection,
    TranslationResult,
)


TokenizerFactory = Callable[[str], Any]
ModelFactory = Callable[[str], Any]


class LocalTranslator:
    """
    Local Bengali/English translator using Hugging Face seq2seq models.

    Bengali -> English:
        Helsinki-NLP/opus-mt-bn-en

    English -> Bengali:
        Helsinki-NLP/opus-mt-en-bn

    Banglish -> English is supported as best-effort input because the
    Banglish sanitizer can route Banglish text here as a Bengali-to-English
    translation request. The Bengali model is not specifically trained for
    Banglish, so Azure fallback is important for production use.
    """

    def __init__(
        self,
        *,
        bn_to_en_model: str = DEFAULT_BN_TO_EN_MODEL,
        en_to_bn_model: str = DEFAULT_EN_TO_BN_MODEL,
        tokenizer_factory: TokenizerFactory | None = None,
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._bn_to_en_model_name = self._validate_model_name(
            bn_to_en_model,
            "bn_to_en_model",
        )
        self._en_to_bn_model_name = self._validate_model_name(
            en_to_bn_model,
            "en_to_bn_model",
        )

        self._tokenizer_factory = tokenizer_factory
        self._model_factory = model_factory

        self._bn_to_en_tokenizer: Any | None = None
        self._bn_to_en_model: Any | None = None

        self._en_to_bn_tokenizer: Any | None = None
        self._en_to_bn_model: Any | None = None

        # These locks protect synchronous model initialization happening
        # inside asyncio.to_thread().
        #
        # Separate locks prevent loading the Bengali->English model from
        # unnecessarily blocking English->Bengali model initialization.
        self._bn_to_en_load_lock = threading.Lock()
        self._en_to_bn_load_lock = threading.Lock()

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """
        Translate text without blocking the asyncio event loop.
        """

        self._validate_input(text, direction)

        try:
            return await asyncio.to_thread(
                self._translate_sync,
                text.strip(),
                direction,
            )
        except LocalTranslationError:
            raise
        except Exception as exc:
            raise LocalTranslationError(
                f"Local translation failed: {exc}"
            ) from exc

    def _translate_sync(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        if direction is TranslationDirection.TO_ENGLISH:
            tokenizer, model = self._get_bn_to_en_components()
            source_language = "bn"
            target_language = "en"
        elif direction is TranslationDirection.TO_BANGLA:
            tokenizer, model = self._get_en_to_bn_components()
            source_language = "en"
            target_language = "bn"
        else:
            raise TypeError(
                "direction must be a TranslationDirection value"
            )

        translated = self._generate(
            tokenizer=tokenizer,
            model=model,
            text=text,
        )

        if not translated:
            raise LocalTranslationError(
                "Local translation produced empty output"
            )

        return TranslationResult(
            text=translated,
            source_language=source_language,
            target_language=target_language,
            provider="local",
        )

    def _get_bn_to_en_components(self) -> tuple[Any, Any]:
        if (
            self._bn_to_en_tokenizer is not None
            and self._bn_to_en_model is not None
        ):
            return self._bn_to_en_tokenizer, self._bn_to_en_model

        with self._bn_to_en_load_lock:
            # Double-check after acquiring the lock. Another worker thread
            # may have initialized the components while this thread waited.
            if (
                self._bn_to_en_tokenizer is not None
                and self._bn_to_en_model is not None
            ):
                return self._bn_to_en_tokenizer, self._bn_to_en_model

            tokenizer_factory, model_factory = self._get_factories()

            try:
                tokenizer = tokenizer_factory(
                    self._bn_to_en_model_name
                )
                model = model_factory(
                    self._bn_to_en_model_name
                )
            except Exception as exc:
                raise LocalTranslationError(
                    "Failed to load Bengali-to-English translation model"
                ) from exc

            self._prepare_model(model)

            self._bn_to_en_tokenizer = tokenizer
            self._bn_to_en_model = model

            return tokenizer, model

    def _get_en_to_bn_components(self) -> tuple[Any, Any]:
        if (
            self._en_to_bn_tokenizer is not None
            and self._en_to_bn_model is not None
        ):
            return self._en_to_bn_tokenizer, self._en_to_bn_model

        with self._en_to_bn_load_lock:
            if (
                self._en_to_bn_tokenizer is not None
                and self._en_to_bn_model is not None
            ):
                return self._en_to_bn_tokenizer, self._en_to_bn_model

            tokenizer_factory, model_factory = self._get_factories()

            try:
                tokenizer = tokenizer_factory(
                    self._en_to_bn_model_name
                )
                model = model_factory(
                    self._en_to_bn_model_name
                )
            except Exception as exc:
                raise LocalTranslationError(
                    "Failed to load English-to-Bengali translation model"
                ) from exc

            self._prepare_model(model)

            self._en_to_bn_tokenizer = tokenizer
            self._en_to_bn_model = model

            return tokenizer, model

    def _get_factories(
        self,
    ) -> tuple[TokenizerFactory, ModelFactory]:
        if (
            self._tokenizer_factory is not None
            and self._model_factory is not None
        ):
            return self._tokenizer_factory, self._model_factory

        try:
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise LocalTranslationError(
                "Local translation requires the 'transformers' package"
            ) from exc

        return (
            AutoTokenizer.from_pretrained,
            AutoModelForSeq2SeqLM.from_pretrained,
        )

    @staticmethod
    def _prepare_model(model: Any) -> None:
        """
        Put the model into evaluation mode when supported.

        Hugging Face models expose eval(), but keeping this defensive makes
        the implementation easier to test with lightweight test doubles.
        """

        eval_method = getattr(model, "eval", None)

        if callable(eval_method):
            try:
                eval_method()
            except Exception as exc:
                raise LocalTranslationError(
                    "Failed to prepare local translation model"
                ) from exc

    @staticmethod
    def _generate(
        *,
        tokenizer: Any,
        model: Any,
        text: str,
    ) -> str:
        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
            )

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
            )

            decoded = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
        except Exception as exc:
            raise LocalTranslationError(
                f"Local translation inference failed: {exc}"
            ) from exc

        if not isinstance(decoded, list):
            raise LocalTranslationError(
                "Translation tokenizer returned an invalid output"
            )

        if not decoded:
            raise LocalTranslationError(
                "Translation tokenizer returned no output"
            )

        translated = decoded[0]

        if not isinstance(translated, str):
            raise LocalTranslationError(
                "Translation output must be a string"
            )

        return translated.strip()

    @staticmethod
    def _validate_model_name(
        model_name: str,
        field_name: str,
    ) -> str:
        if not isinstance(model_name, str):
            raise TypeError(f"{field_name} must be a string")

        normalized = model_name.strip()

        if not normalized:
            raise ValueError(f"{field_name} must not be empty")

        return normalized

    @staticmethod
    def _validate_input(
        text: str,
        direction: TranslationDirection,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(direction, TranslationDirection):
            raise TypeError(
                "direction must be a TranslationDirection value"
            )

    def models_loaded(self) -> dict[TranslationDirection, bool]:
        """
        Return model initialization state.

        Primarily useful for diagnostics and tests.
        """

        return {
            TranslationDirection.TO_ENGLISH: (
                self._bn_to_en_tokenizer is not None
                and self._bn_to_en_model is not None
            ),
            TranslationDirection.TO_BANGLA: (
                self._en_to_bn_tokenizer is not None
                and self._en_to_bn_model is not None
            ),
        }
