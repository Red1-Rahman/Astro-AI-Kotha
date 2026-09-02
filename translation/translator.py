# translation/translator.py
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import httpx


DEFAULT_BN_TO_EN_MODEL = "Helsinki-NLP/opus-mt-bn-en"
DEFAULT_EN_TO_BN_MODEL = "Helsinki-NLP/opus-mt-en-bn"

DEFAULT_AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
DEFAULT_AZURE_API_VERSION = "3.0"


class TranslationDirection(StrEnum):
    """Supported translation directions."""

    TO_ENGLISH = "to_english"
    TO_BANGLA = "to_bangla"


@dataclass(frozen=True)
class TranslationResult:
    """Provider-neutral translation result."""

    text: str


class Translator(Protocol):
    """Provider-neutral translation contract."""

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate text in the requested direction."""
        ...


class TranslationError(RuntimeError):
    """Raised when translation fails."""


class LocalTranslationError(TranslationError):
    """Raised when local translation fails."""


class AzureTranslationError(TranslationError):
    """Raised when Azure translation fails."""


class LocalTranslator:
    """CPU-compatible local Bengali/English translator.

    The model is loaded lazily so importing the application does not
    immediately download or initialize model weights.

    Supported local directions:
        Bengali -> English
        English -> Bengali

    Banglish is not a distinct model language. It is therefore passed
    through the Bengali->English model as a best-effort local translation.
    """

    def __init__(
        self,
        *,
        bn_to_en_model: str = DEFAULT_BN_TO_EN_MODEL,
        en_to_bn_model: str = DEFAULT_EN_TO_BN_MODEL,
        tokenizer_factory: Any | None = None,
        model_factory: Any | None = None,
    ) -> None:
        if not isinstance(bn_to_en_model, str):
            raise TypeError("bn_to_en_model must be a string")

        if not bn_to_en_model.strip():
            raise ValueError("bn_to_en_model must not be empty")

        if not isinstance(en_to_bn_model, str):
            raise TypeError("en_to_bn_model must be a string")

        if not en_to_bn_model.strip():
            raise ValueError("en_to_bn_model must not be empty")

        self.bn_to_en_model = bn_to_en_model.strip()
        self.en_to_bn_model = en_to_bn_model.strip()

        self._tokenizer_factory = tokenizer_factory
        self._model_factory = model_factory

        self._bn_to_en_tokenizer: Any | None = None
        self._bn_to_en_model_instance: Any | None = None

        self._en_to_bn_tokenizer: Any | None = None
        self._en_to_bn_model_instance: Any | None = None

        self._load_lock = asyncio.Lock()

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate text locally without blocking the event loop."""

        self._validate_input(text, direction)

        try:
            return await asyncio.to_thread(
                self._translate_sync,
                text.strip(),
                direction,
            )
        except (TypeError, ValueError):
            raise
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
        """Perform blocking local model inference."""

        if direction is TranslationDirection.TO_ENGLISH:
            tokenizer, model = self._get_bn_to_en_components()
            source_text = text
            target_text = self._generate(
                tokenizer=tokenizer,
                model=model,
                text=source_text,
            )
        elif direction is TranslationDirection.TO_BANGLA:
            tokenizer, model = self._get_en_to_bn_components()
            source_text = text
            target_text = self._generate(
                tokenizer=tokenizer,
                model=model,
                text=source_text,
            )
        else:
            raise LocalTranslationError(
                f"Unsupported translation direction: {direction}"
            )

        translated_text = target_text.strip()

        if not translated_text:
            raise LocalTranslationError(
                "Local translation returned empty text"
            )

        return TranslationResult(text=translated_text)

    def _get_bn_to_en_components(self) -> tuple[Any, Any]:
        """Load Bengali-to-English model components lazily."""

        if (
            self._bn_to_en_tokenizer is not None
            and self._bn_to_en_model_instance is not None
        ):
            return (
                self._bn_to_en_tokenizer,
                self._bn_to_en_model_instance,
            )

        tokenizer_factory, model_factory = self._get_factories()

        try:
            tokenizer = tokenizer_factory.from_pretrained(
                self.bn_to_en_model
            )
            model = model_factory.from_pretrained(
                self.bn_to_en_model
            )
        except Exception as exc:
            raise LocalTranslationError(
                "Failed to load Bengali-to-English translation model"
            ) from exc

        self._bn_to_en_tokenizer = tokenizer
        self._bn_to_en_model_instance = model

        return tokenizer, model

    def _get_en_to_bn_components(self) -> tuple[Any, Any]:
        """Load English-to-Bengali model components lazily."""

        if (
            self._en_to_bn_tokenizer is not None
            and self._en_to_bn_model_instance is not None
        ):
            return (
                self._en_to_bn_tokenizer,
                self._en_to_bn_model_instance,
            )

        tokenizer_factory, model_factory = self._get_factories()

        try:
            tokenizer = tokenizer_factory.from_pretrained(
                self.en_to_bn_model
            )
            model = model_factory.from_pretrained(
                self.en_to_bn_model
            )
        except Exception as exc:
            raise LocalTranslationError(
                "Failed to load English-to-Bengali translation model"
            ) from exc

        self._en_to_bn_tokenizer = tokenizer
        self._en_to_bn_model_instance = model

        return tokenizer, model

    def _get_factories(self) -> tuple[Any, Any]:
        """Resolve Transformers factories lazily."""

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

        return AutoTokenizer, AutoModelForSeq2SeqLM

    @staticmethod
    def _generate(
        *,
        tokenizer: Any,
        model: Any,
        text: str,
    ) -> str:
        """Generate a translation from a loaded sequence-to-sequence model."""

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
            )

            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
            )

            decoded = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
        except Exception as exc:
            raise LocalTranslationError(
                f"Local model inference failed: {exc}"
            ) from exc

        if not isinstance(decoded, list) or not decoded:
            raise LocalTranslationError(
                "Local model returned an invalid result"
            )

        translated_text = decoded[0]

        if not isinstance(translated_text, str):
            raise LocalTranslationError(
                "Local model returned non-text output"
            )

        return translated_text

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


class AzureTranslator:
    """Azure Translator REST API implementation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        endpoint: str = DEFAULT_AZURE_ENDPOINT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_api_key = api_key or os.getenv("AZURE_TRANSLATOR_KEY")
        resolved_region = region or os.getenv("AZURE_TRANSLATOR_REGION")

        if not resolved_api_key:
            raise ValueError(
                "Azure Translator API key is required"
            )

        if not resolved_region:
            raise ValueError(
                "Azure Translator region is required"
            )

        if not isinstance(endpoint, str):
            raise TypeError("endpoint must be a string")

        if not endpoint.strip():
            raise ValueError("endpoint must not be empty")

        self.api_key = resolved_api_key.strip()
        self.region = resolved_region.strip()
        self.endpoint = endpoint.rstrip("/")
        self._client = client

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate text using Azure Translator."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(direction, TranslationDirection):
            raise TypeError(
                "direction must be a TranslationDirection value"
            )

        target_language = self._get_target_language(direction)

        params = {
            "api-version": DEFAULT_AZURE_API_VERSION,
            "to": target_language,
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Ocp-Apim-Subscription-Region": self.region,
            "Content-Type": "application/json",
        }

        payload = [
            {
                "text": text.strip(),
            }
        ]

        client = self._client
        owns_client = client is None

        if client is None:
            client = httpx.AsyncClient()

        try:
            try:
                response = await client.post(
                    f"{self.endpoint}/translate",
                    params=params,
                    headers=headers,
                    json=payload,
                )
            except httpx.HTTPError as exc:
                raise AzureTranslationError(
                    f"Azure Translator request failed: {exc}"
                ) from exc

            if response.status_code >= 400:
                raise AzureTranslationError(
                    "Azure Translator returned HTTP "
                    f"{response.status_code}: {response.text}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise AzureTranslationError(
                    "Azure Translator returned invalid JSON"
                ) from exc

            translated_text = self._extract_translation(data)

            return TranslationResult(text=translated_text)

        except (TypeError, ValueError):
            raise
        except AzureTranslationError:
            raise
        except Exception as exc:
            raise AzureTranslationError(
                f"Azure translation failed: {exc}"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _get_target_language(
        direction: TranslationDirection,
    ) -> str:
        if direction is TranslationDirection.TO_ENGLISH:
            return "en"

        if direction is TranslationDirection.TO_BANGLA:
            return "bn"

        raise AzureTranslationError(
            f"Unsupported translation direction: {direction}"
        )

    @staticmethod
    def _extract_translation(data: Any) -> str:
        """Extract translated text from an Azure response."""

        if not isinstance(data, list) or not data:
            raise AzureTranslationError(
                "Azure Translator returned an invalid response"
            )

        first_item = data[0]

        if not isinstance(first_item, dict):
            raise AzureTranslationError(
                "Azure Translator returned an invalid response"
            )

        translations = first_item.get("translations")

        if not isinstance(translations, list) or not translations:
            raise AzureTranslationError(
                "Azure Translator returned no translations"
            )

        translation = translations[0]

        if not isinstance(translation, dict):
            raise AzureTranslationError(
                "Azure Translator returned an invalid translation"
            )

        translated_text = translation.get("text")

        if not isinstance(translated_text, str):
            raise AzureTranslationError(
                "Azure Translator returned non-text translation"
            )

        translated_text = translated_text.strip()

        if not translated_text:
            raise AzureTranslationError(
                "Azure Translator returned empty text"
            )

        return translated_text


class FallbackTranslator:
    """Primary/fallback translation strategy.

    The primary translator is attempted first. The fallback is used only
    when the primary raises TranslationError.
    """

    def __init__(
        self,
        primary: Translator,
        fallback: Translator,
    ) -> None:
        if primary is None:
            raise ValueError("primary translator is required")

        if fallback is None:
            raise ValueError("fallback translator is required")

        self.primary = primary
        self.fallback = fallback

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate using primary, then fallback on translation failure."""

        try:
            return await self.primary.translate(
                text,
                direction,
            )
        except TranslationError as primary_error:
            try:
                return await self.fallback.translate(
                    text,
                    direction,
                )
            except TranslationError as fallback_error:
                raise TranslationError(
                    "Both primary and fallback translation failed. "
                    f"Primary: {primary_error}; "
                    f"Fallback: {fallback_error}"
                ) from fallback_error


def create_translator(
    *,
    local: Translator | None = None,
    fallback: Translator | None = None,
) -> Translator:
    """Create the application's local-first translator."""

    primary = local or LocalTranslator()

    if fallback is None:
        fallback = AzureTranslator()

    return FallbackTranslator(
        primary=primary,
        fallback=fallback,
    )
