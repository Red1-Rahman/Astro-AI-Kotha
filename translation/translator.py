# translation/translator.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


DEFAULT_BN_TO_EN_MODEL = "Helsinki-NLP/opus-mt-bn-en"
DEFAULT_EN_TO_BN_MODEL = "Helsinki-NLP/opus-mt-en-bn"


class TranslationDirection(StrEnum):
    """Supported translation directions."""

    TO_ENGLISH = "to_english"
    TO_BANGLA = "to_bangla"


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Provider-neutral translation result."""

    text: str
    source_language: str
    target_language: str
    provider: str


class TranslationError(RuntimeError):
    """Base exception for translation failures."""


class LocalTranslationError(TranslationError):
    """Raised when local translation fails."""


class AzureTranslationError(TranslationError):
    """Raised when Azure Translator fails."""


class Translator(Protocol):
    """Provider-independent translation contract."""

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate text in the requested direction."""
        ...


class FallbackTranslator:
    """Try the primary translator and optionally fall back."""

    def __init__(
        self,
        primary: Translator,
        fallback: Translator,
        *,
        enabled: bool = True,
    ) -> None:
        if not isinstance(enabled, bool):
            raise TypeError(
                "enabled must be a boolean"
            )

        self._primary = primary
        self._fallback = fallback
        self._enabled = enabled

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not isinstance(
            direction,
            TranslationDirection,
        ):
            raise TypeError(
                "direction must be a TranslationDirection value"
            )

        try:
            return await self._primary.translate(
                text,
                direction,
            )

        except TranslationError:
            if not self._enabled:
                raise

        return await self._fallback.translate(
            text,
            direction,
        )


def create_translator(
    *,
    provider: str = "local",
    fallback_enabled: bool = True,
    azure_key: str | None = None,
    azure_region: str | None = None,
    azure_endpoint: str = (
        "https://api.cognitive.microsofttranslator.com"
    ),
) -> Translator:
    """Create the configured translation provider."""

    normalized_provider = (
        provider.strip().lower()
    )

    if normalized_provider not in {
        "local",
        "azure",
    }:
        raise ValueError(
            "provider must be one of: 'local', 'azure'"
        )

    if not isinstance(
        fallback_enabled,
        bool,
    ):
        raise TypeError(
            "fallback_enabled must be a boolean"
        )

    if normalized_provider == "azure":
        from translation.azure_translator import (
            AzureTranslator,
        )

        return AzureTranslator(
            api_key=azure_key,
            region=azure_region,
            endpoint=azure_endpoint,
        )

    from translation.local_translator import (
        LocalTranslator,
    )

    local = LocalTranslator()

    if not fallback_enabled:
        return local

    from translation.azure_translator import (
        AzureTranslator,
    )

    azure = AzureTranslator(
        api_key=azure_key,
        region=azure_region,
        endpoint=azure_endpoint,
    )

    return FallbackTranslator(
        primary=local,
        fallback=azure,
        enabled=True,
    )
