# translation/translator.py
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from google.cloud import translate_v2


DEFAULT_TARGET_ENGLISH = "en"
DEFAULT_TARGET_BANGLA = "bn"


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
    """Raised when translation fails unexpectedly."""


class GoogleTranslator:
    """Google Cloud Translation implementation.

    The Google Cloud client is synchronous, so API calls are executed in
    a worker thread to avoid blocking the async application event loop.
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
    ) -> None:
        resolved_project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")

        if resolved_project_id is not None and not isinstance(
            resolved_project_id, str
        ):
            raise TypeError("project_id must be a string or None")

        if resolved_project_id is not None and not resolved_project_id.strip():
            raise ValueError("project_id must not be empty")

        try:
            if resolved_project_id is None:
                self._client = translate_v2.Client()
            else:
                self._client = translate_v2.Client(
                    project=resolved_project_id.strip()
                )
        except Exception as exc:
            raise TranslationError(
                f"Failed to initialize Google Cloud Translation client: {exc}"
            ) from exc

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        """Translate text using Google Cloud Translation."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(direction, TranslationDirection):
            raise TypeError(
                "direction must be a TranslationDirection value"
            )

        target_language = self._get_target_language(direction)

        try:
            result = await asyncio.to_thread(
                self._translate_sync,
                text.strip(),
                target_language,
            )
        except (TypeError, ValueError):
            raise
        except TranslationError:
            raise
        except Exception as exc:
            raise TranslationError(
                f"Google translation failed: {exc}"
            ) from exc

        return result

    def _translate_sync(
        self,
        text: str,
        target_language: str,
    ) -> TranslationResult:
        """Execute the blocking Google Translation API call."""

        try:
            response = self._client.translate(
                text,
                target_language=target_language,
            )
        except Exception as exc:
            raise TranslationError(
                f"Google Translation API request failed: {exc}"
            ) from exc

        translated_text = response.get("translatedText")

        if not isinstance(translated_text, str):
            raise TranslationError(
                "Google Translation returned an invalid response"
            )

        translated_text = translated_text.strip()

        if not translated_text:
            raise TranslationError(
                "Google Translation returned empty text"
            )

        return TranslationResult(text=translated_text)

    @staticmethod
    def _get_target_language(
        direction: TranslationDirection,
    ) -> str:
        """Map the provider-neutral direction to a Google language code."""

        if direction is TranslationDirection.TO_ENGLISH:
            return DEFAULT_TARGET_ENGLISH

        if direction is TranslationDirection.TO_BANGLA:
            return DEFAULT_TARGET_BANGLA

        raise TranslationError(
            f"Unsupported translation direction: {direction}"
        )


def create_translator(
    *,
    project_id: str | None = None,
) -> GoogleTranslator:
    """Create the application's default translator."""

    return GoogleTranslator(project_id=project_id)
