# speech/transcriber.py
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI


DEFAULT_MODEL = "fish-audio/transcribe-1"
DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails."""


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Provider-neutral transcription result.

    Only information required by the application is exposed here.
    Provider-specific response objects must not cross this boundary.
    """

    text: str


class Transcriber(Protocol):
    """Provider-neutral contract for speech-to-text implementations."""

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
    ) -> TranscriptionResult:
        """Transcribe audio into text."""


class FishTranscriber:
    """
    Fish Audio speech-to-text implementation through Vercel AI Gateway.

    The implementation uses the OpenAI-compatible API exposed by AI Gateway,
    while keeping the provider-specific client and model configuration inside
    this class.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")

        if not base_url.strip():
            raise ValueError("base_url must not be empty")

        resolved_api_key = api_key or os.getenv("AI_GATEWAY_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "AI_GATEWAY_API_KEY must be provided through the "
                "api_key argument or environment variable"
            )

        self.model = model
        self.base_url = base_url

        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
        )

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
    ) -> TranscriptionResult:
        """
        Transcribe audio using Fish Audio's transcription model.

        Args:
            audio: Raw audio bytes.
            filename: Filename used to identify the uploaded audio format.

        Returns:
            Provider-neutral transcription result.

        Raises:
            TypeError: If audio or filename has an invalid type.
            ValueError: If audio is empty or filename is blank.
            TranscriptionError: If the provider request fails or returns
                an unusable transcription.
        """
        if not isinstance(audio, bytes):
            raise TypeError("audio must be bytes")

        if not audio:
            raise ValueError("audio must not be empty")

        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        if not filename.strip():
            raise ValueError("filename must not be empty")

        try:
            audio_file = io.BytesIO(audio)
            audio_file.name = filename

            response = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
            )

            text = response.text.strip()

            if not text:
                raise TranscriptionError(
                    "Transcription provider returned an empty transcript"
                )

            return TranscriptionResult(text=text)

        except (TypeError, ValueError):
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Audio transcription failed: {exc}"
            ) from exc


def create_transcriber(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> FishTranscriber:
    """Create the configured speech transcriber."""
    return FishTranscriber(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
