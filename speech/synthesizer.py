# speech/synthesizer.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI


DEFAULT_MODEL = "fish-audio/s2.1-pro"
DEFAULT_BASE_URL = "https://ai-gateway.vercel.sh/v1"


class SynthesisError(RuntimeError):
    """Raised when text-to-speech synthesis fails."""


@dataclass(frozen=True)
class SynthesisResult:
    """Provider-neutral text-to-speech result."""

    audio: bytes
    content_type: str


class Synthesizer(Protocol):
    """Provider-neutral text-to-speech contract."""

    async def synthesize(
        self,
        text: str,
    ) -> SynthesisResult:
        """Synthesize text into audio."""
        ...


class FishSynthesizer:
    """Fish Audio TTS implementation through Vercel AI Gateway."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        voice: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError(
                "model must not be empty"
            )

        if not base_url.strip():
            raise ValueError(
                "base_url must not be empty"
            )

        if (
            voice is not None
            and not voice.strip()
        ):
            raise ValueError(
                "voice must not be empty when provided"
            )

        resolved_api_key = (
            api_key
            or os.getenv("AI_GATEWAY_API_KEY")
        )

        if not resolved_api_key:
            raise ValueError(
                "AI_GATEWAY_API_KEY must be provided through "
                "the api_key argument or environment variable"
            )

        self.model = model
        self.base_url = base_url
        self.voice = voice

        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
        )

    async def synthesize(
        self,
        text: str,
    ) -> SynthesisResult:
        """Synthesize speech from text."""

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if not text.strip():
            raise ValueError(
                "text must not be empty"
            )

        try:
            request_kwargs: dict[str, object] = {
                "model": self.model,
                "input": text,
            }

            if self.voice is not None:
                request_kwargs["voice"] = self.voice

            response = (
                await self._client.audio.speech.create(
                    **request_kwargs,
                )
            )

            audio = await response.aread()

            if not audio:
                raise SynthesisError(
                    "Text-to-speech provider returned "
                    "empty audio"
                )

            content_type = getattr(
                response,
                "headers",
                {},
            ).get(
                "content-type",
                "audio/mpeg",
            )

            if (
                not isinstance(content_type, str)
                or not content_type.strip()
            ):
                content_type = "audio/mpeg"

            return SynthesisResult(
                audio=audio,
                content_type=content_type,
            )

        except (TypeError, ValueError):
            raise

        except SynthesisError:
            raise

        except Exception as exc:
            raise SynthesisError(
                f"Text-to-speech synthesis failed: {exc}"
            ) from exc


def create_synthesizer(
    *,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    voice: str | None = None,
) -> Synthesizer:
    """Create the configured text-to-speech provider."""

    return FishSynthesizer(
        api_key=api_key,
        model=model,
        base_url=base_url,
        voice=voice,
    )
