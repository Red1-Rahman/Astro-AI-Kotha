# main.py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from chatbot.faq_loader import FAQDatabase, load_faq_database
from chatbot.language_detector import (
    Language,
    LanguageDetectionError,
    detect_language,
)
from chatbot.matcher import FAQMatcher
from chatbot.response_builder import ChatResponse, build_response
from chatbot.sanitizer.router import SanitizerRouterError, sanitize
from speech.synthesizer import SynthesisError, Synthesizer
from speech.transcriber import TranscriptionError, Transcriber
from translation.translator import (
    TranslationDirection,
    TranslationError,
    Translator,
    create_translator,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAQ_PATH = BASE_DIR / "data" / "faqs.json"

MAX_AUDIO_SIZE = 25 * 1024 * 1024
DEFAULT_AUDIO_FILENAME = "audio.wav"


class ChatRequest(BaseModel):
    """Backward-compatible text chat request."""

    query: str = Field(min_length=1)


@dataclass(slots=True)
class AppComponents:
    """Application dependencies composed at startup."""

    faq_database: FAQDatabase
    matcher: FAQMatcher
    transcriber: Transcriber
    translator: Translator
    synthesizer: Synthesizer


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_components() -> AppComponents:
    """Build all application dependencies."""

    faq_path = Path(os.getenv("FAQ_PATH", str(DEFAULT_FAQ_PATH)))

    faq_database = load_faq_database(faq_path)
    matcher = FAQMatcher(faq_database)

    transcriber = Transcriber()
    synthesizer = Synthesizer()

    translator = create_translator(
        provider=os.getenv("TRANSLATION_PROVIDER", "local"),
        fallback_enabled=_env_bool(
            "TRANSLATION_FALLBACK_ENABLED",
            True,
        ),
        azure_key=os.getenv("AZURE_TRANSLATOR_KEY"),
        azure_region=os.getenv("AZURE_TRANSLATOR_REGION"),
        azure_endpoint=os.getenv(
            "AZURE_TRANSLATOR_ENDPOINT",
            "https://api.cognitive.microsofttranslator.com",
        ),
    )

    return AppComponents(
        faq_database=faq_database,
        matcher=matcher,
        transcriber=transcriber,
        translator=translator,
        synthesizer=synthesizer,
    )


def _build_chat_response(
    components: AppComponents,
    query: str,
) -> ChatResponse:
    """Run the canonical English FAQ pipeline."""

    try:
        language = detect_language(query)

        if language is not Language.ENGLISH:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The text chat endpoint currently accepts English "
                    "queries. Use /api/chat/voice for Bangla or Banglish "
                    "voice input."
                ),
            )

        sanitized_query = sanitize(
            query,
            language,
        )

        match = components.matcher.match(sanitized_query)

        return build_response(match)

    except HTTPException:
        raise
    except (LanguageDetectionError, SanitizerRouterError) as exc:
        logger.warning("Text query processing failed: %s", exc)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


async def _transcribe_audio(
    components: AppComponents,
    audio: UploadFile,
) -> str:
    """Read and transcribe an uploaded audio file."""

    filename = audio.filename or DEFAULT_AUDIO_FILENAME
    content = await audio.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty.",
        )

    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file exceeds the {MAX_AUDIO_SIZE} byte limit.",
        )

    try:
        result = await components.transcriber.transcribe(
            content,
            filename=filename,
        )
    except TranscriptionError as exc:
        logger.exception("Speech transcription failed.")

        raise HTTPException(
            status_code=502,
            detail="Speech transcription failed.",
        ) from exc

    text = result.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Speech transcription returned an empty result.",
        )

    return text


async def _translate_to_english(
    components: AppComponents,
    text: str,
) -> str:
    """Translate a Bangla/Banglish query into English."""

    try:
        result = await components.translator.translate(
            text,
            TranslationDirection.TO_ENGLISH,
        )
    except TranslationError as exc:
        logger.exception("Input translation failed.")

        raise HTTPException(
            status_code=502,
            detail="Input translation failed.",
        ) from exc

    translated = result.text.strip()

    if not translated:
        raise HTTPException(
            status_code=502,
            detail="Input translation returned an empty result.",
        )

    return translated


async def _translate_to_bangla(
    components: AppComponents,
    text: str,
) -> str:
    """Translate an English answer into Bangla."""

    try:
        result = await components.translator.translate(
            text,
            TranslationDirection.TO_BANGLA,
        )
    except TranslationError as exc:
        logger.exception("Answer translation failed.")

        raise HTTPException(
            status_code=502,
            detail="Answer translation failed.",
        ) from exc

    translated = result.text.strip()

    if not translated:
        raise HTTPException(
            status_code=502,
            detail="Answer translation returned an empty result.",
        )

    return translated


async def _synthesize_answer(
    components: AppComponents,
    answer: str,
) -> Response:
    """Synthesize an answer and return the audio."""

    try:
        result = await components.synthesizer.synthesize(answer)
    except SynthesisError as exc:
        logger.exception("Speech synthesis failed.")

        raise HTTPException(
            status_code=502,
            detail="Speech synthesis failed.",
        ) from exc

    if not result.audio:
        raise HTTPException(
            status_code=502,
            detail="Speech synthesis returned empty audio.",
        )

    return Response(
        content=result.audio,
        media_type=result.content_type,
    )


def create_app(
    components: AppComponents | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Astro-AI-Kotha",
        description=(
            "Voice-enabled FAQ and support interface for the "
            "Astro-AI Galaxy Evolution Analysis Platform."
        ),
        version="0.1.0",
    )

    app.state.components = components or _load_components()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return application health."""

        return {"status": "ok"}

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
    )
    async def chat(
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Backward-compatible text FAQ endpoint.

        Example request:

            {"query": "What is Astro-AI?"}
        """

        return _build_chat_response(
            app.state.components,
            request.query,
        )

    @app.post("/api/chat/voice")
    async def voice_chat(
        audio: Annotated[UploadFile, File(...)],
    ) -> Response:
        """
        Process an audio query through the complete voice pipeline.

        English:

            audio
            -> STT
            -> language detection
            -> English sanitizer
            -> FAQ matching
            -> answer
            -> TTS

        Bangla/Banglish:

            audio
            -> STT
            -> language detection
            -> language-specific sanitizer
            -> translation to English
            -> English sanitizer
            -> FAQ matching
            -> English answer
            -> translation to Bangla
            -> TTS
        """

        components = app.state.components

        transcription = await _transcribe_audio(
            components,
            audio,
        )

        try:
            language = detect_language(transcription)
        except LanguageDetectionError as exc:
            logger.warning("Language detection failed: %s", exc)

            raise HTTPException(
                status_code=422,
                detail=(
                    "Unable to determine the language of "
                    "the transcription."
                ),
            ) from exc

        if language is Language.UNKNOWN:
            raise HTTPException(
                status_code=422,
                detail="Unsupported or undetermined language.",
            )

        try:
            sanitized = sanitize(
                transcription,
                language,
            )
        except SanitizerRouterError as exc:
            logger.warning(
                "Sanitizer routing failed: %s",
                exc,
            )

            raise HTTPException(
                status_code=422,
                detail="Unable to process the detected language.",
            ) from exc

        sanitized = sanitized.strip()

        if not sanitized:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The transcription became empty "
                    "after sanitization."
                ),
            )

        if language is Language.ENGLISH:
            english_query = sanitized
        else:
            english_query = await _translate_to_english(
                components,
                sanitized,
            )

            try:
                english_query = sanitize(
                    english_query,
                    Language.ENGLISH,
                ).strip()
            except SanitizerRouterError as exc:
                logger.warning(
                    "English post-translation sanitization failed: %s",
                    exc,
                )

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Unable to process the translated query."
                    ),
                ) from exc

            if not english_query:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "The translated query became empty "
                        "after sanitization."
                    ),
                )

        try:
            match = components.matcher.match(english_query)
            response = build_response(match)
        except Exception as exc:
            logger.exception("FAQ matching failed.")

            raise HTTPException(
                status_code=500,
                detail="FAQ processing failed.",
            ) from exc

        answer = response.answer

        if language in {
            Language.BANGLA,
            Language.BANGLISH,
        }:
            answer = await _translate_to_bangla(
                components,
                answer,
            )

        return await _synthesize_answer(
            components,
            answer,
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=_env_bool("RELOAD", False),
    )
