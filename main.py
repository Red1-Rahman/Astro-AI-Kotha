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

from chatbot.language_detector import (
    Language,
    LanguageDetectionError,
    detect_language,
)
from chatbot.matcher import FAQMatcher
from chatbot.response_builder import (
    ChatResponse,
    ResponseBuilder,
)
from chatbot.sanitizer.router import (
    SanitizerRouterError,
    sanitize_query,
)
from speech.synthesizer import (
    SynthesisError,
    Synthesizer,
    create_synthesizer,
)
from speech.transcriber import (
    TranscriptionError,
    Transcriber,
    create_transcriber,
)
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
    """Fully composed runtime application dependencies."""

    matcher: FAQMatcher
    response_builder: ResponseBuilder
    transcriber: Transcriber
    translator: Translator
    synthesizer: Synthesizer


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _load_components() -> AppComponents:
    """
    Compose the complete application dependency graph.

    This function is intentionally NOT called during module import.
    """

    faq_path = Path(
        os.getenv(
            "FAQ_PATH",
            str(DEFAULT_FAQ_PATH),
        )
    )

    matcher = FAQMatcher(
        faq_path,
    )

    response_builder = ResponseBuilder(
        matcher,
    )

    transcriber = create_transcriber()

    synthesizer = create_synthesizer()

    translator = create_translator(
        provider=os.getenv(
            "TRANSLATION_PROVIDER",
            "local",
        ),
        fallback_enabled=_env_bool(
            "TRANSLATION_FALLBACK_ENABLED",
            True,
        ),
        azure_key=os.getenv(
            "AZURE_TRANSLATOR_KEY",
        ),
        azure_region=os.getenv(
            "AZURE_TRANSLATOR_REGION",
        ),
        azure_endpoint=os.getenv(
            "AZURE_TRANSLATOR_ENDPOINT",
            "https://api.cognitive.microsofttranslator.com",
        ),
    )

    return AppComponents(
        matcher=matcher,
        response_builder=response_builder,
        transcriber=transcriber,
        translator=translator,
        synthesizer=synthesizer,
    )


def _get_components(
    app: FastAPI,
) -> AppComponents:
    """
    Return composed dependencies, composing them lazily once.

    Importing ``main`` does not invoke this function.
    """

    components = getattr(
        app.state,
        "components",
        None,
    )

    if components is None:
        components = _load_components()
        app.state.components = components

    return components


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
                    "The text chat endpoint currently accepts "
                    "English queries. Use /api/chat/voice for "
                    "Bangla or Banglish voice input."
                ),
            )

        sanitized_query = sanitize_query(
            query,
            language,
        )

        match = components.matcher.match(
            sanitized_query,
        )

        return components.response_builder.build(
            match,
        )

    except HTTPException:
        raise

    except (
        LanguageDetectionError,
        SanitizerRouterError,
    ) as exc:
        logger.warning(
            "Text query processing failed: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "FAQ processing failed.",
        )

        raise HTTPException(
            status_code=500,
            detail="FAQ processing failed.",
        ) from exc


async def _transcribe_audio(
    components: AppComponents,
    audio: UploadFile,
) -> str:
    """Read and transcribe an uploaded audio file."""

    filename = (
        audio.filename
        or DEFAULT_AUDIO_FILENAME
    )

    content = await audio.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded audio file is empty.",
        )

    if len(content) > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Audio file exceeds the "
                f"{MAX_AUDIO_SIZE} byte limit."
            ),
        )

    try:
        result = await components.transcriber.transcribe(
            content,
            filename=filename,
        )

    except TranscriptionError as exc:
        logger.exception(
            "Speech transcription failed.",
        )

        raise HTTPException(
            status_code=502,
            detail="Speech transcription failed.",
        ) from exc

    text = result.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail=(
                "Speech transcription returned "
                "an empty result."
            ),
        )

    return text


async def _translate_to_english(
    components: AppComponents,
    text: str,
) -> str:
    """Translate Bangla/Banglish input into English."""

    try:
        result = await components.translator.translate(
            text,
            TranslationDirection.TO_ENGLISH,
        )

    except TranslationError as exc:
        logger.exception(
            "Input translation failed.",
        )

        raise HTTPException(
            status_code=502,
            detail="Input translation failed.",
        ) from exc

    translated = result.text.strip()

    if not translated:
        raise HTTPException(
            status_code=502,
            detail=(
                "Input translation returned "
                "an empty result."
            ),
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
        logger.exception(
            "Answer translation failed.",
        )

        raise HTTPException(
            status_code=502,
            detail="Answer translation failed.",
        ) from exc

    translated = result.text.strip()

    if not translated:
        raise HTTPException(
            status_code=502,
            detail=(
                "Answer translation returned "
                "an empty result."
            ),
        )

    return translated


async def _synthesize_answer(
    components: AppComponents,
    answer: str,
) -> Response:
    """Synthesize an answer and return the audio."""

    try:
        result = await components.synthesizer.synthesize(
            answer,
        )

    except SynthesisError as exc:
        logger.exception(
            "Speech synthesis failed.",
        )

        raise HTTPException(
            status_code=502,
            detail="Speech synthesis failed.",
        ) from exc

    if not result.audio:
        raise HTTPException(
            status_code=502,
            detail=(
                "Speech synthesis returned "
                "empty audio."
            ),
        )

    return Response(
        content=result.audio,
        media_type=result.content_type,
    )


def create_app(
    components: AppComponents | None = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    If components are supplied, they are used directly. This is the
    dependency-injection path used by tests.

    If components are not supplied, runtime composition is deferred until
    the first request.
    """

    app = FastAPI(
        title="Astro-AI-Kotha",
        description=(
            "Voice-enabled FAQ and support interface for the "
            "Astro-AI Galaxy Evolution Analysis Platform."
        ),
        version="0.1.0",
    )

    app.state.components = components

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
        """Process an English text FAQ query."""

        components = _get_components(app)

        return _build_chat_response(
            components,
            request.query,
        )

    @app.post("/api/chat/voice")
    async def voice_chat(
        audio: Annotated[
            UploadFile,
            File(...),
        ],
    ) -> Response:
        """Process an audio query through the voice pipeline."""

        components = _get_components(app)

        transcription = await _transcribe_audio(
            components,
            audio,
        )

        try:
            language = detect_language(
                transcription,
            )

        except LanguageDetectionError as exc:
            logger.warning(
                "Language detection failed: %s",
                exc,
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "Unable to determine the language "
                    "of the transcription."
                ),
            ) from exc

        if language is Language.UNKNOWN:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported or undetermined language."
                ),
            )

        try:
            sanitized = sanitize_query(
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
                detail=(
                    "Unable to process the detected language."
                ),
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
                english_query = sanitize_query(
                    english_query,
                    Language.ENGLISH,
                ).strip()

            except SanitizerRouterError as exc:
                logger.warning(
                    "English post-translation "
                    "sanitization failed: %s",
                    exc,
                )

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Unable to process the "
                        "translated query."
                    ),
                ) from exc

            if not english_query:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "The translated query became "
                        "empty after sanitization."
                    ),
                )

        try:
            match = components.matcher.match(
                english_query,
            )

            response = (
                components.response_builder.build(
                    match,
                )
            )

        except Exception as exc:
            logger.exception(
                "FAQ matching failed.",
            )

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


# Import boundary:
#
# Creating the ASGI application object is safe.
# It only declares routes and stores ``None`` as the dependency graph.
#
# Actual dependency composition happens inside _get_components() when the
# first runtime request requires it.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv(
            "HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=_env_bool(
            "RELOAD",
            False,
        ),
    )
