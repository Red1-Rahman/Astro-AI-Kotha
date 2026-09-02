from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from chatbot.matcher import FAQMatcher
from chatbot.response_builder import ResponseBuilder
from main import AppComponents, create_app
from speech.synthesizer import SynthesisResult
from speech.transcriber import TranscriptionResult
from translation.translator import (
    TranslationDirection,
    TranslationResult,
)


@pytest.fixture
def e2e_components() -> AppComponents:
    """
    Build the application with the real FAQ retrieval and response layers.

    External services are replaced with async mocks:
        - Fish STT
        - Azure/local translation
        - Fish TTS

    The following application layers remain real:
        - FAQ loading
        - NLP processing
        - TF-IDF indexing
        - FAQ matching
        - response construction
    """

    matcher = FAQMatcher()
    response_builder = ResponseBuilder(matcher)

    transcriber = AsyncMock()
    translator = AsyncMock()
    synthesizer = AsyncMock()

    transcriber.transcribe.return_value = TranscriptionResult(
        text="What is Astro AI?",
    )

    translator.translate.return_value = TranslationResult(
        text="What is Astro AI?",
        source_language="en",
        target_language="en",
        provider="local",
    )

    synthesizer.synthesize.return_value = SynthesisResult(
        audio=b"fake-e2e-audio",
        content_type="audio/mpeg",
    )

    return AppComponents(
        matcher=matcher,
        response_builder=response_builder,
        transcriber=transcriber,
        translator=translator,
        synthesizer=synthesizer,
    )


@pytest.fixture
def e2e_client(
    e2e_components: AppComponents,
) -> TestClient:
    """Create a test client using the real FAQ pipeline."""

    app = create_app(e2e_components)
    return TestClient(app)


def test_e2e_health(
    e2e_client: TestClient,
) -> None:
    """The application exposes a healthy HTTP service."""

    response = e2e_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_e2e_text_chat(
    e2e_client: TestClient,
) -> None:
    """
    Exercise the complete text API pipeline.

    HTTP
        -> request validation
        -> language detection
        -> sanitization
        -> FAQ matching
        -> response building
        -> JSON response
    """

    response = e2e_client.post(
        "/api/chat",
        json={"query": "What is Astro AI?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert set(body) == {
        "answer",
        "score",
        "related_questions",
    }

    assert isinstance(body["answer"], str)
    assert body["answer"]
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["related_questions"], list)


def test_e2e_text_chat_accepts_sanitizable_query(
    e2e_client: TestClient,
) -> None:
    """The real sanitizer and matcher operate on a noisy English query."""

    response = e2e_client.post(
        "/api/chat",
        json={
            "query": "Um... WHAT is Astro-AI?!",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"]
    assert 0.0 <= body["score"] <= 1.0


def test_e2e_english_voice_chat(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """
    Exercise the complete English voice pipeline.

    HTTP
        -> audio upload
        -> STT
        -> language detection
        -> sanitization
        -> FAQ matching
        -> response building
        -> TTS
        -> audio response
    """

    e2e_components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="What is Astro AI?",
        )
    )

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"fake-input-audio",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-e2e-audio"
    assert response.headers["content-type"].startswith("audio/mpeg")

    e2e_components.transcriber.transcribe.assert_awaited_once()

    transcription_call = (
        e2e_components.transcriber.transcribe.await_args
    )

    assert transcription_call.args[0] == b"fake-input-audio"
    assert transcription_call.kwargs["filename"] == "question.wav"

    e2e_components.translator.translate.assert_not_awaited()

    e2e_components.synthesizer.synthesize.assert_awaited_once()

    synthesis_call = (
        e2e_components.synthesizer.synthesize.await_args
    )

    assert synthesis_call.args[0]
    assert "Astro" in synthesis_call.args[0]


def test_e2e_bangla_voice_chat(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """
    Exercise the complete Bangla voice pipeline.

    HTTP
        -> audio
        -> STT
        -> language detection
        -> Bangla sanitization
        -> Bangla -> English translation
        -> English sanitization
        -> FAQ matching
        -> response building
        -> English -> Bangla translation
        -> TTS
        -> audio
    """

    e2e_components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="অ্যাস্ট্রো এআই কী?",
        )
    )

    e2e_components.translator.translate.side_effect = [
        TranslationResult(
            text="What is Astro AI?",
            source_language="bn",
            target_language="en",
            provider="local",
        ),
        TranslationResult(
            text="অ্যাস্ট্রো-এআই গ্যালাক্সি বিবর্তন বিশ্লেষণ করে।",
            source_language="en",
            target_language="bn",
            provider="local",
        ),
    ]

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"fake-input-audio",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-e2e-audio"

    assert (
        e2e_components.translator.translate.await_count
        == 2
    )

    first_translation = (
        e2e_components.translator.translate.await_args_list[0]
    )
    second_translation = (
        e2e_components.translator.translate.await_args_list[1]
    )

    assert first_translation.args[0] == "অ্যাস্ট্রো এআই কী?"
    assert (
        first_translation.args[1]
        is TranslationDirection.TO_ENGLISH
    )

    assert second_translation.args[1] is TranslationDirection.TO_BANGLA

    e2e_components.synthesizer.synthesize.assert_awaited_once_with(
        "অ্যাস্ট্রো-এআই গ্যালাক্সি বিবর্তন বিশ্লেষণ করে。",
    )


def test_e2e_banglish_voice_chat(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """Exercise the Banglish -> English -> Bangla voice path."""

    e2e_components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="bhai astro ai ki?",
        )
    )

    e2e_components.translator.translate.side_effect = [
        TranslationResult(
            text="What is Astro AI?",
            source_language="bn",
            target_language="en",
            provider="local",
        ),
        TranslationResult(
            text="অ্যাস্ট্রো-এআই কী?",
            source_language="en",
            target_language="bn",
            provider="local",
        ),
    ]

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.webm",
                b"fake-input-audio",
                "audio/webm",
            )
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-e2e-audio"

    assert (
        e2e_components.translator.translate.await_count
        == 2
    )

    first_translation = (
        e2e_components.translator.translate.await_args_list[0]
    )
    second_translation = (
        e2e_components.translator.translate.await_args_list[1]
    )

    assert (
        first_translation.args[1]
        is TranslationDirection.TO_ENGLISH
    )
    assert (
        second_translation.args[1]
        is TranslationDirection.TO_BANGLA
    )

    e2e_components.synthesizer.synthesize.assert_awaited_once_with(
        "অ্যাস্ট্রো-এআই কী?",
    )


def test_e2e_voice_requires_audio(
    e2e_client: TestClient,
) -> None:
    """The voice endpoint rejects requests without audio."""

    response = e2e_client.post("/api/chat/voice")

    assert response.status_code == 422


def test_e2e_voice_rejects_empty_audio(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """Empty uploads are rejected before STT."""

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    e2e_components.transcriber.transcribe.assert_not_awaited()


def test_e2e_voice_transcription_failure(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """STT failures terminate the pipeline cleanly."""

    from speech.transcriber import TranscriptionError

    e2e_components.transcriber.transcribe.side_effect = (
        TranscriptionError("provider unavailable")
    )

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"fake-input-audio",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Speech transcription failed.",
    }

    e2e_components.synthesizer.synthesize.assert_not_awaited()


def test_e2e_voice_input_translation_failure(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """Input translation failures stop FAQ processing."""

    from translation.translator import TranslationError

    e2e_components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="অ্যাস্ট্রো এআই কী?",
        )
    )

    e2e_components.translator.translate.side_effect = (
        TranslationError("translation unavailable")
    )

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"fake-input-audio",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Input translation failed.",
    }

    e2e_components.synthesizer.synthesize.assert_not_awaited()


def test_e2e_voice_synthesis_failure(
    e2e_client: TestClient,
    e2e_components: AppComponents,
) -> None:
    """TTS failures are returned as API errors."""

    from speech.synthesizer import SynthesisError

    e2e_components.synthesizer.synthesize.side_effect = (
        SynthesisError("provider unavailable")
    )

    response = e2e_client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"fake-input-audio",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Speech synthesis failed.",
    }


def test_e2e_application_uses_real_faq_matcher(
    e2e_components: AppComponents,
) -> None:
    """The E2E fixture must use the real FAQ matcher."""

    assert isinstance(
        e2e_components.matcher,
        FAQMatcher,
    )

    assert isinstance(
        e2e_components.response_builder,
        ResponseBuilder,
    )

    assert len(e2e_components.matcher) > 0


def test_e2e_app_uses_injected_components(
    e2e_components: AppComponents,
) -> None:
    """create_app must preserve the injected dependency graph."""

    app = create_app(e2e_components)

    assert app.state.components is e2e_components
