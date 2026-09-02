# tests/test_api.py
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from chatbot.language_detector import Language
from chatbot.response_builder import ChatResponse
from main import AppComponents, create_app
from speech.synthesizer import SynthesisError, SynthesisResult
from speech.transcriber import TranscriptionError, TranscriptionResult
from translation.translator import (
    TranslationDirection,
    TranslationError,
    TranslationResult,
)


@dataclass
class FakeMatch:
    answer: str = "Astro-AI analyzes galaxy evolution data."
    score: float = 0.91
    related_questions: list[str] | None = None


@pytest.fixture
def components() -> AppComponents:
    faq_database = MagicMock()
    matcher = MagicMock()
    transcriber = AsyncMock()
    translator = AsyncMock()
    synthesizer = AsyncMock()

    matcher.match.return_value = FakeMatch(
        answer="Astro-AI analyzes galaxy evolution data.",
        score=0.91,
        related_questions=[
            "What data does Astro-AI use?",
        ],
    )

    transcriber.transcribe.return_value = TranscriptionResult(
        text="What is Astro AI?",
    )

    translator.translate.return_value = TranslationResult(
        text="What is Astro AI?",
        source_language="bn",
        target_language="en",
        provider="local",
    )

    synthesizer.synthesize.return_value = SynthesisResult(
        audio=b"fake-audio",
        content_type="audio/mpeg",
    )

    return AppComponents(
        faq_database=faq_database,
        matcher=matcher,
        transcriber=transcriber,
        translator=translator,
        synthesizer=synthesizer,
    )


@pytest.fixture
def client(
    components: AppComponents,
) -> TestClient:
    app = create_app(components)
    return TestClient(app)


def test_health_endpoint(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_success(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat",
        json={"query": "What is Astro AI?"},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == (
        "Astro-AI analyzes galaxy evolution data."
    )
    assert body["score"] == pytest.approx(0.91)
    assert body["related_questions"] == [
        "What data does Astro-AI use?",
    ]

    components.matcher.match.assert_called_once_with(
        "what is astro ai",
    )


def test_chat_is_backward_compatible(
    client: TestClient,
) -> None:
    response = client.post(
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


def test_chat_rejects_empty_query(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_chat_rejects_missing_query(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={},
    )

    assert response.status_code == 422


def test_chat_rejects_invalid_query_type(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/chat",
        json={"query": 123},
    )

    assert response.status_code == 422


def test_chat_rejects_non_english_query(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat",
        json={
            "query": "আমি Astro AI সম্পর্কে জানতে চাই",
        },
    )

    assert response.status_code == 400
    assert "English" in response.json()["detail"]

    components.matcher.match.assert_not_called()


def test_chat_does_not_call_voice_components(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat",
        json={"query": "What is Astro AI?"},
    )

    assert response.status_code == 200

    components.transcriber.transcribe.assert_not_awaited()
    components.translator.translate.assert_not_awaited()
    components.synthesizer.synthesize.assert_not_awaited()


def test_voice_requires_audio(
    client: TestClient,
) -> None:
    response = client.post("/api/chat/voice")

    assert response.status_code == 422


def test_voice_rejects_empty_audio(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    components.transcriber.transcribe.assert_not_awaited()


def test_voice_rejects_oversized_audio(
    client: TestClient,
    components: AppComponents,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "main.MAX_AUDIO_SIZE",
        3,
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"1234",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]

    components.transcriber.transcribe.assert_not_awaited()


def test_voice_transcription_failure(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.side_effect = (
        TranscriptionError("provider unavailable")
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Speech transcription failed."
    )


def test_voice_empty_transcription(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(text="   ")
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 422
    assert "empty" in response.json()["detail"].lower()


def test_voice_english_pipeline(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="Um, what is Astro AI?",
        )
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-audio"
    assert response.headers["content-type"].startswith(
        "audio/mpeg"
    )

    components.transcriber.transcribe.assert_awaited_once()

    components.translator.translate.assert_not_awaited()

    components.matcher.match.assert_called_once_with(
        "what is astro ai",
    )

    components.synthesizer.synthesize.assert_awaited_once_with(
        "Astro-AI analyzes galaxy evolution data.",
    )


def test_voice_bangla_pipeline(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="অ্যাস্ট্রো এআই কী?",
        )
    )

    components.translator.translate.side_effect = [
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

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-audio"

    assert components.translator.translate.await_count == 2

    first_call = (
        components.translator.translate.await_args_list[0]
    )
    second_call = (
        components.translator.translate.await_args_list[1]
    )

    assert (
        first_call.args[1]
        is TranslationDirection.TO_ENGLISH
    )
    assert (
        second_call.args[1]
        is TranslationDirection.TO_BANGLA
    )

    components.matcher.match.assert_called_once_with(
        "what is astro ai",
    )

    components.synthesizer.synthesize.assert_awaited_once_with(
        "অ্যাস্ট্রো-এআই গ্যালাক্সি বিবর্তন বিশ্লেষণ করে。",
    )


def test_voice_banglish_pipeline(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="bhai astro ai ki?",
        )
    )

    components.translator.translate.side_effect = [
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

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 200
    assert response.content == b"fake-audio"

    assert components.translator.translate.await_count == 2


def test_voice_input_translation_failure(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="অ্যাস্ট্রো এআই কী?",
        )
    )

    components.translator.translate.side_effect = (
        TranslationError("translation failed")
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Input translation failed."
    )

    components.matcher.match.assert_not_called()


def test_voice_answer_translation_failure(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.transcriber.transcribe.return_value = (
        TranscriptionResult(
            text="অ্যাস্ট্রো এআই কী?",
        )
    )

    components.translator.translate.side_effect = [
        TranslationResult(
            text="What is Astro AI?",
            source_language="bn",
            target_language="en",
            provider="local",
        ),
        TranslationError("translation failed"),
    ]

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Answer translation failed."
    )

    components.synthesizer.synthesize.assert_not_awaited()


def test_voice_synthesis_failure(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.synthesizer.synthesize.side_effect = (
        SynthesisError("tts provider unavailable")
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Speech synthesis failed."
    )


def test_voice_empty_synthesis_result(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.synthesizer.synthesize.return_value = (
        SynthesisResult(
            audio=b"",
            content_type="audio/mpeg",
        )
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 502
    assert "empty audio" in response.json()["detail"].lower()


def test_voice_preserves_filename(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "recording.webm",
                b"audio",
                "audio/webm",
            ),
        },
    )

    assert response.status_code == 200

    call = components.transcriber.transcribe.await_args

    assert call.args[0] == b"audio"
    assert call.kwargs["filename"] == "recording.webm"


def test_voice_uses_default_filename(
    client: TestClient,
    components: AppComponents,
) -> None:
    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 200

    call = components.transcriber.transcribe.await_args

    assert call.kwargs["filename"] == "audio.wav"


def test_voice_matcher_failure(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.matcher.match.side_effect = RuntimeError(
        "unexpected matcher failure"
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "FAQ processing failed."
    )


def test_voice_returns_synthesized_audio(
    client: TestClient,
    components: AppComponents,
) -> None:
    components.synthesizer.synthesize.return_value = (
        SynthesisResult(
            audio=b"wav-data",
            content_type="audio/wav",
        )
    )

    response = client.post(
        "/api/chat/voice",
        files={
            "audio": (
                "question.wav",
                b"audio",
                "audio/wav",
            ),
        },
    )

    assert response.status_code == 200
    assert response.content == b"wav-data"
    assert response.headers["content-type"].startswith(
        "audio/wav"
    )


def test_create_app_accepts_injected_components(
    components: AppComponents,
) -> None:
    app = create_app(components)

    assert app.state.components is components
