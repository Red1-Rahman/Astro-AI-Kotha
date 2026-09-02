# tests/test_transcriber.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from speech.transcriber import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    FishTranscriber,
    TranscriptionError,
    TranscriptionResult,
    create_transcriber,
)


@pytest.fixture
def api_key() -> str:
    return "test-api-key"


@pytest.fixture
def transcriber(api_key: str) -> FishTranscriber:
    return FishTranscriber(api_key=api_key)


def test_transcription_result_contract() -> None:
    result = TranscriptionResult(text="What is Astro AI?")

    assert result.text == "What is Astro AI?"


def test_transcription_result_is_immutable() -> None:
    result = TranscriptionResult(text="What is Astro AI?")

    with pytest.raises(AttributeError):
        result.text = "changed"  # type: ignore[misc]


def test_transcriber_requires_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(
            ValueError,
            match=(
                "AI_GATEWAY_API_KEY must be provided through the "
                "api_key argument or environment variable"
            ),
        ):
            FishTranscriber()


def test_transcriber_reads_api_key_from_environment() -> None:
    with patch.dict(
        "os.environ",
        {"AI_GATEWAY_API_KEY": "environment-key"},
        clear=True,
    ):
        transcriber = FishTranscriber()

    assert transcriber.model == DEFAULT_MODEL
    assert transcriber.base_url == DEFAULT_BASE_URL


def test_transcriber_rejects_empty_model(api_key: str) -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        FishTranscriber(
            api_key=api_key,
            model=" ",
        )


def test_transcriber_rejects_empty_base_url(api_key: str) -> None:
    with pytest.raises(ValueError, match="base_url must not be empty"):
        FishTranscriber(
            api_key=api_key,
            base_url=" ",
        )


@pytest.mark.asyncio
async def test_transcribe_success(
    transcriber: FishTranscriber,
) -> None:
    response = SimpleNamespace(
        text="What is Astro AI?",
    )

    transcriber._client.audio.transcriptions.create = AsyncMock(
        return_value=response,
    )

    result = await transcriber.transcribe(
        b"fake-audio",
        filename="question.webm",
    )

    assert result == TranscriptionResult(
        text="What is Astro AI?",
    )

    transcriber._client.audio.transcriptions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcribe_strips_transcript_whitespace(
    transcriber: FishTranscriber,
) -> None:
    response = SimpleNamespace(
        text="  What is Astro AI?  ",
    )

    transcriber._client.audio.transcriptions.create = AsyncMock(
        return_value=response,
    )

    result = await transcriber.transcribe(b"fake-audio")

    assert result.text == "What is Astro AI?"


@pytest.mark.asyncio
async def test_transcribe_passes_configured_model(
    transcriber: FishTranscriber,
) -> None:
    response = SimpleNamespace(
        text="How does galaxy evolution work?",
    )

    create_mock = AsyncMock(return_value=response)
    transcriber._client.audio.transcriptions.create = create_mock

    await transcriber.transcribe(
        b"fake-audio",
        filename="question.wav",
    )

    create_mock.assert_awaited_once()

    call_kwargs = create_mock.await_args.kwargs

    assert call_kwargs["model"] == DEFAULT_MODEL

    uploaded_file = call_kwargs["file"]

    assert uploaded_file.name == "question.wav"
    assert uploaded_file.read() == b"fake-audio"


@pytest.mark.asyncio
async def test_transcribe_rejects_non_bytes_audio(
    transcriber: FishTranscriber,
) -> None:
    with pytest.raises(TypeError, match="audio must be bytes"):
        await transcriber.transcribe(
            "not-audio",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_audio(
    transcriber: FishTranscriber,
) -> None:
    with pytest.raises(ValueError, match="audio must not be empty"):
        await transcriber.transcribe(b"")


@pytest.mark.asyncio
async def test_transcribe_rejects_non_string_filename(
    transcriber: FishTranscriber,
) -> None:
    with pytest.raises(TypeError, match="filename must be a string"):
        await transcriber.transcribe(
            b"fake-audio",
            filename=123,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_filename(
    transcriber: FishTranscriber,
) -> None:
    with pytest.raises(ValueError, match="filename must not be empty"):
        await transcriber.transcribe(
            b"fake-audio",
            filename=" ",
        )


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_provider_response(
    transcriber: FishTranscriber,
) -> None:
    response = SimpleNamespace(text="   ")

    transcriber._client.audio.transcriptions.create = AsyncMock(
        return_value=response,
    )

    with pytest.raises(
        TranscriptionError,
        match="provider returned an empty transcript",
    ):
        await transcriber.transcribe(b"fake-audio")


@pytest.mark.asyncio
async def test_transcribe_wraps_provider_failure(
    transcriber: FishTranscriber,
) -> None:
    transcriber._client.audio.transcriptions.create = AsyncMock(
        side_effect=RuntimeError("provider unavailable"),
    )

    with pytest.raises(
        TranscriptionError,
        match="Audio transcription failed: provider unavailable",
    ):
        await transcriber.transcribe(b"fake-audio")


def test_create_transcriber(api_key: str) -> None:
    transcriber = create_transcriber(api_key=api_key)

    assert isinstance(transcriber, FishTranscriber)
    assert transcriber.model == DEFAULT_MODEL
    assert transcriber.base_url == DEFAULT_BASE_URL
