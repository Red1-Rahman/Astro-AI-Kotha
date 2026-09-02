# tests/test_azure_translator.py
from __future__ import annotations

import json

import httpx
import pytest

from translation.azure_translator import (
    DEFAULT_AZURE_ENDPOINT,
    AzureTranslator,
)
from translation.translator import (
    AzureTranslationError,
    TranslationDirection,
)


def make_client(
    handler,
) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)

    return httpx.AsyncClient(
        transport=transport,
    )


@pytest.mark.asyncio
async def test_bengali_to_english_translation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/translate"

        assert request.url.params["api-version"] == "3.0"
        assert request.url.params["to"] == "en"

        assert (
            request.headers["Ocp-Apim-Subscription-Key"]
            == "test-key"
        )
        assert (
            request.headers["Ocp-Apim-Subscription-Region"]
            == "test-region"
        )

        body = json.loads(request.content)

        assert body == [
            {"text": "প্রকল্পটি কী?"}
        ]

        return httpx.Response(
            200,
            json=[
                {
                    "translations": [
                        {
                            "text": "What is the project?",
                            "to": "en",
                        }
                    ]
                }
            ],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="test-key",
        region="test-region",
        client=client,
    )

    result = await translator.translate(
        "প্রকল্পটি কী?",
        TranslationDirection.TO_ENGLISH,
    )

    assert result.text == "What is the project?"
    assert result.source_language == "bn"
    assert result.target_language == "en"
    assert result.provider == "azure"

    await client.aclose()


@pytest.mark.asyncio
async def test_english_to_bengali_translation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["to"] == "bn"

        return httpx.Response(
            200,
            json=[
                {
                    "translations": [
                        {
                            "text": "প্রকল্পটি কী?",
                            "to": "bn",
                        }
                    ]
                }
            ],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="test-key",
        region="test-region",
        client=client,
    )

    result = await translator.translate(
        "What is the project?",
        TranslationDirection.TO_BANGLA,
    )

    assert result.text == "প্রকল্পটি কী?"
    assert result.source_language == "en"
    assert result.target_language == "bn"
    assert result.provider == "azure"

    await client.aclose()


@pytest.mark.asyncio
async def test_input_is_stripped() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)

        assert body == [{"text": "hello"}]

        return httpx.Response(
            200,
            json=[
                {
                    "translations": [
                        {
                            "text": "হ্যালো",
                            "to": "bn",
                        }
                    ]
                }
            ],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    await translator.translate(
        "   hello   ",
        TranslationDirection.TO_BANGLA,
    )

    await client.aclose()


@pytest.mark.asyncio
async def test_http_error_becomes_azure_translation_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": "unauthorized"},
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(AzureTranslationError, match="HTTP 401"):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_network_error_becomes_azure_translation_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(AzureTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_json_structure_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"unexpected": "object"},
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(
        AzureTranslationError,
        match="must be a list",
    ):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_empty_response_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(AzureTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_missing_translations_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{}],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(AzureTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_empty_translation_is_rejected() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "translations": [
                        {
                            "text": "   ",
                            "to": "bn",
                        }
                    ]
                }
            ],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(AzureTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_input_is_rejected() -> None:
    client = make_client(
        lambda _: httpx.Response(500)
    )

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(ValueError):
        await translator.translate(
            "   ",
            TranslationDirection.TO_BANGLA,
        )

    await client.aclose()


@pytest.mark.asyncio
async def test_invalid_direction_is_rejected() -> None:
    client = make_client(
        lambda _: httpx.Response(500)
    )

    translator = AzureTranslator(
        api_key="key",
        region="region",
        client=client,
    )

    with pytest.raises(TypeError):
        await translator.translate(
            "hello",
            "to_bangla",  # type: ignore[arg-type]
        )

    await client.aclose()


def test_api_key_can_be_loaded_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AZURE_TRANSLATOR_KEY",
        "environment-key",
    )
    monkeypatch.setenv(
        "AZURE_TRANSLATOR_REGION",
        "eastus",
    )

    translator = AzureTranslator()

    assert translator._api_key == "environment-key"
    assert translator._region == "eastus"


def test_explicit_credentials_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AZURE_TRANSLATOR_KEY",
        "environment-key",
    )
    monkeypatch.setenv(
        "AZURE_TRANSLATOR_REGION",
        "environment-region",
    )

    translator = AzureTranslator(
        api_key="explicit-key",
        region="explicit-region",
    )

    assert translator._api_key == "explicit-key"
    assert translator._region == "explicit-region"


def test_missing_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "AZURE_TRANSLATOR_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="AZURE_TRANSLATOR_KEY is required",
    ):
        AzureTranslator(
            region="eastus",
        )


def test_missing_region_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "AZURE_TRANSLATOR_REGION",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="AZURE_TRANSLATOR_REGION is required",
    ):
        AzureTranslator(
            api_key="key",
        )


def test_endpoint_is_normalized() -> None:
    translator = AzureTranslator(
        api_key="key",
        region="eastus",
        endpoint="https://example.com///",
    )

    assert translator._endpoint == "https://example.com"


def test_default_endpoint() -> None:
    translator = AzureTranslator(
        api_key="key",
        region="eastus",
    )

    assert translator._endpoint == DEFAULT_AZURE_ENDPOINT


def test_invalid_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        AzureTranslator(
            api_key="key",
            region="eastus",
            timeout=0,
        )


@pytest.mark.asyncio
async def test_injected_client_is_not_closed_by_aclose() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "translations": [
                        {
                            "text": "হ্যালো",
                            "to": "bn",
                        }
                    ]
                }
            ],
        )

    client = make_client(handler)

    translator = AzureTranslator(
        api_key="key",
        region="eastus",
        client=client,
    )

    await translator.aclose()

    assert client.is_closed is False

    await client.aclose()
