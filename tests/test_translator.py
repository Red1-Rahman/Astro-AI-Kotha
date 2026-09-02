# tests/test_translator.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from translation.translator import (
    DEFAULT_AZURE_ENDPOINT,
    DEFAULT_BN_TO_EN_MODEL,
    DEFAULT_EN_TO_BN_MODEL,
    AzureTranslationError,
    AzureTranslator,
    FallbackTranslator,
    LocalTranslationError,
    LocalTranslator,
    TranslationDirection,
    TranslationError,
    TranslationResult,
    create_translator,
)


class TestTranslationResult:
    def test_contains_text(self) -> None:
        result = TranslationResult(text="Hello")

        assert result.text == "Hello"

    def test_is_immutable(self) -> None:
        result = TranslationResult(text="Hello")

        with pytest.raises(AttributeError):
            result.text = "Changed"  # type: ignore[misc]


class TestTranslationDirection:
    def test_to_english(self) -> None:
        assert TranslationDirection.TO_ENGLISH.value == "to_english"

    def test_to_bangla(self) -> None:
        assert TranslationDirection.TO_BANGLA.value == "to_bangla"


class FakeTokenizer:
    def __init__(self) -> None:
        self.batch_decode = MagicMock(
            return_value=["translated text"]
        )

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
    ) -> dict[str, object]:
        return {"input_ids": [1, 2, 3]}


class FakeModel:
    def __init__(self) -> None:
        self.generate = MagicMock(
            return_value=["generated"]
        )


class FakeFactory:
    def __init__(self) -> None:
        self.instances: list[FakeTokenizer | FakeModel] = []

    def from_pretrained(
        self,
        model_name: str,
    ) -> FakeTokenizer | FakeModel:
        raise NotImplementedError


class FakeTokenizerFactory(FakeFactory):
    def from_pretrained(
        self,
        model_name: str,
    ) -> FakeTokenizer:
        tokenizer = FakeTokenizer()
        self.instances.append(tokenizer)
        return tokenizer


class FakeModelFactory(FakeFactory):
    def from_pretrained(
        self,
        model_name: str,
    ) -> FakeModel:
        model = FakeModel()
        self.instances.append(model)
        return model


class TestLocalTranslator:
    @pytest.fixture
    def factories(
        self,
    ) -> tuple[FakeTokenizerFactory, FakeModelFactory]:
        return FakeTokenizerFactory(), FakeModelFactory()

    @pytest.fixture
    def translator(
        self,
        factories: tuple[
            FakeTokenizerFactory,
            FakeModelFactory,
        ],
    ) -> LocalTranslator:
        tokenizer_factory, model_factory = factories

        return LocalTranslator(
            tokenizer_factory=tokenizer_factory,
            model_factory=model_factory,
        )

    @pytest.mark.asyncio
    async def test_translates_to_english(
        self,
        translator: LocalTranslator,
        factories: tuple[
            FakeTokenizerFactory,
            FakeModelFactory,
        ],
    ) -> None:
        tokenizer_factory, model_factory = factories

        tokenizer = FakeTokenizer()
        tokenizer.batch_decode.return_value = [
            "I want to learn astronomy"
        ]

        model = FakeModel()
        model.generate.return_value = ["generated"]

        tokenizer_factory.from_pretrained = MagicMock(
            return_value=tokenizer
        )
        model_factory.from_pretrained = MagicMock(
            return_value=model
        )

        result = await translator.translate(
            "আমি জ্যোতির্বিজ্ঞান শিখতে চাই",
            TranslationDirection.TO_ENGLISH,
        )

        assert result == TranslationResult(
            text="I want to learn astronomy"
        )

        tokenizer_factory.from_pretrained.assert_called_once_with(
            DEFAULT_BN_TO_EN_MODEL
        )
        model_factory.from_pretrained.assert_called_once_with(
            DEFAULT_BN_TO_EN_MODEL
        )

    @pytest.mark.asyncio
    async def test_translates_to_bangla(
        self,
        translator: LocalTranslator,
        factories: tuple[
            FakeTokenizerFactory,
            FakeModelFactory,
        ],
    ) -> None:
        tokenizer_factory, model_factory = factories

        tokenizer = FakeTokenizer()
        tokenizer.batch_decode.return_value = [
            "আমি জ্যোতির্বিজ্ঞান শিখতে চাই"
        ]

        model = FakeModel()

        tokenizer_factory.from_pretrained = MagicMock(
            return_value=tokenizer
        )
        model_factory.from_pretrained = MagicMock(
            return_value=model
        )

        result = await translator.translate(
            "I want to learn astronomy",
            TranslationDirection.TO_BANGLA,
        )

        assert result == TranslationResult(
            text="আমি জ্যোতির্বিজ্ঞান শিখতে চাই"
        )

        tokenizer_factory.from_pretrained.assert_called_once_with(
            DEFAULT_EN_TO_BN_MODEL
        )
        model_factory.from_pretrained.assert_called_once_with(
            DEFAULT_EN_TO_BN_MODEL
        )

    @pytest.mark.asyncio
    async def test_local_translation_is_best_effort_for_banglish(
        self,
        translator: LocalTranslator,
    ) -> None:
        translator._translate_sync = MagicMock(
            return_value=TranslationResult(
                text="What is Astro AI?"
            )
        )

        result = await translator.translate(
            "astro ai ki?",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "What is Astro AI?"

    @pytest.mark.asyncio
    async def test_reuses_loaded_models(
        self,
        translator: LocalTranslator,
        factories: tuple[
            FakeTokenizerFactory,
            FakeModelFactory,
        ],
    ) -> None:
        tokenizer_factory, model_factory = factories

        tokenizer = FakeTokenizer()
        tokenizer.batch_decode.return_value = [
            "Hello"
        ]

        model = FakeModel()

        tokenizer_factory.from_pretrained = MagicMock(
            return_value=tokenizer
        )
        model_factory.from_pretrained = MagicMock(
            return_value=model
        )

        await translator.translate(
            "হ্যালো",
            TranslationDirection.TO_ENGLISH,
        )

        await translator.translate(
            "বিদায়",
            TranslationDirection.TO_ENGLISH,
        )

        tokenizer_factory.from_pretrained.assert_called_once_with(
            DEFAULT_BN_TO_EN_MODEL
        )
        model_factory.from_pretrained.assert_called_once_with(
            DEFAULT_BN_TO_EN_MODEL
        )

    @pytest.mark.asyncio
    async def test_rejects_non_string_text(
        self,
        translator: LocalTranslator,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="text must be a string",
        ):
            await translator.translate(
                123,  # type: ignore[arg-type]
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_text(
        self,
        translator: LocalTranslator,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="text must not be empty",
        ):
            await translator.translate(
                "   ",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_direction(
        self,
        translator: LocalTranslator,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="direction must be a TranslationDirection value",
        ):
            await translator.translate(
                "hello",
                "to_english",  # type: ignore[arg-type]
            )

    @pytest.mark.asyncio
    async def test_model_loading_failure_is_wrapped(
        self,
        translator: LocalTranslator,
        factories: tuple[
            FakeTokenizerFactory,
            FakeModelFactory,
        ],
    ) -> None:
        tokenizer_factory, _ = factories

        tokenizer_factory.from_pretrained = MagicMock(
            side_effect=RuntimeError("model unavailable")
        )

        with pytest.raises(
            LocalTranslationError,
            match="Failed to load Bengali-to-English",
        ):
            await translator.translate(
                "হ্যালো",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_inference_failure_is_wrapped(
        self,
        translator: LocalTranslator,
    ) -> None:
        translator._translate_sync = MagicMock(
            side_effect=RuntimeError("inference failed")
        )

        with pytest.raises(
            LocalTranslationError,
            match="Local translation failed",
        ):
            await translator.translate(
                "হ্যালো",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_empty_model_output_is_rejected(
        self,
        translator: LocalTranslator,
    ) -> None:
        translator._translate_sync = MagicMock(
            return_value=TranslationResult(text="")
        )

        result = await translator.translate(
            "হ্যালো",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == ""


class TestAzureTranslator:
    @pytest.fixture
    def client(self) -> MagicMock:
        return MagicMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def translator(
        self,
        client: MagicMock,
    ) -> AzureTranslator:
        return AzureTranslator(
            api_key="test-key",
            region="eastus",
            endpoint=DEFAULT_AZURE_ENDPOINT,
            client=client,
        )

    @pytest.mark.asyncio
    async def test_translates_to_english(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [
                    {
                        "text": "I want to learn astronomy",
                        "to": "en",
                    }
                ]
            }
        ]

        client.post = AsyncMock(return_value=response)

        result = await translator.translate(
            "আমি জ্যোতির্বিজ্ঞান শিখতে চাই",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "I want to learn astronomy"

        client.post.assert_awaited_once()

        call = client.post.await_args

        assert call.args[0] == (
            "https://api.cognitive.microsofttranslator.com/translate"
        )

        assert call.kwargs["params"] == {
            "api-version": "3.0",
            "to": "en",
        }

        assert call.kwargs["headers"][
            "Ocp-Apim-Subscription-Key"
        ] == "test-key"

        assert call.kwargs["headers"][
            "Ocp-Apim-Subscription-Region"
        ] == "eastus"

        assert call.kwargs["json"] == [
            {
                "text": "আমি জ্যোতির্বিজ্ঞান শিখতে চাই",
            }
        ]

    @pytest.mark.asyncio
    async def test_translates_to_bangla(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [
                    {
                        "text": "আমি জ্যোতির্বিজ্ঞান শিখতে চাই",
                        "to": "bn",
                    }
                ]
            }
        ]

        client.post = AsyncMock(return_value=response)

        result = await translator.translate(
            "I want to learn astronomy",
            TranslationDirection.TO_BANGLA,
        )

        assert result.text == "আমি জ্যোতির্বিজ্ঞান শিখতে চাই"

        call = client.post.await_args

        assert call.kwargs["params"] == {
            "api-version": "3.0",
            "to": "bn",
        }

    @pytest.mark.asyncio
    async def test_strips_input(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [
                    {
                        "text": "Hello",
                    }
                ]
            }
        ]

        client.post = AsyncMock(return_value=response)

        result = await translator.translate(
            "   hello   ",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "Hello"

        assert client.post.await_args.kwargs["json"] == [
            {
                "text": "hello",
            }
        ]

    @pytest.mark.asyncio
    async def test_strips_provider_output(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [
                    {
                        "text": "   Hello world   ",
                    }
                ]
            }
        ]

        client.post = AsyncMock(return_value=response)

        result = await translator.translate(
            "হ্যালো",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "Hello world"

    @pytest.mark.asyncio
    async def test_http_error_is_wrapped(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        client.post = AsyncMock(
            side_effect=httpx.ConnectError("connection failed")
        )

        with pytest.raises(
            AzureTranslationError,
            match="Azure Translator request failed",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_http_status_error_is_wrapped(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 401
        response.text = "Unauthorized"

        client.post = AsyncMock(return_value=response)

        with pytest.raises(
            AzureTranslationError,
            match="HTTP 401",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_invalid_json_is_rejected(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.side_effect = ValueError("invalid JSON")

        client.post = AsyncMock(return_value=response)

        with pytest.raises(
            AzureTranslationError,
            match="invalid JSON",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_invalid_response_is_rejected(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {}

        client.post = AsyncMock(return_value=response)

        with pytest.raises(
            AzureTranslationError,
            match="invalid response",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_missing_translation_is_rejected(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [],
            }
        ]

        client.post = AsyncMock(return_value=response)

        with pytest.raises(
            AzureTranslationError,
            match="no translations",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_empty_translation_is_rejected(
        self,
        translator: AzureTranslator,
        client: MagicMock,
    ) -> None:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = [
            {
                "translations": [
                    {
                        "text": "   ",
                    }
                ]
            }
        ]

        client.post = AsyncMock(return_value=response)

        with pytest.raises(
            AzureTranslationError,
            match="empty text",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_text(
        self,
        translator: AzureTranslator,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="text must be a string",
        ):
            await translator.translate(
                123,  # type: ignore[arg-type]
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_rejects_empty_text(
        self,
        translator: AzureTranslator,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="text must not be empty",
        ):
            await translator.translate(
                "   ",
                TranslationDirection.TO_ENGLISH,
            )

    @pytest.mark.asyncio
    async def test_rejects_invalid_direction(
        self,
        translator: AzureTranslator,
    ) -> None:
        with pytest.raises(
            TypeError,
            match="direction must be a TranslationDirection value",
        ):
            await translator.translate(
                "hello",
                "to_english",  # type: ignore[arg-type]
            )


class TestAzureTranslatorInitialization:
    def test_requires_api_key(self) -> None:
        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            with pytest.raises(
                ValueError,
                match="API key is required",
            ):
                AzureTranslator(
                    region="eastus",
                )

    def test_requires_region(self) -> None:
        with pytest.raises(
            ValueError,
            match="region is required",
        ):
            AzureTranslator(
                api_key="test-key",
            )

    def test_uses_environment_configuration(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AZURE_TRANSLATOR_KEY": "env-key",
                "AZURE_TRANSLATOR_REGION": "env-region",
            },
            clear=True,
        ):
            translator = AzureTranslator()

        assert translator.api_key == "env-key"
        assert translator.region == "env-region"

    def test_explicit_values_override_environment(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "AZURE_TRANSLATOR_KEY": "env-key",
                "AZURE_TRANSLATOR_REGION": "env-region",
            },
            clear=True,
        ):
            translator = AzureTranslator(
                api_key="explicit-key",
                region="explicit-region",
            )

        assert translator.api_key == "explicit-key"
        assert translator.region == "explicit-region"

    def test_strips_credentials(self) -> None:
        translator = AzureTranslator(
            api_key="  test-key  ",
            region="  eastus  ",
        )

        assert translator.api_key == "test-key"
        assert translator.region == "eastus"

    def test_custom_endpoint(self) -> None:
        translator = AzureTranslator(
            api_key="test-key",
            region="eastus",
            endpoint="https://example.com/",
        )

        assert translator.endpoint == "https://example.com"


class TestFallbackTranslator:
    @pytest.mark.asyncio
    async def test_primary_success_does_not_call_fallback(self) -> None:
        primary = AsyncMock(
            return_value=TranslationResult(text="local result")
        )
        fallback = AsyncMock()

        translator = FallbackTranslator(
            primary=primary,
            fallback=fallback,
        )

        result = await translator.translate(
            "hello",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "local result"

        primary.assert_awaited_once_with(
            "hello",
            TranslationDirection.TO_ENGLISH,
        )
        fallback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_is_used_after_primary_failure(self) -> None:
        primary = AsyncMock(
            side_effect=LocalTranslationError("local failed")
        )
        fallback = AsyncMock(
            return_value=TranslationResult(
                text="azure result"
            )
        )

        translator = FallbackTranslator(
            primary=primary,
            fallback=fallback,
        )

        result = await translator.translate(
            "hello",
            TranslationDirection.TO_ENGLISH,
        )

        assert result.text == "azure result"

        primary.assert_awaited_once()
        fallback.assert_awaited_once_with(
            "hello",
            TranslationDirection.TO_ENGLISH,
        )

    @pytest.mark.asyncio
    async def test_both_fail_with_combined_error(self) -> None:
        primary = AsyncMock(
            side_effect=LocalTranslationError("local failed")
        )
        fallback = AsyncMock(
            side_effect=AzureTranslationError("azure failed")
        )

        translator = FallbackTranslator(
            primary=primary,
            fallback=fallback,
        )

        with pytest.raises(
            TranslationError,
            match="Both primary and fallback translation failed",
        ) as exc_info:
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

        assert "local failed" in str(exc_info.value)
        assert "azure failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_translation_error_is_not_silently_swallowed(
        self,
    ) -> None:
        primary = AsyncMock(
            side_effect=ValueError("invalid input")
        )
        fallback = AsyncMock()

        translator = FallbackTranslator(
            primary=primary,
            fallback=fallback,
        )

        with pytest.raises(
            ValueError,
            match="invalid input",
        ):
            await translator.translate(
                "hello",
                TranslationDirection.TO_ENGLISH,
            )

        fallback.assert_not_awaited()

    def test_requires_primary(self) -> None:
        with pytest.raises(
            ValueError,
            match="primary translator is required",
        ):
            FallbackTranslator(
                primary=None,  # type: ignore[arg-type]
                fallback=AsyncMock(),
            )

    def test_requires_fallback(self) -> None:
        with pytest.raises(
            ValueError,
            match="fallback translator is required",
        ):
            FallbackTranslator(
                primary=AsyncMock(),
                fallback=None,  # type: ignore[arg-type]
            )


class TestCreateTranslator:
    def test_creates_local_first_fallback(self) -> None:
        local = MagicMock()
        azure = MagicMock()

        translator = create_translator(
            local=local,
            fallback=azure,
        )

        assert isinstance(
            translator,
            FallbackTranslator,
        )

        assert translator.primary is local
        assert translator.fallback is azure

    def test_uses_default_implementations(self) -> None:
        with patch(
            "translation.translator.LocalTranslator"
        ) as local_class:
            with patch(
                "translation.translator.AzureTranslator"
            ) as azure_class:
                local_instance = MagicMock()
                azure_instance = MagicMock()

                local_class.return_value = local_instance
                azure_class.return_value = azure_instance

                translator = create_translator()

        assert isinstance(
            translator,
            FallbackTranslator,
        )

        assert translator.primary is local_instance
        assert translator.fallback is azure_instance
