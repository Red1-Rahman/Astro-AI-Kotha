# tests/test_local_translator.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from translation.local_translator import LocalTranslator
from translation.translator import (
    DEFAULT_BN_TO_EN_MODEL,
    DEFAULT_EN_TO_BN_MODEL,
    LocalTranslationError,
    TranslationDirection,
)


class FakeTokenizer:
    def __init__(
        self,
        output_text: str = "translated text",
        call_error: Exception | None = None,
        decode_error: Exception | None = None,
    ) -> None:
        self.output_text = output_text
        self.call_error = call_error
        self.decode_error = decode_error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self,
        text: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.call_error is not None:
            raise self.call_error

        self.calls.append((text, kwargs))

        return {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

    def batch_decode(
        self,
        outputs: Any,
        **kwargs: Any,
    ) -> list[str]:
        if self.decode_error is not None:
            raise self.decode_error

        return [self.output_text]


class FakeModel:
    def __init__(
        self,
        output: Any = [[1, 2, 3]],
        generate_error: Exception | None = None,
    ) -> None:
        self.output = output
        self.generate_error = generate_error
        self.generate_calls: list[dict[str, Any]] = []
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> Any:
        if self.generate_error is not None:
            raise self.generate_error

        self.generate_calls.append(kwargs)

        return self.output


@dataclass
class FakeFactories:
    tokenizer_output: str = "translated text"
    model_output: Any = None

    def __post_init__(self) -> None:
        self.tokenizers: list[FakeTokenizer] = []
        self.models: list[FakeModel] = []
        self.tokenizer_names: list[str] = []
        self.model_names: list[str] = []

    def tokenizer_factory(self, model_name: str) -> FakeTokenizer:
        tokenizer = FakeTokenizer(
            output_text=self.tokenizer_output,
        )

        self.tokenizers.append(tokenizer)
        self.tokenizer_names.append(model_name)

        return tokenizer

    def model_factory(self, model_name: str) -> FakeModel:
        model = FakeModel(
            output=(
                self.model_output
                if self.model_output is not None
                else [[1, 2, 3]]
            ),
        )

        self.models.append(model)
        self.model_names.append(model_name)

        return model


def make_translator(
    factories: FakeFactories | None = None,
    **kwargs: Any,
) -> tuple[LocalTranslator, FakeFactories]:
    if factories is None:
        factories = FakeFactories()

    translator = LocalTranslator(
        tokenizer_factory=factories.tokenizer_factory,
        model_factory=factories.model_factory,
        **kwargs,
    )

    return translator, factories


def test_default_model_names() -> None:
    translator, _ = make_translator()

    assert translator._bn_to_en_model_name == DEFAULT_BN_TO_EN_MODEL
    assert translator._en_to_bn_model_name == DEFAULT_EN_TO_BN_MODEL


def test_model_names_are_stripped() -> None:
    translator, _ = make_translator(
        bn_to_en_model="  bn-model  ",
        en_to_bn_model="  en-model  ",
    )

    assert translator._bn_to_en_model_name == "bn-model"
    assert translator._en_to_bn_model_name == "en-model"


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        LocalTranslator(bn_to_en_model="   ")


def test_non_string_model_name_is_rejected() -> None:
    with pytest.raises(TypeError):
        LocalTranslator(
            bn_to_en_model=123,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_bengali_to_english_translation() -> None:
    factories = FakeFactories(
        tokenizer_output="What is the project?"
    )

    translator, _ = make_translator(factories)

    result = await translator.translate(
        "প্রকল্পটি কী?",
        TranslationDirection.TO_ENGLISH,
    )

    assert result.text == "What is the project?"
    assert result.source_language == "bn"
    assert result.target_language == "en"
    assert result.provider == "local"

    assert factories.tokenizer_names == [
        DEFAULT_BN_TO_EN_MODEL
    ]
    assert factories.model_names == [
        DEFAULT_BN_TO_EN_MODEL
    ]


@pytest.mark.asyncio
async def test_english_to_bengali_translation() -> None:
    factories = FakeFactories(
        tokenizer_output="প্রকল্পটি কী?"
    )

    translator, _ = make_translator(factories)

    result = await translator.translate(
        "What is the project?",
        TranslationDirection.TO_BANGLA,
    )

    assert result.text == "প্রকল্পটি কী?"
    assert result.source_language == "en"
    assert result.target_language == "bn"
    assert result.provider == "local"

    assert factories.tokenizer_names == [
        DEFAULT_EN_TO_BN_MODEL
    ]
    assert factories.model_names == [
        DEFAULT_EN_TO_BN_MODEL
    ]


@pytest.mark.asyncio
async def test_input_is_stripped_before_translation() -> None:
    factories = FakeFactories()

    translator, _ = make_translator(factories)

    await translator.translate(
        "   hello world   ",
        TranslationDirection.TO_BANGLA,
    )

    tokenizer = factories.tokenizers[0]

    assert tokenizer.calls[0][0] == "hello world"


@pytest.mark.asyncio
async def test_empty_input_is_rejected() -> None:
    translator, _ = make_translator()

    with pytest.raises(ValueError):
        await translator.translate(
            "   ",
            TranslationDirection.TO_ENGLISH,
        )


@pytest.mark.asyncio
async def test_non_string_input_is_rejected() -> None:
    translator, _ = make_translator()

    with pytest.raises(TypeError):
        await translator.translate(
            123,  # type: ignore[arg-type]
            TranslationDirection.TO_ENGLISH,
        )


@pytest.mark.asyncio
async def test_invalid_direction_is_rejected() -> None:
    translator, _ = make_translator()

    with pytest.raises(TypeError):
        await translator.translate(
            "hello",
            "to_english",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_model_is_loaded_only_once() -> None:
    factories = FakeFactories()

    translator, _ = make_translator(factories)

    await translator.translate(
        "hello",
        TranslationDirection.TO_BANGLA,
    )

    await translator.translate(
        "how are you",
        TranslationDirection.TO_BANGLA,
    )

    assert len(factories.tokenizers) == 1
    assert len(factories.models) == 1


@pytest.mark.asyncio
async def test_different_directions_load_different_models() -> None:
    factories = FakeFactories()

    translator, _ = make_translator(factories)

    await translator.translate(
        "hello",
        TranslationDirection.TO_BANGLA,
    )

    await translator.translate(
        "আপনি কেমন আছেন?",
        TranslationDirection.TO_ENGLISH,
    )

    assert factories.tokenizer_names == [
        DEFAULT_EN_TO_BN_MODEL,
        DEFAULT_BN_TO_EN_MODEL,
    ]

    assert factories.model_names == [
        DEFAULT_EN_TO_BN_MODEL,
        DEFAULT_BN_TO_EN_MODEL,
    ]


@pytest.mark.asyncio
async def test_model_is_put_into_eval_mode() -> None:
    factories = FakeFactories()

    translator, _ = make_translator(factories)

    await translator.translate(
        "hello",
        TranslationDirection.TO_BANGLA,
    )

    assert factories.models[0].eval_called is True


@pytest.mark.asyncio
async def test_tokenizer_failure_becomes_local_translation_error() -> None:
    tokenizer = FakeTokenizer(
        call_error=RuntimeError("tokenizer failed")
    )
    model = FakeModel()

    def tokenizer_factory(_: str) -> FakeTokenizer:
        return tokenizer

    def model_factory(_: str) -> FakeModel:
        return model

    translator = LocalTranslator(
        tokenizer_factory=tokenizer_factory,
        model_factory=model_factory,
    )

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


@pytest.mark.asyncio
async def test_model_generation_failure_becomes_local_translation_error() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(
        generate_error=RuntimeError("generation failed")
    )

    translator = LocalTranslator(
        tokenizer_factory=lambda _: tokenizer,
        model_factory=lambda _: model,
    )

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


@pytest.mark.asyncio
async def test_decode_failure_becomes_local_translation_error() -> None:
    tokenizer = FakeTokenizer(
        decode_error=RuntimeError("decode failed")
    )
    model = FakeModel()

    translator = LocalTranslator(
        tokenizer_factory=lambda _: tokenizer,
        model_factory=lambda _: model,
    )

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


@pytest.mark.asyncio
async def test_empty_model_output_is_rejected() -> None:
    factories = FakeFactories(
        tokenizer_output="   ",
    )

    translator, _ = make_translator(factories)

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


@pytest.mark.asyncio
async def test_empty_decoded_list_is_rejected() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()

    def empty_decode(
        outputs: Any,
        **kwargs: Any,
    ) -> list[str]:
        return []

    tokenizer.batch_decode = empty_decode  # type: ignore[method-assign]

    translator = LocalTranslator(
        tokenizer_factory=lambda _: tokenizer,
        model_factory=lambda _: model,
    )

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


@pytest.mark.asyncio
async def test_invalid_decoded_output_is_rejected() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()

    def invalid_decode(
        outputs: Any,
        **kwargs: Any,
    ) -> list[int]:
        return [123]

    tokenizer.batch_decode = invalid_decode  # type: ignore[method-assign]

    translator = LocalTranslator(
        tokenizer_factory=lambda _: tokenizer,
        model_factory=lambda _: model,
    )

    with pytest.raises(LocalTranslationError):
        await translator.translate(
            "hello",
            TranslationDirection.TO_BANGLA,
        )


def test_models_are_initially_unloaded() -> None:
    translator, _ = make_translator()

    assert translator.models_loaded() == {
        TranslationDirection.TO_ENGLISH: False,
        TranslationDirection.TO_BANGLA: False,
    }


@pytest.mark.asyncio
async def test_models_loaded_reports_initialized_direction() -> None:
    translator, _ = make_translator()

    await translator.translate(
        "hello",
        TranslationDirection.TO_BANGLA,
    )

    assert translator.models_loaded() == {
        TranslationDirection.TO_ENGLISH: False,
        TranslationDirection.TO_BANGLA: True,
    }


@pytest.mark.asyncio
async def test_concurrent_requests_do_not_load_model_twice() -> None:
    class CountingFactories:
        def __init__(self) -> None:
            self.tokenizer_calls = 0
            self.model_calls = 0

        def tokenizer_factory(self, _: str) -> FakeTokenizer:
            self.tokenizer_calls += 1

            # Yield to make a race much more likely if locking is broken.
            return FakeTokenizer(
                output_text="translated"
            )

        def model_factory(self, _: str) -> FakeModel:
            self.model_calls += 1

            return FakeModel()

    factories = CountingFactories()

    translator = LocalTranslator(
        tokenizer_factory=factories.tokenizer_factory,
        model_factory=factories.model_factory,
    )

    await asyncio.gather(
        *(
            translator.translate(
                f"hello {index}",
                TranslationDirection.TO_BANGLA,
            )
            for index in range(20)
        )
    )

    assert factories.tokenizer_calls == 1
    assert factories.model_calls == 1


def test_factories_are_optional() -> None:
    translator = LocalTranslator()

    assert translator._tokenizer_factory is None
    assert translator._model_factory is None
