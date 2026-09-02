# tests/test_translator.py
from __future__ import annotations

import os
from typing import Any

import httpx

from translation.translator import (
    AzureTranslationError,
    TranslationDirection,
    TranslationResult,
)


DEFAULT_AZURE_ENDPOINT = (
    "https://api.cognitive.microsofttranslator.com"
)
DEFAULT_TIMEOUT = 30.0


class AzureTranslator:
    """
    Azure AI Translator REST client.

    Authentication:
        Ocp-Apim-Subscription-Key
        Ocp-Apim-Subscription-Region
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        region: str | None = None,
        endpoint: str = DEFAULT_AZURE_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = self._resolve_value(
            api_key,
            "AZURE_TRANSLATOR_KEY",
        )
        self._region = self._resolve_value(
            region,
            "AZURE_TRANSLATOR_REGION",
        )

        if not isinstance(endpoint, str):
            raise TypeError("endpoint must be a string")

        normalized_endpoint = endpoint.strip().rstrip("/")

        if not normalized_endpoint:
            raise ValueError("endpoint must not be empty")

        if not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number")

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        if client is not None and not isinstance(
            client,
            httpx.AsyncClient,
        ):
            raise TypeError("client must be an httpx.AsyncClient")

        self._endpoint = normalized_endpoint
        self._timeout = float(timeout)
        self._client = client
        self._owns_client = client is None

    async def translate(
        self,
        text: str,
        direction: TranslationDirection,
    ) -> TranslationResult:
        self._validate_input(text, direction)

        target_language = self._target_language(direction)

        client = self._get_client()

        url = f"{self._endpoint}/translate"

        params = {
            "api-version": "3.0",
            "to": target_language,
        }

        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
            "Ocp-Apim-Subscription-Region": self._region,
            "Content-Type": "application/json",
        }

        payload = [{"text": text.strip()}]

        try:
            response = await client.post(
                url,
                params=params,
                headers=headers,
                json=payload,
            )

            response.raise_for_status()

            data = response.json()
            translated = self._extract_translation(data)

            return TranslationResult(
                text=translated,
                source_language=(
                    "bn"
                    if direction is TranslationDirection.TO_ENGLISH
                    else "en"
                ),
                target_language=target_language,
                provider="azure",
            )

        except AzureTranslationError:
            raise
        except httpx.HTTPStatusError as exc:
            raise AzureTranslationError(
                f"Azure Translator returned HTTP {exc.response.status_code}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AzureTranslationError(
                f"Azure Translator request failed: {exc}"
            ) from exc
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise AzureTranslationError(
                f"Azure Translator returned an invalid response: {exc}"
            ) from exc
        except Exception as exc:
            raise AzureTranslationError(
                f"Azure translation failed: {exc}"
            ) from exc

    async def aclose(self) -> None:
        """Close the internally-created HTTP client."""

        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
            )

        return self._client

    @staticmethod
    def _target_language(
        direction: TranslationDirection,
    ) -> str:
        if direction is TranslationDirection.TO_ENGLISH:
            return "en"

        if direction is TranslationDirection.TO_BANGLA:
            return "bn"

        raise TypeError(
            "direction must be a TranslationDirection value"
        )

    @staticmethod
    def _extract_translation(data: Any) -> str:
        if not isinstance(data, list):
            raise AzureTranslationError(
                "Azure response must be a list"
            )

        if not data:
            raise AzureTranslationError(
                "Azure response is empty"
            )

        first_item = data[0]

        if not isinstance(first_item, dict):
            raise AzureTranslationError(
                "Azure response item must be an object"
            )

        translations = first_item.get("translations")

        if not isinstance(translations, list):
            raise AzureTranslationError(
                "Azure response is missing translations"
            )

        if not translations:
            raise AzureTranslationError(
                "Azure response contains no translations"
            )

        first_translation = translations[0]

        if not isinstance(first_translation, dict):
            raise AzureTranslationError(
                "Azure translation item must be an object"
            )

        text = first_translation.get("text")

        if not isinstance(text, str):
            raise AzureTranslationError(
                "Azure translation text must be a string"
            )

        normalized = text.strip()

        if not normalized:
            raise AzureTranslationError(
                "Azure translation text is empty"
            )

        return normalized

    @staticmethod
    def _resolve_value(
        value: str | None,
        environment_name: str,
    ) -> str:
        if value is not None:
            if not isinstance(value, str):
                raise TypeError(
                    f"{environment_name} value must be a string"
                )

            normalized = value.strip()

            if normalized:
                return normalized

        environment_value = os.getenv(environment_name, "").strip()

        if not environment_value:
            raise ValueError(
                f"{environment_name} is required"
            )

        return environment_value

    @staticmethod
    def _validate_input(
        text: str,
        direction: TranslationDirection,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(direction, TranslationDirection):
            raise TypeError(
                "direction must be a TranslationDirection value"
            )
