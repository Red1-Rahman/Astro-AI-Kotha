# chatbot/faq_loader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FAQ_PATH = PROJECT_ROOT / "data" / "faqs.json"


class FAQMetadata(BaseModel):
    """Metadata describing the FAQ knowledge base."""

    model_config = ConfigDict(extra="allow")

    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    nlp_processor: str = Field(min_length=1)
    nlp_model: str = Field(min_length=1)
    preprocessing_steps: list[str] = Field(min_length=1)
    matching_algorithm: str = Field(min_length=1)
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    total_faqs: int = Field(ge=0)
    last_updated: str = Field(min_length=1)


class FAQ(BaseModel):
    """A single FAQ entry."""

    model_config = ConfigDict(extra="allow")

    id: int = Field(gt=0)
    category: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    keywords: list[str]
    entities: list[str]
    related_ids: list[int]

    @field_validator("keywords", "entities")
    @classmethod
    def validate_string_list(cls, value: list[str]) -> list[str]:
        """Reject empty values inside string lists."""
        if any(not item.strip() for item in value):
            raise ValueError("list items must not be empty")

        return value

    @field_validator("related_ids")
    @classmethod
    def validate_related_ids(cls, value: list[int]) -> list[int]:
        """Reject invalid related FAQ IDs."""
        if any(item <= 0 for item in value):
            raise ValueError("related FAQ IDs must be greater than zero")

        if len(value) != len(set(value)):
            raise ValueError("related FAQ IDs must be unique")

        return value


class FAQDatabase(BaseModel):
    """Validated FAQ knowledge base."""

    model_config = ConfigDict(extra="allow")

    metadata: FAQMetadata
    faqs: list[FAQ] = Field(min_length=1)

    @field_validator("faqs")
    @classmethod
    def validate_faqs(cls, value: list[FAQ]) -> list[FAQ]:
        """Validate FAQ IDs and internal relationships."""
        ids = [faq.id for faq in value]

        if len(ids) != len(set(ids)):
            raise ValueError("FAQ IDs must be unique")

        faq_ids = set(ids)

        for faq in value:
            missing_related_ids = set(faq.related_ids) - faq_ids

            if missing_related_ids:
                missing = sorted(missing_related_ids)
                raise ValueError(
                    f"FAQ {faq.id} references missing related FAQ IDs: {missing}"
                )

        return value

    def model_post_init(self, __context: Any) -> None:
        """Validate the metadata FAQ count against the actual dataset."""
        if self.metadata.total_faqs != len(self.faqs):
            raise ValueError(
                "metadata.total_faqs does not match the number of FAQ records: "
                f"expected {self.metadata.total_faqs}, found {len(self.faqs)}"
            )


class FAQLoaderError(RuntimeError):
    """Raised when the FAQ knowledge base cannot be loaded or validated."""


def load_faq_database(
    path: str | Path = DEFAULT_FAQ_PATH,
) -> FAQDatabase:
    """
    Load and validate the FAQ knowledge base.

    Args:
        path: Path to the FAQ JSON file.

    Returns:
        A validated FAQDatabase instance.

    Raises:
        FAQLoaderError: If the file cannot be read, parsed, or validated.
    """
    faq_path = Path(path)

    if not faq_path.is_file():
        raise FAQLoaderError(f"FAQ file does not exist: {faq_path}")

    try:
        raw_data = faq_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FAQLoaderError(
            f"Unable to read FAQ file '{faq_path}': {exc}"
        ) from exc

    try:
        data = json.loads(raw_data)
    except json.JSONDecodeError as exc:
        raise FAQLoaderError(
            f"Invalid JSON in FAQ file '{faq_path}' "
            f"at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise FAQLoaderError(
            f"FAQ file '{faq_path}' must contain a JSON object at the top level"
        )

    try:
        return FAQDatabase.model_validate(data)
    except ValidationError as exc:
        raise FAQLoaderError(
            f"FAQ data validation failed for '{faq_path}': {exc}"
        ) from exc


def load_faqs(
    path: str | Path = DEFAULT_FAQ_PATH,
) -> list[FAQ]:
    """
    Load and return the validated FAQ records.

    This is a convenience wrapper around ``load_faq_database``.
    """
    return load_faq_database(path).faqs


def get_faq_by_id(
    faq_id: int,
    path: str | Path = DEFAULT_FAQ_PATH,
) -> FAQ | None:
    """
    Find a FAQ by its stable numeric ID.

    Args:
        faq_id: FAQ identifier.
        path: Path to the FAQ JSON file.

    Returns:
        The matching FAQ, or None if it does not exist.
    """
    if faq_id <= 0:
        raise ValueError("faq_id must be greater than zero")

    database = load_faq_database(path)

    return next(
        (faq for faq in database.faqs if faq.id == faq_id),
        None,
    )
