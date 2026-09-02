# tests/test_faq_loader.py
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from chatbot.faq_loader import (
    FAQ,
    FAQDatabase,
    FAQLoaderError,
    FAQMetadata,
    get_faq_by_id,
    load_faq_database,
    load_faqs,
)


def make_faq(
    faq_id: int = 1,
    *,
    related_ids: list[int] | None = None,
) -> dict[str, object]:
    """Create a valid FAQ fixture."""
    return {
        "id": faq_id,
        "category": "Installation & Setup",
        "intent": "install_dependencies",
        "question": "What are the system requirements for installing Astro-AI?",
        "answer": "Astro-AI requires Python 3.10 or later.",
        "keywords": ["requirements", "python", "memory", "storage"],
        "entities": ["Python 3.10", "4GB RAM", "2GB storage"],
        "related_ids": related_ids if related_ids is not None else [],
    }


def make_database(
    faqs: list[dict[str, object]] | None = None,
    *,
    total_faqs: int | None = None,
) -> dict[str, object]:
    """Create a valid FAQ database fixture."""
    records = faqs if faqs is not None else [make_faq()]

    return {
        "metadata": {
            "version": "1.0",
            "description": (
                "FAQ Knowledge Base for Astro-AI Galaxy Evolution "
                "Analysis Platform"
            ),
            "nlp_processor": "spacy",
            "nlp_model": "en_core_web_sm",
            "preprocessing_steps": [
                "tokenization",
                "lemmatization",
                "stopword_removal",
                "lowercase_normalization",
            ],
            "matching_algorithm": "tfidf_cosine_similarity",
            "similarity_threshold": 0.4,
            "total_faqs": (
                len(records) if total_faqs is None else total_faqs
            ),
            "last_updated": "2026-03-25",
        },
        "faqs": records,
    }


def write_json(path: Path, data: object) -> None:
    """Write JSON test data to disk."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class TestFAQMetadata:
    """Tests for FAQMetadata validation."""

    def test_valid_metadata(self) -> None:
        metadata = FAQMetadata(
            version="1.0",
            description="Test FAQ database",
            nlp_processor="spacy",
            nlp_model="en_core_web_sm",
            preprocessing_steps=["tokenization"],
            matching_algorithm="tfidf_cosine_similarity",
            similarity_threshold=0.4,
            total_faqs=1,
            last_updated="2026-03-25",
        )

        assert metadata.version == "1.0"
        assert metadata.similarity_threshold == 0.4
        assert metadata.total_faqs == 1

    def test_rejects_invalid_similarity_threshold(self) -> None:
        with pytest.raises(ValidationError):
            FAQMetadata(
                version="1.0",
                description="Test FAQ database",
                nlp_processor="spacy",
                nlp_model="en_core_web_sm",
                preprocessing_steps=["tokenization"],
                matching_algorithm="tfidf_cosine_similarity",
                similarity_threshold=1.5,
                total_faqs=1,
                last_updated="2026-03-25",
            )


class TestFAQ:
    """Tests for individual FAQ validation."""

    def test_valid_faq(self) -> None:
        faq = FAQ.model_validate(make_faq())

        assert faq.id == 1
        assert faq.intent == "install_dependencies"
        assert faq.question.startswith("What are")

    def test_rejects_non_positive_id(self) -> None:
        data = make_faq(faq_id=0)

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)

    def test_rejects_empty_question(self) -> None:
        data = make_faq()
        data["question"] = ""

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)

    def test_rejects_empty_answer(self) -> None:
        data = make_faq()
        data["answer"] = ""

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)

    def test_rejects_empty_keyword(self) -> None:
        data = make_faq()
        data["keywords"] = ["requirements", ""]

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)

    def test_rejects_non_positive_related_id(self) -> None:
        data = make_faq(related_ids=[0])

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)

    def test_rejects_duplicate_related_ids(self) -> None:
        data = make_faq(related_ids=[2, 2])

        with pytest.raises(ValidationError):
            FAQ.model_validate(data)


class TestFAQDatabase:
    """Tests for complete FAQ database validation."""

    def test_valid_database(self) -> None:
        data = make_database(
            faqs=[
                make_faq(1, related_ids=[2]),
                make_faq(2, related_ids=[1]),
            ]
        )

        database = FAQDatabase.model_validate(data)

        assert len(database.faqs) == 2
        assert database.metadata.total_faqs == 2

    def test_rejects_duplicate_faq_ids(self) -> None:
        data = make_database(
            faqs=[
                make_faq(1),
                make_faq(1),
            ],
            total_faqs=2,
        )

        with pytest.raises(ValidationError, match="FAQ IDs must be unique"):
            FAQDatabase.model_validate(data)

    def test_rejects_missing_related_faq(self) -> None:
        data = make_database(
            faqs=[
                make_faq(1, related_ids=[999]),
            ]
        )

        with pytest.raises(
            ValidationError,
            match="references missing related FAQ IDs",
        ):
            FAQDatabase.model_validate(data)

    def test_rejects_incorrect_metadata_count(self) -> None:
        data = make_database(
            faqs=[make_faq()],
            total_faqs=38,
        )

        with pytest.raises(
            ValidationError,
            match="metadata.total_faqs does not match",
        ):
            FAQDatabase.model_validate(data)


class TestLoadFAQDatabase:
    """Tests for loading FAQ JSON files."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        write_json(path, make_database())

        database = load_faq_database(path)

        assert isinstance(database, FAQDatabase)
        assert len(database.faqs) == 1
        assert database.faqs[0].id == 1

    def test_load_faqs_returns_records(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database(
            faqs=[
                make_faq(1, related_ids=[2]),
                make_faq(2, related_ids=[1]),
            ]
        )
        write_json(path, data)

        faqs = load_faqs(path)

        assert len(faqs) == 2
        assert all(isinstance(faq, FAQ) for faq in faqs)
        assert [faq.id for faq in faqs] == [1, 2]

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "missing.json"

        with pytest.raises(FAQLoaderError, match="does not exist"):
            load_faq_database(path)

    def test_raises_for_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        path.write_text(
            '{"metadata": invalid}',
            encoding="utf-8",
        )

        with pytest.raises(FAQLoaderError, match="Invalid JSON"):
            load_faq_database(path)

    def test_raises_for_non_object_json(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        path.write_text(
            json.dumps(["faq", "faq"]),
            encoding="utf-8",
        )

        with pytest.raises(
            FAQLoaderError,
            match="must contain a JSON object",
        ):
            load_faq_database(path)

    def test_raises_for_missing_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database()
        del data["metadata"]

        write_json(path, data)

        with pytest.raises(FAQLoaderError, match="validation failed"):
            load_faq_database(path)

    def test_raises_for_missing_faqs(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database()
        del data["faqs"]

        write_json(path, data)

        with pytest.raises(FAQLoaderError, match="validation failed"):
            load_faq_database(path)

    def test_raises_for_empty_faq_list(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database(faqs=[], total_faqs=0)
        write_json(path, data)

        with pytest.raises(FAQLoaderError, match="validation failed"):
            load_faq_database(path)

    def test_preserves_unicode(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database()
        data["faqs"][0]["answer"] = "Astro-AI supports Unicode: বাংলা."

        write_json(path, data)

        database = load_faq_database(path)

        assert "বাংলা" in database.faqs[0].answer


class TestGetFAQByID:
    """Tests for FAQ lookup."""

    def test_returns_matching_faq(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"

        data = make_database(
            faqs=[
                make_faq(1, related_ids=[2]),
                make_faq(2, related_ids=[1]),
            ]
        )
        write_json(path, data)

        faq = get_faq_by_id(2, path)

        assert faq is not None
        assert faq.id == 2

    def test_returns_none_for_unknown_id(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        write_json(path, make_database())

        faq = get_faq_by_id(999, path)

        assert faq is None

    def test_rejects_invalid_id(self, tmp_path: Path) -> None:
        path = tmp_path / "faqs.json"
        write_json(path, make_database())

        with pytest.raises(ValueError, match="greater than zero"):
            get_faq_by_id(0, path)
          
