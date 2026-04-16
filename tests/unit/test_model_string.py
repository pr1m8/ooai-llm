"""Unit tests for typed model-string utilities."""

from __future__ import annotations

import pytest

from ooai_llm import ModelString
from ooai_llm.providers import Provider


@pytest.mark.unit
def test_model_string_infers_provider() -> None:
    """It should infer providers from bare model names."""
    model = ModelString.parse("gpt-5.4-mini")
    assert model.provider == Provider.OPENAI
    assert model.model_name == "gpt-5.4-mini"
    assert str(model.canonical()) == "openai:gpt-5.4-mini"


@pytest.mark.unit
def test_model_string_embedding_inference() -> None:
    """It should infer OpenAI for embedding model names."""
    model = ModelString.parse("text-embedding-3-small")
    assert model.provider == Provider.OPENAI
    assert str(model.canonical()) == "openai:text-embedding-3-small"


@pytest.mark.unit
def test_model_string_from_parts() -> None:
    """It should build provider-prefixed strings from parts."""
    model = ModelString.from_parts("claude-sonnet-4-20250514", provider="anthropic")
    assert str(model) == "anthropic:claude-sonnet-4-20250514"
    assert model.provider == Provider.ANTHROPIC


@pytest.mark.unit
def test_model_string_with_provider() -> None:
    """It should add a provider prefix to a bare model name."""
    model = ModelString.parse("gemini-2.5-pro").with_provider("google")
    assert str(model) == "google_genai:gemini-2.5-pro"


@pytest.mark.unit
def test_model_string_rejects_empty_values() -> None:
    """It should reject empty input strings."""
    with pytest.raises(ValueError):
        ModelString.parse("  ")
