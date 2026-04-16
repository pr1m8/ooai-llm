"""Unit tests for provider normalization helpers."""

from __future__ import annotations

import pytest

from ooai_llm.providers import (
    Provider,
    infer_provider_from_model_name,
    normalize_provider_name,
    split_model_string,
)


@pytest.mark.unit
def test_normalize_provider_aliases() -> None:
    """It should normalize common provider aliases."""
    assert normalize_provider_name("gpt") == Provider.OPENAI
    assert normalize_provider_name("claude") == Provider.ANTHROPIC
    assert normalize_provider_name("gemini") == Provider.GOOGLE_GENAI


@pytest.mark.unit
def test_normalize_provider_none() -> None:
    """It should allow absent providers."""
    assert normalize_provider_name(None) is None


@pytest.mark.unit
def test_normalize_provider_rejects_unknown_values() -> None:
    """It should reject unknown providers."""
    with pytest.raises(ValueError):
        normalize_provider_name("not-a-provider")


@pytest.mark.unit
def test_infer_provider_from_prefixed_and_unprefixed_names() -> None:
    """It should infer providers from common model prefixes."""
    assert infer_provider_from_model_name("openai:gpt-5.4") == Provider.OPENAI
    assert infer_provider_from_model_name("claude-opus-4-1-20250805") == Provider.ANTHROPIC
    assert infer_provider_from_model_name("gemini-2.5-flash") == Provider.GOOGLE_GENAI
    assert infer_provider_from_model_name("deepseek-chat") == Provider.DEEPSEEK
    assert infer_provider_from_model_name("mistral-small-2603") == Provider.MISTRAL
    assert infer_provider_from_model_name("") is None


@pytest.mark.unit
def test_infer_provider_invalid_prefixed_value_returns_none() -> None:
    """It should return ``None`` for invalid explicit prefixes."""
    assert infer_provider_from_model_name("unknown:model") is None


@pytest.mark.unit
def test_split_model_string() -> None:
    """It should split prefixed and bare model strings."""
    provider, name = split_model_string("openai:gpt-5.4-mini")
    assert provider == Provider.OPENAI
    assert name == "gpt-5.4-mini"

    provider, name = split_model_string("gpt-5.4-mini")
    assert provider is None
    assert name == "gpt-5.4-mini"
