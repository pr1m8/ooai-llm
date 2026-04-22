"""Unit tests for LangChain + LiteLLM metadata helpers."""

from __future__ import annotations

import sys
import types
from decimal import Decimal

import pytest

from ooai_llm import AppSettings, ModelString
from ooai_llm.metadata import (
    build_capability_profile,
    build_usage_snapshot,
    calculate_cost,
    get_model_info,
    resolve_litellm_model_name,
    resolve_model_meta_from_langchain_model,
)


@pytest.mark.unit
def test_model_string_converts_to_litellm_style() -> None:
    """It should convert LangChain-style model strings to LiteLLM style."""
    model = ModelString.parse("openai:gpt-5.4-mini")
    assert model.as_litellm() == "openai/gpt-5.4-mini"
    assert ModelString.from_litellm("anthropic/claude-sonnet-4").as_langchain() == "anthropic:claude-sonnet-4"


@pytest.mark.unit
def test_resolve_litellm_model_name_uses_settings_override() -> None:
    """It should honor LiteLLM provider-prefix overrides from settings."""
    settings = AppSettings(litellm={"provider_prefixes": {"google_genai": "vertex_ai"}})
    resolved = resolve_litellm_model_name("google_genai:gemini-2.5-pro", settings=settings)
    assert resolved == "vertex_ai/gemini-2.5-pro"


@pytest.mark.unit
def test_resolve_litellm_model_name_keeps_unknown_providerless_model() -> None:
    """It should keep a bare model name when no provider can be inferred."""
    assert resolve_litellm_model_name("custom-local-model") == "custom-local-model"


@pytest.mark.unit
def test_resolve_model_meta_uses_litellm_get_model_info(monkeypatch) -> None:
    """It should enrich pricing metadata from LiteLLM's get_model_info."""

    def fake_get_model_info(model_name: str):
        assert model_name in {"openai/gpt-5.4-mini", "gpt-5.4-mini"}
        return {
            "input_cost_per_token": "0.00000025",
            "output_cost_per_token": "0.00000200",
            "max_input_tokens": 200000,
        }

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.get_model_info = fake_get_model_info
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    meta = get_model_info(
        "openai:gpt-5.4-mini",
        profile={"tool_calling": True, "max_input_tokens": 128000},
    )

    assert meta.identity.litellm_model == "openai/gpt-5.4-mini"
    assert meta.pricing.input_cost_per_token == Decimal("0.00000025")
    assert meta.capabilities.tool_calling is True
    assert meta.capabilities.field_sources["tool_calling"] == "profile"
    assert meta.max_input_tokens == 128000


@pytest.mark.unit
def test_resolve_model_meta_falls_back_to_litellm_model_cost_map(monkeypatch) -> None:
    """It should fall back to LiteLLM's model-cost map when get_model_info is absent."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "deepseek/deepseek-chat": {
            "input_cost_per_token": "0.00000010",
            "output_cost_per_token": "0.00000020",
            "max_tokens": 64000,
        }
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    meta = get_model_info("deepseek:deepseek-chat")

    assert meta.pricing.source == "litellm.model_cost"
    assert meta.pricing.max_tokens == 64000
    assert meta.pricing.output_cost_per_token == Decimal("0.00000020")


@pytest.mark.unit
def test_build_usage_snapshot_and_calculate_cost(monkeypatch) -> None:
    """It should normalize usage and calculate cost from pricing metadata."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.4-nano": {
            "input_cost_per_token": "0.00000005",
            "output_cost_per_token": "0.00000010",
        }
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    meta = get_model_info("openai:gpt-5.4-nano")
    usage = build_usage_snapshot({"input_tokens": 1000, "output_tokens": 500})
    assert usage.resolved_total_tokens == 1500
    assert calculate_cost(meta, usage) == Decimal("0.00010")


@pytest.mark.unit
def test_build_capability_profile_handles_explicit_parallel_values() -> None:
    """It should trust explicit profile values before provider heuristics."""
    profile = build_capability_profile(
        {
            "tool_calling": False,
            "parallel_tool_calls": True,
            "max_output_tokens": 4096,
        },
        provider="openai",
    )

    assert profile.tool_calling is False
    assert profile.parallel_tool_calls is False
    assert profile.field_sources["parallel_tool_calls"] == "profile"
    assert profile.max_output_tokens == 4096


@pytest.mark.unit
def test_get_model_info_with_litellm_disabled() -> None:
    """It should return disabled pricing metadata when configured off."""
    info = get_model_info(
        "openai:gpt-5.4-mini",
        settings=AppSettings(litellm={"enabled": False}),
    )

    assert info.pricing.source == "disabled"
    assert info.pricing.billing_model_name == "openai/gpt-5.4-mini"


@pytest.mark.unit
def test_get_model_info_when_litellm_unavailable(monkeypatch) -> None:
    """It should degrade gracefully when LiteLLM is not installed."""
    import ooai_llm.metadata as metadata

    def raise_import_error():
        raise ImportError("litellm missing")

    monkeypatch.setattr(metadata, "_import_litellm", raise_import_error)

    info = get_model_info("openai:gpt-5.4-mini")

    assert info.pricing.source == "unavailable"


@pytest.mark.unit
def test_get_model_info_requires_a_model_name() -> None:
    """It should reject calls where no model can be inferred."""
    with pytest.raises(ValueError, match="Pass either a model string"):
        get_model_info()

    with pytest.raises(ValueError, match="Could not infer"):
        get_model_info(llm=object())


@pytest.mark.unit
def test_deprecated_resolve_model_meta_from_langchain_model() -> None:
    """It should keep the deprecated compatibility helper working."""

    class FakeLLM:
        model = "openai:gpt-5.4-mini"
        profile = {"parallel_tool_calls": False}

    with pytest.deprecated_call():
        info = resolve_model_meta_from_langchain_model(FakeLLM())

    assert info.identity.langchain_model == "openai:gpt-5.4-mini"
    assert info.capabilities.parallel_tool_calls is False
