"""Unit tests for reasoning resolution."""

from __future__ import annotations

import pytest

from ooai_llm import ModelString
from ooai_llm.providers import Provider
from ooai_llm.reasoning import ReasoningConfig, build_reasoning_resolution


@pytest.mark.unit
def test_reasoning_config_from_preset_balanced() -> None:
    """It should normalize the balanced preset."""
    config = ReasoningConfig.from_preset("balanced")
    assert config.effort == "medium"
    assert config.summary == "auto"


@pytest.mark.unit
def test_openai_reasoning_maps_to_reasoning_dict() -> None:
    """It should emit OpenAI reasoning kwargs."""
    resolution = build_reasoning_resolution(
        model=ModelString.parse("openai:gpt-5.4-mini"),
        reasoning="deep",
    )
    assert resolution is not None
    assert resolution.provider is Provider.OPENAI
    assert resolution.constructor_kwargs["reasoning"] == {
        "effort": "high",
        "summary": "auto",
    }


@pytest.mark.unit
def test_anthropic_reasoning_maps_to_adaptive_thinking() -> None:
    """It should emit Anthropic adaptive thinking kwargs."""
    resolution = build_reasoning_resolution(
        model="anthropic:claude-sonnet-4-20250514",
        reasoning=ReasoningConfig(effort="xhigh", summary="auto"),
    )
    assert resolution is not None
    assert resolution.provider is Provider.ANTHROPIC
    assert resolution.constructor_kwargs["effort"] == "xhigh"
    assert resolution.constructor_kwargs["thinking"]["type"] == "adaptive"
    assert resolution.constructor_kwargs["thinking"]["display"] == "summarized"


@pytest.mark.unit
def test_gemini_25_budget_is_used_directly() -> None:
    """It should emit Gemini 2.5 thinking_budget when provided."""
    resolution = build_reasoning_resolution(
        model="google_genai:gemini-2.5-flash",
        reasoning=ReasoningConfig(budget_tokens=1024, include_thoughts=True),
    )
    assert resolution is not None
    assert resolution.provider is Provider.GOOGLE_GENAI
    assert resolution.constructor_kwargs["thinking_budget"] == 1024
    assert resolution.constructor_kwargs["include_thoughts"] is True


@pytest.mark.unit
def test_gemini_3_maps_to_thinking_level() -> None:
    """It should map Gemini 3 reasoning to thinking_level."""
    resolution = build_reasoning_resolution(
        model="google_genai:gemini-3.1-pro-preview",
        reasoning="fast",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["thinking_level"] == "low"


@pytest.mark.unit
def test_xai_multi_agent_maps_reasoning_to_agent_count_control() -> None:
    """It should emit xAI multi-agent reasoning kwargs and a note."""
    resolution = build_reasoning_resolution(
        model="xai:grok-4.20-multi-agent",
        reasoning="deep",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["reasoning"] == {"effort": "high"}
    assert resolution.notes


@pytest.mark.unit
def test_deepseek_reasoner_adds_note_but_no_constructor_kwargs() -> None:
    """It should record a note for DeepSeek reasoner summary preferences."""
    resolution = build_reasoning_resolution(
        model="deepseek:deepseek-reasoner",
        reasoning="balanced",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs == {}
    assert resolution.notes


@pytest.mark.unit
def test_mistral_small_latest_maps_to_adjustable_reasoning() -> None:
    """It should map adjustable reasoning for mistral-small-latest."""
    resolution = build_reasoning_resolution(
        model="mistral:mistral-small-latest",
        reasoning="deep",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["reasoning_effort"] == "high"


@pytest.mark.unit
def test_reasoning_normalize_accepts_effort_strings() -> None:
    """It should normalize raw effort strings."""
    config = ReasoningConfig.normalize("xhigh")
    assert config is not None
    assert config.effort == "xhigh"


@pytest.mark.unit
def test_reasoning_normalize_rejects_unknown_string() -> None:
    """It should reject unknown reasoning strings."""
    with pytest.raises(ValueError):
        ReasoningConfig.normalize("banana")


@pytest.mark.unit
def test_openai_summary_only_defaults_effort_to_medium() -> None:
    """It should default OpenAI effort when only a summary is requested."""
    resolution = build_reasoning_resolution(
        model="openai:gpt-5.4",
        reasoning=ReasoningConfig(effort="off", summary="auto"),
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert resolution.notes


@pytest.mark.unit
def test_openai_budget_note_and_xhigh_mapping() -> None:
    """It should note budget ignoring and xhigh mapping for OpenAI."""
    resolution = build_reasoning_resolution(
        model="openai:gpt-5.4",
        reasoning=ReasoningConfig(effort="xhigh", budget_tokens=100),
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["reasoning"]["effort"] == "high"
    assert len(resolution.notes) >= 2


@pytest.mark.unit
def test_anthropic_detailed_summary_adds_note() -> None:
    """It should note the lossy Anthropic summary mapping."""
    resolution = build_reasoning_resolution(
        model="anthropic:claude-opus-4-1-20250805",
        reasoning=ReasoningConfig(summary="detailed"),
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["thinking"]["display"] == "summarized"
    assert resolution.notes


@pytest.mark.unit
def test_gemini_25_dynamic_budget_is_supported() -> None:
    """It should map Gemini 2.5 dynamic budgets to -1."""
    resolution = build_reasoning_resolution(
        model="google_genai:gemini-2.5-flash",
        reasoning=ReasoningConfig(dynamic_budget=True),
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["thinking_budget"] == -1


@pytest.mark.unit
def test_gemini_3_ignores_budget_and_notes_mapping() -> None:
    """It should note ignored Gemini 3 budgets and map xhigh to high."""
    resolution = build_reasoning_resolution(
        model="google_genai:gemini-3.1-pro-preview",
        reasoning=ReasoningConfig(effort="xhigh", budget_tokens=50, summary="detailed"),
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["thinking_level"] == "high"
    assert resolution.constructor_kwargs["include_thoughts"] is True
    assert len(resolution.notes) >= 2


@pytest.mark.unit
def test_unknown_gemini_family_can_raise_in_strict_mode() -> None:
    """It should raise for unsupported Gemini families in strict mode."""
    with pytest.raises(ValueError):
        build_reasoning_resolution(
            model="google_genai:gemini-1.0-pro",
            reasoning=ReasoningConfig(strict=True),
        )


@pytest.mark.unit
def test_xai_non_multi_agent_adds_note() -> None:
    """It should note that normal xAI reasoning models auto-reason."""
    resolution = build_reasoning_resolution(
        model="xai:grok-4.20-reasoning",
        reasoning="balanced",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs == {}
    assert resolution.notes


@pytest.mark.unit
def test_deepseek_reasoner_off_and_budget_add_notes() -> None:
    """It should note DeepSeek native reasoning behavior."""
    resolution = build_reasoning_resolution(
        model="deepseek:deepseek-reasoner",
        reasoning=ReasoningConfig(effort="off", budget_tokens=100, summary="auto"),
    )
    assert resolution is not None
    assert len(resolution.notes) >= 2


@pytest.mark.unit
def test_mistral_small_off_maps_to_none() -> None:
    """It should disable adjustable Mistral reasoning with none."""
    resolution = build_reasoning_resolution(
        model="mistral:mistral-small-latest",
        reasoning="off",
    )
    assert resolution is not None
    assert resolution.constructor_kwargs["reasoning_effort"] == "none"


@pytest.mark.unit
def test_magistral_off_can_raise_in_strict_mode() -> None:
    """It should raise when strict mode tries to disable Magistral reasoning."""
    with pytest.raises(ValueError):
        build_reasoning_resolution(
            model="mistral:magistral-small-latest",
            reasoning=ReasoningConfig(effort="off", strict=True),
        )


@pytest.mark.unit
def test_unknown_mistral_model_notes_when_enabled() -> None:
    """It should note unsupported Mistral models when reasoning is requested."""
    resolution = build_reasoning_resolution(
        model="mistral:pixtral-large-latest",
        reasoning="balanced",
    )
    assert resolution is not None
    assert resolution.notes


@pytest.mark.unit
def test_reasoning_resolution_requires_resolvable_provider() -> None:
    """It should raise when a provider cannot be inferred or supplied."""
    with pytest.raises(ValueError):
        build_reasoning_resolution(
            model="mystery-model",
            reasoning="balanced",
        )
