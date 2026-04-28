"""Unit tests for refreshed provider model defaults."""

from __future__ import annotations

import sys
import types

import pytest

from ooai_llm import AppSettings, refresh_model_defaults, update_model_defaults
from ooai_llm.catalog import ModelListResult, ProviderModelInfo
from ooai_llm.model_defaults import ModelDefaultCandidate, recommend_provider_model_presets
from ooai_llm.providers import Provider, normalize_provider_name


def _candidate(model_id: str, *, provider: Provider = Provider.OPENAI, **kwargs: object) -> ModelDefaultCandidate:
    return ModelDefaultCandidate(
        provider=provider,
        model_id=model_id,
        source="test",
        mode="chat",
        **kwargs,
    )


@pytest.mark.unit
def test_recommend_provider_model_presets_uses_latest_non_special_default() -> None:
    """It should keep pro/opus-style models for reasoning instead of default."""
    recommendation = recommend_provider_model_presets(
        "openai",
        [
            _candidate("gpt-5.5", input_cost_per_token="0.000005", output_cost_per_token="0.000030"),
            _candidate("gpt-5.5-pro", input_cost_per_token="0.000030", output_cost_per_token="0.000180"),
            _candidate("gpt-5.4-mini", input_cost_per_token="0.00000075", output_cost_per_token="0.0000045"),
            _candidate("gpt-5.4-nano", input_cost_per_token="0.0000002", output_cost_per_token="0.0000010"),
            _candidate("text-embedding-3-large"),
        ],
    )

    assert recommendation.presets.default == "openai:gpt-5.5"
    assert recommendation.presets.latest == "openai:gpt-5.5"
    assert recommendation.presets.reasoning == "openai:gpt-5.5-pro"
    assert recommendation.presets.cheap == "openai:gpt-5.4-nano"
    assert all("embedding" not in candidate.model_id for candidate in recommendation.candidates)


@pytest.mark.unit
def test_refresh_model_defaults_uses_litellm_for_multiple_providers(monkeypatch) -> None:
    """It should refresh OpenAI, Anthropic, and Mistral from LiteLLM metadata."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.5": {
            "mode": "chat",
            "input_cost_per_token": "0.000005",
            "output_cost_per_token": "0.000030",
        },
        "openai/gpt-5.5-pro": {
            "mode": "chat",
            "input_cost_per_token": "0.000030",
            "output_cost_per_token": "0.000180",
        },
        "openai/gpt-5.4-nano": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000010",
        },
        "anthropic/claude-sonnet-4-20250514": {
            "mode": "chat",
            "input_cost_per_token": "0.000003",
            "output_cost_per_token": "0.000015",
        },
        "anthropic/claude-opus-4-1-20250805": {
            "mode": "chat",
            "input_cost_per_token": "0.000015",
            "output_cost_per_token": "0.000075",
        },
        "anthropic/claude-3-5-haiku-20241022": {
            "mode": "chat",
            "input_cost_per_token": "0.0000008",
            "output_cost_per_token": "0.000004",
        },
        "gemini/gemini-3-flash": {
            "mode": "chat",
            "input_cost_per_token": "0.0000003",
            "output_cost_per_token": "0.0000025",
        },
        "gemini/gemini-3-pro": {
            "mode": "chat",
            "input_cost_per_token": "0.000002",
            "output_cost_per_token": "0.000010",
        },
        "gemini/gemini-2.5-flash-lite": {
            "mode": "chat",
            "input_cost_per_token": "0.0000001",
            "output_cost_per_token": "0.0000004",
        },
        "xai/grok-4-2-fast-non-reasoning": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000008",
        },
        "xai/grok-4-2-fast-reasoning": {
            "mode": "chat",
            "input_cost_per_token": "0.000001",
            "output_cost_per_token": "0.000004",
        },
        "deepseek/deepseek-chat-202702": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000008",
        },
        "deepseek/deepseek-reasoner-202701": {
            "mode": "chat",
            "input_cost_per_token": "0.0000005",
            "output_cost_per_token": "0.0000020",
        },
        "deepseek/deepseek-coder-202701": {
            "mode": "chat",
            "input_cost_per_token": "0.0000003",
            "output_cost_per_token": "0.0000012",
        },
        "mistral/mistral-small-latest": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000006",
        },
        "mistral/magistral-small-latest": {
            "mode": "chat",
            "input_cost_per_token": "0.0000005",
            "output_cost_per_token": "0.0000015",
        },
        "mistral/devstral-2512": {
            "mode": "chat",
            "input_cost_per_token": "0.0000004",
            "output_cost_per_token": "0.0000012",
        },
        "mistral/ministral-3b-2512": {
            "mode": "chat",
            "input_cost_per_token": "0.00000004",
            "output_cost_per_token": "0.00000012",
        },
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    result = refresh_model_defaults(
        AppSettings(),
        providers=["openai", "anthropic", "google", "xai", "deepseek", "mistral"],
        source="litellm",
    )

    assert result.notes == []
    assert result.settings.resolve_model(alias="latest") == "openai:gpt-5.5"
    assert result.settings.resolve_model() == "openai:gpt-5.5"
    assert result.settings.resolve_model(provider="anthropic", preset="default") == (
        "anthropic:claude-sonnet-4-20250514"
    )
    assert result.settings.resolve_model(provider="anthropic", preset="reasoning") == (
        "anthropic:claude-opus-4-1-20250805"
    )
    assert result.settings.resolve_model(provider="google", preset="latest") == "google_genai:gemini-3-flash"
    assert result.settings.resolve_model(provider="google", preset="reasoning") == "google_genai:gemini-3-pro"
    assert result.settings.resolve_model(provider="xai", preset="latest") == "xai:grok-4-2-fast-non-reasoning"
    assert result.settings.resolve_model(provider="xai", preset="reasoning") == "xai:grok-4-2-fast-reasoning"
    assert result.settings.resolve_model(provider="deepseek", preset="latest") == "deepseek:deepseek-chat-202702"
    assert result.settings.resolve_model(provider="deepseek", preset="reasoning") == (
        "deepseek:deepseek-reasoner-202701"
    )
    assert result.settings.resolve_model(provider="deepseek", preset="coding") == "deepseek:deepseek-coder-202701"
    assert result.settings.resolve_model(provider="mistral", preset="coding") == "mistral:devstral-2512"
    assert result.settings.resolve_model(provider="mistral", preset="cheap") == "mistral:ministral-3b-2512"


@pytest.mark.unit
def test_update_model_defaults_exports_json_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """It should update settings and render reusable non-secret overrides."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.5": {
            "mode": "chat",
            "input_cost_per_token": "0.000005",
            "output_cost_per_token": "0.000030",
        },
        "openai/gpt-5.5-pro": {
            "mode": "chat",
            "input_cost_per_token": "0.000030",
            "output_cost_per_token": "0.000180",
        },
        "openai/gpt-5.4-nano": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000010",
        },
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    json_result = update_model_defaults(
        AppSettings(),
        providers=["openai"],
        source="litellm",
        output_format="json",
    )

    assert json_result.settings.resolve_model() == "openai:gpt-5.5"
    assert json_result.output_text is not None
    assert '"default_model": "openai:gpt-5.5"' in json_result.output_text
    assert json_result.overrides["llm"]["aliases"]["latest"] == "openai:gpt-5.5"

    env_path = tmp_path / "models.env"
    env_result = update_model_defaults(
        AppSettings(),
        providers=["openai"],
        source="litellm",
        output_path=env_path,
        output_format="env",
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert env_result.output_text is None
    assert env_result.output_path == env_path.resolve()
    assert "OOAI_LLM__DEFAULT_MODEL=openai:gpt-5.5" in env_text
    assert "OOAI_LLM__DEFAULTS_BY_PROVIDER__OPENAI__LATEST=openai:gpt-5.5" in env_text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "expected_latest", "expected_reasoning", "expected_coding"),
    [
        ("openai", "openai:gpt-5.5", "openai:gpt-5.5-pro", "openai:gpt-5.5-pro"),
        (
            "anthropic",
            "anthropic:claude-sonnet-4-20250514",
            "anthropic:claude-opus-4-1-20250805",
            "anthropic:claude-opus-4-1-20250805",
        ),
        (
            "google",
            "google_genai:gemini-3-flash",
            "google_genai:gemini-3-pro",
            "google_genai:gemini-3-pro",
        ),
        (
            "xai",
            "xai:grok-4-2-fast-non-reasoning",
            "xai:grok-4-2-fast-reasoning",
            "xai:grok-4-2-fast-reasoning",
        ),
        (
            "deepseek",
            "deepseek:deepseek-chat-202702",
            "deepseek:deepseek-reasoner-202701",
            "deepseek:deepseek-coder-202701",
        ),
        ("mistral", "mistral:mistral-small-latest", "mistral:magistral-small-latest", "mistral:devstral-2512"),
    ],
)
def test_refresh_model_defaults_uses_provider_catalog_for_every_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected_latest: str,
    expected_reasoning: str,
    expected_coding: str,
) -> None:
    """It should refresh latest/reasoning/coding presets from every provider catalog."""
    import ooai_llm.model_defaults as model_defaults

    catalog_models = {
        Provider.OPENAI: ["gpt-5.5", "gpt-5.5-pro", "gpt-5.4-nano", "text-embedding-3-large"],
        Provider.ANTHROPIC: [
            "claude-sonnet-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-3-5-haiku-20241022",
        ],
        Provider.GOOGLE_GENAI: [
            "models/gemini-3-flash",
            "models/gemini-3-pro",
            "models/gemini-2.5-flash-lite",
            "models/text-embedding-004",
        ],
        Provider.XAI: ["grok-4-2-fast-non-reasoning", "grok-4-2-fast-reasoning", "grok-4-1-mini"],
        Provider.DEEPSEEK: ["deepseek-chat-202702", "deepseek-reasoner-202701", "deepseek-coder-202701"],
        Provider.MISTRAL: ["mistral-small-latest", "magistral-small-latest", "devstral-2512", "ministral-3b-2512"],
    }

    def fake_list_available_models(provider_arg, *, settings, config=None, **kwargs):
        resolved_provider = normalize_provider_name(provider_arg)
        assert resolved_provider is not None
        models = []
        for model_id in catalog_models[resolved_provider]:
            supported_actions = []
            if resolved_provider is Provider.GOOGLE_GENAI:
                supported_actions = ["generateContent"] if "embedding" not in model_id else ["embedContent"]
            models.append(
                ProviderModelInfo(
                    provider=resolved_provider,
                    model_id=model_id,
                    supported_actions=supported_actions,
                )
            )
        return ModelListResult(provider=resolved_provider, models=models, used_sdk=False)

    monkeypatch.setattr(model_defaults, "list_available_models", fake_list_available_models)

    result = refresh_model_defaults(AppSettings(), providers=[provider], source="provider")

    assert result.notes == []
    resolved_provider = normalize_provider_name(provider)
    assert resolved_provider is not None
    assert result.recommendations[resolved_provider.value].source == "provider"
    assert result.settings.resolve_model(provider=provider, preset="latest") == expected_latest
    assert result.settings.resolve_model(provider=provider, preset="reasoning") == expected_reasoning
    assert result.settings.resolve_model(provider=provider, preset="coding") == expected_coding


@pytest.mark.unit
def test_refresh_model_defaults_preserves_provider_when_refresh_fails(monkeypatch) -> None:
    """It should leave existing provider defaults intact on non-strict failures."""
    import ooai_llm.model_defaults as model_defaults

    def raise_import_error(_: str):
        raise ImportError("missing litellm")

    monkeypatch.setattr(model_defaults.importlib, "import_module", raise_import_error)

    result = refresh_model_defaults(
        AppSettings(),
        providers=["deepseek"],
        source="litellm",
    )

    assert result.recommendations == {}
    assert result.notes
    assert result.settings.resolve_model(provider="deepseek", preset="reasoning") == "deepseek:deepseek-reasoner"
