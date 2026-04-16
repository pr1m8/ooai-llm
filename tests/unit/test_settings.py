"""Unit tests for ``ooai_llm`` settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from ooai_llm import AppSettings, ModelString


@pytest.mark.unit
def test_settings_resolve_alias() -> None:
    """It should resolve semantic aliases."""
    settings = AppSettings()
    assert settings.resolve_model(alias="cheap") == "openai:gpt-5.4-nano"


@pytest.mark.unit
def test_settings_resolve_provider_preset() -> None:
    """It should resolve provider-specific presets."""
    settings = AppSettings()
    assert settings.resolve_model(provider="google", preset="reasoning") == "google_genai:gemini-2.5-pro"


@pytest.mark.unit
def test_default_cache_path_is_under_hidden_app_dir(tmp_path: Path) -> None:
    """It should place the default cache under the hidden app directory."""
    settings = AppSettings(app_root=tmp_path)
    assert settings.default_llm_cache_path == (
        tmp_path / ".ooai" / "cache" / "llm" / "langchain_llm_cache.sqlite3"
    ).resolve()


@pytest.mark.unit
def test_resolve_model_string_returns_typed_value() -> None:
    """It should expose a typed model-string resolver."""
    settings = AppSettings()
    model = settings.resolve_model_string(alias="testing")
    assert isinstance(model, ModelString)
    assert model.model_name == "gpt-5.4-nano"


@pytest.mark.unit
def test_resolve_model_rejects_alias_and_provider_together() -> None:
    """It should reject ambiguous resolution inputs."""
    settings = AppSettings()
    with pytest.raises(ValueError):
        settings.resolve_model(alias="cheap", provider="openai")



@pytest.mark.unit
def test_catalog_settings_expose_provider_defaults() -> None:
    """It should expose provider-scoped catalog defaults separately from LLM settings."""
    settings = AppSettings()
    assert settings.catalog.xai.base_url == "https://api.x.ai"
    assert settings.catalog.build_list_models_options("openai")["prefer_sdk"] is True
