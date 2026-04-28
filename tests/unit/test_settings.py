"""Unit tests for ``ooai_llm`` settings."""

from __future__ import annotations

from pathlib import Path

import pytest

from ooai_llm.providers import PROVIDER_API_KEY_ENV_VARS
from ooai_llm import AppSettings, ModelString


@pytest.mark.unit
def test_settings_resolve_alias() -> None:
    """It should resolve semantic aliases."""
    settings = AppSettings()
    assert settings.resolve_model(alias="cheap") == "openai:gpt-5.4-nano"
    assert settings.resolve_model(alias="latest") == "openai:gpt-5.5"


@pytest.mark.unit
def test_settings_resolve_provider_preset() -> None:
    """It should resolve provider-specific presets."""
    settings = AppSettings()
    assert settings.resolve_model(provider="google", preset="reasoning") == "google_genai:gemini-2.5-pro"
    assert settings.resolve_model(provider="google", preset="latest") == "google_genai:gemini-2.5-flash"


@pytest.mark.unit
def test_model_auto_refresh_settings_are_opt_in() -> None:
    """It should keep factory-time model refresh disabled by default."""
    settings = AppSettings()
    assert settings.llm.auto_refresh_models.enabled is False
    assert settings.llm.auto_refresh_models.source == "auto"
    assert settings.llm.auto_refresh_models.cache_seconds == 3600


@pytest.mark.unit
def test_model_auto_refresh_settings_load_from_env(monkeypatch) -> None:
    """It should load factory-time model refresh settings from nested env vars."""
    monkeypatch.setenv("OOAI_LLM__AUTO_REFRESH_MODELS__ENABLED", "true")
    monkeypatch.setenv("OOAI_LLM__AUTO_REFRESH_MODELS__SOURCE", "litellm")
    monkeypatch.setenv("OOAI_LLM__AUTO_REFRESH_MODELS__PROVIDERS", '["openai","mistral"]')
    monkeypatch.setenv("OOAI_LLM__AUTO_REFRESH_MODELS__CACHE_SECONDS", "0")

    settings = AppSettings()

    assert settings.llm.auto_refresh_models.enabled is True
    assert settings.llm.auto_refresh_models.source == "litellm"
    assert settings.llm.auto_refresh_models.providers == ["openai", "mistral"]
    assert settings.llm.auto_refresh_models.cache_seconds == 0


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


@pytest.mark.unit
def test_app_settings_loads_dotenv_file(tmp_path: Path, monkeypatch) -> None:
    """It should load local .env files for developer and live-test workflows."""
    for env_var in PROVIDER_API_KEY_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("OOAI_OPENAI_API_KEY", raising=False)

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-dotenv\n", encoding="utf-8")

    settings = AppSettings()

    if settings.credentials.get_api_key("openai") != "sk-dotenv":
        pytest.fail("Expected OPENAI_API_KEY to be loaded from the local .env file.")
