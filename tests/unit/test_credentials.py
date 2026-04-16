"""Unit tests for provider credential handling."""

from __future__ import annotations

import pytest

from ooai_llm.settings import AppSettings, ProviderCredentials


@pytest.mark.unit
def test_provider_credentials_to_native_environment() -> None:
    """It should map configured secrets to provider-native env vars."""
    creds = ProviderCredentials(openai_api_key="sk-openai", google_api_key="gcp-key")
    env = creds.to_native_environment()
    assert env["OPENAI_API_KEY"] == "sk-openai"
    assert env["GOOGLE_API_KEY"] == "gcp-key"


@pytest.mark.unit
def test_app_settings_coerces_nested_credentials_dict() -> None:
    """It should coerce nested credential dictionaries into typed credentials."""
    settings = AppSettings(credentials={"openai_api_key": "sk-test-openai"})
    assert settings.credentials.openai_api_key is not None
    assert settings.credentials.openai_api_key.get_secret_value() == "sk-test-openai"
