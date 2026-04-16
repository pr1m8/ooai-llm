"""Integration tests for factory helpers."""

from __future__ import annotations

import os
import sys
import types

import pytest

from ooai_llm import AppSettings, create_llm
from ooai_llm.factory import native_environment_overrides, resolve_model_string


@pytest.mark.integration
def test_resolve_model_string_returns_typed_model() -> None:
    """It should resolve aliases into typed model strings."""
    settings = AppSettings()
    model = resolve_model_string(settings=settings, alias="testing")
    assert model.model_name == "gpt-5.4-nano"


@pytest.mark.integration
def test_native_environment_overrides_mirrors_credentials(monkeypatch) -> None:
    """It should temporarily expose native provider env vars from settings."""
    settings = AppSettings(
        credentials={"openai_api_key": "sk-test-openai"},
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with native_environment_overrides(settings):
        assert os.environ["OPENAI_API_KEY"] == "sk-test-openai"

    assert "OPENAI_API_KEY" not in os.environ


@pytest.mark.integration
def test_create_llm_passes_model_and_provider(monkeypatch) -> None:
    """It should call LangChain's init_chat_model with the resolved values."""
    calls: list[dict[str, object]] = []

    def fake_init_chat_model(model: str | None = None, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})
        return {"model": model, "kwargs": kwargs}

    fake_chat_models = types.ModuleType("langchain.chat_models")
    fake_chat_models.init_chat_model = fake_init_chat_model

    fake_langchain = types.ModuleType("langchain")
    fake_langchain.chat_models = fake_chat_models

    monkeypatch.setitem(sys.modules, "langchain", fake_langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake_chat_models)

    settings = AppSettings()
    result = create_llm(
        model="gpt-5.4-mini",
        provider="openai",
        settings=settings,
        temperature=0,
        cache=False,
    )

    assert result["model"] == "gpt-5.4-mini"
    assert result["kwargs"]["model_provider"] == "openai"
    assert result["kwargs"]["temperature"] == 0
    assert result["kwargs"]["cache"] is False


@pytest.mark.integration
def test_create_llm_applies_reasoning_kwargs(monkeypatch) -> None:
    """It should merge provider-specific reasoning kwargs into model creation."""
    calls: list[dict[str, object]] = []

    def fake_init_chat_model(model: str | None = None, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})
        return {"model": model, "kwargs": kwargs}

    fake_chat_models = types.ModuleType("langchain.chat_models")
    fake_chat_models.init_chat_model = fake_init_chat_model

    fake_langchain = types.ModuleType("langchain")
    fake_langchain.chat_models = fake_chat_models

    monkeypatch.setitem(sys.modules, "langchain", fake_langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake_chat_models)

    result = create_llm(
        model="gpt-5.4-mini",
        provider="openai",
        reasoning="deep",
    )

    assert result["kwargs"]["reasoning"] == {"effort": "high", "summary": "auto"}
