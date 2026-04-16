"""End-to-end tests of public package flows."""

from __future__ import annotations

import sys
import types

import pytest

from ooai_llm import AppSettings, ModelString, create_llm


@pytest.mark.e2e
def test_public_settings_and_factory_flow(monkeypatch) -> None:
    """It should support the main public flow from settings to model creation."""
    captured: dict[str, object] = {}

    def fake_init_chat_model(model: str | None = None, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return {"model": model, "kwargs": kwargs}

    fake_chat_models = types.ModuleType("langchain.chat_models")
    fake_chat_models.init_chat_model = fake_init_chat_model

    fake_langchain = types.ModuleType("langchain")
    fake_langchain.chat_models = fake_chat_models

    monkeypatch.setitem(sys.modules, "langchain", fake_langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake_chat_models)

    settings = AppSettings()
    model = settings.resolve_model_string(alias="testing")

    result = create_llm(model=model, settings=settings, temperature=0)

    assert isinstance(model, ModelString)
    assert result["model"] == "openai:gpt-5.4-nano"
    assert result["kwargs"]["temperature"] == 0
