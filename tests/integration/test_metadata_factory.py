"""Integration tests for metadata-aware factory helpers."""

from __future__ import annotations

import sys
import types

import pytest

from ooai_llm import create_llm_bundle


@pytest.mark.integration
def test_create_llm_bundle_returns_metadata(monkeypatch) -> None:
    """It should create an LLM and resolve merged metadata in one step."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.4-mini": {
            "input_cost_per_token": "0.00000025",
            "output_cost_per_token": "0.00000200",
            "max_input_tokens": 200000,
        }
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    class FakeModel(dict):
        model = "gpt-5.4-mini"
        profile = {"tool_calling": True, "reasoning_output": True, "max_input_tokens": 128000}

    def fake_init_chat_model(model: str | None = None, **kwargs):
        result = FakeModel(model=model, kwargs=kwargs)
        return result

    fake_chat_models = types.ModuleType("langchain.chat_models")
    fake_chat_models.init_chat_model = fake_init_chat_model

    fake_langchain = types.ModuleType("langchain")
    fake_langchain.chat_models = fake_chat_models

    monkeypatch.setitem(sys.modules, "langchain", fake_langchain)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake_chat_models)

    bundle = create_llm_bundle(model="gpt-5.4-mini", provider="openai", reasoning="deep")

    assert bundle.model.as_langchain() == "openai:gpt-5.4-mini"
    assert bundle.metadata.identity.litellm_model == "openai/gpt-5.4-mini"
    assert bundle.metadata.capabilities.reasoning_output is True
    assert bundle.metadata.pricing.source == "litellm.model_cost"
    assert bundle.reasoning is not None
