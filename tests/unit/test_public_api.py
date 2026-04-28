"""Unit tests for the trimmed public API."""

from __future__ import annotations

import pytest

import ooai_llm
from ooai_llm import ModelString


@pytest.mark.unit
def test_public_api_is_curated() -> None:
    """The top-level package should expose the curated public API in ``__all__``."""
    exported = set(ooai_llm.__all__)
    assert {
        "ListModelsConfig",
        "Provider",
        "auto_refresh_model_defaults",
        "build_llm_cache",
        "build_reasoning_resolution",
        "configure_global_llm_cache",
        "create_llm",
        "get_model_info",
        "list_model_ids",
        "list_models",
        "make_litellm_cost_callback",
        "ModelInfo",
        "ModelDefaultsAutoRefreshSettings",
        "refresh_model_defaults",
        "resolve_factory_settings",
        "update_model_defaults",
    }.issubset(exported)
    assert "resolve_model_meta" not in exported


@pytest.mark.unit
def test_deprecated_top_level_alias_still_resolves() -> None:
    """Deprecated names should still resolve lazily for compatibility."""
    with pytest.deprecated_call():
        alias = ooai_llm.resolve_model_meta
    assert callable(alias)


@pytest.mark.unit
def test_model_string_style_properties() -> None:
    """ModelString should expose style and prefix helpers."""
    langchain_model = ModelString.parse("openai:gpt-5.4-mini")
    litellm_model = ModelString.parse("openai/gpt-5.4-mini")
    bare_model = ModelString.parse("gpt-5.4-mini")

    assert langchain_model.is_langchain_style is True
    assert langchain_model.is_litellm_style is False
    assert bare_model.is_bare is True
    assert litellm_model.is_litellm_style is True
