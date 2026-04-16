"""Optional live end-to-end tests against provider SDKs and APIs.

These tests are skipped by default unless the relevant provider credentials and
packages are installed in the local environment.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

from ooai_llm import AppSettings, create_llm, list_available_models
from ooai_llm.catalog import ListModelsConfig
from ooai_llm.providers import PROVIDER_EXTRAS, PROVIDER_LANGCHAIN_PACKAGES, Provider


LIVE_PROVIDERS: tuple[Provider, ...] = (
    Provider.OPENAI,
    Provider.ANTHROPIC,
    Provider.GOOGLE_GENAI,
    Provider.XAI,
    Provider.DEEPSEEK,
    Provider.MISTRAL,
)


def _has_module(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return importlib.util.find_spec(module_name.replace("-", "_")) is not None


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("provider", LIVE_PROVIDERS)
def test_live_model_listing(provider: Provider) -> None:
    """It should list at least one model when the provider is configured.

    This test is intended to be run locally by package consumers who have the
    relevant provider credentials and SDKs installed.
    """
    settings = AppSettings()
    api_key = settings.credentials.get_api_key(provider)
    if api_key is None:
        pytest.skip(f"No API key configured for {provider.value}.")

    result = list_available_models(
        provider,
        settings=settings,
        config=ListModelsConfig(limit=3),
    )

    assert result.models


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("provider", LIVE_PROVIDERS)
def test_live_create_llm_instantiation(provider: Provider) -> None:
    """It should instantiate a provider-backed LangChain chat model.

    The test only validates construction, not a network invocation, so it is a
    lightweight smoke test for package wiring.
    """
    settings = AppSettings()
    api_key = settings.credentials.get_api_key(provider)
    if api_key is None:
        pytest.skip(f"No API key configured for {provider.value}.")

    langchain_package = PROVIDER_LANGCHAIN_PACKAGES.get(provider)
    if not _has_module(langchain_package):
        extra_name = PROVIDER_EXTRAS[provider]
        pytest.skip(f"Install ooai-llm[{extra_name}] to run this smoke test.")

    llm = create_llm(provider=provider, preset="testing", settings=settings)
    assert llm is not None
