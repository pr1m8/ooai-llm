"""Optional live end-to-end tests against provider SDKs and APIs.

These tests are skipped by default unless the relevant provider credentials and
packages are installed in the local environment.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from ooai_llm import AppSettings, create_llm, list_available_models
from ooai_llm.catalog import ListModelsConfig
from ooai_llm.providers import PROVIDER_EXTRAS, PROVIDER_LANGCHAIN_PACKAGES, Provider, normalize_provider_name


LIVE_PROVIDERS: tuple[Provider, ...] = (
    Provider.OPENAI,
    Provider.ANTHROPIC,
    Provider.GOOGLE_GENAI,
    Provider.XAI,
    Provider.DEEPSEEK,
    Provider.MISTRAL,
)


def _dotenv_value(name: str) -> str | None:
    """Read a simple key from local .env without exposing secret values."""
    path = Path(".env")
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip().removeprefix("export ") != name:
            continue
        return value.strip().strip("\"'")
    return None


def _env_option(name: str) -> str | None:
    return os.environ.get(name) or _dotenv_value(name)


def _selected_live_providers() -> tuple[Provider, ...]:
    """Return live providers selected by ``OOAI_LIVE_PROVIDERS`` when set."""
    raw = _env_option("OOAI_LIVE_PROVIDERS")
    if not raw:
        return LIVE_PROVIDERS

    selected: list[Provider] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        provider = normalize_provider_name(name)
        if provider is None:
            raise ValueError(f"Unknown live provider: {name!r}.")
        selected.append(provider)
    return tuple(selected)


def _require_live() -> bool:
    """Return whether missing live prerequisites should fail instead of skip."""
    return (_env_option("OOAI_REQUIRE_LIVE") or "").strip().lower() in {"1", "true", "yes", "on"}


def _has_module(module_name: str | None) -> bool:
    if module_name is None:
        return False
    return importlib.util.find_spec(module_name.replace("-", "_")) is not None


def _skip_or_fail(message: str) -> None:
    if _require_live():
        pytest.fail(message)
    pytest.skip(message)


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("provider", _selected_live_providers())
def test_live_model_listing(provider: Provider) -> None:
    """It should list at least one model when the provider is configured.

    This test is intended to be run locally by package consumers who have the
    relevant provider credentials and SDKs installed.
    """
    settings = AppSettings()
    api_key = settings.credentials.get_api_key(provider)
    if api_key is None:
        _skip_or_fail(f"No API key configured for {provider.value}.")

    result = list_available_models(
        provider,
        settings=settings,
        config=ListModelsConfig(limit=3),
    )

    assert result.models


@pytest.mark.e2e
@pytest.mark.live
@pytest.mark.parametrize("provider", _selected_live_providers())
def test_live_create_llm_instantiation(provider: Provider) -> None:
    """It should instantiate a provider-backed LangChain chat model.

    The test only validates construction, not a network invocation, so it is a
    lightweight smoke test for package wiring.
    """
    settings = AppSettings()
    api_key = settings.credentials.get_api_key(provider)
    if api_key is None:
        _skip_or_fail(f"No API key configured for {provider.value}.")

    langchain_package = PROVIDER_LANGCHAIN_PACKAGES.get(provider)
    if not _has_module(langchain_package):
        extra_name = PROVIDER_EXTRAS[provider]
        _skip_or_fail(f"Install ooai-llm[{extra_name}] to run this smoke test.")

    llm = create_llm(provider=provider, preset="testing", settings=settings)
    assert llm is not None
