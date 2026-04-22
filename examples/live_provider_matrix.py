"""List models for configured providers, skipping missing credentials.

Set ``OOAI_EXAMPLE_PROVIDERS`` to a comma-separated provider list when you want
something other than the default OpenAI, Anthropic, DeepSeek, and Mistral pass.
"""

from __future__ import annotations

import os

from ooai_llm import AppSettings, ListModelsConfig, Provider, list_available_models, normalize_provider_name

DEFAULT_PROVIDERS = ("openai", "anthropic", "deepseek", "mistral")


def _selected_providers() -> tuple[Provider, ...]:
    raw = os.environ.get("OOAI_EXAMPLE_PROVIDERS")
    names = raw.split(",") if raw else DEFAULT_PROVIDERS
    providers: list[Provider] = []
    for name in names:
        provider = normalize_provider_name(name.strip())
        if provider is not None:
            providers.append(provider)
    return tuple(providers)


def main() -> None:
    """List a few model IDs for each configured provider."""
    settings = AppSettings()
    for provider in _selected_providers():
        if settings.credentials.get_api_key(provider) is None:
            print(f"{provider.value}: skipped, no API key configured")
            continue

        result = list_available_models(
            provider,
            settings=settings,
            config=ListModelsConfig(limit=5),
        )
        model_ids = ", ".join(model.model_id for model in result.models)
        print(f"{provider.value}: {model_ids}")


if __name__ == "__main__":
    main()
