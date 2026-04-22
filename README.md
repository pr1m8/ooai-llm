# ooai-llm

[![CI](https://github.com/pr1m8/ooai-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/pr1m8/ooai-llm/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/ooai-llm/badge/?version=latest)](https://ooai-llm.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/ooai-llm.svg)](https://pypi.org/project/ooai-llm/)
[![Python](https://img.shields.io/pypi/pyversions/ooai-llm.svg)](https://pypi.org/project/ooai-llm/)
[![Coverage](https://img.shields.io/codecov/c/github/pr1m8/ooai-llm.svg)](https://codecov.io/gh/pr1m8/ooai-llm)

Typed LLM settings, provider-aware model-string parsing, LangChain-first chat
model creation, live provider model discovery, LiteLLM pricing enrichment, and
usage/cost callback helpers for Python applications.

## What This Is

`ooai-llm` is a small integration layer for application code that already wants
to use LangChain model classes directly, but does not want to repeat the same
provider configuration, model defaults, env-var handling, cache setup, metadata
lookup, and usage accounting in every project.

It is not a router, proxy, agent framework, or hosted model catalog.

## Features

- Typed `ModelString` parsing for bare, LangChain-style, and LiteLLM-style model names.
- Provider inference for OpenAI, Anthropic, Google GenAI, xAI, DeepSeek, and Mistral.
- `AppSettings` with provider credentials, default aliases, provider presets, cache settings, catalog settings, and LiteLLM settings.
- Native and app-prefixed credential env vars, such as `OPENAI_API_KEY` and `OOAI_OPENAI_API_KEY`.
- SQLite-backed LangChain global cache bootstrap.
- `create_llm(...)`, a thin wrapper around LangChain `init_chat_model(...)`.
- `create_llm_bundle(...)`, which returns the model, resolved metadata, and reasoning resolution together.
- Live model listing through provider SDKs or REST fallbacks.
- LangChain profile + LiteLLM pricing metadata in one `ModelInfo` object.
- Provider-aware reasoning kwargs for OpenAI, Anthropic, Gemini, xAI, DeepSeek, and Mistral.
- Usage and cost helpers for LangChain metadata and LiteLLM callbacks.
- Unit, integration, e2e, live-provider tests, coverage reports, docs builds, package builds, and PyPI release workflow.

## Installation

Base package:

```bash
pip install ooai-llm
```

With PDM:

```bash
pdm add ooai-llm
```

Install only the provider extras you use:

```bash
pdm add ooai-llm[openai]
pdm add ooai-llm[anthropic]
pdm add ooai-llm[deepseek]
pdm add ooai-llm[mistral]
pdm add ooai-llm[litellm]
```

Gemini and xAI are available as `ooai-llm[google]` and `ooai-llm[xai]`, but you
can skip those extras entirely if you do not have those keys.

## Factory Quick Start

```python
from ooai_llm import AppSettings, configure_global_llm_cache, create_llm

settings = AppSettings()
configure_global_llm_cache(settings)

llm = create_llm(
    alias="testing",
    settings=settings,
    temperature=0,
    reasoning="fast",
)

print(type(llm).__name__)
```

Most applications only need the factory plus settings:

```python
from ooai_llm import AppSettings, create_llm

settings = AppSettings()

testing_llm = create_llm(alias="testing", settings=settings, temperature=0)
reasoning_llm = create_llm(provider="anthropic", preset="reasoning", settings=settings)
explicit_llm = create_llm("openai:gpt-5.4-mini", settings=settings)
```

Use `create_llm_bundle(...)` when you want the created model and resolved
metadata together:

```python
from ooai_llm import create_llm_bundle

bundle = create_llm_bundle(
    alias="testing",
    reasoning="fast",
    temperature=0,
)

print(bundle.model.as_langchain())
print(bundle.metadata.identity.litellm_model)
print(bundle.reasoning.constructor_kwargs if bundle.reasoning else None)
```

## Environment

The package accepts both native provider variables and `OOAI_` aliases:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MISTRAL_API_KEY="..."

export OOAI_OPENAI_API_KEY="..."
export OOAI_ANTHROPIC_API_KEY="..."
export OOAI_DEEPSEEK_API_KEY="..."
export OOAI_MISTRAL_API_KEY="..."
```

Google/Gemini and xAI variables are supported too, but are optional:

```bash
export GOOGLE_API_KEY="..."
export GEMINI_API_KEY="..."
export XAI_API_KEY="..."
```

## Model Strings

```python
from ooai_llm import ModelString

model = ModelString.parse("gpt-5.4-mini")
print(model.provider)       # Provider.OPENAI
print(model.model_name)     # gpt-5.4-mini
print(model.canonical())    # openai:gpt-5.4-mini
print(model.as_litellm())   # openai/gpt-5.4-mini
```

## Settings And Defaults

```python
from ooai_llm import AppSettings

settings = AppSettings()

print(settings.resolve_model(alias="cheap"))
print(settings.resolve_model(provider="anthropic", preset="reasoning"))
print(settings.default_llm_cache_path)
```

Default aliases and provider presets include:

- `default`
- `cheap`
- `testing`
- `fast`
- `balanced`
- `reasoning`
- `coding`
- `vision`

## Live Model Discovery

```python
from ooai_llm import AppSettings, ListModelsConfig, list_available_models

settings = AppSettings()
result = list_available_models(
    "openai",
    settings=settings,
    config=ListModelsConfig(limit=5),
)

for model in result.models:
    print(model.model_string, model.display_name)
```

Provider SDKs are preferred when installed. Supported REST fallbacks are used
when SDK listing is unavailable or when you pass `ListModelsConfig(prefer_sdk=False)`.

## Reasoning

```python
from ooai_llm import ReasoningConfig, build_reasoning_resolution, create_llm

resolution = build_reasoning_resolution(
    model="openai:gpt-5.4-mini",
    reasoning="deep",
)
print(resolution.constructor_kwargs)

llm = create_llm(
    "anthropic:claude-sonnet-4-20250514",
    reasoning=ReasoningConfig(effort="medium", summary="auto"),
)
```

## Metadata And Usage

```python
from ooai_llm import BudgetPolicy, UsageRecorder, create_llm_bundle, make_litellm_cost_callback

bundle = create_llm_bundle(
    "openai:gpt-5.4-mini",
    reasoning="fast",
)

print(bundle.metadata.identity.litellm_model)
print(bundle.metadata.capabilities.raw_profile)
print(bundle.metadata.pricing.input_cost_per_token)

recorder = UsageRecorder()
callback = make_litellm_cost_callback(
    recorder,
    budget=BudgetPolicy(warn_total_tokens=5000),
)
```

## Cache Bootstrap

```python
from ooai_llm import AppSettings, configure_global_llm_cache

settings = AppSettings()
cache = configure_global_llm_cache(settings)
print(cache)
```

By default the SQLite cache is placed under:

```text
{app_root}/.ooai/cache/llm/langchain_llm_cache.sqlite3
```

Override it with `OOAI_LLM__CACHE__PATH` or `AppSettings(llm={"cache": {"path": ...}})`.

## Development

Install the development dependencies:

```bash
pdm install -G test -G docs -G dev
```

Run the full checked suite with coverage:

```bash
pdm run pytest
```

Live provider tests are skipped by the default suite so local keys in `.env`
do not trigger network calls accidentally.

Run tiers directly without the global coverage gate:

```bash
pdm run pytest -m unit --no-cov
pdm run pytest -m integration --no-cov
pdm run pytest -m "e2e and not live" --no-cov
```

Run live provider tests for your configured providers. This skips Gemini and xAI:

```bash
OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
```

To make live e2e fail instead of skip when a selected provider is missing a key
or SDK package:

```bash
OOAI_REQUIRE_LIVE=true OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
```

`AppSettings` loads a local `.env` file. Keep real keys in `.env`, not
`.env.example`.

Build docs and distributions:

```bash
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
pdm build
pdm run twine check dist/*
```

## Project Layout

```text
src/ooai_llm/
  cache.py       LangChain cache setup
  callbacks.py   usage and cost events
  catalog.py     live provider model listing
  factory.py     LangChain chat-model creation
  messages.py    message normalization
  metadata.py    LangChain + LiteLLM metadata
  providers.py   provider normalization
  reasoning.py   provider reasoning kwargs
  settings.py    Pydantic settings
  types.py       ModelString

docs/             Sphinx + MyST docs
examples/         runnable examples
tests/            unit, integration, e2e, and live tests
```

## Publishing

The repository includes:

- `.github/workflows/ci.yml` for tests, coverage, docs build, and package build
- `.github/workflows/docs.yml` for standalone docs validation
- `.github/workflows/release.yml` for tagged PyPI releases with trusted publishing
- `.readthedocs.yaml` for Read the Docs builds

Before publishing, configure the PyPI trusted publisher for `release.yml` and
environment `pypi`, import the repo in Read the Docs, and update
`docs/changelog.md`.

## License

MIT
