# ooai-llm

[![CI](https://github.com/OWNER/ooai-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/ooai-llm/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/ooai-llm/badge/?version=latest)](https://ooai-llm.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/ooai-llm.svg)](https://pypi.org/project/ooai-llm/)
[![Python](https://img.shields.io/pypi/pyversions/ooai-llm.svg)](https://pypi.org/project/ooai-llm/)
[![Coverage](https://img.shields.io/codecov/c/github/OWNER/ooai-llm.svg)](https://codecov.io/gh/OWNER/ooai-llm)

Typed LLM settings, provider-aware model-string parsing, LangChain-first model creation, LiteLLM pricing enrichment, and ergonomic callback helpers for Python applications.

> Update `OWNER` and the Read the Docs project slug in the badge links after creating your repository.

## Why this package exists

`ooai-llm` gives you one small, typed layer for the parts of LLM applications that tend to repeat across projects:

- parsing and normalizing model strings like `openai:gpt-5.4-mini`
- inferring providers from bare model names such as `gpt-5.4-nano`
- resolving provider defaults like `cheap`, `testing`, `reasoning`, or `vision`
- loading API keys from both app-prefixed and native provider env vars
- wiring a global SQLite-backed LangChain cache with sensible local paths
- creating LangChain chat models with a thin wrapper around `init_chat_model(...)`
- listing available models from native provider APIs and SDKs with one typed return shape

The package is intentionally focused. It does not try to be a router, proxy, or model catalog service.

## Features

- Fully typed `ModelString` root model that is reusable for chat, embeddings, rerankers, and other model families.
- Canonical provider normalization and inference helpers.
- `AppSettings` with nested `LLMSettings`, `ProviderCredentials`, and `LLMCacheSettings`.
- Default models by provider and by semantic alias.
- Optional global LangChain cache bootstrap using SQLite.
- Thin `create_llm(...)` wrapper for LangChain chat models.
- `create_llm_bundle(...)` helper that resolves LangChain profiles together with native LiteLLM pricing metadata.
- LiteLLM-style model-string conversion for pricing, callbacks, and future routing.
- `list_models(...)` and `list_model_ids(...)` for live provider model discovery.
- `get_model_info(...)` for LangChain-profile capabilities plus LiteLLM pricing enrichment.
- `normalize_messages(...)` for ergonomic string / dict / LangChain-message handling.
- Provider-aware reasoning adapters for OpenAI, Anthropic, Gemini, xAI, DeepSeek, and Mistral.
- Unit, integration, and end-to-end tests with coverage reporting.
- Sphinx + AutoAPI docs scaffolding with Read the Docs configuration.

## Installation

### Base package

```bash
pdm add ooai-llm
```

### Provider integrations

Install the providers you actually use.

```bash
pdm add ooai-llm[openai]
pdm add ooai-llm[anthropic]
pdm add ooai-llm[google]
pdm add ooai-llm[xai]
pdm add ooai-llm[deepseek]
pdm add ooai-llm[mistral]
pdm add ooai-llm[litellm]
pdm add ooai-llm[providers]
pdm add ooai-llm[all]
```

### Developer setup

```bash
pdm install -G test -G docs -G dev
```

## Quick start

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

## Model-string parsing

```python
from ooai_llm import ModelString

model = ModelString.parse("gpt-5.4-mini")
print(model.provider)       # Provider.OPENAI
print(model.model_name)     # gpt-5.4-mini
print(model.canonical())    # openai:gpt-5.4-mini
```

`ModelString` is a `RootModel[str]`, so it is easy to embed in other typed settings or request models.

## Settings

```python
from ooai_llm import AppSettings

settings = AppSettings()

print(settings.resolve_model(alias="cheap"))
print(settings.resolve_model(provider="anthropic", preset="reasoning"))
print(settings.default_llm_cache_path)
```

### Supported environment variables

The package accepts both app-prefixed and provider-native env vars.

Examples:

```bash
export OOAI_OPENAI_API_KEY="..."
export OPENAI_API_KEY="..."
export OOAI_ANTHROPIC_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OOAI_GOOGLE_API_KEY="..."
export GOOGLE_API_KEY="..."
export GEMINI_API_KEY="..."
```

The factory temporarily mirrors app-prefixed values into the native provider variables expected by LangChain integration packages.

## Default model presets

Each provider ships with the same preset names:

- `default`
- `cheap`
- `testing`
- `fast`
- `balanced`
- `reasoning`
- `coding`
- `vision`

Global aliases of the same names are also available.

## Live model discovery

```python
from ooai_llm import AppSettings, ListModelsConfig, get_model_info, list_model_ids, list_models

settings = AppSettings()

result = list_models(
    "openai",
    settings=settings,
    config=ListModelsConfig(limit=5),
)

for model in result.models:
    print(model.model_id, model.display_name)

print(list_model_ids("deepseek", settings=settings))
```

The discovery layer prefers official provider SDKs when they are available and
falls back to documented REST endpoints when needed.


## Reasoning

```python
from ooai_llm import ReasoningConfig, build_reasoning_resolution, create_llm

resolution = build_reasoning_resolution(
    model="google_genai:gemini-2.5-pro",
    reasoning=ReasoningConfig(budget_tokens=2048, include_thoughts=True),
)

print(resolution.constructor_kwargs)
# {'thinking_budget': 2048, 'include_thoughts': True}

llm = create_llm(
    "openai:gpt-5.4-mini",
    reasoning="deep",
    temperature=0,
)
```

String values such as `"fast"`, `"balanced"`, and `"deep"` are normalized
into provider-specific kwargs. You can also pass a typed `ReasoningConfig` when
you want more control.


## LangChain + LiteLLM metadata

```python
from ooai_llm import create_llm_bundle

bundle = create_llm_bundle(
    alias="testing",
    reasoning="deep",
)

print(bundle.model.as_langchain())
print(bundle.metadata.identity.litellm_model)
print(bundle.metadata.capabilities.raw_profile)
print(bundle.metadata.pricing.input_cost_per_token)
```

The package stays LangChain-first for runtime creation, but uses the native
`litellm` package to enrich pricing and canonical provider/model naming when
it is installed. LangChain `.profile` remains the primary capability source.

## Usage callbacks and budget tracking

```python
from ooai_llm import BudgetPolicy, UsageRecorder, make_litellm_cost_callback

recorder = UsageRecorder()
callback = make_litellm_cost_callback(
    recorder,
    budget=BudgetPolicy(warn_total_tokens=5000),
)

# later: litellm.success_callback = [callback]
```

For LangChain-native usage metadata, use `build_langchain_usage_event(...)` or
`estimate_and_record_langchain_usage(...)`.

## Cache bootstrap

```python
from ooai_llm import AppSettings, configure_global_llm_cache

settings = AppSettings()
cache = configure_global_llm_cache(settings)
print(cache)
```

By default, the SQLite cache is placed under:

```text
{app_root}/.ooai/cache/llm/langchain_llm_cache.sqlite3
```

You can override this with `OOAI_LLM__CACHE__PATH` or directly on `AppSettings`.

## Testing

Run the full suite:

```bash
pdm run pytest
```

Run only unit tests:

```bash
pdm run pytest -m unit
```

Generate HTML coverage:

```bash
pdm run pytest
open build/reports/coverage/html/index.html
```

## Documentation

Build docs locally:

```bash
pdm install -G docs
pdm run sphinx-build -b html docs docs/_build/html
```

Autobuild during development:

```bash
pdm run sphinx-autobuild docs docs/_build/html
```

## Project layout

```text
src/ooai_llm/
├── __init__.py
├── cache.py
├── factory.py
├── providers.py
├── py.typed
├── reasoning.py
├── settings.py
└── types.py

docs/
examples/
tests/
```

## Included workflows

- `ci.yml` for tests, coverage, docs build, and package build
- `docs.yml` for standalone documentation validation
- `release.yml` for tagged releases and trusted PyPI publishing
- live `pytest -m live` smoke tests for provider model listing and model instantiation

## Roadmap

Short-term extensions that fit naturally with this package:

- capability and pricing metadata merged from LangChain and LiteLLM
- capability and pricing metadata merged from LangChain and LiteLLM
- embedding-model factory helpers built on the same `ModelString` type

## License

MIT


## Configuration

`ooai_llm` now separates runtime LLM configuration from live model-discovery
configuration:

- `AppSettings.llm` for defaults, aliases, and cache behavior
- `AppSettings.credentials` for provider secrets and native env-var mapping
- `AppSettings.catalog` for model-listing preferences, page sizes, SDK-vs-REST
  behavior, and provider transport overrides

Example:

```python
from ooai_llm import AppSettings

settings = AppSettings(
    catalog={
        "deepseek": {"base_url": "https://api.deepseek.com/v1"},
        "xai": {"prefer_sdk": False},
    }
)
```

