# AGENTS.md

This file gives coding agents enough context to work safely in this repository.

## Project Purpose

`ooai-llm` is a Python package that provides:

- typed LLM settings through Pydantic
- provider-aware model-string parsing
- LangChain-first chat-model factories
- provider model discovery
- LangChain + LiteLLM metadata enrichment
- usage and cost callbacks
- provider-specific reasoning kwarg adaptation

The main user-facing runtime entry points are `create_llm(...)` and
`create_llm_bundle(...)`.

## Repository Layout

- `src/ooai_llm/`: package source
- `tests/unit/`: fast pure tests
- `tests/integration/`: package interaction tests
- `tests/e2e/`: public flow and live provider tests
- `examples/`: runnable examples
- `docs/`: Sphinx/MyST docs
- `.github/workflows/`: CI, docs, and release automation

## Development Commands

Install all development dependencies:

```bash
pdm install -G test -G docs -G dev
```

Run the full suite with coverage:

```bash
pdm run pytest
```

Run warning-strict docs:

```bash
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
```

Validate docs style:

```bash
pdm run doc8 docs
```

Build and validate distributions:

```bash
pdm build
pdm run twine check dist/*
```

## Live Provider Tests

Live tests use real API credentials and network calls. Select providers and
require hard failures with:

```bash
OOAI_REQUIRE_LIVE=true OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
```

Gemini and xAI are supported by the package but are not required for the normal
live release pass unless explicitly selected.

## Secret Handling

- Never commit `.env`.
- Keep `.env.example` blank for secret values.
- Do not print API key values in logs or summaries.
- It is acceptable to report which credential variable names are present.
- If secrets accidentally land in tracked files, copy them to ignored `.env`
  first if needed, then redact the tracked file.

## Factory Usage Rules

Prefer documenting and testing from the public factory surface:

```python
from ooai_llm import AppSettings, create_llm, create_llm_bundle

settings = AppSettings()
llm = create_llm(alias="testing", settings=settings, temperature=0)
bundle = create_llm_bundle(alias="testing", settings=settings, reasoning="fast")
```

Use provider presets for provider-specific defaults:

```python
llm = create_llm(provider="anthropic", preset="reasoning", settings=settings)
```

Use explicit model strings for exact runtime choices:

```python
llm = create_llm("openai:gpt-5.4-mini", settings=settings)
```

## Release Flow

Before publishing:

```bash
pdm run pytest
OOAI_REQUIRE_LIVE=true OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
pdm run doc8 docs
pdm build
pdm run twine check dist/*
```

The GitHub release workflow publishes to PyPI on `v*` tags. Make sure PyPI
trusted publishing is configured for repository `pr1m8/ooai-llm`, workflow
`release.yml`, and environment `pypi`.

## Editing Guidance

- Keep public docs and README oriented around the factory API.
- Keep optional provider dependencies optional.
- Do not make Gemini or xAI mandatory for tests or examples.
- Prefer focused tests for changed behavior.
- Do not commit generated docs output under `docs/_build` or generated AutoAPI
  files under `docs/autoapi`.
