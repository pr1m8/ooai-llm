# Testing

The repository has three test tiers:

- `unit`: fast tests for pure helpers, settings, parsing, catalog normalization,
  metadata, callbacks, and reasoning mappings.
- `integration`: package-level interactions, especially factory behavior and
  metadata-aware model creation.
- `e2e`: public flows. Live provider tests are marked with both `e2e` and
  `live`, are skipped by the default suite, and run only when selected with
  `-m live`.

## Common commands

Run the full checked suite with coverage:

```bash
pdm run pytest
```

Run a single tier without the global coverage threshold:

```bash
pdm run pytest -m unit --no-cov
pdm run pytest -m integration --no-cov
pdm run pytest -m "e2e and not live" --no-cov
```

Generate HTML coverage:

```bash
pdm run pytest
open build/reports/coverage/html/index.html
```

The coverage gate is configured in `pyproject.toml` and currently requires at
least 80% total coverage.

## Live provider tests

Install the provider extras you want to exercise:

```bash
pdm install -G test -G openai -G anthropic -G deepseek -G mistral
```

Set credentials with either native provider variables or the `OOAI_` aliases:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MISTRAL_API_KEY="..."
```

Run live checks only for configured providers. This example intentionally skips
Gemini and xAI:

```bash
OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
```

If `OOAI_LIVE_PROVIDERS` is omitted, the explicit live suite parametrizes all
supported providers and skips the ones without credentials.

To make missing credentials or missing optional SDK packages fail instead of
skip, set `OOAI_REQUIRE_LIVE=true`:

```bash
OOAI_REQUIRE_LIVE=true OOAI_LIVE_PROVIDERS=openai,anthropic,deepseek,mistral pdm run pytest -m live --no-cov
```

`AppSettings` loads a local `.env` file, so putting keys in `.env` is enough for
these live tests. Keep `.env.example` as a blank template.
