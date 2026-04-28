# Changelog

## 0.3.1

- Added Redis, Upstash Redis, memory, and SQLAlchemy cache bootstrap support
  alongside the default SQLite cache.
- Added provider-generic model default refresh from live catalogs or LiteLLM
  metadata, plus a `latest` preset/alias for convenience factories.
- Added `update_model_defaults(...)` and `ooai-llm models update` for printing
  or writing reusable model-default overrides.
- Added opt-in automatic factory refresh through
  `llm.auto_refresh_models.enabled`, `create_llm(..., auto_refresh_models=True)`,
  and `force_model_refresh=True`.

## 0.3.0

- Added live model discovery for OpenAI, Anthropic, Google GenAI, xAI,
  DeepSeek, and Mistral.
- Added LangChain + LiteLLM model metadata helpers and usage/cost callbacks.
- Added provider-aware reasoning configuration.
- Added focused unit coverage for catalog fallback paths, callbacks, metadata,
  and public imports.
- Added PyPI and Read the Docs publishing guidance.

## 0.2.0

- Added a fuller project scaffold with docs, tests, coverage, and workflows.
- Added a typed `ModelString` root model reusable beyond chat-model configuration.
- Added provider defaults, semantic aliases, and local cache bootstrap helpers.
