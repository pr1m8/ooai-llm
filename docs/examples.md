# Examples

Runnable examples live in `examples/`.

## Basic model creation

```bash
python examples/basic_usage.py
```

This loads `AppSettings`, configures the SQLite-backed LangChain cache, and
creates the configured `testing` chat model.

## Model strings

```bash
python examples/model_string_usage.py
```

This demonstrates provider inference and conversion between bare, LangChain,
and LiteLLM model-string styles.

## Factory usage

```bash
python examples/factory_usage.py
```

This demonstrates the recommended runtime entry points: `create_llm(...)` for
model construction and `create_llm_bundle(...)` when the caller also needs
metadata.

## Live model discovery

```bash
python examples/model_discovery.py
```

The example lists a few OpenAI models by default. Change the provider argument
in the file or pass a configured provider to `list_available_models(...)`.

For a local provider sweep that skips missing credentials and defaults to
OpenAI, Anthropic, DeepSeek, and Mistral:

```bash
python examples/live_provider_matrix.py
```

Override the provider list when needed:

```bash
OOAI_EXAMPLE_PROVIDERS=openai,anthropic,deepseek,mistral python examples/live_provider_matrix.py
```

## Reasoning

```bash
python examples/reasoning_usage.py
```

This prints provider-specific constructor kwargs produced from semantic
reasoning presets.

## Metadata and callbacks

```bash
python examples/litellm_metadata_usage.py
```

This shows the metadata-aware bundle helper and the LiteLLM usage/cost callback
factory.
