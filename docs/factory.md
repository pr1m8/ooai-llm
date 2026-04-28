# Factory guide

The factory is the main runtime entry point for applications. It resolves a
model string from settings, mirrors configured credentials into provider-native
environment variables, adapts reasoning options for the selected provider, and
then calls LangChain's `init_chat_model(...)`.

## Create a model

Use an alias for app defaults:

```python
from ooai_llm import AppSettings, create_llm

settings = AppSettings()

llm = create_llm(
    alias="testing",
    settings=settings,
    temperature=0,
)
```

Use a provider preset when you want a provider-specific choice:

```python
from ooai_llm import create_llm

llm = create_llm(
    provider="anthropic",
    preset="reasoning",
    temperature=0,
)
```

Use an explicit model string when the application should choose the exact
provider and model:

```python
from ooai_llm import create_llm

llm = create_llm("openai:gpt-5.4-mini", temperature=0)
```

Bare model names can be paired with a provider:

```python
from ooai_llm import create_llm

llm = create_llm("claude-3-5-haiku-20241022", provider="anthropic")
```

## Add reasoning

Reasoning can be a simple preset:

```python
from ooai_llm import create_llm

llm = create_llm(
    "openai:gpt-5.4-mini",
    reasoning="deep",
    temperature=0,
)
```

Or a typed config:

```python
from ooai_llm import ReasoningConfig, create_llm

llm = create_llm(
    "google_genai:gemini-2.5-flash",
    reasoning=ReasoningConfig(budget_tokens=1024, include_thoughts=True),
)
```

## Get metadata with the model

Use `create_llm_bundle(...)` when callers need the model plus normalized
metadata:

```python
from ooai_llm import create_llm_bundle

bundle = create_llm_bundle(
    alias="testing",
    reasoning="fast",
    temperature=0,
)

print(bundle.model.as_langchain())
print(bundle.metadata.identity.litellm_model)
print(bundle.metadata.capabilities.tool_calling)
print(bundle.metadata.pricing.input_cost_per_token)
```

The bundle contains:

- `model`: typed `ModelString`
- `llm`: the LangChain chat model instance
- `metadata`: LangChain capability profile plus LiteLLM pricing when available
- `reasoning`: the provider-specific reasoning resolution, if requested

## Customize defaults

Override app-level aliases and provider presets with `AppSettings`:

```python
from ooai_llm import AppSettings, create_llm

settings = AppSettings(
    llm={
        "aliases": {
            "testing": "openai:gpt-5.4-nano",
            "reasoning": "anthropic:claude-sonnet-4-20250514",
        }
    }
)

llm = create_llm(alias="reasoning", settings=settings)
```

Refresh aliases and provider presets from live provider catalogs or LiteLLM
metadata when you want convenience factories to track newer models:

```python
from ooai_llm import AppSettings, create_llm, refresh_model_defaults

settings = AppSettings()
refresh = refresh_model_defaults(
    settings,
    providers=["openai", "anthropic", "mistral"],
    source="auto",
)

settings = refresh.settings
llm = create_llm(alias="latest", settings=settings)
```

Use `source="provider"` to require live provider model-listing APIs, or
`source="litellm"` to use LiteLLM's local registry without provider-listing
credentials.

## Cache behavior

Configure the global LangChain cache once at application startup:

```python
from ooai_llm import AppSettings, configure_global_llm_cache, create_llm

settings = AppSettings()
configure_global_llm_cache(settings)

llm = create_llm(alias="testing", settings=settings)
```

Pass `cache=False` to disable caching for one model:

```python
llm = create_llm(alias="testing", cache=False)
```

Configure Redis, Upstash Redis, memory, or SQLAlchemy by changing
`llm.cache.backend`:

```python
settings = AppSettings(
    llm={
        "cache": {
            "backend": "redis",
            "redis_url": "redis://localhost:6379/0",
            "ttl": 3600,
        }
    }
)
configure_global_llm_cache(settings)
```

```python
settings = AppSettings(
    llm={
        "cache": {
            "backend": "upstash_redis",
            "upstash_url": "...",
            "upstash_token": "...",
            "ttl": 3600,
        }
    }
)
configure_global_llm_cache(settings)
```

## Recommended application wrapper

For application code, define one small wrapper so the rest of the app does not
need to know about aliases, providers, or presets:

```python
from ooai_llm import AppSettings, create_llm

settings = AppSettings()


def build_chat_model(kind: str = "default"):
    if kind == "cheap":
        return create_llm(alias="cheap", settings=settings, temperature=0)
    if kind == "reasoning":
        return create_llm(alias="reasoning", settings=settings, reasoning="deep")
    return create_llm(alias="default", settings=settings)
```
