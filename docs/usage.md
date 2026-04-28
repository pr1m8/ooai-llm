# Usage

## Model strings

```python
from ooai_llm import ModelString

model = ModelString.parse("gpt-5.4-mini")
assert model.provider_prefix == "openai"
assert model.model_name == "gpt-5.4-mini"
```

## Provider defaults

```python
from ooai_llm import AppSettings

settings = AppSettings()

assert settings.resolve_model(alias="cheap") == "openai:gpt-5.4-nano"
assert settings.resolve_model(alias="latest") == "openai:gpt-5.5"
reasoning_model = settings.resolve_model(provider="google", preset="reasoning")
assert reasoning_model == "google_genai:gemini-2.5-pro"
```

## Cache bootstrap

```python
from ooai_llm import AppSettings, configure_global_llm_cache

settings = AppSettings()
cache = configure_global_llm_cache(settings)
print(cache)
```

Supported cache backends are `sqlite`, `memory`, `sqlalchemy`, `redis`, and
`upstash_redis`:

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

## Factory helper

```python
from ooai_llm import create_llm

llm = create_llm("openai:gpt-5.4-mini", temperature=0)
```

Bare model names can be paired with a provider when you want the provider to be
explicit in code:

```python
from ooai_llm import create_llm

llm = create_llm("claude-3-5-haiku-20241022", provider="anthropic")
```

## Live model discovery

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

Provider SDKs are preferred when installed. REST fallbacks are used for
supported providers where SDK listing is unavailable or explicitly disabled:

```python
from ooai_llm import ListModelsConfig, list_available_models

result = list_available_models(
    "anthropic",
    config=ListModelsConfig(prefer_sdk=False, page_size=20),
)
```

Update factory aliases and provider presets from live catalogs or LiteLLM
metadata:

```python
from ooai_llm import AppSettings, update_model_defaults

settings = AppSettings()
update = update_model_defaults(
    settings,
    providers=["openai", "anthropic", "mistral"],
    source="litellm",
)

settings = update.settings
print(settings.resolve_model(alias="latest"))
print(settings.resolve_model(provider="mistral", preset="coding"))
```

Write reusable overrides from the CLI:

```bash
ooai-llm models update --source litellm --providers openai,anthropic,mistral --format json
ooai-llm models update --source auto --provider openai --format env --output .env.models
```

Enable factory-time automatic refresh when the application should refresh
aliases before model creation:

```python
from ooai_llm import AppSettings, create_llm

settings = AppSettings(
    llm={
        "auto_refresh_models": {
            "enabled": True,
            "source": "auto",
            "providers": ["openai", "anthropic", "mistral"],
        }
    }
)

llm = create_llm(alias="latest", settings=settings)
```

This path is opt-in and cached for one hour by default. Use
`force_model_refresh=True` on a factory call to bypass the cache once.


## Reasoning

```python
from ooai_llm import ReasoningConfig, build_reasoning_resolution, create_llm

resolution = build_reasoning_resolution(
    model="anthropic:claude-sonnet-4-20250514",
    reasoning="deep",
)
assert resolution is not None
assert resolution.constructor_kwargs["thinking"]["type"] == "adaptive"

llm = create_llm(
    "google_genai:gemini-2.5-flash",
    reasoning=ReasoningConfig(budget_tokens=1024, include_thoughts=True),
)
```

## Metadata and cost accounting

```python
from ooai_llm import BudgetPolicy, UsageRecorder, create_llm_bundle, make_litellm_cost_callback

bundle = create_llm_bundle("openai:gpt-5.4-mini", reasoning="fast")
print(bundle.metadata.capabilities.raw_profile)
print(bundle.metadata.pricing.input_cost_per_token)

recorder = UsageRecorder()
callback = make_litellm_cost_callback(
    recorder,
    budget=BudgetPolicy(warn_total_tokens=5000),
)
```
