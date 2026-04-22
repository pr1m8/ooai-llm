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
