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
assert settings.resolve_model(provider="google", preset="reasoning") == "google_genai:gemini-2.5-pro"
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
