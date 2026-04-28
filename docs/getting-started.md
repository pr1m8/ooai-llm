# Getting started

## Install the base package

```bash
pdm add ooai-llm
```

Install only the provider integrations your app uses:

```bash
pdm add ooai-llm[openai]
pdm add ooai-llm[anthropic]
pdm add ooai-llm[deepseek]
pdm add ooai-llm[mistral]
pdm add ooai-llm[litellm]
pdm add ooai-llm[redis]
pdm add ooai-llm[upstash]
pdm add ooai-llm[caches]
```

Gemini and xAI support are available through the `google` and `xai` extras, but
you do not need those extras unless you plan to use those providers.

## Configure environment

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

`AppSettings` loads `.env` automatically. Keep `.env.example` as a blank
template and put real keys only in `.env`.

At minimum, set one provider key:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export DEEPSEEK_API_KEY="..."
export MISTRAL_API_KEY="..."
```

You can also use the app-prefixed names, such as `OOAI_OPENAI_API_KEY`.

## Minimal example

```python
from ooai_llm import AppSettings, configure_global_llm_cache, create_llm

settings = AppSettings()
configure_global_llm_cache(settings)

llm = create_llm(alias="testing", settings=settings, temperature=0)
print(llm)
```

Refresh convenience defaults before creating models when you want aliases such
as `latest` to follow provider catalogs or LiteLLM metadata:

```bash
ooai-llm models update --source litellm --providers openai,anthropic,mistral
```

Or let factories refresh automatically by opting in:

```bash
OOAI_LLM__AUTO_REFRESH_MODELS__ENABLED=true
OOAI_LLM__AUTO_REFRESH_MODELS__SOURCE=litellm
OOAI_LLM__AUTO_REFRESH_MODELS__PROVIDERS='["openai","anthropic","mistral"]'
```

## Developer setup

```bash
pdm install -G test -G docs -G dev
pdm run pytest
pdm run sphinx-build -E -W --keep-going -b html docs docs/_build/html
pdm build
```
