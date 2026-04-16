# Getting started

## Install

```bash
pdm add ooai-llm
```

Add provider integrations as needed:

```bash
pdm add ooai-llm[openai]
pdm add ooai-llm[anthropic]
```

## Configure environment

Create a `.env` file from the example template:

```bash
cp .env.example .env
```

At minimum, set the provider key you need.

## Minimal example

```python
from ooai_llm import AppSettings, configure_global_llm_cache, create_llm

settings = AppSettings()
configure_global_llm_cache(settings)

llm = create_llm(alias="testing", settings=settings, temperature=0)
print(llm)
```
