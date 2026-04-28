# ooai-llm

Typed LLM settings, provider-aware model parsing, LangChain-first chat-model
creation, live model discovery, LiteLLM metadata enrichment, and usage/cost
callback helpers.

```{toctree}
:maxdepth: 2
:caption: Guide

getting-started
factory
usage
examples
testing
publishing
api/index
changelog
```

## What this package solves

`ooai-llm` gives you one small typed layer for:

- parsing and canonicalizing model strings
- inferring providers from common model IDs
- resolving model defaults like `latest`, `cheap`, `testing`, or `reasoning`
- loading credentials from both app-prefixed and native provider env vars
- configuring a global LangChain cache with SQLite, memory, SQLAlchemy, Redis, or Upstash Redis
- creating chat models through a thin wrapper over LangChain's unified initializer
- listing available models from provider SDKs and REST APIs
- refreshing convenience defaults from live provider catalogs or LiteLLM metadata
- joining LangChain capability profiles with LiteLLM pricing metadata
- recording usage and cost events from LangChain or LiteLLM callbacks

```{admonition} Scope
:class: tip
This package is intentionally focused on configuration and model construction.
It does not try to be a full routing layer or model catalog service.
```

## Package overview

```{mermaid}
flowchart LR
    A[AppSettings] --> B[Model resolution]
    A --> C[Credential env mapping]
    B --> D[ModelString]
    C --> E[create_llm]
    D --> E[create_llm]
    A --> F[configure_global_llm_cache]
    D --> G[list_available_models]
    E --> H[get_model_info]
```
