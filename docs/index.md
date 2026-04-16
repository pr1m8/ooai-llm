# ooai-llm

Typed settings, provider-aware model parsing, cache bootstrap, and ergonomic LangChain model creation.

```{toctree}
:maxdepth: 2
:caption: Guide

getting-started
usage
api/index
changelog
```

## What this package solves

`ooai-llm` gives you one small typed layer for:

- parsing and canonicalizing model strings
- inferring providers from common model IDs
- resolving model defaults like `cheap`, `testing`, or `reasoning`
- loading credentials from both app-prefixed and native provider env vars
- configuring a global SQLite-backed LangChain cache
- creating chat models through a thin wrapper over LangChain's unified initializer

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
```
