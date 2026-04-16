"""Public package API for ``ooai_llm``.

Purpose:
    Expose a small, ergonomic public surface centered on chat-model creation,
    model discovery, and model inspection.

Design:
    - Keep the top-level API intentionally small.
    - Preserve older names through deprecated lazy aliases to avoid sudden
      breakage for early adopters.
    - Re-export the core settings and typed model-string helpers that users are
      most likely to configure directly.

Attributes:
    __all__: Curated public API.

Examples:
    >>> from ooai_llm import AppSettings, create_llm, get_model_info
    >>> settings = AppSettings()
    >>> callable(create_llm) and callable(get_model_info)
    True
"""

from __future__ import annotations

from warnings import warn

from .callbacks import BudgetExceededError, BudgetPolicy, UsageEvent, UsageRecorder
from .catalog import list_available_models
from .factory import create_llm, create_llm_bundle
from .messages import MessageEstimate, NormalizedMessages, normalize_messages
from .metadata import ModelInfo, get_model_info
from .reasoning import ReasoningConfig
from .settings import AppSettings
from .types import ModelString

list_models = list_available_models

__all__ = [
    "AppSettings",
    "BudgetExceededError",
    "BudgetPolicy",
    "MessageEstimate",
    "ModelInfo",
    "ModelString",
    "NormalizedMessages",
    "ReasoningConfig",
    "UsageEvent",
    "UsageRecorder",
    "create_llm",
    "create_llm_bundle",
    "get_model_info",
    "list_available_models",
    "list_models",
    "normalize_messages",
]

_DEPRECATED_IMPORTS = {
    "CapabilityProfile": (".metadata", "CapabilityProfile"),
    "CreatedLLMBundle": (".metadata", "CreatedLLMBundle"),
    "PriceEntry": (".metadata", "PriceEntry"),
    "ResolvedModelIdentity": (".metadata", "ResolvedModelIdentity"),
    "ResolvedModelMeta": (".metadata", "ResolvedModelMeta"),
    "UsageSnapshot": (".metadata", "UsageSnapshot"),
    "build_capability_profile": (".metadata", "build_capability_profile"),
    "build_langchain_usage_event": (".callbacks", "build_langchain_usage_event"),
    "build_model_profile": (".metadata", "build_model_profile"),
    "build_usage_snapshot": (".metadata", "build_usage_snapshot"),
    "calculate_cost": (".metadata", "calculate_cost"),
    "configure_global_llm_cache": (".cache", "configure_global_llm_cache"),
    "estimate_and_record_langchain_usage": (".callbacks", "estimate_and_record_langchain_usage"),
    "get_litellm_provider_prefix": (".providers", "get_litellm_provider_prefix"),
    "get_provider_client_info": (".catalog", "get_provider_client_info"),
    "infer_provider_from_model_name": (".providers", "infer_provider_from_model_name"),
    "ListModelsConfig": (".catalog", "ListModelsConfig"),
    "LiteLLMSettings": (".settings", "LiteLLMSettings"),
    "LLMCacheSettings": (".settings", "LLMCacheSettings"),
    "LLMSettings": (".settings", "LLMSettings"),
    "make_litellm_cost_callback": (".callbacks", "make_litellm_cost_callback"),
    "native_environment_overrides": (".factory", "native_environment_overrides"),
    "normalize_langchain_model_name": (".metadata", "normalize_langchain_model_name"),
    "normalize_model_name": (".metadata", "normalize_model_name"),
    "normalize_provider_name": (".providers", "normalize_provider_name"),
    "Provider": (".providers", "Provider"),
    "ProviderClientInfo": (".catalog", "ProviderClientInfo"),
    "ProviderCredentials": (".settings", "ProviderCredentials"),
    "ProviderModelInfo": (".catalog", "ProviderModelInfo"),
    "ProviderModelPresets": (".settings", "ProviderModelPresets"),
    "resolve_litellm_model_name": (".metadata", "resolve_litellm_model_name"),
    "resolve_llm_cache_path": (".cache", "resolve_llm_cache_path"),
    "resolve_model_meta": (".metadata", "resolve_model_meta"),
    "resolve_model_meta_from_langchain_model": (".metadata", "resolve_model_meta_from_langchain_model"),
    "resolve_model_string": (".factory", "resolve_model_string"),
    "ReasoningResolution": (".reasoning", "ReasoningResolution"),
    "build_reasoning_resolution": (".reasoning", "build_reasoning_resolution"),
    "CatalogProviderSettings": (".settings", "CatalogProviderSettings"),
    "CatalogSettings": (".settings", "CatalogSettings"),
    "DefaultModelAliases": (".settings", "DefaultModelAliases"),
    "DefaultModelsByProvider": (".settings", "DefaultModelsByProvider"),
    "ModelListResult": (".catalog", "ModelListResult"),
    "list_model_ids": (".catalog", "list_model_ids"),
}


def __getattr__(name: str):
    """Resolve deprecated top-level imports lazily.

    Args:
        name: Requested attribute name.

    Returns:
        Deprecated object imported from its home module.

    Raises:
        AttributeError: If the name is unknown.
    """
    target = _DEPRECATED_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = target
    warn(
        f"`ooai_llm.{name}` is deprecated as a top-level import. Import it from {module_name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    module = __import__(f"{__name__}{module_name}", fromlist=[attribute_name])
    return getattr(module, attribute_name)
