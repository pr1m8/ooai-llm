"""Chat-model factory helpers.

Purpose:
    Provide an ergonomic wrapper around LangChain's ``init_chat_model`` that
    integrates app settings, default model resolution, optional reasoning
    adaptation, temporary native environment-variable injection, and optional metadata
    bundling that merges LangChain profiles with native LiteLLM pricing.

Design:
    - Keep ``create_llm`` thin and transparent.
    - Reuse :class:`ooai_llm.types.ModelString`,
      :class:`ooai_llm.settings.AppSettings`, and
      :mod:`ooai_llm.reasoning`.
    - Use a context manager to mirror app-prefixed credentials into the
      provider-native environment variable names expected by integration
      packages.
    - Allow explicit ``**kwargs`` to override any auto-generated constructor
      kwargs such as cache or provider-specific reasoning settings.

Examples:
    >>> settings = AppSettings()
    >>> resolved = resolve_model_string(settings=settings, alias="testing")
    >>> resolved.model_name == "gpt-5.4-nano"
    True
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from .cache import normalize_cache_argument
from .messages import MessagesLike
from .metadata import CreatedLLMBundle, get_model_info
from .providers import Provider, normalize_provider_name
from .reasoning import ReasoningInput, build_reasoning_resolution
from .settings import AppSettings, ModelAliasName, ModelPresetName
from .types import ModelString

if TYPE_CHECKING:
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.caches import BaseCache


def resolve_model_string(
    *,
    settings: AppSettings,
    model: str | ModelString | None = None,
    alias: ModelAliasName | None = None,
    provider: Provider | str | None = None,
    preset: ModelPresetName = "default",
) -> ModelString:
    """Resolve the effective model string from settings and call arguments.

    Args:
        settings: Application settings.
        model: Explicit model string or typed model-string object.
        alias: Optional semantic alias.
        provider: Optional provider enum or alias.
        preset: Provider-specific preset name.

    Returns:
        Typed model-string object.
    """
    if isinstance(model, ModelString):
        return model
    resolved = settings.resolve_model(model=model, alias=alias, provider=provider, preset=preset)
    return ModelString.parse(resolved)


def resolve_factory_settings(
    settings: AppSettings | None = None,
    *,
    auto_refresh_models: bool | None = None,
    force_model_refresh: bool = False,
) -> AppSettings:
    """Resolve settings for a factory call, applying opt-in model refresh.

    Args:
        settings: Optional application settings.
        auto_refresh_models: Optional per-call override for
            ``settings.llm.auto_refresh_models.enabled``.
        force_model_refresh: Whether to bypass the process-local refresh cache.

    Returns:
        Original or refreshed settings.
    """
    resolved_settings = settings or AppSettings()
    if auto_refresh_models is False:
        return resolved_settings
    if auto_refresh_models is True or resolved_settings.llm.auto_refresh_models.enabled:
        from .model_defaults import auto_refresh_model_defaults

        return auto_refresh_model_defaults(
            resolved_settings,
            enabled=auto_refresh_models,
            force=force_model_refresh,
        ).settings
    return resolved_settings


@contextmanager
def native_environment_overrides(settings: AppSettings, *, force: bool = False) -> Iterator[None]:
    """Temporarily set provider-native environment variables from settings.

    Args:
        settings: Application settings.
        force: Whether to overwrite existing environment variables.

    Yields:
        ``None`` while the temporary environment is active.
    """
    desired = settings.credentials.to_native_environment()
    previous: dict[str, str | None] = {}

    try:
        for key, value in desired.items():
            if not force and key in os.environ:
                continue
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def create_llm(
    model: str | ModelString | None = None,
    *,
    settings: AppSettings | None = None,
    alias: ModelAliasName | None = None,
    provider: Provider | str | None = None,
    preset: ModelPresetName = "default",
    cache: "BaseCache | bool | None" = None,
    reasoning: ReasoningInput = None,
    auto_refresh_models: bool | None = None,
    force_model_refresh: bool = False,
    configurable_fields: str | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> "BaseChatModel":
    """Create a LangChain chat model.

    Args:
        model: Explicit model string or typed model-string object.
        settings: Optional application settings. Defaults to ``AppSettings()``.
        alias: Optional semantic alias used when ``model`` is omitted.
        provider: Optional provider used when ``model`` is omitted or bare.
        preset: Provider preset used with ``provider``.
        cache: Optional per-model cache override.
        reasoning: Optional semantic reasoning preset, reasoning-effort string,
            or typed :class:`ooai_llm.reasoning.ReasoningConfig`.
        auto_refresh_models: Opt-in model-default refresh before model
            resolution. Defaults to ``settings.llm.auto_refresh_models.enabled``.
        force_model_refresh: Bypass the process-local model-default refresh
            cache when automatic refresh is enabled.
        configurable_fields: Optional LangChain configurable field spec.
        config_prefix: Optional LangChain configuration prefix.
        **kwargs: Additional keyword arguments passed to ``init_chat_model``.
            Explicit kwargs override auto-generated cache and reasoning kwargs.

    Returns:
        LangChain chat model instance.

    Raises:
        ImportError: If ``langchain`` is not installed.
    """
    from langchain.chat_models import init_chat_model

    resolved_settings = resolve_factory_settings(
        settings,
        auto_refresh_models=auto_refresh_models,
        force_model_refresh=force_model_refresh,
    )
    resolved_model = resolve_model_string(
        settings=resolved_settings,
        model=model,
        alias=alias,
        provider=provider,
        preset=preset,
    )

    ctor_kwargs: dict[str, Any] = {}
    if cache is not None:
        ctor_kwargs["cache"] = normalize_cache_argument(cache)

    reasoning_resolution = build_reasoning_resolution(
        model=resolved_model,
        provider=provider,
        reasoning=reasoning,
    )
    if reasoning_resolution is not None:
        ctor_kwargs.update(reasoning_resolution.constructor_kwargs)

    model_provider = normalize_provider_name(provider) or resolved_model.provider
    bare_model_name = resolved_model.model_name
    model_argument = str(resolved_model) if resolved_model.is_prefixed else bare_model_name

    if configurable_fields is not None:
        ctor_kwargs["configurable_fields"] = configurable_fields
    if config_prefix is not None:
        ctor_kwargs["config_prefix"] = config_prefix
    if model_provider is not None and not resolved_model.is_prefixed:
        ctor_kwargs["model_provider"] = str(model_provider)

    ctor_kwargs.update(kwargs)

    with native_environment_overrides(resolved_settings):
        return init_chat_model(model_argument, **ctor_kwargs)



def create_llm_bundle(
    model: str | ModelString | None = None,
    *,
    settings: AppSettings | None = None,
    alias: ModelAliasName | None = None,
    provider: Provider | str | None = None,
    preset: ModelPresetName = "default",
    cache: "BaseCache | bool | None" = None,
    reasoning: ReasoningInput = None,
    auto_refresh_models: bool | None = None,
    force_model_refresh: bool = False,
    billing_model_name: str | None = None,
    messages: MessagesLike | None = None,
    tools: list[Any] | tuple[Any, ...] | None = None,
    configurable_fields: str | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> CreatedLLMBundle:
    """Create a chat model and resolve merged metadata for it.

    Args:
        model: Explicit model string or typed model-string object.
        settings: Optional application settings. Defaults to ``AppSettings()``.
        alias: Optional semantic alias used when ``model`` is omitted.
        provider: Optional provider used when ``model`` is omitted or bare.
        preset: Provider preset used with ``provider``.
        cache: Optional per-model cache override.
        reasoning: Optional reasoning preset or typed config.
        auto_refresh_models: Opt-in model-default refresh before model
            resolution. Defaults to ``settings.llm.auto_refresh_models.enabled``.
        force_model_refresh: Bypass the process-local model-default refresh
            cache when automatic refresh is enabled.
        billing_model_name: Optional explicit LiteLLM billing-model override.
        messages: Optional message input used for best-effort token estimates.
        tools: Optional tool schema list used for token estimation.
        configurable_fields: Optional LangChain configurable field spec.
        config_prefix: Optional LangChain configuration prefix.
        **kwargs: Additional keyword arguments passed to ``init_chat_model``.

    Returns:
        Convenience bundle containing the created LLM, the typed model string,
        resolved metadata, and the applied reasoning resolution.
    """
    resolved_settings = resolve_factory_settings(
        settings,
        auto_refresh_models=auto_refresh_models,
        force_model_refresh=force_model_refresh,
    )
    resolved_model = resolve_model_string(
        settings=resolved_settings,
        model=model,
        alias=alias,
        provider=provider,
        preset=preset,
    )
    llm = create_llm(
        model=resolved_model,
        settings=resolved_settings,
        provider=provider,
        cache=cache,
        reasoning=reasoning,
        auto_refresh_models=False,
        configurable_fields=configurable_fields,
        config_prefix=config_prefix,
        **kwargs,
    )
    reasoning_resolution = build_reasoning_resolution(
        model=resolved_model,
        provider=provider,
        reasoning=reasoning,
    )
    metadata = get_model_info(
        model=resolved_model,
        llm=llm,
        settings=resolved_settings,
        provider=provider,
        billing_model_name=billing_model_name,
        messages=messages,
        tools=tools,
    )
    return CreatedLLMBundle(
        model=resolved_model,
        llm=llm,
        metadata=metadata,
        reasoning=reasoning_resolution,
    )
