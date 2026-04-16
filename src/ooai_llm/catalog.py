"""Live model discovery helpers.

Purpose:
    Discover available models from provider-native APIs or SDKs and normalize
    the results into one typed shape reusable across chat, embeddings, and
    other future model families.

Design:
    - Use official provider SDKs when available.
    - Fall back to documented REST endpoints when an SDK is absent or a simple
      models-list endpoint is easier to support robustly.
    - Keep the returned model objects provider-agnostic while preserving the
      raw payload for debugging or deeper inspection.
    - Focus on read-only discovery so the module stays useful even before more
      advanced routing, pricing, or profile-merging features are added.

Attributes:
    DEFAULT_DEEPSEEK_BASE_URL: OpenAI-compatible DeepSeek base URL.
    DEFAULT_XAI_BASE_URL: xAI REST API base URL.

Examples:
    >>> from ooai_llm import ProviderModelInfo
    >>> info = ProviderModelInfo(provider="openai", model_id="gpt-5.4-mini")
    >>> str(info.model_string)
    'openai:gpt-5.4-mini'
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .providers import (
    PROVIDER_EXTRAS,
    PROVIDER_NATIVE_SDK_PACKAGES,
    Provider,
    normalize_provider_name,
)
from .settings import AppSettings
from .types import ModelString

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_XAI_BASE_URL = "https://api.x.ai"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class ListModelsConfig(BaseModel):
    """Configuration for live model discovery.

    Args:
        limit: Optional cap on the number of models returned.
        page_size: Provider-specific page size when supported.
        query_base: Google GenAI SDK flag controlling whether base models are
            listed alongside tuned models.
        prefer_sdk: Whether the helper should prefer an installed SDK over a
            direct REST call when both are supported.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1)
    query_base: bool | None = None
    prefer_sdk: bool = True


class ProviderModelInfo(BaseModel):
    """Normalized model metadata from a provider listing.

    Args:
        provider: Canonical provider.
        model_id: Provider-native model identifier.
        display_name: Human-readable name when available.
        type: Provider object type.
        owned_by: Owner identifier when provided.
        created: Unix timestamp when provided.
        created_at: ISO-style timestamp when provided.
        aliases: Optional model aliases.
        supported_actions: Optional supported actions such as
            ``generateContent`` or ``embedContent``.
        input_token_limit: Maximum input tokens when returned by the provider.
        output_token_limit: Maximum output tokens when returned by the provider.
        raw: Original provider payload as a plain dictionary.

    Examples:
        >>> info = ProviderModelInfo(provider="openai", model_id="gpt-5.4")
        >>> str(info.model_string)
        'openai:gpt-5.4'
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model_id: str
    display_name: str | None = None
    type: str | None = None
    owned_by: str | None = None
    created: int | None = None
    created_at: str | None = None
    aliases: list[str] = Field(default_factory=list)
    supported_actions: list[str] = Field(default_factory=list)
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_string(self) -> ModelString:
        """Return the canonical provider-prefixed model string."""
        bare_model_id = self.model_id.removeprefix("models/")
        return ModelString.parse(f"{self.provider}:{bare_model_id}")


class ModelListResult(BaseModel):
    """Normalized result of a provider model-list call.

    Args:
        provider: Canonical provider.
        models: Listed models.
        next_page_token: Provider-native continuation token when exposed.
        used_sdk: Whether a native SDK path was used.
        notes: Informational notes about fallbacks or partial support.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    models: list[ProviderModelInfo] = Field(default_factory=list)
    next_page_token: str | None = None
    used_sdk: bool | None = None
    notes: list[str] = Field(default_factory=list)


class ProviderClientInfo(BaseModel):
    """Metadata about a provider's native SDK integration.

    Args:
        provider: Canonical provider.
        extra_name: Optional package extra that installs support.
        native_sdk_package: Native SDK import/package name when known.
        langchain_package: LangChain integration package when known.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Provider
    extra_name: str | None = None
    native_sdk_package: str | None = None
    langchain_package: str | None = None


def get_provider_client_info(provider: Provider | str) -> ProviderClientInfo:
    """Return installation metadata for a provider.

    Args:
        provider: Canonical provider or alias.

    Returns:
        Installation metadata.
    """
    resolved = normalize_provider_name(provider)
    assert resolved is not None
    return ProviderClientInfo(
        provider=resolved,
        extra_name=PROVIDER_EXTRAS.get(resolved),
        native_sdk_package=PROVIDER_NATIVE_SDK_PACKAGES.get(resolved),
        langchain_package=None,
    )


def list_model_ids(
    provider: Provider | str,
    *,
    settings: AppSettings | None = None,
    api_key: str | None = None,
    config: ListModelsConfig | None = None,
    **kwargs: Any,
) -> list[str]:
    """List only model identifiers for a provider.

    Args:
        provider: Canonical provider or alias.
        settings: Optional application settings.
        api_key: Optional explicit API key override.
        config: Optional listing configuration.
        **kwargs: Provider-specific constructor or transport kwargs.

    Returns:
        List of provider-native model IDs.
    """
    result = list_available_models(
        provider,
        settings=settings,
        api_key=api_key,
        config=config,
        **kwargs,
    )
    return [model.model_id for model in result.models]


def list_available_models(
    provider: Provider | str,
    *,
    settings: AppSettings | None = None,
    api_key: str | None = None,
    config: ListModelsConfig | None = None,
    **kwargs: Any,
) -> ModelListResult:
    """List available models from a provider.

    Args:
        provider: Canonical provider or alias.
        settings: Optional application settings.
        api_key: Optional explicit API key override.
        config: Optional listing configuration.
        **kwargs: Provider-specific client kwargs such as ``base_url``.

    Returns:
        Structured model-list result.

    Raises:
        ValueError: If the provider cannot be resolved.
        RuntimeError: If no API key can be resolved when the provider requires
            one.
        ImportError: If a provider SDK is required but unavailable and no REST
            fallback is implemented.
    """
    resolved_provider = normalize_provider_name(provider)
    if resolved_provider is None:
        raise ValueError(f"Unknown provider: {provider!r}.")

    resolved_settings = settings or AppSettings()
    resolved_config = _resolve_list_models_config(
        provider=resolved_provider,
        settings=resolved_settings,
        config=config,
    )
    transport_kwargs = _resolve_catalog_transport_kwargs(
        provider=resolved_provider,
        settings=resolved_settings,
        kwargs=kwargs,
    )
    resolved_api_key = api_key or resolved_settings.credentials.get_api_key(resolved_provider)
    if not resolved_api_key:
        raise RuntimeError(
            f"No API key configured for provider {resolved_provider.value!r}. "
            f"Set the native env var or the app-prefixed equivalent."
        )

    if resolved_provider is Provider.OPENAI:
        return _list_openai_models(
            api_key=resolved_api_key,
            config=resolved_config,
            **transport_kwargs,
        )
    if resolved_provider is Provider.ANTHROPIC:
        return _list_anthropic_models(
            api_key=resolved_api_key,
            config=resolved_config,
            **transport_kwargs,
        )
    if resolved_provider is Provider.GOOGLE_GENAI:
        return _list_google_models(
            api_key=resolved_api_key,
            settings=resolved_settings,
            config=resolved_config,
            **transport_kwargs,
        )
    if resolved_provider is Provider.XAI:
        return _list_xai_models(
            api_key=resolved_api_key,
            config=resolved_config,
            **transport_kwargs,
        )
    if resolved_provider is Provider.DEEPSEEK:
        return _list_deepseek_models(
            api_key=resolved_api_key,
            config=resolved_config,
            **transport_kwargs,
        )
    if resolved_provider is Provider.MISTRAL:
        return _list_mistral_models(
            api_key=resolved_api_key,
            config=resolved_config,
            **kwargs,
        )

    raise AssertionError(f"Unhandled provider: {resolved_provider!r}")



def _resolve_list_models_config(
    *,
    provider: Provider,
    settings: AppSettings,
    config: ListModelsConfig | None,
) -> ListModelsConfig:
    """Resolve an effective ``ListModelsConfig`` from app settings and overrides.

    Args:
        provider: Canonical provider.
        settings: Application settings.
        config: Optional explicit list-models config.

    Returns:
        Effective list-models config.
    """
    seed = settings.catalog.build_list_models_options(provider)
    if config is not None:
        seed.update(config.model_dump(exclude_none=True))
    return ListModelsConfig(**seed)



def _resolve_catalog_transport_kwargs(
    *,
    provider: Provider,
    settings: AppSettings,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve transport kwargs from settings and explicit overrides.

    Args:
        provider: Canonical provider.
        settings: Application settings.
        kwargs: Explicit function-call overrides.

    Returns:
        Effective transport kwargs.
    """
    resolved = settings.catalog.build_transport_kwargs(provider)
    resolved.update(dict(kwargs))
    return resolved

def _iter_limited(items: Iterable[Any], limit: int | None) -> Iterator[Any]:
    if limit is None:
        yield from items
    else:
        yield from islice(items, limit)


def _as_dict(value: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK objects into plain dictionaries."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if hasattr(value, "to_dict"):
        return value.to_dict()  # type: ignore[no-any-return]
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[no-any-return]
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": value}


def _normalize_model_info(provider: Provider, raw_model: Any) -> ProviderModelInfo:
    raw = _as_dict(raw_model)
    model_id = (
        raw.get("id")
        or raw.get("name")
        or raw.get("model")
        or raw.get("model_id")
        or raw.get("endpoint")
    )
    if model_id is None:
        raise ValueError(f"Could not determine model identifier from payload: {raw!r}")

    aliases = raw.get("aliases") or []
    if isinstance(aliases, tuple):
        aliases = list(aliases)
    if isinstance(aliases, str):
        aliases = [aliases]

    supported_actions = raw.get("supported_actions") or []
    if isinstance(supported_actions, tuple):
        supported_actions = list(supported_actions)

    created_at = raw.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()

    return ProviderModelInfo(
        provider=provider,
        model_id=str(model_id),
        display_name=raw.get("display_name") or raw.get("name") or raw.get("id"),
        type=raw.get("type") or raw.get("object"),
        owned_by=raw.get("owned_by") or raw.get("owner"),
        created=raw.get("created"),
        created_at=created_at,
        aliases=[str(item) for item in aliases],
        supported_actions=[str(item) for item in supported_actions],
        input_token_limit=raw.get("input_token_limit"),
        output_token_limit=raw.get("output_token_limit"),
        raw=raw,
    )


def _json_get(url: str, *, headers: Mapping[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, method="GET", headers=dict(headers or {}))
    with urlopen(request) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _list_openai_models(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, **kwargs)
    response = client.models.list()
    models = [_normalize_model_info(Provider.OPENAI, item) for item in _iter_limited(response.data, config.limit)]
    return ModelListResult(provider=Provider.OPENAI, models=models, used_sdk=True)


def _list_deepseek_models(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    from openai import OpenAI

    base_url = kwargs.pop("base_url", DEFAULT_DEEPSEEK_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    response = client.models.list()
    models = [_normalize_model_info(Provider.DEEPSEEK, item) for item in _iter_limited(response.data, config.limit)]
    return ModelListResult(provider=Provider.DEEPSEEK, models=models, used_sdk=True)


def _list_mistral_models(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    from mistralai import Mistral

    client = Mistral(api_key=api_key, **kwargs)
    response = client.models.list()
    data = getattr(response, "data", response)
    models = [_normalize_model_info(Provider.MISTRAL, item) for item in _iter_limited(data, config.limit)]
    return ModelListResult(provider=Provider.MISTRAL, models=models, used_sdk=True)


def _list_google_models(
    *,
    api_key: str,
    settings: AppSettings,
    config: ListModelsConfig,
    **kwargs: Any,
) -> ModelListResult:
    from google import genai

    client_kwargs: dict[str, Any] = dict(kwargs)
    client_kwargs.setdefault("api_key", api_key)
    if settings.credentials.google_use_vertexai is not None:
        client_kwargs.setdefault("vertexai", settings.credentials.google_use_vertexai)
    if settings.credentials.google_cloud_project:
        client_kwargs.setdefault("project", settings.credentials.google_cloud_project)
    if settings.credentials.google_cloud_location:
        client_kwargs.setdefault("location", settings.credentials.google_cloud_location)

    client = genai.Client(**client_kwargs)

    list_config: dict[str, Any] = {}
    if config.page_size is not None:
        list_config["page_size"] = config.page_size
    if config.query_base is not None:
        list_config["query_base"] = config.query_base

    pager = client.models.list(config=list_config or None)
    models = [_normalize_model_info(Provider.GOOGLE_GENAI, item) for item in _iter_limited(pager, config.limit)]
    return ModelListResult(provider=Provider.GOOGLE_GENAI, models=models, used_sdk=True)


def _list_anthropic(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    query: dict[str, Any] = {}
    if config.page_size is not None:
        query["limit"] = config.page_size

    url = "https://api.anthropic.com/v1/models"
    if query:
        url = f"{url}?{urlencode(query)}"

    payload = _json_get(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": kwargs.pop("anthropic_version", DEFAULT_ANTHROPIC_API_VERSION),
        },
    )
    data = payload.get("data", [])
    models = [_normalize_model_info(Provider.ANTHROPIC, item) for item in _iter_limited(data, config.limit)]
    next_page_token = payload.get("last_id") if payload.get("has_more") else None
    return ModelListResult(
        provider=Provider.ANTHROPIC,
        models=models,
        next_page_token=next_page_token,
        used_sdk=False,
        notes=["Used Anthropic REST models endpoint."],
    )


def _list_xai(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    url = kwargs.pop("base_url", DEFAULT_XAI_BASE_URL).rstrip("/") + "/v1/models"
    payload = _json_get(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
    )
    data = payload.get("data", [])
    models = [_normalize_model_info(Provider.XAI, item) for item in _iter_limited(data, config.limit)]
    return ModelListResult(
        provider=Provider.XAI,
        models=models,
        used_sdk=False,
        notes=["Used xAI REST models endpoint; use xai-sdk directly for richer language-model metadata."],
    )


def _list_xai_models(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    if config.prefer_sdk:
        try:
            import xai_sdk

            client = xai_sdk.Client(api_key=api_key, **kwargs)
            data = client.models.list_language_models()
            models = [
                _normalize_model_info(Provider.XAI, item)
                for item in _iter_limited(data, config.limit)
            ]
            return ModelListResult(
                provider=Provider.XAI,
                models=models,
                used_sdk=True,
                notes=["Used xAI SDK language-model listing."],
            )
        except ImportError:
            pass
    return _list_xai(api_key=api_key, config=config, **kwargs)


def _list_anthropic_models(*, api_key: str, config: ListModelsConfig, **kwargs: Any) -> ModelListResult:
    if config.prefer_sdk:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, **kwargs)
            try:
                response = client.models.list(limit=config.page_size) if config.page_size is not None else client.models.list()
            except TypeError:
                response = client.models.list()
            data = getattr(response, "data", response)
            models = [
                _normalize_model_info(Provider.ANTHROPIC, item)
                for item in _iter_limited(data, config.limit)
            ]
            next_page_token = getattr(response, "last_id", None) if getattr(response, "has_more", False) else None
            return ModelListResult(
                provider=Provider.ANTHROPIC,
                models=models,
                next_page_token=next_page_token,
                used_sdk=True,
            )
        except ImportError:
            pass
        except AttributeError:
            pass
    return _list_anthropic(api_key=api_key, config=config, **kwargs)
