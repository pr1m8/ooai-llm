"""Unified LangChain + LiteLLM model information.

Purpose:
    Combine LangChain model profile data with LiteLLM pricing metadata and
    optional message-derived estimates to expose one normalized object for
    capability checks, cost estimation, and post-call accounting.

Design:
    - LangChain ``profile`` is treated as the capability/limits source.
    - LiteLLM is treated as the pricing and provider-string normalization
      source.
    - LangChain ``usage_metadata`` is treated as the usage source of truth
      after invocation.
    - Optional message input is normalized lazily and used for best-effort
      token/context estimates when a LangChain model instance is available.

Examples:
    >>> identity = ResolvedModelIdentity.from_model("openai:gpt-5.4-mini")
    >>> identity.litellm_model
    'openai/gpt-5.4-mini'
    >>> usage = UsageSnapshot(input_tokens=100, output_tokens=50)
    >>> usage.resolved_total_tokens
    150
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .messages import MessageEstimate, MessagesLike, normalize_messages
from .providers import Provider, get_litellm_provider_prefix, normalize_provider_name
from .reasoning import ReasoningResolution
from .settings import AppSettings
from .types import ModelString

DEFAULT_PROVIDER_PREFIXES: dict[str, str] = {
    "openai": "openai",
    "azure_openai": "openai",
    "anthropic": "anthropic",
    "google_genai": "gemini",
    "google_vertexai": "vertex_ai",
    "vertex_ai": "vertex_ai",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "openrouter": "openrouter",
    "xai": "xai",
    "mistralai": "mistral",
    "deepseek": "deepseek",
}


class PriceEntry(BaseModel):
    """Normalized token pricing for a model.

    Args:
        input_cost_per_token: USD cost per input token.
        output_cost_per_token: USD cost per output token.
        max_tokens: Maximum supported total/context tokens if known.
        billing_model_name: Model name used for billing or pricing lookup.
        source: Where this pricing entry came from.
        raw_info: Original LiteLLM pricing payload when available.
    """

    model_config = ConfigDict(frozen=True)

    input_cost_per_token: Decimal = Field(default=Decimal("0"))
    output_cost_per_token: Decimal = Field(default=Decimal("0"))
    max_tokens: int | None = None
    billing_model_name: str | None = None
    source: str = "unknown"
    raw_info: dict[str, Any] = Field(default_factory=dict)


class CapabilityProfile(BaseModel):
    """Normalized capability view derived from LangChain ``profile``.

    Args:
        max_input_tokens: Maximum input/context size if known.
        max_output_tokens: Maximum output size if known.
        tool_calling: Whether tool calling is supported.
        tool_choice: Whether tool choice is supported.
        parallel_tool_calls: Whether parallel tool calls are supported.
        structured_output: Whether native/provider structured output is supported.
        reasoning_output: Whether reasoning output is supported.
        field_sources: Per-field provenance, such as ``profile`` or ``heuristic``.
        notes: Optional explanatory notes for inferred capability values.
        raw_profile: Original LangChain profile dictionary.
    """

    model_config = ConfigDict(frozen=True)

    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    tool_calling: bool | None = None
    tool_choice: bool | None = None
    parallel_tool_calls: bool | None = None
    structured_output: bool | None = None
    reasoning_output: bool | None = None
    field_sources: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    raw_profile: dict[str, Any] = Field(default_factory=dict)


class UsageSnapshot(BaseModel):
    """Normalized usage metadata from LangChain.

    Args:
        input_tokens: Input token count.
        output_tokens: Output token count.
        total_tokens: Total token count if already provided.
        raw_usage: Original usage metadata dictionary.
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int | None = None
    raw_usage: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_total_tokens(self) -> int:
        """Return total tokens, inferring when absent."""
        return self.total_tokens if self.total_tokens is not None else self.input_tokens + self.output_tokens


class ResolvedModelIdentity(BaseModel):
    """Normalized model identity across LangChain and LiteLLM naming styles."""

    model_config = ConfigDict(frozen=True)

    provider: Provider | None = None
    model_name: str
    langchain_model: str
    litellm_model: str

    @classmethod
    def from_model(
        cls,
        model: str | ModelString,
        *,
        provider: Provider | str | None = None,
        settings: AppSettings | None = None,
    ) -> ResolvedModelIdentity:
        """Build a resolved model identity from a model string."""
        parsed = ModelString.parse(model).ensure_provider(provider)
        resolved_provider = normalize_provider_name(provider) or parsed.provider
        langchain_model = parsed.as_langchain()
        litellm_model = resolve_litellm_model_name(parsed, settings=settings)
        return cls(
            provider=resolved_provider,
            model_name=parsed.model_name,
            langchain_model=langchain_model,
            litellm_model=litellm_model,
        )


class ModelInfo(BaseModel):
    """Merged capability and pricing metadata for one model identity."""

    model_config = ConfigDict(frozen=True)

    identity: ResolvedModelIdentity
    capabilities: CapabilityProfile
    pricing: PriceEntry
    message_estimate: MessageEstimate | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider(self) -> Provider | None:
        """Return the canonical provider when known."""
        return self.identity.provider

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        """Return the bare model name."""
        return self.identity.model_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def billing_model_name(self) -> str:
        """Return the model name used for pricing lookup."""
        return self.pricing.billing_model_name or self.identity.model_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_input_tokens(self) -> int | None:
        """Prefer LangChain profile, then pricing metadata."""
        return self.capabilities.max_input_tokens or self.pricing.max_tokens


ResolvedModelMeta = ModelInfo


@dataclass(slots=True, frozen=True)
class CreatedLLMBundle:
    """Convenience bundle combining a model instance with resolved metadata."""

    model: ModelString
    llm: Any
    metadata: ModelInfo
    reasoning: ReasoningResolution | None = None


def normalize_langchain_model_name(raw_model: str | ModelString) -> tuple[str | None, str]:
    """Split a LangChain model spec into provider and model parts."""
    parsed = ModelString.parse(raw_model)
    provider = parsed.provider_prefix
    return provider, parsed.model_name


normalize_model_name = normalize_langchain_model_name


def _infer_parallel_tool_calls(
    raw_profile: Mapping[str, Any],
    *,
    llm: Any | None = None,
    provider: Provider | None = None,
) -> tuple[bool | None, str | None, list[str]]:
    """Infer parallel-tool-call support conservatively.

    Args:
        raw_profile: Raw LangChain profile mapping.
        llm: Optional live LangChain model instance.
        provider: Optional canonical provider.

    Returns:
        Tuple of inferred value, source label, and explanatory notes.
    """
    notes: list[str] = []
    tool_calling = raw_profile.get("tool_calling")
    if tool_calling is False:
        return False, "profile", notes

    explicit_value = raw_profile.get("parallel_tool_calls")
    if explicit_value is not None:
        return bool(explicit_value), "profile", notes

    disabled_params = getattr(llm, "disabled_params", None)
    if isinstance(disabled_params, Mapping) and "parallel_tool_calls" in disabled_params:
        disabled_value = disabled_params.get("parallel_tool_calls")
        if disabled_value is None:
            notes.append("parallel_tool_calls disabled by model disabled_params")
            return False, "model_attr", notes

    if tool_calling is True and provider in {Provider.OPENAI, Provider.ANTHROPIC}:
        notes.append("parallel_tool_calls inferred from provider tool-calling defaults")
        return True, "heuristic", notes
    return None, None, notes


def build_capability_profile(
    profile: Mapping[str, Any] | None,
    *,
    llm: Any | None = None,
    provider: Provider | str | None = None,
) -> CapabilityProfile:
    """Build a normalized capability profile from LangChain ``profile``."""
    raw = dict(profile or {})
    resolved_provider = normalize_provider_name(provider)
    field_sources: dict[str, str] = {}
    notes: list[str] = []

    def value_from_profile(name: str) -> Any:
        value = raw.get(name)
        if value is not None:
            field_sources[name] = "profile"
        return value

    parallel_tool_calls, parallel_source, parallel_notes = _infer_parallel_tool_calls(
        raw,
        llm=llm,
        provider=resolved_provider,
    )
    if parallel_source is not None:
        field_sources["parallel_tool_calls"] = parallel_source
    notes.extend(parallel_notes)

    return CapabilityProfile(
        max_input_tokens=value_from_profile("max_input_tokens"),
        max_output_tokens=value_from_profile("max_output_tokens"),
        tool_calling=value_from_profile("tool_calling"),
        tool_choice=value_from_profile("tool_choice"),
        parallel_tool_calls=parallel_tool_calls,
        structured_output=value_from_profile("structured_output"),
        reasoning_output=value_from_profile("reasoning_output"),
        field_sources=field_sources,
        notes=notes,
        raw_profile=raw,
    )


build_model_profile = build_capability_profile


def build_usage_snapshot(usage_metadata: Mapping[str, Any] | None) -> UsageSnapshot:
    """Build a normalized usage snapshot from LangChain usage metadata."""
    raw = dict(usage_metadata or {})
    return UsageSnapshot(
        input_tokens=int(raw.get("input_tokens", 0) or 0),
        output_tokens=int(raw.get("output_tokens", 0) or 0),
        total_tokens=raw.get("total_tokens"),
        raw_usage=raw,
    )


def calculate_cost(meta: ModelInfo, usage: UsageSnapshot) -> Decimal:
    """Calculate actual post-call cost from normalized usage."""
    input_cost = Decimal(usage.input_tokens) * meta.pricing.input_cost_per_token
    output_cost = Decimal(usage.output_tokens) * meta.pricing.output_cost_per_token
    return input_cost + output_cost


def resolve_litellm_model_name(
    model: str | ModelString,
    *,
    settings: AppSettings | None = None,
    provider: Provider | str | None = None,
) -> str:
    """Return the LiteLLM-style model string for a model."""
    parsed = ModelString.parse(model).ensure_provider(provider)
    resolved_provider = normalize_provider_name(provider) or parsed.provider
    if resolved_provider is None:
        return parsed.model_name

    if settings is not None:
        prefix = settings.litellm.provider_prefixes.get(resolved_provider.value)
        if prefix:
            return f"{prefix}/{parsed.model_name}"
    prefix = get_litellm_provider_prefix(resolved_provider) or resolved_provider.value
    return f"{prefix}/{parsed.model_name}"


def _best_effort_message_estimate(
    *,
    llm: Any | None,
    messages: MessagesLike | None,
    max_input_tokens: int | None,
    tools: Sequence[Any] | None = None,
) -> MessageEstimate | None:
    """Build a message estimate when message input is supplied."""
    if messages is None:
        return None

    normalized = normalize_messages(messages)
    estimated_input_tokens: int | None = None
    warning: str | None = None
    if llm is not None and hasattr(llm, "get_num_tokens_from_messages"):
        try:
            estimated_input_tokens = int(
                llm.get_num_tokens_from_messages(normalized.langchain_messages, tools=tools)
            )
        except Exception as exc:  # pragma: no cover - best effort
            warning = f"Token estimation failed: {exc}"
    else:
        warning = "Token estimation unavailable without a LangChain model instance."

    fits_context_window: bool | None = None
    if estimated_input_tokens is not None and max_input_tokens is not None:
        fits_context_window = estimated_input_tokens <= max_input_tokens

    return MessageEstimate(
        message_count=normalized.message_count,
        estimated_input_tokens=estimated_input_tokens,
        fits_context_window=fits_context_window,
        warning=warning,
    )


def _build_model_info(
    model: str | ModelString,
    *,
    settings: AppSettings | None = None,
    provider: Provider | str | None = None,
    profile: Mapping[str, Any] | None = None,
    billing_model_name: str | None = None,
    llm: Any | None = None,
    messages: MessagesLike | None = None,
    tools: Sequence[Any] | None = None,
) -> ModelInfo:
    """Build normalized model information without compatibility warnings."""
    resolved_settings = settings or AppSettings()
    resolved_provider = normalize_provider_name(provider) or ModelString.parse(model).provider
    identity = ResolvedModelIdentity.from_model(
        model,
        provider=resolved_provider,
        settings=resolved_settings,
    )
    capabilities = build_capability_profile(profile, llm=llm, provider=resolved_provider)
    pricing = resolve_litellm_price_entry(
        identity,
        settings=resolved_settings,
        billing_model_name=billing_model_name,
    )
    max_input_tokens = capabilities.max_input_tokens or pricing.max_tokens
    message_estimate = _best_effort_message_estimate(
        llm=llm,
        messages=messages,
        max_input_tokens=max_input_tokens,
        tools=tools,
    )
    return ModelInfo(
        identity=identity,
        capabilities=capabilities,
        pricing=pricing,
        message_estimate=message_estimate,
    )


def resolve_model_meta(
    model: str | ModelString,
    *,
    settings: AppSettings | None = None,
    provider: Provider | str | None = None,
    profile: Mapping[str, Any] | None = None,
    billing_model_name: str | None = None,
    llm: Any | None = None,
    messages: MessagesLike | None = None,
    tools: Sequence[Any] | None = None,
) -> ResolvedModelMeta:
    """Resolve merged LangChain capability and LiteLLM pricing metadata."""
    resolved_settings = settings or AppSettings()
    resolved_provider = normalize_provider_name(provider) or ModelString.parse(model).provider
    identity = ResolvedModelIdentity.from_model(
        model,
        provider=resolved_provider,
        settings=resolved_settings,
    )
    capabilities = build_capability_profile(profile, llm=llm, provider=resolved_provider)
    pricing = resolve_litellm_price_entry(
        identity,
        settings=resolved_settings,
        billing_model_name=billing_model_name,
    )
    max_input_tokens = capabilities.max_input_tokens or pricing.max_tokens
    message_estimate = _best_effort_message_estimate(
        llm=llm,
        messages=messages,
        max_input_tokens=max_input_tokens,
        tools=tools,
    )
    return ResolvedModelMeta(
        identity=identity,
        capabilities=capabilities,
        pricing=pricing,
        message_estimate=message_estimate,
    )


def get_model_info(
    model: str | ModelString | None = None,
    *,
    llm: Any | None = None,
    profile: Mapping[str, Any] | None = None,
    settings: AppSettings | None = None,
    provider: Provider | str | None = None,
    billing_model_name: str | None = None,
    messages: MessagesLike | None = None,
    tools: Sequence[Any] | None = None,
) -> ModelInfo:
    """Return normalized model information.

    Args:
        model: Explicit model string.
        llm: Existing LangChain model instance.
        profile: Optional explicit profile mapping.
        settings: Optional application settings.
        provider: Optional provider override.
        billing_model_name: Optional explicit LiteLLM billing model.
        messages: Optional message input for best-effort token estimation.
        tools: Optional tool schema list used in token estimation.

    Returns:
        Normalized model information.

    Raises:
        ValueError: If no model name can be determined.
    """
    if llm is not None:
        inferred_model = model
        if inferred_model is None:
            for attr in ("model", "model_name"):
                value = getattr(llm, attr, None)
                if value:
                    inferred_model = str(value)
                    break
        if inferred_model is None:
            raise ValueError("Could not infer a model name from the LangChain model instance.")
        return _build_model_info(
            inferred_model,
            settings=settings,
            provider=provider,
            profile=profile or getattr(llm, "profile", None),
            billing_model_name=billing_model_name,
            llm=llm,
            messages=messages,
            tools=tools,
        )

    if model is None:
        raise ValueError("Pass either a model string or an llm instance.")

    return _build_model_info(
        model,
        settings=settings,
        provider=provider,
        profile=profile,
        billing_model_name=billing_model_name,
        llm=None,
        messages=messages,
        tools=tools,
    )


def resolve_model_meta_from_langchain_model(
    llm: Any,
    *,
    model: str | ModelString | None = None,
    settings: AppSettings | None = None,
    provider: Provider | str | None = None,
    billing_model_name: str | None = None,
    messages: MessagesLike | None = None,
    tools: Sequence[Any] | None = None,
) -> ModelInfo:
    """Resolve metadata for an existing LangChain model instance.

    Deprecated:
        Use :func:`get_model_info` for new code.
    """
    import warnings

    warnings.warn(
        "`resolve_model_meta_from_langchain_model` is deprecated; use `get_model_info(model=llm, ...)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_model_info(

        model=model,
        llm=llm,
        settings=settings,
        provider=provider,
        billing_model_name=billing_model_name,
        messages=messages,
        tools=tools,
    )


def _import_litellm() -> Any:
    """Import the native LiteLLM package lazily."""
    import importlib

    return importlib.import_module("litellm")


def _coerce_decimal(value: Any) -> Decimal:
    """Coerce a numeric-like value into ``Decimal``."""
    if value in (None, ""):
        return Decimal("0")
    return Decimal(str(value))


def _coerce_plain_dict(value: Any) -> dict[str, Any]:
    """Convert an object or mapping into a plain dictionary."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    if hasattr(value, "__dict__"):
        raw = {key: val for key, val in vars(value).items() if not key.startswith("_")}
        return dict(raw)
    return {}


def _extract_litellm_raw_info(litellm_module: Any, billing_model_name: str) -> tuple[dict[str, Any], str]:
    """Extract raw pricing info for a model from LiteLLM."""
    candidates = [billing_model_name]
    if "/" in billing_model_name:
        candidates.append(billing_model_name.split("/", 1)[1])
    if ":" in billing_model_name:
        candidates.append(billing_model_name.split(":", 1)[1])

    get_model_info = getattr(litellm_module, "get_model_info", None)
    if callable(get_model_info):
        for candidate in candidates:
            try:
                raw = _coerce_plain_dict(get_model_info(candidate))
            except Exception:
                continue
            if raw:
                return raw, "litellm.get_model_info"

    for attr in ("model_cost", "model_prices_and_context_window_json"):
        mapping = getattr(litellm_module, attr, None)
        if isinstance(mapping, Mapping):
            for candidate in candidates:
                raw = _coerce_plain_dict(mapping.get(candidate))
                if raw:
                    return raw, f"litellm.{attr}"

    return {}, "litellm"


def resolve_litellm_price_entry(
    identity: ResolvedModelIdentity,
    *,
    settings: AppSettings,
    billing_model_name: str | None = None,
) -> PriceEntry:
    """Resolve pricing metadata for a model through native LiteLLM."""
    billing_name = billing_model_name or identity.litellm_model
    if not settings.litellm.enabled or not settings.litellm.enrich_metadata:
        return PriceEntry(billing_model_name=billing_name, source="disabled")

    try:
        litellm_module = _import_litellm()
    except ImportError:
        return PriceEntry(billing_model_name=billing_name, source="unavailable")

    raw_info, source = _extract_litellm_raw_info(litellm_module, billing_name)
    if not raw_info:
        return PriceEntry(billing_model_name=billing_name, source=source)

    max_tokens = raw_info.get("max_input_tokens") or raw_info.get("max_tokens")
    return PriceEntry(
        input_cost_per_token=_coerce_decimal(raw_info.get("input_cost_per_token")),
        output_cost_per_token=_coerce_decimal(raw_info.get("output_cost_per_token")),
        max_tokens=int(max_tokens) if max_tokens not in (None, "") else None,
        billing_model_name=billing_name,
        source=source,
        raw_info=raw_info,
    )
