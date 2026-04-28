"""Provider-generic model default refresh helpers.

Purpose:
    Build refreshed provider presets from live provider catalogs or LiteLLM's
    local model registry so convenience factories can track newer models
    without hard-coding every provider release.

Design:
    - Keep factory refresh opt-in and cache automatic refresh results in-process.
    - Prefer provider-native model listings when credentials are available.
    - Use LiteLLM metadata as an optional no-credential fallback.
    - Select presets with transparent name/cost/capability heuristics and
      return notes when a provider cannot be refreshed.

Examples:
    >>> from ooai_llm import AppSettings, refresh_model_defaults
    >>> result = refresh_model_defaults(
    ...     AppSettings(),
    ...     providers=["openai"],
    ...     source="litellm",
    ... )
    >>> isinstance(result.settings, AppSettings)
    True
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import importlib
import json
from pathlib import Path
import re
from time import monotonic
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .catalog import ListModelsConfig, ProviderModelInfo, list_available_models
from .providers import Provider, get_litellm_provider_prefix, normalize_provider_name
from .settings import AppSettings, DefaultModelAliases, ModelPresetName, ProviderModelPresets
from .types import ModelString

ModelDefaultSource = Literal["auto", "provider", "litellm"]
ModelDefaultsExportFormat = Literal["json", "env"]

_AUTO_REFRESH_CACHE: dict[tuple[Any, ...], tuple[float, "ModelDefaultsRefreshResult"]] = {}

_PRESET_NAMES: tuple[ModelPresetName, ...] = (
    "default",
    "latest",
    "cheap",
    "testing",
    "fast",
    "balanced",
    "reasoning",
    "coding",
    "vision",
)

_EXCLUDED_NAME_PARTS = (
    "audio",
    "babbage",
    "dall-e",
    "dalle",
    "davinci",
    "edit",
    "embedding",
    "embed",
    "image",
    "moderation",
    "realtime",
    "rerank",
    "sora",
    "speech",
    "transcribe",
    "translation",
    "tts",
    "whisper",
)
_CHAT_MODES = {"chat", "completion", "responses", "messages"}
_SMALL_MODEL_WORDS = (
    "nano",
    "mini",
    "haiku",
    "flash",
    "lite",
    "small",
    "fast",
    "8b",
    "3b",
)
_EXPENSIVE_SPECIAL_WORDS = ("pro", "opus", "max", "ultra")
_REASONING_WORDS = (
    "reasoning",
    "think",
    "thinking",
    "magistral",
    "opus",
    "pro",
    "reasoner",
    "o1",
    "o3",
    "o4",
)
_CODING_WORDS = ("code", "codex", "codestral", "devstral", "coder")
_VISION_WORDS = ("vision", "visual", "pixtral", "vl", "multimodal")


class ModelDefaultCandidate(BaseModel):
    """Candidate chat model used for provider preset selection."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model_id: str
    source: str
    display_name: str | None = None
    created: int | None = None
    created_at: str | None = None
    supported_actions: list[str] = Field(default_factory=list)
    input_cost_per_token: Decimal | None = None
    output_cost_per_token: Decimal | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    mode: str | None = None
    supports_vision: bool | None = None
    supports_function_calling: bool | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_string(self) -> ModelString:
        """Return the provider-prefixed model string."""
        bare_model_id = self.model_id.removeprefix("models/")
        return ModelString.from_parts(bare_model_id, provider=self.provider)


class ModelPresetRecommendation(BaseModel):
    """Recommended provider presets and the candidates used to pick them."""

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    presets: ProviderModelPresets
    source: str
    candidates: list[ModelDefaultCandidate] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ModelDefaultsRefreshResult(BaseModel):
    """Result of refreshing model defaults for one or more providers."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    settings: AppSettings
    recommendations: dict[str, ModelPresetRecommendation] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ModelDefaultsUpdateResult(BaseModel):
    """Result of updating model defaults for immediate or persisted use."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    settings: AppSettings
    recommendations: dict[str, ModelPresetRecommendation] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    output_path: Path | None = None
    output_format: ModelDefaultsExportFormat | None = None
    output_text: str | None = None


def _coerce_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return {}


def _coerce_decimal(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    return Decimal(str(value))


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _candidate_from_provider_info(info: ProviderModelInfo, *, source: str) -> ModelDefaultCandidate:
    raw = dict(info.raw)
    return ModelDefaultCandidate(
        provider=info.provider,
        model_id=info.model_id,
        source=source,
        display_name=info.display_name,
        created=info.created,
        created_at=info.created_at,
        supported_actions=list(info.supported_actions),
        max_input_tokens=info.input_token_limit,
        max_output_tokens=info.output_token_limit,
        raw=raw,
    )


def _candidate_from_litellm_entry(
    *,
    provider: Provider,
    model_key: str,
    raw: Mapping[str, Any],
) -> ModelDefaultCandidate | None:
    raw_dict = dict(raw)
    provider_prefixes = {
        provider.value,
        get_litellm_provider_prefix(provider) or provider.value,
    }
    raw_provider = str(
        raw_dict.get("litellm_provider") or raw_dict.get("provider") or ""
    ).lower()

    model_id = model_key
    if "/" in model_key:
        prefix, remainder = model_key.split("/", 1)
        if prefix.lower() not in provider_prefixes:
            return None
        model_id = remainder
    elif raw_provider and raw_provider not in provider_prefixes:
        return None
    elif not raw_provider and ModelString.parse(model_key).provider is not provider:
        return None

    return ModelDefaultCandidate(
        provider=provider,
        model_id=model_id,
        source="litellm",
        display_name=str(raw_dict.get("display_name") or model_id),
        input_cost_per_token=_coerce_decimal(raw_dict.get("input_cost_per_token")),
        output_cost_per_token=_coerce_decimal(raw_dict.get("output_cost_per_token")),
        max_input_tokens=_coerce_int(raw_dict.get("max_input_tokens") or raw_dict.get("max_tokens")),
        max_output_tokens=_coerce_int(
            raw_dict.get("max_output_tokens") or raw_dict.get("max_output_tokens_per_response")
        ),
        mode=str(raw_dict.get("mode")).lower() if raw_dict.get("mode") is not None else None,
        supports_vision=raw_dict.get("supports_vision"),
        supports_function_calling=raw_dict.get("supports_function_calling"),
        raw=raw_dict,
    )


def _load_litellm_candidates(provider: Provider) -> list[ModelDefaultCandidate]:
    litellm_module = importlib.import_module("litellm")
    model_registry: Mapping[str, Any] | None = None
    for attr in ("model_cost", "model_prices_and_context_window_json"):
        value = getattr(litellm_module, attr, None)
        if isinstance(value, Mapping):
            model_registry = value
            break
    if model_registry is None:
        return []

    candidates: list[ModelDefaultCandidate] = []
    for model_key, raw_value in model_registry.items():
        raw = _coerce_plain_dict(raw_value)
        if not raw:
            continue
        candidate = _candidate_from_litellm_entry(
            provider=provider,
            model_key=str(model_key),
            raw=raw,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _load_provider_candidates(
    provider: Provider,
    *,
    settings: AppSettings,
    config: ListModelsConfig | None,
) -> list[ModelDefaultCandidate]:
    result = list_available_models(provider, settings=settings, config=config)
    return [_candidate_from_provider_info(item, source="provider") for item in result.models]


def _normalize_providers(providers: Iterable[Provider | str] | None) -> list[Provider]:
    if providers is None:
        return list(Provider)
    normalized: list[Provider] = []
    for provider in providers:
        resolved = normalize_provider_name(provider)
        if resolved is None:
            continue
        if resolved not in normalized:
            normalized.append(resolved)
    return normalized


def _date_score_from_text(text: str) -> int:
    normalized = text.lower()
    best = 0

    for year, month, day in re.findall(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", normalized):
        best = max(best, int(year) * 10000 + int(month) * 100 + int(day))

    for year, month in re.findall(r"\b(20\d{2})[-_]?([01]\d)\b", normalized):
        best = max(best, int(year) * 10000 + int(month) * 100)

    for suffix in re.findall(r"(?:^|[-_])(\d{4})(?:$|[-_])", normalized):
        year = int(suffix[:2])
        month = int(suffix[2:])
        if 1 <= month <= 12 and 24 <= year <= 40:
            best = max(best, (2000 + year) * 10000 + month * 100)

    return best


def _created_score(candidate: ModelDefaultCandidate) -> int:
    if candidate.created is not None:
        return candidate.created
    if candidate.created_at:
        try:
            parsed = datetime.fromisoformat(candidate.created_at.replace("Z", "+00:00"))
        except ValueError:
            return 0
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    return 0


def _version_score(name: str) -> int:
    score = 0
    scale = 10**15
    for number in re.findall(r"\d+", name)[:6]:
        score += min(int(number), 999) * scale
        scale //= 1000
    return score


def _recency_score(candidate: ModelDefaultCandidate) -> int:
    name = candidate.model_id.lower()
    latest_score = 1 if "latest" in name else 0
    return (
        latest_score * 10**30
        + max(_created_score(candidate), _date_score_from_text(name)) * 10**18
        + _version_score(name)
    )


def _name_contains(candidate: ModelDefaultCandidate, words: Sequence[str]) -> bool:
    name = candidate.model_id.lower()
    return any(word in name for word in words)


def _candidate_cost(candidate: ModelDefaultCandidate) -> Decimal | None:
    costs = [candidate.input_cost_per_token, candidate.output_cost_per_token]
    known = [cost for cost in costs if cost is not None]
    if not known:
        return None
    return sum(known, Decimal("0"))


def _is_expensive_special(candidate: ModelDefaultCandidate) -> bool:
    return _name_contains(candidate, _EXPENSIVE_SPECIAL_WORDS)


def _is_reasoning_candidate(candidate: ModelDefaultCandidate) -> bool:
    name = candidate.model_id.lower()
    if "non-reasoning" in name or "non_reasoning" in name:
        return False
    return _name_contains(candidate, _REASONING_WORDS)


def _is_coding_candidate(candidate: ModelDefaultCandidate) -> bool:
    return _name_contains(candidate, _CODING_WORDS)


def _cheap_name_score(candidate: ModelDefaultCandidate) -> int:
    name = candidate.model_id.lower()
    return sum(1 for word in _SMALL_MODEL_WORDS if word in name)


def _is_chat_candidate(candidate: ModelDefaultCandidate) -> bool:
    name = candidate.model_id.lower()
    if any(part in name for part in _EXCLUDED_NAME_PARTS):
        return False
    if candidate.mode is not None and candidate.mode not in _CHAT_MODES:
        return False
    if candidate.provider is Provider.GOOGLE_GENAI and candidate.supported_actions:
        return "generateContent" in candidate.supported_actions
    return True


def _choose_latest(candidates: Sequence[ModelDefaultCandidate]) -> ModelDefaultCandidate:
    return max(candidates, key=lambda candidate: (_recency_score(candidate), candidate.model_id))


def _choose_cheapest(candidates: Sequence[ModelDefaultCandidate]) -> ModelDefaultCandidate:
    infinity = Decimal("Infinity")
    return min(
        candidates,
        key=lambda candidate: (
            _candidate_cost(candidate) is None,
            _candidate_cost(candidate) or infinity,
            -_cheap_name_score(candidate),
            -_recency_score(candidate),
        ),
    )


def _as_model_string(candidate: ModelDefaultCandidate) -> str:
    return candidate.model_string.as_langchain()


def recommend_provider_model_presets(
    provider: Provider | str,
    candidates: Sequence[ModelDefaultCandidate],
    *,
    source: str = "custom",
) -> ModelPresetRecommendation:
    """Recommend provider presets from candidate chat models.

    Args:
        provider: Provider being refreshed.
        candidates: Candidate models from a provider catalog or LiteLLM.
        source: Human-readable source label.

    Returns:
        Preset recommendation for the provider.

    Raises:
        ValueError: If no chat-like candidates are available.
    """
    resolved_provider = normalize_provider_name(provider)
    if resolved_provider is None:
        raise ValueError("Provider cannot be None.")

    chat_candidates = [candidate for candidate in candidates if _is_chat_candidate(candidate)]
    if not chat_candidates:
        raise ValueError(
            f"No chat-like model candidates found for provider {resolved_provider.value!r}."
        )

    balanced_pool = [
        candidate
        for candidate in chat_candidates
        if not _is_expensive_special(candidate)
        and not _is_reasoning_candidate(candidate)
        and not _is_coding_candidate(candidate)
    ] or [
        candidate for candidate in chat_candidates if not _is_expensive_special(candidate)
    ] or chat_candidates
    reasoning_pool = [
        candidate
        for candidate in chat_candidates
        if _is_reasoning_candidate(candidate) or _is_expensive_special(candidate)
    ] or chat_candidates
    fast_pool = [
        candidate for candidate in balanced_pool if _cheap_name_score(candidate) > 0
    ] or balanced_pool
    cheap_pool = [
        candidate for candidate in chat_candidates if _cheap_name_score(candidate) > 0
    ] or chat_candidates
    coding_pool = [candidate for candidate in chat_candidates if _is_coding_candidate(candidate)] or reasoning_pool
    vision_pool = [
        candidate
        for candidate in chat_candidates
        if candidate.supports_vision is True or _name_contains(candidate, _VISION_WORDS)
    ] or balanced_pool

    default = _choose_latest(balanced_pool)
    reasoning = _choose_latest(reasoning_pool)
    fast = _choose_latest(fast_pool)
    cheap = _choose_cheapest(cheap_pool)
    coding = _choose_latest(coding_pool)
    vision = _choose_latest(vision_pool)

    presets = ProviderModelPresets(
        default=_as_model_string(default),
        latest=_as_model_string(default),
        cheap=_as_model_string(cheap),
        testing=_as_model_string(cheap),
        fast=_as_model_string(fast),
        balanced=_as_model_string(default),
        reasoning=_as_model_string(reasoning),
        coding=_as_model_string(coding),
        vision=_as_model_string(vision),
    )
    notes = [
        f"Selected presets from {len(chat_candidates)} chat-like candidates.",
        "The `latest` preset intentionally avoids high-cost special variants "
        "such as pro/opus when a general model is available.",
    ]
    return ModelPresetRecommendation(
        provider=resolved_provider,
        presets=presets,
        source=source,
        candidates=chat_candidates,
        notes=notes,
    )


def _load_candidates_for_source(
    provider: Provider,
    *,
    settings: AppSettings,
    source: ModelDefaultSource,
    config: ListModelsConfig | None,
) -> tuple[list[ModelDefaultCandidate], str]:
    if source == "provider":
        return _load_provider_candidates(provider, settings=settings, config=config), "provider"
    if source == "litellm":
        return _load_litellm_candidates(provider), "litellm"

    if settings.credentials.get_api_key(provider):
        try:
            return _load_provider_candidates(provider, settings=settings, config=config), "provider"
        except Exception:
            pass
    return _load_litellm_candidates(provider), "litellm"


def _updated_aliases_from_provider_presets(
    aliases: DefaultModelAliases,
    presets: ProviderModelPresets,
) -> DefaultModelAliases:
    updates = {preset: presets.get(preset) for preset in _PRESET_NAMES}
    return aliases.model_copy(update=updates)


def refresh_model_defaults(
    settings: AppSettings | None = None,
    *,
    providers: Iterable[Provider | str] | None = None,
    source: ModelDefaultSource = "auto",
    config: ListModelsConfig | None = None,
    primary_alias_provider: Provider | str = Provider.OPENAI,
    strict: bool = False,
) -> ModelDefaultsRefreshResult:
    """Return settings with refreshed provider presets.

    Args:
        settings: Base settings. Defaults to ``AppSettings()``.
        providers: Providers to refresh. Defaults to every supported provider.
        source: ``"provider"`` for live provider APIs, ``"litellm"`` for
            LiteLLM's local registry, or ``"auto"`` to prefer live APIs when
            credentials are present and fall back to LiteLLM.
        config: Optional provider-listing configuration.
        primary_alias_provider: Provider whose refreshed presets should update
            global aliases like ``alias="latest"`` and ``alias="cheap"``.
        strict: Raise on the first provider refresh failure instead of
            returning notes and leaving that provider unchanged.

    Returns:
        Refresh result containing copied settings and per-provider recommendations.
    """
    resolved_settings = settings or AppSettings()
    provider_list = _normalize_providers(providers)
    primary_provider = normalize_provider_name(primary_alias_provider)
    notes: list[str] = []
    recommendations: dict[str, ModelPresetRecommendation] = {}

    defaults_by_provider = resolved_settings.llm.defaults_by_provider.model_copy(deep=True)
    aliases = resolved_settings.llm.aliases.model_copy(deep=True)

    for provider in provider_list:
        try:
            candidates, used_source = _load_candidates_for_source(
                provider,
                settings=resolved_settings,
                source=source,
                config=config,
            )
            recommendation = recommend_provider_model_presets(
                provider,
                candidates,
                source=used_source,
            )
        except Exception as exc:
            message = f"Could not refresh {provider.value} defaults from {source}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            notes.append(message)
            continue

        setattr(defaults_by_provider, provider.value, recommendation.presets)
        recommendations[provider.value] = recommendation
        if primary_provider is provider:
            aliases = _updated_aliases_from_provider_presets(aliases, recommendation.presets)

    updated_llm = resolved_settings.llm.model_copy(
        update={
            "aliases": aliases,
            "defaults_by_provider": defaults_by_provider,
            "default_model": aliases.default,
        }
    )
    updated_settings = resolved_settings.model_copy(update={"llm": updated_llm})
    return ModelDefaultsRefreshResult(
        settings=updated_settings,
        recommendations=recommendations,
        notes=notes,
    )


def _auto_refresh_cache_key(settings: AppSettings) -> tuple[Any, ...]:
    refresh_config = settings.llm.auto_refresh_models
    providers = _normalize_providers(refresh_config.providers)
    primary_provider = normalize_provider_name(refresh_config.primary_alias_provider)
    credential_presence = tuple(
        (provider.value, settings.credentials.get_api_key(provider) is not None)
        for provider in providers
    )
    llm_seed = settings.llm.model_dump(
        mode="json",
        exclude={"auto_refresh_models", "cache"},
    )
    return (
        json.dumps(llm_seed, sort_keys=True),
        json.dumps(settings.catalog.model_dump(mode="json"), sort_keys=True),
        json.dumps(settings.litellm.model_dump(mode="json"), sort_keys=True),
        refresh_config.source,
        tuple(provider.value for provider in providers),
        primary_provider.value if primary_provider is not None else refresh_config.primary_alias_provider,
        refresh_config.strict,
        credential_presence,
    )


def auto_refresh_model_defaults(
    settings: AppSettings | None = None,
    *,
    enabled: bool | None = None,
    force: bool = False,
) -> ModelDefaultsRefreshResult:
    """Refresh model defaults according to factory auto-refresh settings.

    Args:
        settings: Base settings. Defaults to ``AppSettings()``.
        enabled: Optional override for ``settings.llm.auto_refresh_models.enabled``.
        force: Bypass the process-local refresh cache.

    Returns:
        Refresh result. When automatic refresh is disabled, the original
        settings are returned unchanged with no recommendations.
    """
    resolved_settings = settings or AppSettings()
    refresh_config = resolved_settings.llm.auto_refresh_models
    should_refresh = refresh_config.enabled if enabled is None else enabled
    if not should_refresh:
        return ModelDefaultsRefreshResult(settings=resolved_settings)

    cache_seconds = refresh_config.cache_seconds
    cache_key = _auto_refresh_cache_key(resolved_settings)
    now = monotonic()
    if not force and cache_seconds != 0:
        cached = _AUTO_REFRESH_CACHE.get(cache_key)
        if cached is not None:
            cached_at, cached_result = cached
            if cache_seconds is None or now - cached_at <= cache_seconds:
                return cached_result

    result = refresh_model_defaults(
        resolved_settings,
        providers=refresh_config.providers,
        source=refresh_config.source,
        primary_alias_provider=refresh_config.primary_alias_provider,
        strict=refresh_config.strict,
    )
    if cache_seconds != 0:
        _AUTO_REFRESH_CACHE[cache_key] = (now, result)
    return result


def build_model_default_overrides(
    settings: AppSettings,
    *,
    providers: Iterable[Provider | str] | None = None,
    include_aliases: bool = True,
) -> dict[str, Any]:
    """Build a non-secret settings override payload for model defaults.

    Args:
        settings: Settings containing the desired model defaults.
        providers: Providers to include. Defaults to every supported provider.
        include_aliases: Whether to include global aliases and default model.

    Returns:
        Nested dictionary suitable for ``AppSettings(**payload)``.
    """
    provider_list = _normalize_providers(providers)
    llm_payload: dict[str, Any] = {
        "defaults_by_provider": {},
    }
    if include_aliases:
        llm_payload["default_model"] = settings.llm.default_model
        llm_payload["aliases"] = settings.llm.aliases.model_dump(mode="json")

    for provider in provider_list:
        presets = settings.llm.defaults_by_provider.get(provider)
        llm_payload["defaults_by_provider"][provider.value] = presets.model_dump(mode="json")

    return {"llm": llm_payload}


def _flatten_env(prefix: str, value: Mapping[str, Any]) -> list[tuple[str, str]]:
    lines: list[tuple[str, str]] = []
    for key, item in value.items():
        separator = "_" if prefix == "OOAI" else "__"
        env_key = f"{prefix}{separator}{key.upper()}"
        if isinstance(item, Mapping):
            lines.extend(_flatten_env(env_key, item))
            continue
        if item is None:
            continue
        lines.append((env_key, str(item)))
    return lines


def model_default_overrides_to_env(overrides: Mapping[str, Any]) -> str:
    """Render model-default overrides as ``OOAI_`` nested environment variables."""
    lines = _flatten_env("OOAI", overrides)
    return "\n".join(f"{key}={value}" for key, value in lines) + "\n"


def model_default_overrides_to_json(overrides: Mapping[str, Any]) -> str:
    """Render model-default overrides as formatted JSON."""
    return json.dumps(overrides, indent=2, sort_keys=True) + "\n"


def render_model_default_overrides(
    overrides: Mapping[str, Any],
    *,
    output_format: ModelDefaultsExportFormat,
) -> str:
    """Render model-default overrides in a supported persistence format."""
    if output_format == "json":
        return model_default_overrides_to_json(overrides)
    if output_format == "env":
        return model_default_overrides_to_env(overrides)
    raise ValueError(f"Unsupported model-default output format: {output_format!r}.")


def update_model_defaults(
    settings: AppSettings | None = None,
    *,
    providers: Iterable[Provider | str] | None = None,
    source: ModelDefaultSource = "auto",
    config: ListModelsConfig | None = None,
    primary_alias_provider: Provider | str = Provider.OPENAI,
    strict: bool = False,
    output_path: str | Path | None = None,
    output_format: ModelDefaultsExportFormat = "json",
    include_aliases: bool = True,
) -> ModelDefaultsUpdateResult:
    """Refresh model defaults and optionally write reusable overrides.

    Args:
        settings: Base settings. Defaults to ``AppSettings()``.
        providers: Providers to update. Defaults to every supported provider.
        source: Model source, passed through to ``refresh_model_defaults``.
        config: Optional provider-listing configuration.
        primary_alias_provider: Provider whose presets update global aliases.
        strict: Raise on provider refresh failure.
        output_path: Optional file path to write rendered overrides.
        output_format: ``"json"`` or ``"env"``.
        include_aliases: Whether exported overrides include global aliases.

    Returns:
        Update result with refreshed settings, overrides, and rendered text when
        ``output_path`` is omitted.
    """
    refresh = refresh_model_defaults(
        settings,
        providers=providers,
        source=source,
        config=config,
        primary_alias_provider=primary_alias_provider,
        strict=strict,
    )
    overrides = build_model_default_overrides(
        refresh.settings,
        providers=providers,
        include_aliases=include_aliases,
    )
    rendered = render_model_default_overrides(overrides, output_format=output_format)

    resolved_output_path: Path | None = None
    output_text: str | None = rendered
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(rendered, encoding="utf-8")
        output_text = None

    return ModelDefaultsUpdateResult(
        settings=refresh.settings,
        recommendations=refresh.recommendations,
        overrides=overrides,
        notes=refresh.notes,
        output_path=resolved_output_path,
        output_format=output_format,
        output_text=output_text,
    )
