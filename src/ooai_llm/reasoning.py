"""Provider-aware reasoning configuration.

Purpose:
    Normalize one ergonomic reasoning policy into provider-specific keyword
    arguments for chat-model construction.

Design:
    - Accept either string presets or a typed configuration object.
    - Keep provider differences explicit through a structured resolution object
      that records notes about lossy mappings and unsupported options.
    - Focus on constructor kwargs consumed by ``create_llm(...)`` while leaving
      room for future invoke-time configuration.

Attributes:
    ReasoningPresetName: Semantic presets for common reasoning modes.
    ReasoningEffortName: Provider-agnostic effort scale.
    ReasoningSummaryName: Cross-provider summary visibility preference.

Examples:
    >>> from ooai_llm.types import ModelString
    >>> resolution = build_reasoning_resolution(
    ...     model=ModelString.parse("openai:gpt-5.4-mini"),
    ...     reasoning="deep",
    ... )
    >>> resolution.constructor_kwargs["reasoning"]["effort"]
    'high'
    >>> resolution.provider.value
    'openai'
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .providers import Provider, normalize_provider_name
from .types import ModelString

ReasoningPresetName = Literal["off", "testing", "fast", "balanced", "deep", "max"]
ReasoningEffortName = Literal[
    "auto",
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]
ReasoningSummaryName = Literal["off", "auto", "detailed"]


class ReasoningConfig(BaseModel):
    """Provider-agnostic reasoning preferences.

    Args:
        effort: Normalized reasoning effort.
        summary: Desired reasoning-summary visibility.
        budget_tokens: Optional explicit reasoning-budget token count.
        dynamic_budget: Whether the provider should choose the budget when it
            exposes a dynamic mode.
        include_thoughts: Whether provider-visible thought summaries should be
            included when supported.
        strict: Whether unsupported or lossy mappings should raise errors.

    Examples:
        >>> config = ReasoningConfig.from_preset("balanced")
        >>> config.effort
        'medium'
        >>> config.summary
        'auto'
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    effort: ReasoningEffortName = "auto"
    summary: ReasoningSummaryName = "auto"
    budget_tokens: int | None = Field(default=None, ge=-1)
    dynamic_budget: bool | None = None
    include_thoughts: bool = False
    strict: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def enabled(self) -> bool:
        """Whether reasoning is conceptually enabled.

        Returns:
            ``True`` when reasoning is enabled or a nonzero budget is present.
        """
        return self.effort != "off" or self.budget_tokens not in {None, 0}

    @classmethod
    def from_preset(cls, preset: ReasoningPresetName) -> Self:
        """Build a configuration from a semantic preset.

        Args:
            preset: Semantic preset name.

        Returns:
            Normalized reasoning configuration.

        Raises:
            ValueError: If the preset is unknown.
        """
        mapping: dict[ReasoningPresetName, ReasoningConfig] = {
            "off": cls(effort="off", summary="off"),
            "testing": cls(effort="off", summary="off"),
            "fast": cls(effort="low", summary="off"),
            "balanced": cls(effort="medium", summary="auto"),
            "deep": cls(effort="high", summary="auto"),
            "max": cls(effort="max", summary="detailed"),
        }
        return mapping[preset]

    @classmethod
    def normalize(cls, reasoning: ReasoningInput) -> Self | None:
        """Normalize string or typed input into a config object.

        Args:
            reasoning: Reasoning input.

        Returns:
            Typed config or ``None`` when reasoning is absent.

        Raises:
            ValueError: If the string value is unknown.
        """
        if reasoning is None:
            return None
        if isinstance(reasoning, cls):
            return reasoning
        if reasoning in {"off", "testing", "fast", "balanced", "deep", "max"}:
            return cls.from_preset(reasoning)
        if reasoning in {"auto", "minimal", "low", "medium", "high", "xhigh"}:
            return cls(effort=reasoning)
        raise ValueError(f"Unknown reasoning value: {reasoning!r}.")


type ReasoningInput = ReasoningConfig | ReasoningPresetName | ReasoningEffortName | None


class ReasoningResolution(BaseModel):
    """Provider-specific reasoning kwargs resolution.

    Args:
        model: Canonical model string.
        provider: Resolved provider.
        config: Normalized reasoning config.
        constructor_kwargs: Keyword arguments suitable for chat-model
            construction.
        invoke_kwargs: Reserved for future invoke-time reasoning configuration.
        notes: Informational notes about ignored or lossy mappings.

    Examples:
        >>> resolution = ReasoningResolution(
        ...     model=ModelString.parse("openai:gpt-5.4"),
        ...     provider=Provider.OPENAI,
        ...     config=ReasoningConfig(),
        ... )
        >>> resolution.constructor_kwargs
        {}
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelString
    provider: Provider
    config: ReasoningConfig
    constructor_kwargs: dict[str, Any] = Field(default_factory=dict)
    invoke_kwargs: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


def _append_note(
    resolution: ReasoningResolution,
    message: str,
    *,
    strict: bool,
) -> None:
    """Append a note or raise when strict mode is enabled.

    Args:
        resolution: Resolution object to update.
        message: Human-readable note.
        strict: Whether to raise instead.

    Raises:
        ValueError: If ``strict`` is ``True``.
    """
    if strict:
        raise ValueError(message)
    resolution.notes.append(message)


def _map_openai_effort(effort: ReasoningEffortName) -> str | None:
    """Map normalized effort to OpenAI effort.

    Args:
        effort: Normalized effort.

    Returns:
        OpenAI effort or ``None`` when reasoning is off or automatic.
    """
    if effort in {"auto", "off"}:
        return None
    if effort in {"minimal", "low"}:
        return "low"
    if effort == "medium":
        return "medium"
    return "high"


def _map_anthropic_effort(effort: ReasoningEffortName) -> str | None:
    """Map normalized effort to Anthropic effort.

    Args:
        effort: Normalized effort.

    Returns:
        Anthropic effort or ``None`` when omitted.
    """
    if effort in {"auto", "off"}:
        return None
    if effort == "minimal":
        return "low"
    return effort


def _map_gemini_level(effort: ReasoningEffortName, *, model_name: str) -> str:
    """Map normalized effort to Gemini 3 thinking level.

    Args:
        effort: Normalized effort.
        model_name: Bare Gemini model name.

    Returns:
        Gemini thinking level.
    """
    is_flash = "flash" in model_name
    if effort == "off":
        return "minimal" if is_flash else "low"
    if effort == "auto":
        return "high" if "pro" in model_name else "medium"
    if effort == "minimal":
        return "minimal" if is_flash else "low"
    if effort in {"low", "medium", "high"}:
        return effort
    return "high"


def _map_gemini_budget(effort: ReasoningEffortName) -> int:
    """Map normalized effort to Gemini 2.5 thinking budget.

    Args:
        effort: Normalized effort.

    Returns:
        Thinking budget.
    """
    if effort == "off":
        return 0
    if effort == "auto":
        return -1
    if effort == "minimal":
        return 256
    if effort == "low":
        return 1024
    if effort == "medium":
        return 8192
    return 24576


def _resolve_provider(
    model: str | ModelString,
    provider: Provider | str | None = None,
) -> tuple[ModelString, Provider]:
    """Resolve the effective model string and provider.

    Args:
        model: Model string or typed model-string object.
        provider: Optional explicit provider.

    Returns:
        Tuple of typed model string and provider.

    Raises:
        ValueError: If the provider cannot be resolved.
    """
    parsed = ModelString.parse(model)
    resolved_provider = normalize_provider_name(provider) or parsed.provider
    if resolved_provider is None:
        raise ValueError(
            "Could not resolve provider for reasoning configuration. "
            "Pass an explicit provider or a prefixed/inferable model string."
        )
    return parsed, resolved_provider


def build_reasoning_resolution(
    *,
    model: str | ModelString,
    provider: Provider | str | None = None,
    reasoning: ReasoningInput,
) -> ReasoningResolution | None:
    """Resolve provider-specific reasoning kwargs.

    Args:
        model: Model string or typed model-string object.
        provider: Optional explicit provider.
        reasoning: Semantic preset, effort string, typed config, or ``None``.

    Returns:
        Structured reasoning resolution or ``None`` when reasoning is absent.
    """
    config = ReasoningConfig.normalize(reasoning)
    if config is None:
        return None

    parsed_model, resolved_provider = _resolve_provider(model, provider)
    resolution = ReasoningResolution(
        model=parsed_model,
        provider=resolved_provider,
        config=config,
    )

    if resolved_provider is Provider.OPENAI:
        if config.budget_tokens is not None:
            _append_note(
                resolution,
                "OpenAI reasoning does not expose a portable budget_tokens knob; budget_tokens was ignored.",
                strict=config.strict,
            )
        effort = _map_openai_effort(config.effort)
        if effort is None and config.summary == "off":
            return resolution
        chosen_effort = effort or "medium"
        if effort is None and config.summary != "off":
            resolution.notes.append(
                "OpenAI summary requested without explicit effort; defaulted effort to medium."
            )
        reasoning_kwargs: dict[str, Any] = {"effort": chosen_effort}
        if config.summary != "off":
            reasoning_kwargs["summary"] = config.summary
        resolution.constructor_kwargs["reasoning"] = reasoning_kwargs
        if config.effort in {"xhigh", "max"}:
            resolution.notes.append(
                "OpenAI exposes low/medium/high effort through LangChain; xhigh/max were mapped to high."
            )
        return resolution

    if resolved_provider is Provider.ANTHROPIC:
        if config.budget_tokens is not None:
            _append_note(
                resolution,
                "Anthropic adaptive thinking is preferred here; budget_tokens was ignored.",
                strict=config.strict,
            )
        display = "summarized" if (config.summary != "off" or config.include_thoughts) else "omitted"
        resolution.constructor_kwargs["thinking"] = {
            "type": "adaptive",
            "display": display,
        }
        effort = _map_anthropic_effort(config.effort)
        if effort is not None:
            resolution.constructor_kwargs["effort"] = effort
        if config.summary == "detailed":
            resolution.notes.append(
                "Anthropic currently exposes summarized vs omitted thinking here, not a distinct detailed summary mode."
            )
        return resolution

    if resolved_provider is Provider.GOOGLE_GENAI:
        model_name = parsed_model.model_name
        include_thoughts = config.include_thoughts or config.summary != "off"
        if model_name.startswith("gemini-2.5"):
            budget = config.budget_tokens
            if budget is None:
                if config.dynamic_budget is True:
                    budget = -1
                elif config.dynamic_budget is False and config.effort == "auto":
                    budget = _map_gemini_budget("medium")
                else:
                    budget = _map_gemini_budget(config.effort)
            resolution.constructor_kwargs["thinking_budget"] = budget
            if include_thoughts:
                resolution.constructor_kwargs["include_thoughts"] = True
            if config.summary == "detailed":
                resolution.notes.append(
                    "Gemini include_thoughts surfaces provider summaries, but not a distinct detailed-summary level here."
                )
            return resolution
        if model_name.startswith("gemini-3"):
            if config.budget_tokens is not None:
                _append_note(
                    resolution,
                    "Gemini 3 models use thinking_level rather than thinking_budget; budget_tokens was ignored.",
                    strict=config.strict,
                )
            resolution.constructor_kwargs["thinking_level"] = _map_gemini_level(
                config.effort,
                model_name=model_name,
            )
            if include_thoughts:
                resolution.constructor_kwargs["include_thoughts"] = True
            if config.effort in {"xhigh", "max"}:
                resolution.notes.append(
                    "Gemini 3 exposes minimal/low/medium/high thinking levels; xhigh/max were mapped to high."
                )
            if config.summary == "detailed":
                resolution.notes.append(
                    "Gemini include_thoughts surfaces provider summaries, but not a distinct detailed-summary level here."
                )
            return resolution

        _append_note(
            resolution,
            "Unknown Gemini model family; no reasoning kwargs were emitted.",
            strict=config.strict,
        )
        return resolution

    if resolved_provider is Provider.XAI:
        if parsed_model.model_name == "grok-4.20-multi-agent":
            if config.effort not in {"auto", "off"}:
                mapped_effort = config.effort
                if mapped_effort == "minimal":
                    mapped_effort = "low"
                if mapped_effort == "max":
                    mapped_effort = "xhigh"
                resolution.constructor_kwargs["reasoning"] = {"effort": mapped_effort}
            resolution.notes.append(
                "For grok-4.20-multi-agent, reasoning.effort controls agent count rather than pure thinking depth."
            )
            return resolution
        if config.enabled:
            _append_note(
                resolution,
                "Current xAI reasoning models generally reason automatically; no portable reasoning kwargs were emitted.",
                strict=config.strict,
            )
        return resolution

    if resolved_provider is Provider.DEEPSEEK:
        if parsed_model.model_name == "deepseek-reasoner":
            if config.effort == "off":
                _append_note(
                    resolution,
                    "deepseek-reasoner reasons natively and does not expose a disable switch here.",
                    strict=config.strict,
                )
            if config.summary != "off":
                resolution.notes.append(
                    "deepseek-reasoner exposes reasoning_content in responses, but not a summary-control kwarg."
                )
            if config.budget_tokens is not None:
                resolution.notes.append(
                    "DeepSeek exposes max_tokens for total output length; budget_tokens was not mapped separately."
                )
        return resolution

    if resolved_provider is Provider.MISTRAL:
        model_name = parsed_model.model_name
        if model_name == "mistral-small-latest":
            if config.effort == "off":
                resolution.constructor_kwargs["reasoning_effort"] = "none"
                return resolution
            if config.enabled:
                resolution.constructor_kwargs["reasoning_effort"] = "high"
                if config.effort not in {"auto", "high"}:
                    resolution.notes.append(
                        "Mistral adjustable reasoning currently exposes high/none; the requested effort was mapped to high."
                    )
            return resolution
        if model_name.startswith("magistral-"):
            if config.effort == "off":
                _append_note(
                    resolution,
                    "Magistral models reason natively and do not expose an off switch here.",
                    strict=config.strict,
                )
            return resolution
        if config.enabled:
            _append_note(
                resolution,
                "No portable reasoning kwargs were emitted for this Mistral model.",
                strict=config.strict,
            )
        return resolution

    return resolution
