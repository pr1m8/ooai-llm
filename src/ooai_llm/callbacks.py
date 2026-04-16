"""Usage and cost callback helpers.

Purpose:
    Provide ergonomic callback helpers that work with LangChain usage metadata
    and the native LiteLLM callback interface.

Design:
    - Normalize usage and cost into a shared ``UsageEvent`` model.
    - Offer a recorder object that can accumulate usage across many calls.
    - Expose a LiteLLM-compatible success callback factory for cost and usage
      tracking.
    - Keep callback helpers transport-agnostic so they can support chat,
      embeddings, and future model families.

Examples:
    >>> recorder = UsageRecorder()
    >>> callback = make_litellm_cost_callback(recorder)
    >>> callable(callback)
    True
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .metadata import build_usage_snapshot, calculate_cost, get_model_info
from .types import ModelString


class BudgetExceededError(RuntimeError):
    """Raised when a usage or cost budget is exceeded."""


class UsageEvent(BaseModel):
    """Normalized usage and cost event.

    Args:
        source: Origin of the event, such as ``langchain`` or ``litellm``.
        model: Typed model string.
        input_tokens: Input token count.
        output_tokens: Output token count.
        total_tokens: Total token count.
        cost_usd: Actual or estimated USD cost.
        latency_ms: Measured latency in milliseconds when available.
        raw: Original payload fragments.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    model: ModelString
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Decimal | None = None
    latency_ms: Decimal | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class BudgetPolicy(BaseModel):
    """Simple budget and warning thresholds for usage tracking.

    Args:
        warn_cost_usd: Optional single-event cost warning threshold.
        error_cost_usd: Optional single-event hard cost threshold.
        warn_total_tokens: Optional single-event total-token warning threshold.
        error_total_tokens: Optional single-event hard token threshold.
    """

    model_config = ConfigDict(extra="forbid")

    warn_cost_usd: Decimal | None = None
    error_cost_usd: Decimal | None = None
    warn_total_tokens: int | None = None
    error_total_tokens: int | None = None

    def check(self, event: UsageEvent) -> list[str]:
        """Return warning messages or raise when hard thresholds are crossed.

        Args:
            event: Usage event to evaluate.

        Returns:
            Warning messages.

        Raises:
            BudgetExceededError: If a hard threshold is exceeded.
        """
        warnings: list[str] = []
        if self.error_total_tokens is not None and event.total_tokens > self.error_total_tokens:
            raise BudgetExceededError(
                f"Total token budget exceeded: {event.total_tokens} > {self.error_total_tokens}."
            )
        if self.error_cost_usd is not None and event.cost_usd is not None and event.cost_usd > self.error_cost_usd:
            raise BudgetExceededError(
                f"Cost budget exceeded: {event.cost_usd} > {self.error_cost_usd}."
            )
        if self.warn_total_tokens is not None and event.total_tokens > self.warn_total_tokens:
            warnings.append(
                f"Total token warning threshold exceeded: {event.total_tokens} > {self.warn_total_tokens}."
            )
        if self.warn_cost_usd is not None and event.cost_usd is not None and event.cost_usd > self.warn_cost_usd:
            warnings.append(
                f"Cost warning threshold exceeded: {event.cost_usd} > {self.warn_cost_usd}."
            )
        return warnings


class UsageRecorder(BaseModel):
    """In-memory recorder for normalized usage events."""

    model_config = ConfigDict(extra="forbid")

    events: list[UsageEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def record(self, event: UsageEvent, *, budget: BudgetPolicy | None = None) -> UsageEvent:
        """Record an event and optionally apply a budget policy."""
        if budget is not None:
            self.warnings.extend(budget.check(event))
        self.events.append(event)
        return event

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        """Return total tokens recorded so far."""
        return sum(event.total_tokens for event in self.events)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> Decimal:
        """Return total cost recorded so far."""
        total = Decimal("0")
        for event in self.events:
            if event.cost_usd is not None:
                total += event.cost_usd
        return total


def build_langchain_usage_event(
    *,
    model: str | ModelString,
    usage_metadata: Mapping[str, Any] | None,
    cost_usd: Decimal | None = None,
    latency_ms: Decimal | None = None,
) -> UsageEvent:
    """Build a normalized event from LangChain usage metadata."""
    parsed_model = ModelString.parse(model).canonical()
    usage = build_usage_snapshot(usage_metadata)
    return UsageEvent(
        source="langchain",
        model=parsed_model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.resolved_total_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        raw=usage.raw_usage,
    )


def make_litellm_cost_callback(
    recorder: UsageRecorder,
    *,
    budget: BudgetPolicy | None = None,
) -> Any:
    """Return a LiteLLM success callback that records cost and usage.

    Args:
        recorder: Recorder instance to update.
        budget: Optional budget policy to evaluate for each event.

    Returns:
        LiteLLM-compatible callback function.
    """

    def _callback(
        kwargs: Mapping[str, Any],
        completion_response: Any,
        start_time: datetime | float | int,
        end_time: datetime | float | int,
    ) -> None:
        model_name = str(kwargs.get("model") or getattr(completion_response, "model", "unknown"))
        usage_payload = getattr(completion_response, "usage", None)
        if usage_payload is None and isinstance(completion_response, Mapping):
            usage_payload = completion_response.get("usage")
        usage = build_usage_snapshot(_coerce_mapping(usage_payload))
        latency_ms = _compute_latency_ms(start_time, end_time)
        response_cost = kwargs.get("response_cost")
        cost_usd = Decimal(str(response_cost)) if response_cost not in (None, "") else None
        event = UsageEvent(
            source="litellm",
            model=ModelString.parse(model_name).canonical(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.resolved_total_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw={
                "kwargs": dict(kwargs),
                "usage": usage.raw_usage,
            },
        )
        recorder.record(event, budget=budget)

    return _callback


def estimate_and_record_langchain_usage(
    recorder: UsageRecorder,
    *,
    model: str | ModelString,
    usage_metadata: Mapping[str, Any] | None,
    budget: BudgetPolicy | None = None,
    settings: Any = None,
    profile: Mapping[str, Any] | None = None,
) -> UsageEvent:
    """Estimate cost from LangChain usage metadata and record the result.

    Args:
        recorder: Recorder instance to update.
        model: Raw or typed model string.
        usage_metadata: LangChain usage metadata.
        budget: Optional budget policy.
        settings: Optional app settings used for LiteLLM enrichment.
        profile: Optional LangChain model profile.

    Returns:
        Recorded usage event.
    """
    meta = get_model_info(model=model, settings=settings, profile=profile)
    usage = build_usage_snapshot(usage_metadata)
    cost = calculate_cost(meta, usage)
    event = build_langchain_usage_event(
        model=model,
        usage_metadata=usage_metadata,
        cost_usd=cost,
    )
    return recorder.record(event, budget=budget)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Coerce a value into a plain dictionary when possible."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump())
    if hasattr(value, "dict"):
        return dict(value.dict())
    if hasattr(value, "__dict__"):
        return {key: val for key, val in vars(value).items() if not key.startswith("_")}
    return {}


def _compute_latency_ms(start_time: datetime | float | int, end_time: datetime | float | int) -> Decimal | None:
    """Return latency in milliseconds when both timestamps are usable."""
    try:
        if isinstance(start_time, datetime):
            start_value = start_time.timestamp()
        else:
            start_value = float(start_time)
        if isinstance(end_time, datetime):
            end_value = end_time.timestamp()
        else:
            end_value = float(end_time)
    except Exception:
        return None
    return Decimal(str((end_value - start_value) * 1000))
