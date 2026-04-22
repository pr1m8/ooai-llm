"""Unit tests for usage and budget callback helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import sys
from types import SimpleNamespace
import types

import pytest

from ooai_llm import AppSettings, BudgetExceededError, BudgetPolicy, UsageRecorder
from ooai_llm.callbacks import (
    build_langchain_usage_event,
    estimate_and_record_langchain_usage,
    make_litellm_cost_callback,
)


@pytest.mark.unit
def test_build_langchain_usage_event_normalizes_usage() -> None:
    """It should normalize LangChain usage into a reusable event."""
    event = build_langchain_usage_event(
        model="openai:gpt-5.4-mini",
        usage_metadata={"input_tokens": 100, "output_tokens": 25},
        cost_usd=Decimal("0.01"),
    )

    assert event.model.as_langchain() == "openai:gpt-5.4-mini"
    assert event.total_tokens == 125
    assert event.cost_usd == Decimal("0.01")


@pytest.mark.unit
def test_litellm_callback_records_event() -> None:
    """It should record LiteLLM success events with cost and usage."""
    recorder = UsageRecorder()
    callback = make_litellm_cost_callback(recorder)

    response = SimpleNamespace(
        model="openai/gpt-5.4-mini",
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150),
    )
    callback(
        {"model": "openai/gpt-5.4-mini", "response_cost": "0.0123"},
        response,
        10.0,
        10.5,
    )

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.total_tokens == 150
    assert event.cost_usd == Decimal("0.0123")
    assert event.latency_ms == Decimal("500.0")


@pytest.mark.unit
def test_budget_policy_warns_and_errors() -> None:
    """It should warn on soft thresholds and raise on hard thresholds."""
    recorder = UsageRecorder()
    callback = make_litellm_cost_callback(
        recorder,
        budget=BudgetPolicy(warn_total_tokens=50, error_cost_usd=Decimal("0.02")),
    )

    response = SimpleNamespace(
        model="openai/gpt-5.4-mini",
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=20, total_tokens=60),
    )
    callback(
        {"model": "openai/gpt-5.4-mini", "response_cost": "0.01"},
        response,
        0.0,
        0.1,
    )
    assert recorder.warnings

    with pytest.raises(BudgetExceededError):
        callback(
            {"model": "openai/gpt-5.4-mini", "response_cost": "0.03"},
            response,
            0.0,
            0.1,
        )


@pytest.mark.unit
def test_recorder_totals_and_budget_cost_warning() -> None:
    """It should expose aggregate token and cost totals."""
    recorder = UsageRecorder()
    event = build_langchain_usage_event(
        model="openai:gpt-5.4-mini",
        usage_metadata={"input_tokens": 10, "output_tokens": 5},
        cost_usd=Decimal("0.03"),
    )

    recorder.record(event, budget=BudgetPolicy(warn_cost_usd=Decimal("0.02")))

    assert recorder.total_tokens == 15
    assert recorder.total_cost_usd == Decimal("0.03")
    assert "Cost warning threshold exceeded" in recorder.warnings[0]


@pytest.mark.unit
def test_budget_policy_errors_on_total_tokens() -> None:
    """It should raise when a hard token limit is crossed."""
    event = build_langchain_usage_event(
        model="openai:gpt-5.4-mini",
        usage_metadata={"input_tokens": 10, "output_tokens": 5},
    )

    with pytest.raises(BudgetExceededError, match="Total token budget exceeded"):
        BudgetPolicy(error_total_tokens=10).check(event)


@pytest.mark.unit
def test_litellm_callback_accepts_mapping_response_and_datetime_latency() -> None:
    """It should handle mapping-style LiteLLM responses."""
    recorder = UsageRecorder()
    callback = make_litellm_cost_callback(recorder)

    callback(
        {"model": "deepseek/deepseek-chat"},
        {"usage": {"input_tokens": 20, "output_tokens": 7, "total_tokens": 27}},
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 0, 0, 1, 250000, tzinfo=UTC),
    )

    event = recorder.events[0]
    assert event.input_tokens == 20
    assert event.output_tokens == 7
    assert event.cost_usd is None
    assert event.latency_ms == Decimal("1250.0")


@pytest.mark.unit
def test_litellm_callback_handles_invalid_latency() -> None:
    """It should leave latency empty when timestamps cannot be coerced."""
    recorder = UsageRecorder()
    callback = make_litellm_cost_callback(recorder)

    callback(
        {"model": "openai/gpt-5.4-mini"},
        {"usage": {"input_tokens": 1, "output_tokens": 1}},
        "not-a-time",
        1.0,
    )

    assert recorder.events[0].latency_ms is None


@pytest.mark.unit
def test_estimate_and_record_langchain_usage_uses_litellm_pricing(monkeypatch) -> None:
    """It should estimate LangChain event cost from resolved pricing."""
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.4-mini": {
            "input_cost_per_token": "0.01",
            "output_cost_per_token": "0.02",
        }
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    recorder = UsageRecorder()
    event = estimate_and_record_langchain_usage(
        recorder,
        model="openai:gpt-5.4-mini",
        usage_metadata={"input_tokens": 3, "output_tokens": 4},
        settings=AppSettings(),
    )

    assert event.cost_usd == Decimal("0.11")
    assert recorder.total_cost_usd == Decimal("0.11")
