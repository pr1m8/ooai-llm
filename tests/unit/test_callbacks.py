"""Unit tests for usage and budget callback helpers."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from ooai_llm import BudgetExceededError, BudgetPolicy, UsageRecorder
from ooai_llm.callbacks import build_langchain_usage_event, make_litellm_cost_callback


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
