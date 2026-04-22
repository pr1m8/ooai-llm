"""Shared pytest configuration for ``ooai_llm`` tests."""

from __future__ import annotations

import os

import pytest


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep live tests opt-in even when local provider keys exist in ``.env``."""
    marker_expression = (config.option.markexpr or "").lower()
    live_requested = "live" in marker_expression or _truthy(os.environ.get("OOAI_RUN_LIVE"))
    if live_requested:
        return

    skip_live = pytest.mark.skip(reason="live tests are opt-in; run pytest -m live to enable them")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
