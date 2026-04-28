"""Unit tests for the command-line interface."""

from __future__ import annotations

import sys
import types

import pytest

from ooai_llm.cli import main


def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.model_cost = {
        "openai/gpt-5.5": {
            "mode": "chat",
            "input_cost_per_token": "0.000005",
            "output_cost_per_token": "0.000030",
        },
        "openai/gpt-5.5-pro": {
            "mode": "chat",
            "input_cost_per_token": "0.000030",
            "output_cost_per_token": "0.000180",
        },
        "openai/gpt-5.4-nano": {
            "mode": "chat",
            "input_cost_per_token": "0.0000002",
            "output_cost_per_token": "0.0000010",
        },
    }
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)


@pytest.mark.unit
def test_models_update_cli_prints_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """It should print refreshed model defaults as JSON."""
    _install_fake_litellm(monkeypatch)

    exit_code = main(
        [
            "models",
            "update",
            "--source",
            "litellm",
            "--providers",
            "openai",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"default_model": "openai:gpt-5.5"' in captured.out
    assert '"latest": "openai:gpt-5.5"' in captured.out
    assert captured.err == ""


@pytest.mark.unit
def test_models_update_cli_writes_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    """It should write refreshed model defaults as nested env vars."""
    _install_fake_litellm(monkeypatch)
    output_path = tmp_path / "models.env"

    exit_code = main(
        [
            "models",
            "update",
            "--source",
            "litellm",
            "--provider",
            "openai",
            "--format",
            "env",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    env_text = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert captured.out == ""
    assert f"Wrote model defaults to {output_path}" in captured.err
    assert "OOAI_LLM__DEFAULT_MODEL=openai:gpt-5.5" in env_text
    assert "OOAI_LLM__ALIASES__LATEST=openai:gpt-5.5" in env_text
