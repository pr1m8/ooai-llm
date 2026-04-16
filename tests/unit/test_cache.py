"""Unit tests for cache helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ooai_llm.cache import build_sqlite_cache, configure_global_llm_cache, resolve_llm_cache_path
from ooai_llm.settings import AppSettings


def _install_fake_cache_modules(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Install fake LangChain cache modules for isolated cache tests."""
    captured: list[object] = []

    class FakeSQLiteCache:
        def __init__(self, *, database_path: str) -> None:
            self.database_path = database_path

    def fake_set_llm_cache(value: object) -> None:
        captured.append(value)

    fake_community_cache = types.ModuleType("langchain_community.cache")
    fake_community_cache.SQLiteCache = FakeSQLiteCache

    fake_core_globals = types.ModuleType("langchain_core.globals")
    fake_core_globals.set_llm_cache = fake_set_llm_cache

    monkeypatch.setitem(sys.modules, "langchain_community.cache", fake_community_cache)
    monkeypatch.setitem(sys.modules, "langchain_core.globals", fake_core_globals)
    return captured


@pytest.mark.unit
def test_resolve_llm_cache_path_explicit(tmp_path: Path) -> None:
    """It should resolve explicit cache-path overrides."""
    settings = AppSettings(app_root=tmp_path)
    path = resolve_llm_cache_path(settings, path=tmp_path / "custom.sqlite3")
    assert path == (tmp_path / "custom.sqlite3").resolve()


@pytest.mark.unit
def test_build_sqlite_cache_creates_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should create parent directories for the SQLite cache."""
    _install_fake_cache_modules(monkeypatch)
    settings = AppSettings(app_root=tmp_path)
    cache = build_sqlite_cache(settings)
    assert cache is not None
    assert settings.default_llm_cache_path.parent.exists()
    assert cache.database_path == str(settings.default_llm_cache_path)


@pytest.mark.unit
def test_configure_global_llm_cache_disabled_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should skip cache installation when disabled."""
    captured = _install_fake_cache_modules(monkeypatch)
    settings = AppSettings(app_root=tmp_path, llm={"cache": {"enabled": False}})
    assert configure_global_llm_cache(settings) is None
    assert captured == [None]


@pytest.mark.unit
def test_configure_global_llm_cache_rejects_unknown_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should reject unsupported cache backends."""
    _install_fake_cache_modules(monkeypatch)
    settings = AppSettings(app_root=tmp_path, llm={"cache": {"backend": "redis"}})
    with pytest.raises(ValueError):
        configure_global_llm_cache(settings)


@pytest.mark.unit
def test_configure_global_llm_cache_installs_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should install the fake cache globally when enabled."""
    captured = _install_fake_cache_modules(monkeypatch)
    settings = AppSettings(app_root=tmp_path)
    cache = configure_global_llm_cache(settings)
    assert cache is captured[-1]
    assert cache.database_path.endswith("langchain_llm_cache.sqlite3")
