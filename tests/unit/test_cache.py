"""Unit tests for cache helpers."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from ooai_llm.cache import (
    build_llm_cache,
    build_redis_cache,
    build_sqlite_cache,
    build_upstash_redis_cache,
    configure_global_llm_cache,
    resolve_llm_cache_path,
)
from ooai_llm.settings import AppSettings


def _install_fake_cache_modules(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Install fake LangChain cache modules for isolated cache tests."""
    captured: list[object] = []

    class FakeSQLiteCache:
        def __init__(self, *, database_path: str) -> None:
            self.database_path = database_path

    class FakeInMemoryCache:
        pass

    class FakeRedisCache:
        def __init__(self, *, redis_: object, ttl: int | None = None) -> None:
            self.redis = redis_
            self.ttl = ttl

    class FakeUpstashRedisCache:
        def __init__(self, *, redis_: object, ttl: int | None = None) -> None:
            self.redis = redis_
            self.ttl = ttl

    def fake_set_llm_cache(value: object) -> None:
        captured.append(value)

    fake_community_cache = types.ModuleType("langchain_community.cache")
    fake_community_cache.SQLiteCache = FakeSQLiteCache
    fake_community_cache.InMemoryCache = FakeInMemoryCache
    fake_community_cache.RedisCache = FakeRedisCache
    fake_community_cache.UpstashRedisCache = FakeUpstashRedisCache

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
    settings = AppSettings(app_root=tmp_path, llm={"cache": {"backend": "missing"}})
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


@pytest.mark.unit
def test_build_memory_cache_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should build the in-memory backend."""
    _install_fake_cache_modules(monkeypatch)
    settings = AppSettings(app_root=tmp_path, llm={"cache": {"backend": "memory"}})

    cache = build_llm_cache(settings)

    assert type(cache).__name__ == "FakeInMemoryCache"


@pytest.mark.unit
def test_build_redis_cache_from_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should build Redis caches without exposing secret values."""
    _install_fake_cache_modules(monkeypatch)

    class FakeRedisClient:
        @classmethod
        def from_url(cls, url: str, **kwargs: object):
            return {"url": url, "kwargs": kwargs}

    fake_redis = types.ModuleType("redis")
    fake_redis.Redis = FakeRedisClient
    monkeypatch.setitem(sys.modules, "redis", fake_redis)

    settings = AppSettings(
        app_root=tmp_path,
        llm={
            "cache": {
                "backend": "redis",
                "redis_url": "redis://localhost:6379/0",
                "ttl": 60,
                "redis_connection_kwargs": {"decode_responses": True},
            }
        },
    )

    cache = build_redis_cache(settings)

    assert cache.ttl == 60
    assert cache.redis == {
        "url": "redis://localhost:6379/0",
        "kwargs": {"decode_responses": True},
    }


@pytest.mark.unit
def test_build_redis_cache_from_host_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should build Redis clients from host-style settings."""
    _install_fake_cache_modules(monkeypatch)

    class FakeRedisClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_redis = types.ModuleType("redis")
    fake_redis.Redis = FakeRedisClient
    monkeypatch.setitem(sys.modules, "redis", fake_redis)

    settings = AppSettings(
        app_root=tmp_path,
        llm={
            "cache": {
                "backend": "redis",
                "redis_host": "redis.local",
                "redis_port": 6380,
                "redis_db": 2,
                "redis_username": "cache-user",
                "redis_password": "cache-secret",
                "redis_ssl": True,
            }
        },
    )

    cache = build_redis_cache(settings)

    assert cache.redis.kwargs["host"] == "redis.local"
    assert cache.redis.kwargs["port"] == 6380
    assert cache.redis.kwargs["db"] == 2
    assert cache.redis.kwargs["username"] == "cache-user"
    assert cache.redis.kwargs["password"] == "cache-secret"
    assert cache.redis.kwargs["ssl"] is True


@pytest.mark.unit
def test_build_upstash_redis_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should build Upstash Redis caches from REST credentials."""
    _install_fake_cache_modules(monkeypatch)

    class FakeUpstashClient:
        def __init__(self, *, url: str, token: str) -> None:
            self.url = url
            self.token = token

    fake_upstash = types.ModuleType("upstash_redis")
    fake_upstash.Redis = FakeUpstashClient
    monkeypatch.setitem(sys.modules, "upstash_redis", fake_upstash)

    settings = AppSettings(
        app_root=tmp_path,
        llm={
            "cache": {
                "backend": "upstash_redis",
                "upstash_url": "https://upstash.example",
                "upstash_token": "upstash-secret",
                "ttl": 120,
            }
        },
    )

    cache = build_upstash_redis_cache(settings)

    assert cache.ttl == 120
    assert cache.redis.url == "https://upstash.example"
    assert cache.redis.token == "upstash-secret"


@pytest.mark.unit
def test_build_upstash_requires_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It should fail clearly when Upstash credentials are incomplete."""
    _install_fake_cache_modules(monkeypatch)

    fake_upstash = types.ModuleType("upstash_redis")
    fake_upstash.Redis = object
    monkeypatch.setitem(sys.modules, "upstash_redis", fake_upstash)

    settings = AppSettings(app_root=tmp_path, llm={"cache": {"backend": "upstash_redis"}})

    with pytest.raises(ValueError, match="upstash_url"):
        build_upstash_redis_cache(settings)
