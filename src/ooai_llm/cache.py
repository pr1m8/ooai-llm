"""LangChain cache bootstrap helpers.

Purpose:
    Provide small utilities for resolving a project-local LLM cache path and
    configuring the global LangChain cache.

Design:
    - Keep imports lazy so the module can be imported without LangChain.
    - Default to SQLite because it is file-backed, simple, and local.
    - Support networked caches through optional Redis and Upstash Redis clients.
    - Reuse :class:`ooai_llm.settings.AppSettings` for path resolution.

Examples:
    >>> from pathlib import Path
    >>> settings = AppSettings(app_root=Path.cwd())
    >>> resolve_llm_cache_path(settings).name
    'langchain_llm_cache.sqlite3'
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .settings import AppSettings

if TYPE_CHECKING:
    from langchain_core.caches import BaseCache


def resolve_llm_cache_path(settings: AppSettings, *, path: str | Path | None = None) -> Path:
    """Resolve the effective LLM cache path.

    Args:
        settings: Application settings.
        path: Optional explicit cache path override.

    Returns:
        Resolved cache path.
    """
    if path is None:
        return settings.default_llm_cache_path
    return Path(path).expanduser().resolve()


def build_sqlite_cache(settings: AppSettings, *, path: str | Path | None = None) -> Any:
    """Build a SQLite-backed LangChain cache.

    Args:
        settings: Application settings.
        path: Optional explicit cache path override.

    Returns:
        SQLite cache instance.

    Raises:
        ImportError: If ``langchain_community`` is not installed.
    """
    from langchain_community.cache import SQLiteCache

    resolved_path = resolve_llm_cache_path(settings, path=path)
    if settings.llm.cache.create_dirs:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteCache(database_path=str(resolved_path))


def build_memory_cache() -> Any:
    """Build an in-memory LangChain cache.

    Returns:
        In-memory cache instance.

    Raises:
        ImportError: If ``langchain_community`` is not installed.
    """
    from langchain_community.cache import InMemoryCache

    return InMemoryCache()


def build_sqlalchemy_cache(settings: AppSettings, *, path: str | Path | None = None) -> Any:
    """Build a SQLAlchemy-backed LangChain cache.

    Args:
        settings: Application settings.
        path: Optional SQLite path used when ``sqlalchemy_url`` is unset.

    Returns:
        SQLAlchemy cache instance.

    Raises:
        ImportError: If SQLAlchemy or LangChain community caches are missing.
    """
    from langchain_community.cache import SQLAlchemyCache
    from sqlalchemy import create_engine

    sqlalchemy_url = settings.llm.cache.sqlalchemy_url
    if sqlalchemy_url is None:
        resolved_path = resolve_llm_cache_path(settings, path=path)
        if settings.llm.cache.create_dirs:
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
        sqlalchemy_url = f"sqlite:///{resolved_path}"

    return SQLAlchemyCache(engine=create_engine(sqlalchemy_url))


def build_redis_cache(settings: AppSettings) -> Any:
    """Build a Redis-backed LangChain cache.

    Args:
        settings: Application settings.

    Returns:
        Redis cache instance.

    Raises:
        ImportError: If ``redis`` or LangChain community caches are missing.
    """
    from langchain_community.cache import RedisCache

    try:
        import redis
    except ImportError as exc:  # pragma: no cover - exercised with dependency absent
        raise ImportError("Install `ooai-llm[redis]` or `redis` to use the Redis cache backend.") from exc

    cache_settings = settings.llm.cache
    connection_kwargs = dict(cache_settings.redis_connection_kwargs)
    if cache_settings.redis_url:
        client = redis.Redis.from_url(cache_settings.redis_url, **connection_kwargs)
    else:
        if cache_settings.redis_username is not None:
            connection_kwargs.setdefault("username", cache_settings.redis_username)
        if cache_settings.redis_password is not None:
            connection_kwargs.setdefault("password", cache_settings.redis_password.get_secret_value())
        client = redis.Redis(
            host=cache_settings.redis_host,
            port=cache_settings.redis_port,
            db=cache_settings.redis_db,
            ssl=cache_settings.redis_ssl,
            **connection_kwargs,
        )
    return RedisCache(redis_=client, ttl=cache_settings.ttl)


def build_upstash_redis_cache(settings: AppSettings) -> Any:
    """Build an Upstash Redis-backed LangChain cache.

    Args:
        settings: Application settings.

    Returns:
        Upstash Redis cache instance.

    Raises:
        ValueError: If the Upstash URL or token is missing.
        ImportError: If ``upstash_redis`` or LangChain community caches are missing.
    """
    from langchain_community.cache import UpstashRedisCache

    try:
        from upstash_redis import Redis
    except ImportError as exc:  # pragma: no cover - exercised with dependency absent
        raise ImportError(
            "Install `ooai-llm[upstash]` or `upstash-redis` to use the Upstash Redis cache backend."
        ) from exc

    cache_settings = settings.llm.cache
    if not cache_settings.upstash_url:
        raise ValueError("Upstash Redis cache backend requires `llm.cache.upstash_url`.")
    if cache_settings.upstash_token is None:
        raise ValueError("Upstash Redis cache backend requires `llm.cache.upstash_token`.")

    client = Redis(url=cache_settings.upstash_url, token=cache_settings.upstash_token.get_secret_value())
    return UpstashRedisCache(redis_=client, ttl=cache_settings.ttl)


def build_llm_cache(settings: AppSettings, *, path: str | Path | None = None) -> Any:
    """Build the configured LangChain cache backend.

    Args:
        settings: Application settings.
        path: Optional SQLite path override for file-backed backends.

    Returns:
        Cache object.

    Raises:
        ValueError: If the configured backend is unsupported.
        ImportError: If backend-specific dependencies are missing.
    """
    backend = settings.llm.cache.backend.strip().lower().replace("-", "_")
    if backend in {"sqlite", "sqlite3"}:
        return build_sqlite_cache(settings, path=path)
    if backend in {"memory", "in_memory", "inmemory"}:
        return build_memory_cache()
    if backend == "sqlalchemy":
        return build_sqlalchemy_cache(settings, path=path)
    if backend == "redis":
        return build_redis_cache(settings)
    if backend in {"upstash", "upstash_redis"}:
        return build_upstash_redis_cache(settings)

    supported = "sqlite, memory, sqlalchemy, redis, upstash_redis"
    raise ValueError(
        f"Unsupported LLM cache backend: {settings.llm.cache.backend!r}. "
        f"Supported backends: {supported}."
    )


def configure_global_llm_cache(
    settings: AppSettings,
    *,
    path: str | Path | None = None,
) -> Any:
    """Configure the global LangChain LLM cache.

    Args:
        settings: Application settings.
        path: Optional explicit cache path override.

    Returns:
        Cache object when enabled, otherwise ``None``.

    Raises:
        ValueError: If the configured backend is unsupported.
        ImportError: If required LangChain cache packages are not installed.
    """
    from langchain_core.globals import set_llm_cache

    if not settings.llm.cache.enabled:
        set_llm_cache(None)
        return None

    cache = build_llm_cache(settings, path=path)
    set_llm_cache(cache)
    return cache


def normalize_cache_argument(cache: "BaseCache | bool | None") -> "BaseCache | bool | None":
    """Normalize the per-model ``cache`` argument.

    Args:
        cache: ``True`` to force the global cache, ``False`` to disable,
            ``None`` to inherit the global setting, or a concrete cache object.

    Returns:
        Unchanged normalized cache value.
    """
    return cache
