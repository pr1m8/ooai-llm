"""LangChain cache bootstrap helpers.

Purpose:
    Provide small utilities for resolving a project-local LLM cache path and
    configuring the global LangChain cache.

Design:
    - Keep imports lazy so the module can be imported without LangChain.
    - Default to SQLite because it is file-backed, simple, and local.
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

    backend = settings.llm.cache.backend.strip().lower()
    if backend != "sqlite":
        raise ValueError(f"Unsupported LLM cache backend: {settings.llm.cache.backend!r}.")

    cache = build_sqlite_cache(settings, path=path)
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
