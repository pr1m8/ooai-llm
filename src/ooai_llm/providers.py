"""Provider normalization helpers.

Purpose:
    Normalize provider names, aliases, and package metadata used across the
    package.

Design:
    - Use a canonical provider enum that matches the provider strings commonly
      passed to LangChain and other SDKs.
    - Keep alias normalization and model-name inference in one place so both
      settings and typed model-string utilities reuse the same logic.
    - Expose native environment-variable names, LiteLLM provider prefixes,
      native SDK package names, and extra names for installation guidance.

Important mappings:
    PROVIDER_ALIASES: Mapping from common aliases to canonical providers.
    PROVIDER_API_KEY_ENV_VARS: Native environment-variable names by provider.
    PROVIDER_LITELLM_PREFIXES: Native LiteLLM provider prefixes.
    PROVIDER_NATIVE_SDK_PACKAGES: Official or primary Python SDK package names.
    PROVIDER_LANGCHAIN_PACKAGES: LangChain integration package names.
    PROVIDER_EXTRAS: Optional-dependency extra names by provider.

Examples:
    >>> normalize_provider_name("claude") == Provider.ANTHROPIC
    True
    >>> infer_provider_from_model_name("gpt-5.4-mini") == Provider.OPENAI
    True
    >>> get_litellm_provider_prefix("google")
    'gemini'
"""

from __future__ import annotations

from enum import StrEnum


class Provider(StrEnum):
    """Canonical provider identifiers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE_GENAI = "google_genai"
    XAI = "xai"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"


PROVIDER_ALIASES: dict[str, Provider] = {
    "openai": Provider.OPENAI,
    "oai": Provider.OPENAI,
    "gpt": Provider.OPENAI,
    "anthropic": Provider.ANTHROPIC,
    "claude": Provider.ANTHROPIC,
    "google": Provider.GOOGLE_GENAI,
    "gemini": Provider.GOOGLE_GENAI,
    "google_genai": Provider.GOOGLE_GENAI,
    "google-vertexai": Provider.GOOGLE_GENAI,
    "google_vertexai": Provider.GOOGLE_GENAI,
    "vertex_ai": Provider.GOOGLE_GENAI,
    "vertexai": Provider.GOOGLE_GENAI,
    "xai": Provider.XAI,
    "grok": Provider.XAI,
    "deepseek": Provider.DEEPSEEK,
    "mistral": Provider.MISTRAL,
    "mistralai": Provider.MISTRAL,
    "ministral": Provider.MISTRAL,
    "magistral": Provider.MISTRAL,
    "devstral": Provider.MISTRAL,
    "pixtral": Provider.MISTRAL,
    "codestral": Provider.MISTRAL,
}

PROVIDER_API_KEY_ENV_VARS: dict[Provider, str] = {
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.GOOGLE_GENAI: "GOOGLE_API_KEY",
    Provider.XAI: "XAI_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.MISTRAL: "MISTRAL_API_KEY",
}

PROVIDER_LITELLM_PREFIXES: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE_GENAI: "gemini",
    Provider.XAI: "xai",
    Provider.DEEPSEEK: "deepseek",
    Provider.MISTRAL: "mistral",
}

PROVIDER_NATIVE_SDK_PACKAGES: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE_GENAI: "google-genai",
    Provider.XAI: "xai-sdk",
    Provider.DEEPSEEK: "openai",
    Provider.MISTRAL: "mistralai",
}

PROVIDER_LANGCHAIN_PACKAGES: dict[Provider, str] = {
    Provider.OPENAI: "langchain-openai",
    Provider.ANTHROPIC: "langchain-anthropic",
    Provider.GOOGLE_GENAI: "langchain-google-genai",
    Provider.XAI: "langchain-xai",
    Provider.DEEPSEEK: "langchain-deepseek",
    Provider.MISTRAL: "langchain-mistralai",
}

PROVIDER_EXTRAS: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.ANTHROPIC: "anthropic",
    Provider.GOOGLE_GENAI: "google",
    Provider.XAI: "xai",
    Provider.DEEPSEEK: "deepseek",
    Provider.MISTRAL: "mistral",
}


def normalize_provider_name(provider: Provider | str | None) -> Provider | None:
    """Normalize a provider alias to its canonical enum value.

    Args:
        provider: Provider enum, alias, or ``None``.

    Returns:
        Canonical provider enum or ``None`` when ``provider`` is absent.

    Raises:
        ValueError: If the provider value is unknown.
    """
    if provider is None:
        return None
    if isinstance(provider, Provider):
        return provider

    normalized = provider.strip().lower()
    if normalized in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[normalized]
    raise ValueError(f"Unknown provider: {provider!r}.")


def get_litellm_provider_prefix(provider: Provider | str | None) -> str | None:
    """Return the canonical LiteLLM provider prefix for a provider.

    Args:
        provider: Canonical provider or alias.

    Returns:
        LiteLLM provider prefix or ``None`` when ``provider`` is absent.
    """
    normalized = normalize_provider_name(provider)
    if normalized is None:
        return None
    return PROVIDER_LITELLM_PREFIXES[normalized]


def infer_provider_from_model_name(model_name: str) -> Provider | None:
    """Infer a provider from a raw model name.

    Args:
        model_name: Raw or provider-prefixed model string.

    Returns:
        Inferred provider enum, or ``None`` when inference is not possible.
    """
    text = model_name.strip().lower()
    if not text:
        return None

    candidate_provider, remainder = split_model_string(text)
    if candidate_provider is not None:
        return candidate_provider
    text = remainder

    if text.startswith(("gpt-", "o1", "o3", "o4", "gpt5", "text-embedding-")):
        return Provider.OPENAI
    if text.startswith("claude"):
        return Provider.ANTHROPIC
    if text.startswith("gemini"):
        return Provider.GOOGLE_GENAI
    if text.startswith("grok"):
        return Provider.XAI
    if text.startswith("deepseek"):
        return Provider.DEEPSEEK
    if text.startswith(("mistral", "ministral", "magistral", "devstral", "pixtral", "codestral")):
        return Provider.MISTRAL
    return None


def split_model_string(model: str) -> tuple[Provider | None, str]:
    """Split a model string into provider and model parts.

    Args:
        model: Provider-prefixed or bare model string.

    Returns:
        A tuple of ``(provider, model_name)`` where provider may be ``None``.

    Examples:
        >>> split_model_string("openai:gpt-5.4")
        (<Provider.OPENAI: 'openai'>, 'gpt-5.4')
        >>> split_model_string("anthropic/claude-sonnet-4")
        (<Provider.ANTHROPIC: 'anthropic'>, 'claude-sonnet-4')
    """
    text = model.strip()
    for sep in (":", "/"):
        if sep in text:
            provider_text, model_name = text.split(sep, 1)
            try:
                provider = normalize_provider_name(provider_text)
            except ValueError:
                continue
            return provider, model_name
    return None, text
