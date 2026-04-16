"""Typed model-string helpers.

Purpose:
    Provide a reusable Pydantic root model for provider/model strings that can
    be used across chat, embeddings, rerankers, and related model families.

Design:
    - Store the canonical raw model string as a ``RootModel[str]``.
    - Reuse provider normalization and inference logic from
      :mod:`ooai_llm.providers`.
    - Offer ergonomic class methods for parsing, construction, conversion to
      LangChain and LiteLLM naming styles, and provider enrichment.

Examples:
    >>> model = ModelString.parse("gpt-5.4-mini")
    >>> model.provider
    <Provider.OPENAI: 'openai'>
    >>> model.as_litellm()
    'openai/gpt-5.4-mini'
"""

from __future__ import annotations

from typing import Self

from pydantic import ConfigDict, RootModel, computed_field, field_validator

from .providers import (
    Provider,
    get_litellm_provider_prefix,
    infer_provider_from_model_name,
    normalize_provider_name,
    split_model_string,
)


class ModelString(RootModel[str]):
    """Root model for provider/model strings.

    Args:
        root: Raw model string, optionally provider-prefixed.

    Examples:
        >>> ModelString("openai:gpt-5.4-mini").model_name
        'gpt-5.4-mini'
        >>> ModelString.parse("anthropic/claude-sonnet-4").provider
        <Provider.ANTHROPIC: 'anthropic'>
    """

    model_config = ConfigDict(frozen=True)

    @field_validator("root")
    @classmethod
    def _validate_root(cls, value: str) -> str:
        """Normalize and validate the root string.

        Args:
            value: Incoming raw value.

        Returns:
            Normalized model string.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value is empty or malformed.
        """
        if not isinstance(value, str):
            raise TypeError("ModelString must wrap a string.")
        text = value.strip()
        if not text:
            raise ValueError("Model string cannot be empty.")
        if text.endswith(":") or text.endswith("/"):
            raise ValueError("Model string cannot end with a provider separator.")
        return text

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider(self) -> Provider | None:
        """Return the canonical provider when present or inferable."""
        explicit_provider, model_name = split_model_string(self.root)
        return explicit_provider or infer_provider_from_model_name(model_name)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def provider_prefix(self) -> str | None:
        """Return the canonical provider prefix string when available."""
        provider = self.provider
        return None if provider is None else str(provider)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def litellm_provider_prefix(self) -> str | None:
        """Return the canonical LiteLLM provider prefix when available."""
        return get_litellm_provider_prefix(self.provider)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        """Return the unprefixed model name."""
        _, model_name = split_model_string(self.root)
        return model_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_prefixed(self) -> bool:
        """Whether the model string includes an explicit provider prefix."""
        explicit_provider, _ = split_model_string(self.root)
        return explicit_provider is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_litellm_style(self) -> bool:
        """Whether the raw model string uses LiteLLM's slash separator."""
        return "/" in self.root and ":" not in self.root

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_langchain_style(self) -> bool:
        """Whether the raw model string uses LangChain's colon separator."""
        return ":" in self.root

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_bare(self) -> bool:
        """Whether the model string omits an explicit provider prefix."""
        return not self.is_prefixed

    @classmethod
    def parse(cls, value: str | Self) -> Self:
        """Parse a raw string or return the existing model-string instance.

        Args:
            value: Model string or model-string instance.

        Returns:
            Parsed model-string instance.
        """
        if isinstance(value, cls):
            return value
        return cls(value)

    @classmethod
    def from_parts(cls, model_name: str, *, provider: Provider | str | None = None) -> Self:
        """Build a model string from parts.

        Args:
            model_name: Bare model name.
            provider: Optional provider enum or alias.

        Returns:
            Constructed model-string instance.
        """
        normalized_provider = normalize_provider_name(provider)
        if normalized_provider is None:
            return cls(model_name)
        return cls(f"{normalized_provider}:{model_name}")

    @classmethod
    def from_litellm(cls, model_name: str) -> Self:
        """Parse a LiteLLM-style model string.

        Args:
            model_name: Provider-prefixed or bare LiteLLM model string.

        Returns:
            Canonical model-string instance using LangChain-style provider
            separators.
        """
        parsed = cls.parse(model_name)
        return parsed.as_langchain_model_string()

    @classmethod
    def infer(cls, model_name: str) -> Self:
        """Create a provider-prefixed model string when inference succeeds.

        Args:
            model_name: Bare or prefixed model string.

        Returns:
            Canonicalized model-string instance.
        """
        parsed = cls.parse(model_name)
        if parsed.is_prefixed:
            return parsed.as_langchain_model_string()
        inferred = parsed.provider
        if inferred is None:
            return parsed
        return cls.from_parts(parsed.model_name, provider=inferred)

    def split(self) -> tuple[Provider | None, str]:
        """Return the provider and model-name components.

        Returns:
            Tuple of ``(provider, model_name)``.
        """
        return self.provider, self.model_name

    def with_provider(self, provider: Provider | str) -> Self:
        """Return a provider-prefixed model string.

        Args:
            provider: Provider enum or alias.

        Returns:
            New provider-prefixed model-string instance.
        """
        normalized_provider = normalize_provider_name(provider)
        if normalized_provider is None:
            raise ValueError("Provider cannot be None.")
        return self.from_parts(self.model_name, provider=normalized_provider)

    def ensure_provider(self, provider: Provider | str | None = None) -> Self:
        """Return a provider-prefixed model string when possible.

        Args:
            provider: Explicit provider to apply. If omitted, provider
                inference is attempted.

        Returns:
            Provider-prefixed model-string instance when possible, otherwise the
            original value.
        """
        if provider is not None:
            return self.with_provider(provider)
        return self.canonical()

    def without_provider(self) -> str:
        """Return only the bare model name.

        Returns:
            Unprefixed model name.
        """
        return self.model_name

    def canonical(self) -> Self:
        """Return the canonical model string.

        Returns:
            Provider-prefixed model string when the provider can be inferred,
            otherwise the original string.
        """
        return self.infer(self.root)

    def as_langchain(self) -> str:
        """Return the string in LangChain-style ``provider:model`` form.

        Returns:
            LangChain-style model string when possible.
        """
        return str(self.as_langchain_model_string())

    def as_langchain_model_string(self) -> Self:
        """Return a LangChain-style typed model string.

        Returns:
            Canonical model string using ``provider:model`` when possible.
        """
        if self.provider is None:
            return self
        return self.from_parts(self.model_name, provider=self.provider)

    def as_litellm(self) -> str:
        """Return the string in LiteLLM-style ``provider/model`` form.

        Returns:
            LiteLLM-style model string when the provider is known, otherwise
            the bare model name.
        """
        if self.provider is None:
            return self.model_name
        prefix = self.litellm_provider_prefix or str(self.provider)
        return f"{prefix}/{self.model_name}"

    def __str__(self) -> str:
        """Return the raw model string."""
        return self.root
