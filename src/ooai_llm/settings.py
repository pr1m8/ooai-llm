"""Settings models for ``ooai_llm``.

Purpose:
    Centralize application configuration for provider credentials, default
    model selection, and LangChain LLM cache behavior.

Design:
    - Accept both app-prefixed and provider-native environment variables.
    - Keep model defaults configurable through semantic aliases and per-provider
      preset bundles.
    - Expose the application directory and default cache path as computed
      values derived from the working directory.
    - Make model-string defaults reusable through a typed ``ModelString``.

Type aliases:
    ModelPresetName: Semantic preset names available per provider.
    ModelAliasName: Global semantic aliases.

Examples:
    >>> settings = AppSettings()
    >>> settings.resolve_model(alias="testing")
    'openai:gpt-5.4-nano'
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .providers import Provider, PROVIDER_API_KEY_ENV_VARS, normalize_provider_name
from .types import ModelString

ModelPresetName = Literal[
    "default",
    "cheap",
    "testing",
    "fast",
    "balanced",
    "reasoning",
    "coding",
    "vision",
]

ModelAliasName = ModelPresetName


def _load_dotenv_values() -> dict[str, str]:
    """Load non-empty values from a local ``.env`` file when available."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return {}

    values = dotenv_values(".env")
    return {key: str(value) for key, value in values.items() if key and value}


class ProviderCredentials(BaseModel):
    """Provider credentials with native-env fallback aliases.

    Args:
        openai_api_key: OpenAI API key.
        anthropic_api_key: Anthropic API key.
        google_api_key: Google or Gemini API key.
        xai_api_key: xAI API key.
        deepseek_api_key: DeepSeek API key.
        mistral_api_key: Mistral API key.
        google_use_vertexai: Whether to route Gemini through Vertex AI.
        google_cloud_project: Optional Google Cloud project ID.
        google_cloud_location: Optional Google Cloud location.

    Examples:
        >>> creds = ProviderCredentials()
        >>> isinstance(creds, ProviderCredentials)
        True
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    xai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_XAI_API_KEY", "XAI_API_KEY"),
    )
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    )
    mistral_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
    )
    google_use_vertexai: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_GOOGLE_USE_VERTEXAI", "GOOGLE_GENAI_USE_VERTEXAI"),
    )
    google_cloud_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT"),
    )
    google_cloud_location: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OOAI_GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_LOCATION"),
    )

    @classmethod
    def from_environment(cls) -> Self:
        """Build provider credentials from native and app-scoped env vars.

        Returns:
            Credentials populated from environment variables.
        """
        dotenv_values = _load_dotenv_values()

        def first(*names: str) -> str | None:
            for name in names:
                value = os.environ.get(name) or dotenv_values.get(name)
                if value:
                    return value
            return None

        google_vertexai = first("OOAI_GOOGLE_USE_VERTEXAI", "GOOGLE_GENAI_USE_VERTEXAI")

        return cls(
            openai_api_key=first("OOAI_OPENAI_API_KEY", "OPENAI_API_KEY"),
            anthropic_api_key=first("OOAI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
            google_api_key=first("OOAI_GOOGLE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"),
            xai_api_key=first("OOAI_XAI_API_KEY", "XAI_API_KEY"),
            deepseek_api_key=first("OOAI_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
            mistral_api_key=first("OOAI_MISTRAL_API_KEY", "MISTRAL_API_KEY"),
            google_use_vertexai=google_vertexai.lower() in {"1", "true", "yes", "on"}
            if google_vertexai is not None
            else None,
            google_cloud_project=first("OOAI_GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_PROJECT"),
            google_cloud_location=first("OOAI_GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_LOCATION"),
        )

    def get_api_key(self, provider: Provider | str) -> str | None:
        """Return the resolved API key for a provider.

        Args:
            provider: Canonical provider or provider alias.

        Returns:
            The plain API key string, or ``None`` when unavailable.
        """
        resolved = normalize_provider_name(provider)
        if resolved is None:
            return None

        mapping: dict[Provider, SecretStr | None] = {
            Provider.OPENAI: self.openai_api_key,
            Provider.ANTHROPIC: self.anthropic_api_key,
            Provider.GOOGLE_GENAI: self.google_api_key,
            Provider.XAI: self.xai_api_key,
            Provider.DEEPSEEK: self.deepseek_api_key,
            Provider.MISTRAL: self.mistral_api_key,
        }
        value = mapping[resolved]
        return value.get_secret_value() if value is not None else None

    def require_api_key(self, provider: Provider | str) -> str:
        """Return the API key for a provider or raise.

        Args:
            provider: Canonical provider or provider alias.

        Returns:
            The plain API key string.

        Raises:
            ValueError: If no API key is configured for the provider.
        """
        resolved = normalize_provider_name(provider)
        if resolved is None:
            raise ValueError(f"Unknown provider: {provider!r}.")

        api_key = self.get_api_key(resolved)
        if api_key is None:
            raise ValueError(
                f"No API key configured for provider {resolved.value!r}. "
                f"Set {PROVIDER_API_KEY_ENV_VARS[resolved]!r} or its OOAI_ equivalent."
            )
        return api_key

    def to_native_environment(self) -> dict[str, str]:
        """Return provider-native environment variables.

        Returns:
            Mapping of native environment-variable names to string values.
        """
        env: dict[str, str] = {}
        if self.openai_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.OPENAI]] = self.openai_api_key.get_secret_value()
        if self.anthropic_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.ANTHROPIC]] = self.anthropic_api_key.get_secret_value()
        if self.google_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.GOOGLE_GENAI]] = self.google_api_key.get_secret_value()
        if self.xai_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.XAI]] = self.xai_api_key.get_secret_value()
        if self.deepseek_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.DEEPSEEK]] = self.deepseek_api_key.get_secret_value()
        if self.mistral_api_key is not None:
            env[PROVIDER_API_KEY_ENV_VARS[Provider.MISTRAL]] = self.mistral_api_key.get_secret_value()
        if self.google_use_vertexai is not None:
            env["GOOGLE_GENAI_USE_VERTEXAI"] = "true" if self.google_use_vertexai else "false"
        if self.google_cloud_project:
            env["GOOGLE_CLOUD_PROJECT"] = self.google_cloud_project
        if self.google_cloud_location:
            env["GOOGLE_CLOUD_LOCATION"] = self.google_cloud_location
        return env


class ProviderModelPresets(BaseModel):
    """Provider-specific model presets.

    Args:
        default: Recommended default model for general use.
        cheap: Lowest-cost reasonable default.
        testing: Lightweight default for development and smoke tests.
        fast: Lower-latency model.
        balanced: Balanced cost/performance model.
        reasoning: Stronger reasoning-oriented model.
        coding: Coding-oriented model.
        vision: Vision-capable model.

    Examples:
        >>> presets = ProviderModelPresets(
        ...     default="openai:gpt-5.4-mini",
        ...     cheap="openai:gpt-5.4-nano",
        ...     testing="openai:gpt-5.4-nano",
        ...     fast="openai:gpt-5.4-mini",
        ...     balanced="openai:gpt-5.4-mini",
        ...     reasoning="openai:gpt-5.4",
        ...     coding="openai:gpt-5.4",
        ...     vision="openai:gpt-5.4-mini",
        ... )
        >>> presets.get("cheap")
        'openai:gpt-5.4-nano'
    """

    model_config = ConfigDict(extra="forbid")

    default: str
    cheap: str
    testing: str
    fast: str
    balanced: str
    reasoning: str
    coding: str
    vision: str

    def get(self, preset: ModelPresetName) -> str:
        """Return the model configured for a preset.

        Args:
            preset: Preset name.

        Returns:
            Configured model string.
        """
        return getattr(self, preset)


class DefaultModelsByProvider(BaseModel):
    """Default model presets keyed by provider.

    Notes:
        These defaults are intentionally practical application defaults. They
        are not a replacement for a live model catalog or dynamic router.

    Examples:
        >>> defaults = DefaultModelsByProvider()
        >>> defaults.openai.cheap
        'openai:gpt-5.4-nano'
    """

    model_config = ConfigDict(extra="forbid")

    openai: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="openai:gpt-5.4-mini",
            cheap="openai:gpt-5.4-nano",
            testing="openai:gpt-5.4-nano",
            fast="openai:gpt-5.4-mini",
            balanced="openai:gpt-5.4-mini",
            reasoning="openai:gpt-5.4",
            coding="openai:gpt-5.4",
            vision="openai:gpt-5.4-mini",
        )
    )
    anthropic: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="anthropic:claude-sonnet-4-20250514",
            cheap="anthropic:claude-3-5-haiku-20241022",
            testing="anthropic:claude-3-5-haiku-20241022",
            fast="anthropic:claude-3-5-haiku-20241022",
            balanced="anthropic:claude-sonnet-4-20250514",
            reasoning="anthropic:claude-opus-4-1-20250805",
            coding="anthropic:claude-opus-4-1-20250805",
            vision="anthropic:claude-sonnet-4-20250514",
        )
    )
    google_genai: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="google_genai:gemini-2.5-flash",
            cheap="google_genai:gemini-2.5-flash-lite",
            testing="google_genai:gemini-2.5-flash-lite",
            fast="google_genai:gemini-2.5-flash-lite",
            balanced="google_genai:gemini-2.5-flash",
            reasoning="google_genai:gemini-2.5-pro",
            coding="google_genai:gemini-2.5-pro",
            vision="google_genai:gemini-2.5-flash",
        )
    )
    xai: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="xai:grok-4-1-fast-reasoning",
            cheap="xai:grok-4-1-fast-non-reasoning",
            testing="xai:grok-4-1-fast-non-reasoning",
            fast="xai:grok-4-1-fast-non-reasoning",
            balanced="xai:grok-4-1-fast-reasoning",
            reasoning="xai:grok-4.20-0309-reasoning",
            coding="xai:grok-4-1-fast-reasoning",
            vision="xai:grok-4.20-0309-reasoning",
        )
    )
    deepseek: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="deepseek:deepseek-chat",
            cheap="deepseek:deepseek-chat",
            testing="deepseek:deepseek-chat",
            fast="deepseek:deepseek-chat",
            balanced="deepseek:deepseek-chat",
            reasoning="deepseek:deepseek-reasoner",
            coding="deepseek:deepseek-chat",
            vision="deepseek:deepseek-chat",
        )
    )
    mistral: ProviderModelPresets = Field(
        default_factory=lambda: ProviderModelPresets(
            default="mistral:mistral-small-2603",
            cheap="mistral:ministral-3b-2512",
            testing="mistral:ministral-3b-2512",
            fast="mistral:ministral-8b-2512",
            balanced="mistral:mistral-small-2603",
            reasoning="mistral:magistral-small-2509",
            coding="mistral:devstral-2512",
            vision="mistral:mistral-small-2603",
        )
    )

    def get(self, provider: Provider | str) -> ProviderModelPresets:
        """Return the preset bundle for a provider.

        Args:
            provider: Canonical provider or alias.

        Returns:
            Provider-specific preset bundle.
        """
        normalized = normalize_provider_name(provider)
        if normalized is None:
            raise ValueError("Provider cannot be None.")
        return getattr(self, normalized.value)


class DefaultModelAliases(BaseModel):
    """Global semantic aliases for common model selections.

    Examples:
        >>> aliases = DefaultModelAliases()
        >>> aliases.cheap
        'openai:gpt-5.4-nano'
    """

    model_config = ConfigDict(extra="forbid")

    default: str = "openai:gpt-5.4-mini"
    cheap: str = "openai:gpt-5.4-nano"
    testing: str = "openai:gpt-5.4-nano"
    fast: str = "openai:gpt-5.4-mini"
    balanced: str = "openai:gpt-5.4-mini"
    reasoning: str = "openai:gpt-5.4"
    coding: str = "openai:gpt-5.4"
    vision: str = "openai:gpt-5.4-mini"

    def get(self, alias: ModelAliasName) -> str:
        """Return the model configured for an alias.

        Args:
            alias: Alias name.

        Returns:
            Configured model string.
        """
        return getattr(self, alias)


class CatalogProviderSettings(BaseModel):
    """Provider-specific model-catalog settings.

    Args:
        prefer_sdk: Whether SDK-based model listing should be preferred for
            this provider.
        limit: Optional default cap on the number of returned models.
        page_size: Optional provider-native page size for paginated model
            listings.
        query_base: Optional Google GenAI flag controlling whether base models
            are included in listings.
        base_url: Optional provider base URL override for model listing.
        api_version: Optional provider API version override.

    Examples:
        >>> cfg = CatalogProviderSettings(base_url="https://api.example.com/v1")
        >>> cfg.base_url
        'https://api.example.com/v1'
    """

    model_config = ConfigDict(extra="forbid")

    prefer_sdk: bool | None = None
    limit: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1)
    query_base: bool | None = None
    base_url: str | None = None
    api_version: str | None = None


class CatalogSettings(BaseModel):
    """Settings for live model discovery and provider catalog access.

    Args:
        prefer_sdk: Whether SDK-based model listing should be preferred by
            default.
        limit: Optional default cap on the number of returned models.
        page_size: Optional provider-native page size for paginated model
            listings.
        query_base: Optional default Google GenAI flag controlling whether base
            models are included in listings.
        openai: OpenAI-specific catalog settings.
        anthropic: Anthropic-specific catalog settings.
        google_genai: Google GenAI-specific catalog settings.
        xai: xAI-specific catalog settings.
        deepseek: DeepSeek-specific catalog settings.
        mistral: Mistral-specific catalog settings.

    Examples:
        >>> catalog = CatalogSettings()
        >>> catalog.xai.base_url
        'https://api.x.ai'
        >>> catalog.build_list_models_options('openai')['prefer_sdk']
        True
    """

    model_config = ConfigDict(extra="forbid")

    prefer_sdk: bool = True
    limit: int | None = Field(default=None, ge=1)
    page_size: int | None = Field(default=None, ge=1)
    query_base: bool | None = None

    openai: CatalogProviderSettings = Field(default_factory=CatalogProviderSettings)
    anthropic: CatalogProviderSettings = Field(
        default_factory=lambda: CatalogProviderSettings(api_version="2023-06-01")
    )
    google_genai: CatalogProviderSettings = Field(default_factory=CatalogProviderSettings)
    xai: CatalogProviderSettings = Field(
        default_factory=lambda: CatalogProviderSettings(base_url="https://api.x.ai")
    )
    deepseek: CatalogProviderSettings = Field(
        default_factory=lambda: CatalogProviderSettings(base_url="https://api.deepseek.com/v1")
    )
    mistral: CatalogProviderSettings = Field(default_factory=CatalogProviderSettings)

    def get(self, provider: Provider | str) -> CatalogProviderSettings:
        """Return catalog settings for a provider.

        Args:
            provider: Canonical provider or provider alias.

        Returns:
            Provider-specific catalog settings.
        """
        normalized = normalize_provider_name(provider)
        if normalized is None:
            raise ValueError("Provider cannot be None.")
        return getattr(self, normalized.value)

    def build_list_models_options(self, provider: Provider | str) -> dict[str, object]:
        """Build default list-model options for a provider.

        Args:
            provider: Canonical provider or provider alias.

        Returns:
            Dictionary of options that can seed ``ListModelsConfig``.
        """
        provider_settings = self.get(provider)
        return {
            "prefer_sdk": provider_settings.prefer_sdk if provider_settings.prefer_sdk is not None else self.prefer_sdk,
            "limit": provider_settings.limit if provider_settings.limit is not None else self.limit,
            "page_size": provider_settings.page_size if provider_settings.page_size is not None else self.page_size,
            "query_base": provider_settings.query_base if provider_settings.query_base is not None else self.query_base,
        }

    def build_transport_kwargs(self, provider: Provider | str) -> dict[str, str]:
        """Build provider-specific transport overrides for model discovery.

        Args:
            provider: Canonical provider or provider alias.

        Returns:
            Transport keyword arguments such as ``base_url`` or
            ``anthropic_version``.
        """
        provider_settings = self.get(provider)
        kwargs: dict[str, str] = {}
        if provider_settings.base_url is not None:
            kwargs["base_url"] = provider_settings.base_url
        if provider_settings.api_version is not None:
            kwargs["anthropic_version"] = provider_settings.api_version
        return kwargs


class LiteLLMSettings(BaseModel):
    """Native LiteLLM integration settings.

    Args:
        enabled: Whether native LiteLLM integration is enabled.
        enrich_metadata: Whether LiteLLM pricing/model metadata should be used
            to enrich LangChain-first metadata resolution.
        profile_resolution_mode: How LangChain profiles and LiteLLM metadata
            should be combined.
        provider_prefixes: Optional per-provider LiteLLM prefix overrides.
        success_callbacks: Default named LiteLLM success callbacks.
        failure_callbacks: Default named LiteLLM failure callbacks.

    Examples:
        >>> cfg = LiteLLMSettings()
        >>> cfg.provider_prefixes["google_genai"]
        'gemini'
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    enrich_metadata: bool = True
    profile_resolution_mode: Literal[
        "langchain_only",
        "langchain_then_litellm",
        "litellm_only",
    ] = "langchain_then_litellm"
    provider_prefixes: dict[str, str] = Field(
        default_factory=lambda: {
            "openai": "openai",
            "anthropic": "anthropic",
            "google_genai": "gemini",
            "xai": "xai",
            "deepseek": "deepseek",
            "mistral": "mistral",
        }
    )
    success_callbacks: list[str] = Field(default_factory=list)
    failure_callbacks: list[str] = Field(default_factory=list)


class LLMCacheSettings(BaseModel):
    """LLM cache configuration.

    Args:
        enabled: Whether global LLM caching should be enabled by default.
        backend: Cache backend identifier.
        path: Optional explicit cache path.
        filename: Cache filename when ``path`` is not explicitly provided.
        create_dirs: Whether parent directories should be created automatically.

    Examples:
        >>> cache = LLMCacheSettings()
        >>> cache.backend
        'sqlite'
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    backend: str = "sqlite"
    path: Path | None = None
    filename: str = "langchain_llm_cache.sqlite3"
    create_dirs: bool = True


class LLMSettings(BaseModel):
    """Top-level LLM settings.

    Args:
        default_model: Fallback model when no alias or provider default is used.
        defaults_by_provider: Default model preset bundles per provider.
        aliases: Global semantic aliases.
        cache: Global cache settings.

    Examples:
        >>> llm = LLMSettings()
        >>> llm.default_model
        'openai:gpt-5.4-mini'
    """

    model_config = ConfigDict(extra="forbid")

    default_model: str = "openai:gpt-5.4-mini"
    defaults_by_provider: DefaultModelsByProvider = Field(default_factory=DefaultModelsByProvider)
    aliases: DefaultModelAliases = Field(default_factory=DefaultModelAliases)
    cache: LLMCacheSettings = Field(default_factory=LLMCacheSettings)


class AppSettings(BaseSettings):
    """Application settings for the LLM layer.

    Args:
        app_name: Human-readable application name.
        app_root: Root directory used for derived paths.
        app_dir_name: Hidden directory rooted under ``app_root`` for local
            cache and application files.
        credentials: Provider credential settings.
        llm: LLM settings.
        catalog: Live model-discovery settings.
        litellm: Native LiteLLM integration settings.

    Examples:
        >>> settings = AppSettings()
        >>> settings.resolve_model(alias="reasoning")
        'openai:gpt-5.4'
        >>> settings.catalog.xai.base_url
        'https://api.x.ai'
        >>> settings.litellm.profile_resolution_mode
        'langchain_then_litellm'
    """

    model_config = SettingsConfigDict(
        env_prefix="OOAI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )


    @field_validator("credentials", mode="before")
    @classmethod
    def _coerce_credentials(cls, value: ProviderCredentials | dict[str, object]) -> ProviderCredentials | dict[str, object]:
        """Coerce nested credential dictionaries into typed credentials.

        Args:
            value: Incoming credentials value.

        Returns:
            Typed credentials object or untouched value.
        """
        if isinstance(value, dict):
            return ProviderCredentials.model_validate(value)
        return value

    app_name: str = "ooai"
    app_root: Path = Field(default_factory=Path.cwd)
    app_dir_name: str = ".ooai"
    credentials: ProviderCredentials = Field(default_factory=ProviderCredentials.from_environment)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    litellm: LiteLLMSettings = Field(default_factory=LiteLLMSettings)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def app_dir(self) -> Path:
        """Return the resolved application directory.

        Returns:
            Resolved application directory.
        """
        return (self.app_root / self.app_dir_name).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def default_llm_cache_path(self) -> Path:
        """Return the resolved default LLM cache path.

        Returns:
            Effective cache file path.
        """
        explicit_path = self.llm.cache.path
        if explicit_path is not None:
            return explicit_path.expanduser().resolve()
        return (self.app_dir / "cache" / "llm" / self.llm.cache.filename).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def default_model_string(self) -> ModelString:
        """Return the configured top-level default as a typed model string.

        Returns:
            Typed default model string.
        """
        return ModelString.parse(self.llm.default_model)

    def model_copy_with_root(self, app_root: Path) -> Self:
        """Return a copy of the settings with a new application root.

        Args:
            app_root: New application root.

        Returns:
            Updated settings copy.
        """
        return self.model_copy(update={"app_root": app_root})

    def resolve_model(
        self,
        *,
        model: str | None = None,
        alias: ModelAliasName | None = None,
        provider: Provider | str | None = None,
        preset: ModelPresetName = "default",
    ) -> str:
        """Resolve the effective model string.

        Resolution order:
            1. Explicit ``model``.
            2. Global semantic ``alias``.
            3. Provider-specific ``preset``.
            4. Global fallback ``llm.default_model``.

        Args:
            model: Explicit model string.
            alias: Optional semantic alias.
            provider: Optional provider name or alias.
            preset: Provider-specific preset name.

        Returns:
            Resolved model string.

        Raises:
            ValueError: If an alias and provider are both supplied.
        """
        if model is not None:
            return model
        if alias is not None and provider is not None:
            raise ValueError("Pass either alias or provider, not both.")
        if alias is not None:
            return self.llm.aliases.get(alias)
        if provider is not None:
            return self.llm.defaults_by_provider.get(provider).get(preset)
        return self.llm.default_model

    def resolve_model_string(
        self,
        *,
        model: str | ModelString | None = None,
        alias: ModelAliasName | None = None,
        provider: Provider | str | None = None,
        preset: ModelPresetName = "default",
    ) -> ModelString:
        """Resolve the effective model as a typed model string.

        Args:
            model: Explicit model string or typed model string.
            alias: Optional semantic alias.
            provider: Optional provider name or alias.
            preset: Provider-specific preset name.

        Returns:
            Typed model string.
        """
        if isinstance(model, ModelString):
            return model
        return ModelString.parse(
            self.resolve_model(model=model, alias=alias, provider=provider, preset=preset)
        )
