"""Unit tests for live model-discovery helpers."""

from __future__ import annotations

import sys
import types

import pytest

from ooai_llm import list_available_models
from ooai_llm.catalog import ListModelsConfig, list_model_ids
from ooai_llm.catalog import get_provider_client_info
from ooai_llm.providers import Provider
from ooai_llm.settings import AppSettings


class _FakeOpenAIModel:
    def __init__(self, model_id: str, owned_by: str = "openai") -> None:
        self.id = model_id
        self.object = "model"
        self.owned_by = owned_by


@pytest.mark.unit
def test_list_available_models_openai_uses_sdk(monkeypatch) -> None:
    """It should list OpenAI models through the official SDK wrapper."""

    class FakeClient:
        def __init__(self, *, api_key: str, **_: object) -> None:
            assert api_key == "sk-openai"
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[_FakeOpenAIModel("gpt-5.4-mini"), _FakeOpenAIModel("gpt-5.4-nano")]
                )
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = list_available_models(
        "openai",
        settings=AppSettings(credentials={"openai_api_key": "sk-openai"}),
    )

    assert result.provider == Provider.OPENAI
    assert result.used_sdk is True
    assert [item.model_id for item in result.models] == ["gpt-5.4-mini", "gpt-5.4-nano"]
    assert str(result.models[0].model_string) == "openai:gpt-5.4-mini"


@pytest.mark.unit
def test_list_available_models_google_uses_native_sdk(monkeypatch) -> None:
    """It should list Gemini models through the Google GenAI SDK."""

    class FakeGoogleModel:
        def __init__(self, name: str) -> None:
            self.name = name
            self.supported_actions = ["generateContent"]
            self.input_token_limit = 1000
            self.output_token_limit = 200

    class FakeClient:
        def __init__(self, *, api_key: str, **kwargs: object) -> None:
            assert api_key == "sk-google"
            assert kwargs["vertexai"] is True
            assert kwargs["project"] == "demo-project"
            assert kwargs["location"] == "us-central1"
            self.models = types.SimpleNamespace(
                list=self.list
            )

        def list(self, config=None):
            assert config == {"page_size": 20, "query_base": True}
            return [
                FakeGoogleModel("models/gemini-2.5-flash"),
                FakeGoogleModel("models/gemini-2.5-pro"),
            ]

    fake_google = types.ModuleType("google")
    fake_google.genai = types.SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)

    result = list_available_models(
        "google",
        settings=AppSettings(
            credentials={
                "google_api_key": "sk-google",
                "google_use_vertexai": True,
                "google_cloud_project": "demo-project",
                "google_cloud_location": "us-central1",
            }
        ),
        config=ListModelsConfig(limit=1, page_size=20, query_base=True),
    )

    assert result.provider == Provider.GOOGLE_GENAI
    assert result.used_sdk is True
    assert [item.model_id for item in result.models] == ["models/gemini-2.5-flash"]
    assert result.models[0].supported_actions == ["generateContent"]


@pytest.mark.unit
def test_list_available_models_anthropic_uses_sdk(monkeypatch) -> None:
    """It should list Anthropic models through the SDK when available."""

    class FakeClient:
        def __init__(self, *, api_key: str, **kwargs: object) -> None:
            assert api_key == "sk-anthropic"
            assert "anthropic_version" not in kwargs
            self.models = types.SimpleNamespace(list=self.list)

        def list(self, *, limit: int | None = None):
            assert limit == 1
            return types.SimpleNamespace(
                data=[{"id": "claude-sonnet-4-20250514", "type": "model"}],
                has_more=True,
                last_id="next-page",
            )

    fake_anthropic = types.ModuleType("anthropic")
    fake_anthropic.Anthropic = FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    result = list_available_models(
        "anthropic",
        settings=AppSettings(credentials={"anthropic_api_key": "sk-anthropic"}),
        config=ListModelsConfig(page_size=1),
    )

    assert result.used_sdk is True
    assert result.next_page_token == "next-page"
    assert result.models[0].model_id == "claude-sonnet-4-20250514"


@pytest.mark.unit
def test_list_available_models_xai_uses_sdk_without_rest_base_url(monkeypatch) -> None:
    """It should avoid passing REST-only kwargs into the xAI SDK."""

    class FakeClient:
        def __init__(self, *, api_key: str, **kwargs: object) -> None:
            assert api_key == "sk-xai"
            assert "base_url" not in kwargs
            self.models = types.SimpleNamespace(
                list_language_models=lambda: [
                    {"id": "grok-4.20-reasoning", "object": "model", "aliases": ("grok",)}
                ]
            )

    fake_xai = types.ModuleType("xai_sdk")
    fake_xai.Client = FakeClient
    monkeypatch.setitem(sys.modules, "xai_sdk", fake_xai)

    result = list_available_models(
        "xai",
        settings=AppSettings(credentials={"xai_api_key": "sk-xai"}),
        base_url="https://custom.xai.example",
    )

    assert result.used_sdk is True
    assert result.models[0].aliases == ["grok"]
    assert result.notes == ["Used xAI SDK language-model listing."]


@pytest.mark.unit
def test_list_available_models_mistral_uses_sdk(monkeypatch) -> None:
    """It should list Mistral models through the native SDK."""

    class FakeClient:
        def __init__(self, *, api_key: str, **_: object) -> None:
            assert api_key == "sk-mistral"
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[{"id": "mistral-small-2603", "created_at": "2026-01-01T00:00:00Z"}]
                )
            )

    fake_mistral = types.ModuleType("mistralai")
    fake_mistral.Mistral = FakeClient
    monkeypatch.setitem(sys.modules, "mistralai", fake_mistral)

    result = list_available_models(
        "mistral",
        settings=AppSettings(credentials={"mistral_api_key": "sk-mistral"}),
    )

    assert result.provider == Provider.MISTRAL
    assert result.used_sdk is True
    assert result.models[0].model_id == "mistral-small-2603"


@pytest.mark.unit
def test_list_available_models_xai_falls_back_to_rest(monkeypatch) -> None:
    """It should fall back to xAI's REST models endpoint when the SDK is absent."""

    monkeypatch.delitem(sys.modules, "xai_sdk", raising=False)

    import ooai_llm.catalog as catalog

    def fake_json_get(url: str, *, headers=None):
        assert url.endswith("/v1/models")
        assert headers["Authorization"] == "Bearer sk-xai"
        return {
            "data": [
                {"id": "grok-4.20-reasoning", "object": "model", "owned_by": "xai"},
            ]
        }

    monkeypatch.setattr(catalog, "_json_get", fake_json_get)

    result = list_available_models(
        "xai",
        settings=AppSettings(credentials={"xai_api_key": "sk-xai"}),
        config=ListModelsConfig(prefer_sdk=False),
    )

    assert result.provider == Provider.XAI
    assert result.used_sdk is False
    assert result.models[0].model_id == "grok-4.20-reasoning"
    assert result.notes


@pytest.mark.unit
def test_list_available_models_anthropic_rest_fallback(monkeypatch) -> None:
    """It should support Anthropic via the documented REST models endpoint."""

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)

    import ooai_llm.catalog as catalog

    def fake_json_get(url: str, *, headers=None):
        assert url.startswith("https://api.anthropic.com/v1/models")
        assert headers["x-api-key"] == "sk-anthropic"
        return {
            "data": [
                {
                    "id": "claude-sonnet-4-20250514",
                    "display_name": "Claude Sonnet 4",
                    "type": "model",
                    "created_at": "2025-02-19T00:00:00Z",
                }
            ],
            "has_more": False,
            "last_id": None,
        }

    monkeypatch.setattr(catalog, "_json_get", fake_json_get)

    result = list_available_models(
        "anthropic",
        settings=AppSettings(credentials={"anthropic_api_key": "sk-anthropic"}),
        config=ListModelsConfig(prefer_sdk=False),
    )

    assert result.provider == Provider.ANTHROPIC
    assert result.models[0].display_name == "Claude Sonnet 4"
    assert result.models[0].created_at == "2025-02-19T00:00:00Z"


@pytest.mark.unit
def test_list_model_ids_returns_only_ids(monkeypatch) -> None:
    """It should provide a convenience API for bare model identifiers."""

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str, **_: object) -> None:
            assert api_key == "sk-deepseek"
            assert base_url.endswith("/v1")
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[_FakeOpenAIModel("deepseek-chat", owned_by="deepseek")]
                )
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    ids = list_model_ids(
        "deepseek",
        settings=AppSettings(credentials={"deepseek_api_key": "sk-deepseek"}),
    )

    assert ids == ["deepseek-chat"]


@pytest.mark.unit
def test_get_provider_client_info_exposes_install_metadata() -> None:
    """It should expose installation guidance for provider extras and SDKs."""
    info = get_provider_client_info("google")
    assert info.provider == Provider.GOOGLE_GENAI
    assert info.extra_name == "google"
    assert info.native_sdk_package == "google-genai"
    assert info.langchain_package == "langchain-google-genai"



@pytest.mark.unit
def test_catalog_settings_feed_transport_defaults(monkeypatch) -> None:
    """It should use catalog transport settings when explicit kwargs are absent."""

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str, **_: object) -> None:
            assert api_key == "sk-deepseek"
            assert base_url == "https://custom.deepseek.example/v1"
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[_FakeOpenAIModel("deepseek-chat", owned_by="deepseek")]
                )
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = list_available_models(
        "deepseek",
        settings=AppSettings(
            credentials={"deepseek_api_key": "sk-deepseek"},
            catalog={"deepseek": {"base_url": "https://custom.deepseek.example/v1"}},
        ),
    )

    assert result.provider == Provider.DEEPSEEK
    assert result.used_sdk is True


@pytest.mark.unit
def test_explicit_config_overrides_catalog_defaults(monkeypatch) -> None:
    """It should let explicit list config override catalog defaults."""

    class FakeClient:
        def __init__(self, *, api_key: str, **_: object) -> None:
            assert api_key == "sk-openai"
            self.models = types.SimpleNamespace(
                list=lambda: types.SimpleNamespace(
                    data=[_FakeOpenAIModel("gpt-5.4-mini"), _FakeOpenAIModel("gpt-5.4-nano")]
                )
            )

    fake_openai = types.ModuleType("openai")
    fake_openai.OpenAI = FakeClient
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    result = list_available_models(
        "openai",
        settings=AppSettings(
            credentials={"openai_api_key": "sk-openai"},
            catalog={"openai": {"limit": 2}},
        ),
        config=ListModelsConfig(limit=1),
    )

    assert [item.model_id for item in result.models] == ["gpt-5.4-mini"]
