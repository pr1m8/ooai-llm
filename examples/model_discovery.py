"""Example: list available models from provider-native APIs."""

from __future__ import annotations

from ooai_llm import AppSettings, ListModelsConfig, list_available_models


def main() -> None:
    """Run a small live model-discovery example."""
    settings = AppSettings()
    result = list_available_models(
        "openai",
        settings=settings,
        config=ListModelsConfig(limit=5),
    )

    for model in result.models:
        print(model.model_id)


if __name__ == "__main__":
    main()
