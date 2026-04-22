"""Factory-first usage examples for ``ooai_llm``."""

from __future__ import annotations

from ooai_llm import AppSettings, create_llm, create_llm_bundle


def main() -> None:
    """Create chat models through aliases, providers, and explicit models."""
    settings = AppSettings()

    testing_llm = create_llm(alias="testing", settings=settings, temperature=0)
    print("testing:", type(testing_llm).__name__)

    reasoning_llm = create_llm(
        provider="anthropic",
        preset="reasoning",
        settings=settings,
        reasoning="deep",
        temperature=0,
    )
    print("reasoning:", type(reasoning_llm).__name__)

    bundle = create_llm_bundle(
        "openai:gpt-5.4-mini",
        settings=settings,
        reasoning="fast",
        temperature=0,
    )
    print("bundle model:", bundle.model.as_langchain())
    print("billing model:", bundle.metadata.identity.litellm_model)


if __name__ == "__main__":
    main()
