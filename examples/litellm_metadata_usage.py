"""Examples for LangChain-first metadata and LiteLLM enrichment."""

from __future__ import annotations

from ooai_llm import (
    BudgetPolicy,
    UsageRecorder,
    create_llm_bundle,
    make_litellm_cost_callback,
)


def main() -> None:
    """Create a bundle and show how LiteLLM callback helpers fit in."""
    bundle = create_llm_bundle(alias="testing", reasoning="fast")
    print(bundle.model.as_langchain())
    print(bundle.metadata.identity.litellm_model)
    print(bundle.metadata.pricing.source)

    recorder = UsageRecorder()
    callback = make_litellm_cost_callback(
        recorder,
        budget=BudgetPolicy(warn_total_tokens=5_000),
    )
    print(callback)


if __name__ == "__main__":
    main()
