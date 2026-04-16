"""Reasoning examples for ``ooai_llm``.

Purpose:
    Demonstrate semantic reasoning presets and typed reasoning overrides.

Examples:
    .. code-block:: bash

        python examples/reasoning_usage.py
"""

from __future__ import annotations

from ooai_llm import ReasoningConfig, build_reasoning_resolution


def main() -> None:
    """Print a few reasoning-resolution examples."""
    openai = build_reasoning_resolution(
        model="openai:gpt-5.4-mini",
        reasoning="deep",
    )
    print("OpenAI:", openai.model_dump() if openai is not None else None)

    gemini = build_reasoning_resolution(
        model="google_genai:gemini-2.5-flash",
        reasoning=ReasoningConfig(budget_tokens=1024, include_thoughts=True),
    )
    print("Gemini:", gemini.model_dump() if gemini is not None else None)


if __name__ == "__main__":
    main()
