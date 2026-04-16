"""Model-string parsing example for ``ooai_llm``."""

from __future__ import annotations

from ooai_llm import ModelString


def main() -> None:
    """Show provider inference and canonicalization."""
    raw = ModelString.parse("text-embedding-3-small")
    print(raw.provider)
    print(raw.model_name)
    print(raw.canonical())


if __name__ == "__main__":
    main()
