"""Basic usage example for ``ooai_llm``."""

from __future__ import annotations

from ooai_llm import AppSettings, configure_global_llm_cache, create_llm


def main() -> None:
    """Configure settings, bootstrap cache, and create a chat model."""
    settings = AppSettings()
    configure_global_llm_cache(settings)

    llm = create_llm(alias="testing", settings=settings, temperature=0)
    print(llm)


if __name__ == "__main__":
    main()
