"""Example: update convenience factory defaults from model catalogs."""

from __future__ import annotations

from ooai_llm import AppSettings, update_model_defaults


def main() -> None:
    """Update defaults and print the effective provider presets."""
    settings = AppSettings()
    update = update_model_defaults(
        settings,
        providers=["openai", "anthropic", "mistral"],
        source="auto",
        output_format="env",
    )

    for note in update.notes:
        print(f"note: {note}")

    refreshed_settings = update.settings
    print("latest:", refreshed_settings.resolve_model(alias="latest"))
    print(
        "anthropic reasoning:",
        refreshed_settings.resolve_model(provider="anthropic", preset="reasoning"),
    )
    print("mistral coding:", refreshed_settings.resolve_model(provider="mistral", preset="coding"))
    print(update.output_text)


if __name__ == "__main__":
    main()
