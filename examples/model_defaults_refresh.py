"""Example: refresh convenience factory defaults from model catalogs."""

from __future__ import annotations

from ooai_llm import AppSettings, refresh_model_defaults


def main() -> None:
    """Refresh defaults and print the effective provider presets."""
    settings = AppSettings()
    refresh = refresh_model_defaults(
        settings,
        providers=["openai", "anthropic", "mistral"],
        source="auto",
    )

    for note in refresh.notes:
        print(f"note: {note}")

    refreshed_settings = refresh.settings
    print("latest:", refreshed_settings.resolve_model(alias="latest"))
    print("anthropic reasoning:", refreshed_settings.resolve_model(provider="anthropic", preset="reasoning"))
    print("mistral coding:", refreshed_settings.resolve_model(provider="mistral", preset="coding"))


if __name__ == "__main__":
    main()
