"""Command-line interface for ``ooai-llm``."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .catalog import ListModelsConfig
from .model_defaults import update_model_defaults


def _parse_providers(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    providers: list[str] = []
    for value in values:
        providers.extend(part.strip() for part in value.split(",") if part.strip())
    return providers or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ooai-llm",
        description="Utilities for ooai-llm settings, model catalogs, and defaults.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    models = subcommands.add_parser("models", help="Model catalog and default utilities.")
    model_commands = models.add_subparsers(dest="models_command", required=True)

    update = model_commands.add_parser(
        "update",
        help="Refresh convenience factory model defaults.",
    )
    update.add_argument(
        "--source",
        choices=("auto", "provider", "litellm"),
        default="auto",
        help="Refresh source. Defaults to auto.",
    )
    update.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Provider to refresh. Can be repeated.",
    )
    update.add_argument(
        "--providers",
        action="append",
        default=[],
        help="Comma-separated provider list.",
    )
    update.add_argument(
        "--primary-alias-provider",
        default="openai",
        help="Provider used to update global aliases such as latest and cheap.",
    )
    update.add_argument(
        "--format",
        choices=("json", "env"),
        default="json",
        help="Output format for reusable overrides.",
    )
    update.add_argument(
        "--output",
        help="Optional path to write overrides. Prints to stdout when omitted.",
    )
    update.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-provider model-list limit for provider catalogs.",
    )
    update.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any selected provider cannot be refreshed.",
    )
    update.add_argument(
        "--no-aliases",
        action="store_true",
        help="Only export provider presets, not global aliases/default_model.",
    )

    return parser


def _run_models_update(args: argparse.Namespace) -> int:
    providers = _parse_providers([*args.provider, *args.providers])
    config = ListModelsConfig(limit=args.limit) if args.limit is not None else None
    result = update_model_defaults(
        providers=providers,
        source=args.source,
        config=config,
        primary_alias_provider=args.primary_alias_provider,
        strict=args.strict,
        output_path=args.output,
        output_format=args.format,
        include_aliases=not args.no_aliases,
    )

    for note in result.notes:
        print(f"warning: {note}", file=sys.stderr)

    if result.output_text is not None:
        print(result.output_text, end="")
    elif result.output_path is not None:
        print(f"Wrote model defaults to {result.output_path}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ooai-llm`` command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "models" and args.models_command == "update":
            return _run_models_update(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
