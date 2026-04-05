"""os8-launcher CLI — command-line interface for managing local AI models."""

import argparse
import sys
from pathlib import Path

from src.config import ConfigError, load_config, format_config


def _stub(phase, description):
    """Create a stub handler that prints a not-yet-implemented message."""
    def handler(args):
        print(f"{description} — coming in Phase {phase}.")
    return handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launcher",
        description="os8-launcher — manage local open-source AI models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- config ---
    config_parser = subparsers.add_parser("config", help="View configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_show = config_sub.add_parser("show", help="Print the current configuration")
    config_show.set_defaults(handler="config_show")
    config_parser.set_defaults(handler="config_help", _parser=config_parser)

    # --- setup ---
    setup_parser = subparsers.add_parser("setup", help="Install backends and clients")
    setup_parser.add_argument("tool", nargs="?", help="Tool to install (or --all)")
    setup_parser.add_argument("--all", action="store_true", help="Install everything")
    setup_parser.set_defaults(handler=_stub(2, "Tool installation"))

    # --- models ---
    models_parser = subparsers.add_parser("models", help="Manage model weights")
    models_sub = models_parser.add_subparsers(dest="models_command")

    models_list = models_sub.add_parser("list", help="List available models")
    models_list.set_defaults(handler=_stub(2, "Model listing"))

    models_download = models_sub.add_parser("download", help="Download model weights")
    models_download.add_argument("model", help="Model to download")
    models_download.set_defaults(handler=_stub(2, "Model download"))

    models_remove = models_sub.add_parser("remove", help="Remove model weights")
    models_remove.add_argument("model", help="Model to remove")
    models_remove.set_defaults(handler=_stub(2, "Model removal"))

    models_parser.set_defaults(handler="models_help", _parser=models_parser)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start a serving backend")
    serve_parser.add_argument("model", help="Model to serve")
    serve_parser.add_argument("--backend", help="Backend to use (default: model's default)")
    serve_parser.set_defaults(handler=_stub(3, "Backend serving"))

    # --- client ---
    client_parser = subparsers.add_parser("client", help="Start a client")
    client_parser.add_argument("name", help="Client to start")
    client_parser.set_defaults(handler=_stub(3, "Client launch"))

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Show what's running")
    status_parser.set_defaults(handler=_stub(3, "Status reporting"))

    # --- stop ---
    stop_parser = subparsers.add_parser("stop", help="Stop all running services")
    stop_parser.set_defaults(handler=_stub(3, "Service shutdown"))

    return parser


def main(repo_root: Path):
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = args.handler

    # String handlers need special treatment
    if handler == "config_help":
        args._parser.print_help()
        return
    if handler == "models_help":
        args._parser.print_help()
        return
    if handler == "config_show":
        try:
            config = load_config(repo_root)
            print(format_config(config))
        except ConfigError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Callable stub handlers
    handler(args)
