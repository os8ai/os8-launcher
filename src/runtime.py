"""Shared runtime utilities for backends and clients."""

import os
import shlex
import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config


def serve_combo(
    model: str,
    backend: str | None,
    client: str | None,
    config: "Config",
    repo_root: Path,
):
    """Start a backend, then (only if it came up healthy) start a client.

    The single source of truth for "serve = backend + optional client".
    Both the CLI's `./launcher serve --client` and the dashboard's
    /api/serve endpoint call this so the two interfaces stay in sync.
    """
    # Imported lazily to avoid a circular import (backends.py imports runtime).
    from src.backends import start_backend
    from src.clients import start_client

    start_backend(model, backend, config, repo_root)
    if client:
        start_client(client, config, repo_root, model, backend)


def check_port(port: int) -> bool:
    """Return True if the port is already in use on localhost."""
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def expand_template(template: str, variables: dict) -> str:
    """Expand a manifest run template with the given variables.

    Raises KeyError with a clear message if a placeholder is missing.
    """
    try:
        return template.format(**variables)
    except KeyError as e:
        raise KeyError(
            f"Manifest template requires variable {e} but it was not provided.\n"
            f"Template: {template}\n"
            f"Available variables: {', '.join(variables.keys())}"
        ) from None


def build_env_for_venv(venv_path: Path) -> dict:
    """Build an environment dict that activates the given venv."""
    env = os.environ.copy()
    venv_bin = str(venv_path / "bin")
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["VIRTUAL_ENV"] = str(venv_path)
    return env


def parse_command(cmd_string: str) -> list[str]:
    """Parse a command string into a list suitable for subprocess."""
    return shlex.split(cmd_string)
