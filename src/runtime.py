"""Shared runtime utilities for backends and clients."""

import os
import shlex
import socket
from pathlib import Path


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
