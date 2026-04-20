"""Shared runtime utilities for backends and clients."""

import os
import shlex
import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config


class PortAllocationError(Exception):
    """Raised when no free port can be found for a new backend instance."""


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


def allocate_port(
    backend_name: str,
    model_name: str,
    default_port: int,
    reserved_ports: set[int] | None = None,
    scan_window: int = 90,
) -> int:
    """Pick a free localhost port for a new backend instance.

    Priority (highest wins):
      1. Per-instance override in settings.yaml
         (port_overrides["{backend}-{model}"]).
      2. The backend's configured port — already post per-kind override
         from settings.yaml and the Ports tab; this is BackendConfig.port.
      3. First free port in range(default_port + 10, default_port + 10 +
         scan_window). The +10 offset leaves a visible gap between "the
         port you configured" and allocator-chosen siblings so `lsof`
         output stays readable.

    `reserved_ports` is an explicit set of ports to avoid (e.g.
    earmarked for other instances about to start in the same pass).
    Ports of currently-recorded backends in state.yaml are implicitly
    avoided too, so a sibling mid-start whose socket isn't yet listening
    doesn't get its port handed out twice.
    """
    # Lazy imports avoid a circular: settings and state both import runtime.
    from src.settings import get_port_overrides
    from src.state import compute_instance_id, load_state

    instance_id = compute_instance_id(backend_name, model_name)
    reserved = set(reserved_ports or ())
    # Include ports of all recorded backends — shields the allocator from
    # a race where a sibling has written state but not yet bound.
    for entry in (load_state().get("backends") or {}).values():
        p = entry.get("port")
        if isinstance(p, int):
            reserved.add(p)
    overrides = get_port_overrides()

    per_instance = overrides.get(instance_id)
    if per_instance is not None:
        if per_instance in reserved or check_port(per_instance):
            raise PortAllocationError(
                f"Per-instance port override {per_instance} for "
                f"{instance_id} is in use."
            )
        return per_instance

    if default_port not in reserved and not check_port(default_port):
        return default_port

    for offset in range(10, 10 + scan_window):
        candidate = default_port + offset
        if candidate in reserved:
            continue
        if not check_port(candidate):
            return candidate

    raise PortAllocationError(
        f"No free port for {instance_id} in range("
        f"{default_port + 10}, {default_port + 10 + scan_window})."
    )


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
