"""State file management for tracking running backends and clients.

State is stored at ~/.config/os8-launcher/state.yaml and cross-checked
against reality (process liveness) on every read via validate_state().
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

STATE_DIR = Path.home() / ".config" / "os8-launcher"
STATE_FILE = STATE_DIR / "state.yaml"


def load_state() -> dict:
    """Load state from file, or return empty dict."""
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_state(data: dict):
    """Write state to file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def set_backend(
    name: str,
    model: str,
    port: int,
    install_type: str,
    pid: int | None = None,
    container_id: str | None = None,
    health_path: str = "/v1/models",
):
    """Record a running backend in state."""
    data = load_state()
    data["backend"] = {
        "name": name,
        "model": model,
        "port": port,
        "pid": pid,
        "container_id": container_id,
        "install_type": install_type,
        "health_path": health_path,
        "start_time": datetime.now().isoformat(),
    }
    save_state(data)


def clear_backend():
    """Remove the backend entry from state."""
    data = load_state()
    data.pop("backend", None)
    save_state(data)


def set_client(
    name: str,
    port: int,
    install_type: str,
    container_id: str | None = None,
    ready: bool = False,
):
    """Record a running detached client in state."""
    data = load_state()
    if "clients" not in data:
        data["clients"] = {}
    data["clients"][name] = {
        "port": port,
        "container_id": container_id,
        "install_type": install_type,
        "start_time": datetime.now().isoformat(),
        "ready": ready,
    }
    save_state(data)


def mark_client_ready(name: str):
    """Flip an existing client's ready flag to True."""
    data = load_state()
    clients = data.get("clients", {})
    if name in clients:
        clients[name]["ready"] = True
        save_state(data)


def clear_client(name: str):
    """Remove a client entry from state."""
    data = load_state()
    clients = data.get("clients", {})
    clients.pop(name, None)
    if not clients:
        data.pop("clients", None)
    save_state(data)


def clear_all():
    """Wipe the state file."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_container_running(container_id: str) -> bool:
    """Check if a Docker container is running."""
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_id],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() == "true"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _is_entry_alive(entry: dict) -> bool:
    """Check if a state entry (backend or client) is still alive."""
    container_id = entry.get("container_id")
    pid = entry.get("pid")

    if container_id:
        return is_container_running(container_id)
    if pid:
        return is_process_alive(pid)
    return False


def validate_state() -> dict:
    """Load state, verify liveness, remove stale entries, save back.

    This is the orphan recovery mechanism — called before serve, status, and stop.
    """
    data = load_state()
    changed = False

    # Check backend
    backend = data.get("backend")
    if backend and not _is_entry_alive(backend):
        data.pop("backend")
        changed = True

    # Check clients
    clients = data.get("clients", {})
    stale = [name for name, entry in clients.items() if not _is_entry_alive(entry)]
    for name in stale:
        del clients[name]
        changed = True
    if not clients:
        data.pop("clients", None)

    if changed:
        save_state(data)

    return data
