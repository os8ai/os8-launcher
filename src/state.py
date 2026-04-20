"""State file management for tracking running backends and clients.

State is stored at ~/.config/os8-launcher/state.yaml and cross-checked
against reality (process liveness) on every read via validate_state().

Schema (current):

    backends:
      <instance_id>:              # e.g. "vllm-gemma-4-31B-it-nvfp4"
        instance_id: ...
        name: ...                 # backend kind (vllm, ollama, ...)
        model: ...
        port: ...
        pid / container_id: ...
        install_type: ...
        health_path: ...
        start_time: ...
    clients:
      <client_name>: {...}
    active_project: ...

A legacy-read migrator in load_state() upgrades pre-Phase-2 state files
that used a singular `backend:` key; the on-disk schema converges to the
new shape on the first read after upgrade.

Phase 2 is introduced in stages. This module ships the multi-instance
storage but callers still operate under the single-backend-at-a-time
runtime guard (enforced in backends.py). `get_primary_backend()` is the
read shim that lets one-backend-assuming callers keep working unchanged.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

STATE_DIR = Path.home() / ".config" / "os8-launcher"
STATE_FILE = STATE_DIR / "state.yaml"


def compute_instance_id(backend_name: str, model: str) -> str:
    """Deterministic key for a (backend, model) pair.

    Used as the dict key under state["backends"] and as the logical ID a
    caller uses to target an instance (stop, touch, ensure). Deterministic
    so the same (backend, model) request always resolves to the same
    entry — that's what makes /api/serve/ensure idempotent.
    """
    return f"{backend_name}-{model}"


def load_state() -> dict:
    """Load state from file, or return empty dict.

    Legacy-read migrator: if the file contains the pre-Phase-2 singular
    `backend:` key, rewrite it into the new `backends:` dict in-place so
    the rest of the code only ever sees the new shape. Next save flushes
    the migrated form to disk.
    """
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return {}

    legacy = data.pop("backend", None)
    if legacy and isinstance(legacy, dict):
        instance_id = legacy.get("instance_id") or compute_instance_id(
            legacy.get("name", "unknown"),
            legacy.get("model", "unknown"),
        )
        legacy["instance_id"] = instance_id
        backends = data.setdefault("backends", {})
        # Don't clobber an existing new-shape entry (shouldn't happen, but
        # defensive — a partial migration isn't a reason to lose data).
        if instance_id not in backends:
            backends[instance_id] = legacy

    return data


def save_state(data: dict):
    """Write state to file atomically.

    yaml.safe_dump writes to a temp file in the same directory, fsyncs,
    then os.replace()s into place so a mid-write crash can't corrupt the
    live state file. The extra syscall cost is negligible next to a
    docker-stop or model load.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def set_backend(
    name: str,
    model: str,
    port: int,
    install_type: str,
    pid: int | None = None,
    container_id: str | None = None,
    health_path: str = "/v1/models",
    instance_id: str | None = None,
):
    """Record a running backend in state.

    `name` is the backend kind (vllm, ollama, ...). `instance_id` is the
    multi-instance key under state["backends"]; defaults to the
    deterministic (backend, model) pair when omitted.
    """
    instance_id = instance_id or compute_instance_id(name, model)
    data = load_state()
    backends = data.setdefault("backends", {})
    backends[instance_id] = {
        "instance_id": instance_id,
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


def clear_backend(instance_id: str | None = None):
    """Remove a backend entry from state.

    With `instance_id`, removes only that entry. Without, removes every
    backend entry — the legacy "stop all backends" semantics used by the
    single-backend paths that haven't been plumbed through with instance
    IDs yet.
    """
    data = load_state()
    backends = data.get("backends") or {}
    if instance_id is None:
        if backends:
            data.pop("backends", None)
    else:
        backends.pop(instance_id, None)
        if not backends:
            data.pop("backends", None)
    save_state(data)


def get_primary_backend(data: dict) -> dict | None:
    """Return a single backend entry for single-backend-assuming callers.

    Under the Phase-2 schema there can be multiple backends resident, but
    launcher-1 keeps the runtime guard that allows only one at a time —
    so in practice this returns the sole entry or None. Once launcher-2
    lifts the guard, callers that still use this helper will need to be
    made instance-aware; until then it picks the most-recently-started.
    """
    backends = data.get("backends") or {}
    if not backends:
        return None
    if len(backends) == 1:
        return next(iter(backends.values()))
    return max(
        backends.values(),
        key=lambda b: b.get("start_time", ""),
    )


def touch_backend(instance_id: str) -> bool:
    """Mark an instance as recently used (LRU signal for launcher-4).

    Returns True if the instance exists and was updated, False otherwise.
    Used by /api/serve/touch — a cheap hint from OS8 that this instance
    just served a request. The `last_used` field is stored but not yet
    consulted for eviction (that lands in launcher-4). Missing launcher-4
    keeps this a no-op in practice, which is intentional: OS8 fires touch
    unconditionally and we don't want a read-modify-write on every request
    to be load-bearing.
    """
    data = load_state()
    backends = data.get("backends") or {}
    entry = backends.get(instance_id)
    if not entry:
        return False
    entry["last_used"] = datetime.now().isoformat()
    save_state(data)
    return True


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

    # Check backends
    backends = data.get("backends") or {}
    stale_backends = [
        instance_id
        for instance_id, entry in backends.items()
        if not _is_entry_alive(entry)
    ]
    for instance_id in stale_backends:
        del backends[instance_id]
        changed = True
    if not backends and "backends" in data:
        data.pop("backends", None)

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
