"""Client launcher — start clients connected to a running backend."""

import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.actionlog import log_start, log_ready, log_stopped, log_fail
from src.config import Config, ConfigError
from src.preflight import check_docker, run_checks
from src.runtime import check_port, expand_template, build_env_for_venv, parse_command
from src.state import (
    validate_state,
    set_client,
    mark_client_ready,
    clear_client,
    is_container_running,
    get_primary_backend,
)


def _wait_for_client_port(port: int, container_id: str, timeout: int = 120) -> bool:
    """Poll a client's port until it accepts an HTTP response, or the
    container dies, or the timeout expires.

    A bare TCP-accept isn't enough — Open WebUI binds the port well before
    its app is actually serving, and a click during that window still 502s.
    We require an HTTP response (any status code) to call it ready.

    Returns True if ready, False on timeout or container death.
    """
    url = f"http://localhost:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_container_running(container_id):
            return False
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2):
                return True
        except urllib.error.HTTPError:
            # Any HTTP status — including 4xx/5xx — means the app is up
            # enough to respond. Good enough to expose the link.
            return True
        except (urllib.error.URLError, socket.timeout, ConnectionError, OSError):
            time.sleep(1)
    return False


class ClientError(Exception):
    """Raised when a client operation fails."""


def _get_running_backend(state: dict) -> dict:
    """Get a running backend from state, or raise.

    Clients today assume exactly one backend is running. Phase 2's
    storage schema supports many, but launcher-1 keeps the runtime guard
    so this still picks the single entry (via the primary-backend shim).
    """
    backend = get_primary_backend(state)
    if not backend:
        raise ClientError(
            "No backend running.\n"
            "Start one with: ./launcher serve <model>"
        )
    return backend


def _start_attached(
    cmd_string: str,
    venv_path: Path | None,
    cwd: Path | None = None,
    extra_env: dict | None = None,
):
    """Start a client in the foreground (attached to terminal)."""
    cmd = parse_command(cmd_string)

    env = None
    if venv_path:
        env = build_env_for_venv(venv_path)
    if extra_env:
        import os as _os
        if env is None:
            env = _os.environ.copy()
        env.update(extra_env)

    try:
        subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None)
    except KeyboardInterrupt:
        print()


def _start_detached_container(
    cmd_string: str,
    client_name: str,
) -> str:
    """Start a Docker container client in detached mode. Returns container ID."""
    cmd = parse_command(cmd_string)

    # Insert -d --rm --name after 'docker run'
    if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "run":
        cmd = cmd[:2] + ["-d", "--rm", "--name", f"os8-{client_name}"] + cmd[2:]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise ClientError(
            f"Failed to start {client_name} container:\n  {result.stderr.strip()}"
        )

    container_id = result.stdout.strip()
    if not container_id:
        raise ClientError("Docker returned no container ID.")

    return container_id


def start_client(
    client_name: str,
    config: Config,
    repo_root: Path,
    model: str | None = None,
    backend_name: str | None = None,
):
    """Start a client connected to the running backend."""
    log_start("client", client_name)
    try:
        _start_client_inner(client_name, config, repo_root, model, backend_name)
    except Exception as e:
        log_fail("client", client_name, e)
        raise
    log_ready("client", client_name)


def _start_client_inner(
    client_name: str,
    config: Config,
    repo_root: Path,
    model: str | None = None,
    backend_name: str | None = None,
):
    """If `model` (and optionally `backend_name`) are provided and no backend
    is currently running, auto-start the backend first so the client launches
    against it. If a backend is already running, the model/backend hints are
    ignored — we never restart a working backend.
    """
    state = validate_state()
    if not get_primary_backend(state) and model:
        # Auto-start the backend the user picked in the launch controls.
        from src.backends import start_backend
        start_backend(model, backend_name, config, repo_root)
        state = validate_state()

    backend = _get_running_backend(state)
    backend_port = backend["port"]
    backend_model = backend["model"]
    backend_name_actual = backend["name"]

    # Resolve client
    client = config.get_client(client_name)

    # --- Bridge ---
    if client.type == "bridge":
        print(f"{client_name} connection info:")
        print(f"  API base: http://localhost:{backend_port}/v1")
        print(f"  Model:    {backend_model}")
        print(f"Configure {client_name}'s local model adapter to point at this URL.")
        return

    manifest = client.manifest
    if not manifest:
        raise ClientError(f"Client '{client_name}' has no manifest.")

    run_template = manifest.fields.get("run")
    if not run_template:
        raise ClientError(f"Manifest for '{client_name}' has no 'run' field.")

    # Build variables
    from src.backends import served_model_name
    model_cfg = config.get_model(backend_model)
    backend_cfg = config.get_backend(backend_name_actual)
    variables = {
        "port": str(backend_port),
        "backend_port": str(backend_port),
        "backend_name": backend_name_actual,
        "served_model_name": served_model_name(model_cfg, backend_cfg),
        "repo_root": str(repo_root),
    }
    if client.port:
        variables["port"] = str(client.port)
    if "image" in manifest.fields:
        variables["image"] = manifest.fields["image"]

    cmd_string = expand_template(run_template, variables)

    # Manifest-defined env vars (templated). Used for clients like opencode
    # that take their config via an env var instead of CLI flags. Values are
    # plain string-replaced (not .format()) so JSON braces don't collide
    # with the template-variable syntax.
    manifest_env: dict[str, str] = {}
    raw_env = manifest.fields.get("env") or {}
    if isinstance(raw_env, dict):
        for k, v in raw_env.items():
            sv = str(v)
            for var_k, var_v in variables.items():
                sv = sv.replace("{" + var_k + "}", var_v)
            manifest_env[str(k)] = sv

    # --- pip + attached ---
    if manifest.install_type == "pip":
        venv_rel = manifest.fields.get("venv")
        venv_path = None
        if venv_rel:
            venv_path = repo_root / venv_rel
            if not venv_path.exists():
                raise ClientError(
                    f"Client '{client_name}' is not installed.\n"
                    f"Run: ./launcher setup {client_name}"
                )

        from src.projects import get_active_project
        active = get_active_project()
        cwd = active.path if active else None

        print(f"Starting {client_name}...")
        print(f"  Connected to {backend['name']} ({backend_model}) on port {backend_port}")
        if active:
            print(f"  Project: {active.name} ({active.path})")
        else:
            print("  Project: (none — running in launcher repo cwd)")
        print()
        _start_attached(cmd_string, venv_path, cwd=cwd, extra_env=manifest_env or None)
        return

    # --- binary + attached ---
    if manifest.install_type == "binary":
        binary_rel = manifest.fields.get("binary")
        if not binary_rel:
            raise ClientError(
                f"Manifest for '{client_name}' missing 'binary' field."
            )
        binary_path = (repo_root / binary_rel).resolve()
        if not binary_path.exists():
            raise ClientError(
                f"Client '{client_name}' is not installed.\n"
                f"Run: ./launcher setup {client_name}"
            )
        # The client may run in a project cwd, so the relative binary path
        # in the manifest won't resolve. Replace it with the absolute path.
        cmd_string = cmd_string.replace(binary_rel, str(binary_path), 1)

        from src.projects import get_active_project
        active = get_active_project()
        cwd = active.path if active else None

        print(f"Starting {client_name}...")
        print(f"  Connected to {backend['name']} ({backend_model}) on port {backend_port}")
        if active:
            print(f"  Project: {active.name} ({active.path})")
        else:
            print("  Project: (none — running in launcher repo cwd)")
        print()
        _start_attached(cmd_string, None, cwd=cwd, extra_env=manifest_env or None)
        return

    # --- container + detached ---
    if manifest.install_type == "container":
        if not run_checks([("Docker", check_docker())]):
            raise ClientError("Prerequisites not met.")

        # Check if already running
        existing_clients = state.get("clients", {})
        if client_name in existing_clients:
            entry = existing_clients[client_name]
            cid = entry.get("container_id")
            if cid and is_container_running(cid):
                port = entry.get("port", "?")
                raise ClientError(
                    f"{client_name} is already running on port {port}.\n"
                    f"Stop it with: ./launcher stop --client {client_name}"
                )

        # Check port
        target_port = client.port
        if target_port and check_port(target_port):
            raise ClientError(
                f"Port {target_port} is already in use.\n"
                f"Check with: lsof -i :{target_port}"
            )

        print(f"Starting {client_name}...")
        container_id = _start_detached_container(cmd_string, client_name)
        print(f"  Container started: {container_id[:12]}")

        # Record in state immediately so the user sees the row in Active
        # Session and can stop it if needed — but mark ready=False so the
        # frontend hides the URL until the app is actually serving.
        set_client(
            name=client_name,
            port=target_port or 0,
            install_type="container",
            container_id=container_id,
            ready=not target_port,  # no port → nothing to wait for
        )

        if target_port:
            print(f"  Waiting for {client_name} to accept connections on port {target_port}...")
            if _wait_for_client_port(target_port, container_id):
                mark_client_ready(client_name)
                print(f"  {client_name} ready at http://localhost:{target_port}")
            else:
                print(
                    f"  {client_name} did not become ready in time. "
                    f"It may still come up — check 'docker logs os8-{client_name}'."
                )
        return

    raise ClientError(f"Unknown install_type '{manifest.install_type}' for client '{client_name}'.")


def stop_client(client_name: str):
    """Stop a specific detached client."""
    state = validate_state()
    clients = state.get("clients", {})
    entry = clients.get(client_name)

    if not entry:
        print(f"Client '{client_name}' is not running.")
        return

    log_start("client", client_name)
    try:
        container_id = entry.get("container_id")
        if container_id:
            subprocess.run(
                ["docker", "stop", f"os8-{client_name}"],
                capture_output=True, timeout=45,
            )
        clear_client(client_name)
    except Exception as e:
        log_fail("client", client_name, e)
        raise
    log_stopped("client", client_name)
