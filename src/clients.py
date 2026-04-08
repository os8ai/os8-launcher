"""Client launcher — start clients connected to a running backend."""

import subprocess
from pathlib import Path

from src.actionlog import log_start, log_ready, log_stopped, log_fail
from src.config import Config, ConfigError
from src.preflight import check_docker, run_checks
from src.runtime import check_port, expand_template, build_env_for_venv, parse_command
from src.state import (
    validate_state,
    set_client,
    clear_client,
    is_container_running,
)


class ClientError(Exception):
    """Raised when a client operation fails."""


def _get_running_backend(state: dict) -> dict:
    """Get the running backend from state, or raise."""
    backend = state.get("backend")
    if not backend:
        raise ClientError(
            "No backend running.\n"
            "Start one with: ./launcher serve <model>"
        )
    return backend


def _start_attached(cmd_string: str, venv_path: Path | None):
    """Start a client in the foreground (attached to terminal)."""
    cmd = parse_command(cmd_string)

    env = None
    if venv_path:
        env = build_env_for_venv(venv_path)

    try:
        subprocess.run(cmd, env=env)
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
    if not state.get("backend") and model:
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
    variables = {
        "port": str(backend_port),
        "backend_port": str(backend_port),
        "backend_name": backend_name_actual,
        "served_model_name": served_model_name(model_cfg, backend_name_actual),
    }
    if client.port:
        variables["port"] = str(client.port)
    if "image" in manifest.fields:
        variables["image"] = manifest.fields["image"]

    cmd_string = expand_template(run_template, variables)

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

        print(f"Starting {client_name}...")
        print(f"  Connected to {backend['name']} ({backend_model}) on port {backend_port}")
        print()
        _start_attached(cmd_string, venv_path)
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

        # Record in state
        set_client(
            name=client_name,
            port=target_port or 0,
            install_type="container",
            container_id=container_id,
        )

        if target_port:
            print(f"  {client_name} running at http://localhost:{target_port}")
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
