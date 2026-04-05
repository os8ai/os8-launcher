"""Backend lifecycle — start, stop, health-check, and status."""

import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

from src.config import Config, ConfigError
from src.credentials import get_ngc_key, prompt_ngc_key
from src.preflight import (
    check_docker,
    check_nvidia_container_toolkit,
    check_nvidia_gpu,
    run_checks,
)
from src.runtime import check_port, expand_template, build_env_for_venv, parse_command
from src.state import (
    validate_state,
    set_backend,
    clear_backend,
    clear_client,
    is_process_alive,
    is_container_running,
)


class BackendError(Exception):
    """Raised when a backend operation fails."""


def _check_model_downloaded(model_path: Path) -> bool:
    """Check if model weights are present on disk."""
    if not model_path.exists():
        return False
    return any(model_path.rglob("*"))


def _build_variables(model, backend, config, repo_root: Path) -> dict:
    """Build the template variables dict for a backend run command."""
    variables = {
        "port": str(backend.port),
        "model_path": str(repo_root / model.path),
    }

    # NIM-specific
    if model.nim_image:
        variables["nim_image"] = model.nim_image
    ngc_key = get_ngc_key()
    if ngc_key:
        variables["ngc_api_key"] = ngc_key

    # Container image from manifest
    if backend.manifest and "image" in backend.manifest.fields:
        variables["image"] = backend.manifest.fields["image"]

    # Binary-specific
    if backend.manifest and "binary" in backend.manifest.fields:
        variables["binary"] = str(repo_root / backend.manifest.fields["binary"])

    return variables


def _inject_docker_flags(cmd: list[str], name: str) -> list[str]:
    """Insert -d --rm --name os8-{name} after 'docker run' in the command."""
    if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "run":
        return cmd[:2] + ["-d", "--rm", "--name", f"os8-{name}"] + cmd[2:]
    return cmd


def _start_container(cmd_string: str, backend_name: str) -> str:
    """Start a Docker container in detached mode. Returns container ID."""
    cmd = parse_command(cmd_string)
    cmd = _inject_docker_flags(cmd, backend_name)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise BackendError(
            f"Failed to start container:\n  {result.stderr.strip()}"
        )

    container_id = result.stdout.strip()
    if not container_id:
        raise BackendError("Docker returned no container ID.")

    return container_id


def _start_pip_process(cmd_string: str, venv_path: Path, extra_env: dict | None = None) -> subprocess.Popen:
    """Start a pip-based backend as a background subprocess."""
    cmd = parse_command(cmd_string)
    env = build_env_for_venv(venv_path)
    if extra_env:
        env.update(extra_env)

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    return process


def _start_binary_process(cmd_string: str) -> subprocess.Popen:
    """Start a binary backend as a background subprocess."""
    cmd = parse_command(cmd_string)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    return process


def _wait_for_healthy(
    port: int,
    check_alive_fn,
    timeout: int = 300,
    interval: int = 3,
    initial_delay: int = 5,
):
    """Poll the backend's health endpoint until it responds.

    Args:
        port: The port to check.
        check_alive_fn: Callable that returns True if the process is still alive.
        timeout: Maximum seconds to wait.
        interval: Seconds between attempts.
        initial_delay: Seconds to wait before first attempt.
    """
    url = f"http://localhost:{port}/v1/models"

    print(f"  Waiting for backend to be ready on port {port}...", end="", flush=True)
    time.sleep(initial_delay)

    start = time.time()
    while time.time() - start < timeout:
        # Check if process is still alive
        if not check_alive_fn():
            print(" failed.")
            raise BackendError(
                "Backend process exited unexpectedly.\n"
                "Check logs for details."
            )

        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                print(" ready.")
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            print(".", end="", flush=True)
            time.sleep(interval)

    print(" timed out.")
    raise BackendError(
        f"Health check timed out after {timeout}s.\n"
        f"The model may still be loading. Check logs:\n"
        f"  docker logs os8-<backend>   (for containers)\n"
        f"  or check process output     (for pip/binary backends)"
    )


def start_backend(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
):
    """Start a serving backend for the given model."""
    # 1. Validate state — check nothing is already running
    state = validate_state()
    if "backend" in state:
        b = state["backend"]
        raise BackendError(
            f"{b['name']} is already serving {b['model']} on port {b['port']}.\n"
            f"Run ./launcher stop first."
        )

    # 2. Resolve model and backend
    model = config.get_model(model_name)
    backend_name = backend_name or model.default_backend

    if backend_name not in model.backends:
        raise BackendError(
            f"Backend '{backend_name}' is not compatible with model '{model_name}'.\n"
            f"Compatible backends: {', '.join(model.backends)}"
        )

    backend = config.get_backend(backend_name)
    manifest = backend.manifest
    if not manifest:
        raise BackendError(f"Backend '{backend_name}' has no manifest.")

    # 3. Check model is downloaded (for NIM, cache dir just needs to exist)
    model_path = repo_root / model.path
    if manifest.install_type != "container":
        if not _check_model_downloaded(model_path):
            raise BackendError(
                f"Model '{model_name}' is not downloaded.\n"
                f"Run: ./launcher models download {model_name}"
            )

    # 4. Preflight checks
    checks = []
    if manifest.install_type == "container":
        checks.append(("Docker", check_docker()))
        checks.append(("NVIDIA Container Toolkit", check_nvidia_container_toolkit()))
        checks.append(("NVIDIA GPU", check_nvidia_gpu()))
    if not run_checks(checks):
        raise BackendError("Prerequisites not met.")

    # 5. Check port
    if check_port(backend.port):
        raise BackendError(
            f"Port {backend.port} is already in use.\n"
            f"Check with: lsof -i :{backend.port}"
        )

    # 6. Build variables and expand template
    run_template = manifest.fields.get("run")
    if not run_template:
        raise BackendError(f"Manifest for '{backend_name}' has no 'run' field.")

    variables = _build_variables(model, backend, config, repo_root)

    # NIM needs NGC auth
    if manifest.install_type == "container" and "nvcr.io" in (model.nim_image or ""):
        ngc_key = get_ngc_key()
        if not ngc_key:
            ngc_key = prompt_ngc_key()
        if not ngc_key:
            raise BackendError("NGC API key is required to run NIM.")
        variables["ngc_api_key"] = ngc_key

    cmd_string = expand_template(run_template, variables)

    # 7. Start the process
    print(f"Starting {backend_name} for {model_name}...")
    container_id = None
    pid = None

    if manifest.install_type == "container":
        # Create cache dir for NIM
        model_path.mkdir(parents=True, exist_ok=True)
        container_id = _start_container(cmd_string, backend_name)
        print(f"  Container started: {container_id[:12]}")
        check_alive = lambda: is_container_running(container_id)

    elif manifest.install_type == "pip":
        venv_rel = manifest.fields.get("venv")
        if not venv_rel:
            raise BackendError(f"Manifest for '{backend_name}' missing 'venv' field.")
        venv_path = repo_root / venv_rel
        if not venv_path.exists():
            raise BackendError(
                f"Backend '{backend_name}' is not installed.\n"
                f"Run: ./launcher setup {backend_name}"
            )
        extra_env = manifest.fields.get("env") if manifest.fields.get("env") else None
        process = _start_pip_process(cmd_string, venv_path, extra_env)
        pid = process.pid
        print(f"  Process started: PID {pid}")
        check_alive = lambda: is_process_alive(pid)

    elif manifest.install_type == "binary":
        binary_rel = manifest.fields.get("binary")
        if binary_rel and not (repo_root / binary_rel).exists():
            raise BackendError(
                f"Backend '{backend_name}' is not installed.\n"
                f"Run: ./launcher setup {backend_name}"
            )
        process = _start_binary_process(cmd_string)
        pid = process.pid
        print(f"  Process started: PID {pid}")
        check_alive = lambda: is_process_alive(pid)

    else:
        raise BackendError(f"Unknown install_type: {manifest.install_type}")

    # 8. Record state
    set_backend(
        name=backend_name,
        model=model_name,
        port=backend.port,
        install_type=manifest.install_type,
        pid=pid,
        container_id=container_id,
    )

    # 9. Health check
    try:
        _wait_for_healthy(backend.port, check_alive, timeout=900)
    except (BackendError, KeyboardInterrupt):
        print("\n  Cleaning up...")
        stop_backend()
        raise

    # 10. Report success
    print(f"\nBackend ready: {model_name} via {backend_name}")
    print(f"  URL: http://localhost:{backend.port}/v1")


def stop_backend():
    """Stop the running backend."""
    state = validate_state()
    backend = state.get("backend")

    if not backend:
        print("No backend running.")
        return

    name = backend["name"]
    install_type = backend.get("install_type")
    container_id = backend.get("container_id")
    pid = backend.get("pid")

    print(f"Stopping {name}...")

    if install_type == "container" and container_id:
        subprocess.run(
            ["docker", "stop", f"os8-{name}"],
            capture_output=True, timeout=45,
        )
        print(f"  Container stopped.")

    elif pid:
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait up to 10 seconds for graceful shutdown
            for _ in range(20):
                if not is_process_alive(pid):
                    break
                time.sleep(0.5)
            else:
                # Still alive — force kill
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
            print(f"  Process stopped.")
        except ProcessLookupError:
            print(f"  Process already exited.")

    clear_backend()


def stop_all():
    """Stop the backend and all tracked clients."""
    state = validate_state()

    # Stop clients first
    clients = state.get("clients", {})
    for client_name, entry in list(clients.items()):
        container_id = entry.get("container_id")
        if container_id:
            print(f"Stopping client {client_name}...")
            subprocess.run(
                ["docker", "stop", f"os8-{client_name}"],
                capture_output=True, timeout=45,
            )
            print(f"  Stopped.")
        clear_client(client_name)

    # Stop backend
    if "backend" in state:
        stop_backend()
    elif not clients:
        print("Nothing running.")


def _format_uptime(start_time_str: str) -> str:
    """Format uptime as a human-readable string."""
    try:
        start = datetime.fromisoformat(start_time_str)
        delta = datetime.now() - start
        total_seconds = int(delta.total_seconds())

        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m {total_seconds % 60}s"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    except (ValueError, TypeError):
        return "unknown"


def _check_health(port: int) -> str:
    """Quick one-shot health check. Returns 'healthy' or 'unhealthy'."""
    try:
        url = f"http://localhost:{port}/v1/models"
        with urllib.request.urlopen(url, timeout=3):
            return "healthy"
    except Exception:
        return "unhealthy"


def get_status_data() -> dict:
    """Return structured status for JSON serialization."""
    state = validate_state()
    backend = state.get("backend")
    clients = state.get("clients", {})

    result = {"backend": None, "clients": {}}

    if backend:
        health = _check_health(backend["port"])
        uptime = _format_uptime(backend.get("start_time", ""))
        result["backend"] = {
            "name": backend["name"],
            "model": backend["model"],
            "port": backend["port"],
            "health": health,
            "uptime": uptime,
            "start_time": backend.get("start_time"),
            "install_type": backend.get("install_type"),
        }

    for name, entry in clients.items():
        result["clients"][name] = {
            "port": entry.get("port"),
            "uptime": _format_uptime(entry.get("start_time", "")),
            "start_time": entry.get("start_time"),
        }

    return result


def get_status() -> str:
    """Get a formatted status string for all running services."""
    state = validate_state()
    backend = state.get("backend")
    clients = state.get("clients", {})

    if not backend and not clients:
        return "Nothing running."

    lines = []

    if backend:
        health = _check_health(backend["port"])
        uptime = _format_uptime(backend.get("start_time", ""))
        lines.append(f"Backend: {backend['name']} serving {backend['model']}")
        lines.append(f"  URL:    http://localhost:{backend['port']}/v1")
        lines.append(f"  Health: {health}")
        lines.append(f"  Uptime: {uptime}")

    if clients:
        lines.append("")
        lines.append("Clients:")
        for name, entry in clients.items():
            uptime = _format_uptime(entry.get("start_time", ""))
            port_str = f"http://localhost:{entry['port']}" if entry.get("port") else ""
            lines.append(f"  {name:<15} {port_str:<30} running  {uptime}")

    return "\n".join(lines)
