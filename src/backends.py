"""Backend lifecycle — start, stop, health-check, and status."""

import os
import signal
import subprocess
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

from src.actionlog import (
    log_start, log_ready, log_stopped, log_fail,
    log_group_start, log_group_done,
)
from src.config import Config, ConfigError
from src.credentials import (
    get_ngc_key, prompt_ngc_key,
    get_hf_token, prompt_hf_token,
)


def _lookup_credential(name: str) -> str | None:
    """Resolve a credential by manifest name. Returns None if unset."""
    if name == "ngc_api_key":
        return get_ngc_key()
    if name == "hf_token":
        return get_hf_token()
    return None


def _prompt_credential(name: str) -> str | None:
    """Interactively prompt for a credential by manifest name."""
    if name == "ngc_api_key":
        return prompt_ngc_key()
    if name == "hf_token":
        return prompt_hf_token()
    return None
from src.preflight import (
    check_docker,
    check_nvidia_container_toolkit,
    check_nvidia_gpu,
    resolve_image,
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
from src.verification import record_success, record_failure


class BackendError(Exception):
    """Raised when a backend operation fails."""


def served_model_name(model, backend) -> str:
    """Return the name the backend exposes to API clients.

    Honors `model_name_template` from the manifest if set (e.g. ollama uses
    "{ollama_tag}"). Falls back to model.name.
    """
    template = None
    if backend and backend.manifest:
        template = backend.manifest.fields.get("model_name_template")
    if template:
        # Expand against the model's __dict__ so manifests can reference any
        # ModelConfig field by name.
        try:
            value = template.format(**_model_template_vars(model))
            if value:
                return value
        except KeyError:
            pass
    return model.name


def _model_template_vars(model) -> dict:
    """Expose model fields to manifest templates as a flat dict."""
    return {
        "name": model.name,
        "source": model.source,
        "path": model.path,
        "format": model.format,
        "nim_image": model.nim_image or "",
        "ollama_tag": model.ollama_tag or "",
        # Generic alias used by post_start templates so manifests don't need
        # to know which specific tag field a backend uses.
        "model_tag": model.ollama_tag or model.name,
    }


def _run_post_start(manifest, variables: dict, repo_root: Path):
    """Run an optional post_start hook declared in a manifest.

    Shape:
        post_start:
          wait_for: <url>           (optional — poll until 200)
          wait_timeout: <seconds>   (default 30)
          run: <command template>   (expanded against `variables`)
    """
    post = manifest.fields.get("post_start") if manifest else None
    if not post:
        return

    wait_for = post.get("wait_for")
    if wait_for:
        wait_url = expand_template(wait_for, variables)
        wait_timeout = int(post.get("wait_timeout") or 30)
        print(f"  Waiting for {wait_url} ...")
        for _ in range(wait_timeout):
            try:
                with urllib.request.urlopen(wait_url, timeout=2):
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        else:
            raise BackendError(f"post_start wait_for timed out: {wait_url}")

    run_template = post.get("run")
    if run_template:
        cmd_string = expand_template(run_template, variables)
        print(f"  post_start: {cmd_string}")
        result = subprocess.run(parse_command(cmd_string))
        if result.returncode != 0:
            raise BackendError(f"post_start command failed: {cmd_string}")


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
        "model_name": model.name,
        "repo_root": str(repo_root),
    }

    # Pre-create persistent backend cache dir so docker doesn't create it
    # root-owned. Used by manifests that mount {repo_root}/var/cache/<backend>
    # to keep torch.compile / kernel caches across container restarts.
    cache_dir = repo_root / "var" / "cache" / backend.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Expose model fields generically so manifests can reference any of them.
    variables.update(_model_template_vars(model))

    # Inject credentials declared in the manifest. Each name maps to a
    # function in src.credentials. Missing credentials are simply omitted —
    # if the run template references one and it isn't set, expand_template
    # will raise a clear error at start time.
    for cred_name in (backend.manifest.fields.get("credentials") or []) if backend.manifest else []:
        value = _lookup_credential(cred_name)
        if value:
            variables[cred_name] = value

    # Container image from manifest (arch-aware: prefers image_{machine})
    if backend.manifest and "image" in backend.manifest.fields:
        variables["image"] = resolve_image(backend.manifest.fields)

    # Binary-specific
    if backend.manifest and "binary" in backend.manifest.fields:
        variables["binary"] = str(repo_root / backend.manifest.fields["binary"])

    # Per-model env vars become "-e KEY=VAL" flags injected into the run template.
    env_flags = " ".join(
        f"-e {k}={v}" for k, v in (model.backend_env or {}).items()
    )
    variables["backend_env_flags"] = env_flags
    variables["backend_args"] = model.backend_args or ""

    return variables


def _inject_docker_flags(cmd: list[str], name: str) -> list[str]:
    """Insert -d --rm --name os8-{name} after 'docker run' in the command."""
    if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "run":
        return cmd[:2] + ["-d", "--rm", "--name", f"os8-{name}"] + cmd[2:]
    return cmd


def _image_present(image: str) -> bool:
    """Return True if the image is already cached locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0


def _ensure_image_present(image: str, ngc_key: str | None = None):
    """If the image isn't cached, log in (if needed) and pull it.

    Streams progress to stdout so the user can see what's happening on a
    multi-GB pull. No timeout — large NIM images can legitimately take 30+ min.
    """
    if _image_present(image):
        return

    if "nvcr.io" in image:
        if not ngc_key:
            raise BackendError(
                f"Image {image} is not cached locally and requires an NGC API "
                f"key to pull. Set NGC_API_KEY in Settings first."
            )
        print(f"Logging in to nvcr.io...")
        login = subprocess.run(
            ["docker", "login", "nvcr.io",
             "--username", "$oauthtoken",
             "--password-stdin"],
            input=ngc_key, capture_output=True, text=True, timeout=30,
        )
        if login.returncode != 0:
            raise BackendError(
                f"Failed to log in to nvcr.io. Check your NGC API key.\n"
                f"  {login.stderr.strip()}"
            )

    print(f"Pulling image: {image}")
    print(f"This is a one-time download and may take a while.")
    pull = subprocess.run(["docker", "pull", image])
    if pull.returncode != 0:
        raise BackendError(f"Failed to pull image: {image}")
    print(f"Image pulled successfully.")


def container_log_path(backend_name: str, repo_root: Path) -> Path:
    """Return the on-disk path where this backend's container logs are streamed."""
    return repo_root / "var" / "backends" / f"{backend_name}.log"


def _start_container_log_streamer(
    container_name: str,
    log_path: Path,
) -> subprocess.Popen | None:
    """Spawn `docker logs -f` in the background, redirecting all output to
    log_path. Returns the Popen handle (caller doesn't need to manage it —
    `docker logs -f` exits naturally when the container stops).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Truncate at start so the file reflects the current run, not history.
    fh = open(log_path, "wb")
    try:
        return subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            stdout=fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError:
        fh.close()
        return None


def _start_container(cmd_string: str, backend_name: str) -> str:
    """Start a Docker container in detached mode. Returns container ID."""
    cmd = parse_command(cmd_string)
    cmd = _inject_docker_flags(cmd, backend_name)

    # Detached run is near-instant once the image is local; 60s is generous
    # for slow filesystems but tight enough to fail fast on real problems.
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise BackendError(
            f"Failed to start container:\n  {result.stderr.strip()}"
        )

    container_id = result.stdout.strip()
    if not container_id:
        raise BackendError("Docker returned no container ID.")

    return container_id


def _open_backend_log(log_path: Path):
    """Open a backend log file for writing, creating parents and truncating."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return open(log_path, "wb")


def _start_pip_process(
    cmd_string: str,
    venv_path: Path,
    extra_env: dict | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """Start a pip-based backend as a background subprocess.

    Output is redirected to `log_path` (or DEVNULL) rather than to PIPEs —
    PIPEs whose parent has exited can SIGPIPE the child once full, killing
    daemons after `./launcher serve` returns.
    """
    cmd = parse_command(cmd_string)
    env = build_env_for_venv(venv_path)
    if extra_env:
        env.update(extra_env)

    out = _open_backend_log(log_path) if log_path else subprocess.DEVNULL
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process


def _start_binary_process(
    cmd_string: str,
    extra_env: dict | None = None,
    log_path: Path | None = None,
) -> subprocess.Popen:
    """Start a binary backend as a background subprocess.

    Output is redirected to `log_path` (or DEVNULL) rather than to PIPEs —
    PIPEs whose parent has exited can SIGPIPE the child once full, killing
    daemons after `./launcher serve` returns.
    """
    cmd = parse_command(cmd_string)

    env = None
    if extra_env:
        env = {**os.environ, **extra_env}

    out = _open_backend_log(log_path) if log_path else subprocess.DEVNULL
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process


def _wait_for_healthy(
    port: int,
    check_alive_fn,
    timeout: int = 300,
    interval: int = 3,
    initial_delay: int = 5,
    container_name: str | None = None,
    log_path: Path | None = None,
    health_path: str = "/v1/models",
):
    """Poll the backend's health endpoint until it responds.

    Args:
        port: The port to check.
        check_alive_fn: Callable that returns True if the process is still alive.
        timeout: Maximum seconds to wait.
        interval: Seconds between attempts.
        initial_delay: Seconds to wait before first attempt.
        container_name: If set, periodically tail the container's logs into the
            launcher's stdout so the dashboard log panel shows real backend
            progress instead of just polling dots.
    """
    url = f"http://localhost:{port}{health_path}"

    print(f"  Waiting for backend to be ready on port {port}...", flush=True)
    time.sleep(initial_delay)

    start = time.time()
    last_log_dump = 0.0
    log_interval = 15  # seconds between container log snapshots
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
                print("  Backend ready.")
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            # Periodically tail container logs so the user sees real progress.
            # Prefer the on-disk streamed log (full fidelity, no extra docker
            # calls); fall back to `docker logs --tail` for backends that
            # don't have a streamer.
            now = time.time()
            if (log_path or container_name) and now - last_log_dump >= log_interval:
                last_log_dump = now
                elapsed = int(now - start)
                tail: list[str] = []
                if log_path and log_path.exists():
                    try:
                        with open(log_path, "rb") as fh:
                            try:
                                fh.seek(-4096, 2)
                            except OSError:
                                fh.seek(0)
                            data = fh.read().decode("utf-8", errors="replace")
                        tail = data.strip().splitlines()
                    except OSError:
                        pass
                elif container_name:
                    try:
                        logs = subprocess.run(
                            ["docker", "logs", "--tail", "5", container_name],
                            capture_output=True, text=True, timeout=5,
                        )
                        tail = (logs.stdout + logs.stderr).strip().splitlines()
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                if tail:
                    print(f"  [+{elapsed}s] container log tail:")
                    for line in tail[-5:]:
                        print(f"    | {line}")
            time.sleep(interval)

    print("  Health check timed out.")
    log_hint = (
        f"  tail -f {log_path}   (streamed container log)\n"
        if log_path
        else "  docker logs os8-<backend>   (for containers)\n"
    )
    raise BackendError(
        f"Health check timed out after {timeout}s.\n"
        f"The model may still be loading. Check logs:\n"
        f"{log_hint}"
        f"  or check process output     (for pip/binary backends)"
    )


def start_backend(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
):
    """Start a serving backend for the given model."""
    log_start("backend", f"{backend_name or '?'} for {model_name}")
    try:
        _start_backend_inner(model_name, backend_name, config, repo_root)
    except Exception as e:
        log_fail("backend", model_name, e)
        raise
    log_ready("backend", model_name)


def _start_backend_inner(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
):
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

    # 3. Check model is downloaded. Backends that manage their own model store
    #    (download.type: daemon-pull) or that download on first run from a
    #    container (install_type: container) skip the on-disk check.
    model_path = repo_root / model.path
    download_type = (manifest.fields.get("download") or {}).get("type")
    skip_disk_check = (
        manifest.install_type == "container"
        or download_type == "daemon-pull"
    )
    if not skip_disk_check:
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

    # If the run template references a declared credential that wasn't
    # picked up by _build_variables (because it isn't set yet), prompt for it.
    declared_creds = (manifest.fields.get("credentials") or [])
    for cred_name in declared_creds:
        if cred_name in variables:
            continue
        if "{" + cred_name + "}" not in run_template:
            continue
        value = _prompt_credential(cred_name)
        if not value:
            raise BackendError(
                f"Credential '{cred_name}' is required by backend '{backend_name}'."
            )
        variables[cred_name] = value

    cmd_string = expand_template(run_template, variables)

    # 7. Start the process
    print(f"Starting {backend_name} for {model_name}...")
    container_id = None
    pid = None

    if manifest.install_type == "container":
        # Ensure the cache dir exists so the container can mount it.
        model_path.mkdir(parents=True, exist_ok=True)
        # Ensure the image is cached locally before starting; pulls if missing.
        # Per-model image fields (declared via manifest's download.image_field)
        # take precedence over the manifest's static `image` field.
        download_cfg = manifest.fields.get("download") or {}
        image_field = download_cfg.get("image_field")
        image_to_check = None
        if image_field:
            image_to_check = getattr(model, image_field, None)
        if not image_to_check:
            image_to_check = resolve_image(manifest.fields) if manifest.fields else None
        if image_to_check:
            ngc_key_for_pull = get_ngc_key() if "nvcr.io" in image_to_check else None
            _ensure_image_present(image_to_check, ngc_key=ngc_key_for_pull)
        container_id = _start_container(cmd_string, backend_name)
        print(f"  Container started: {container_id[:12]}")
        # Stream the container's full logs to disk so the user (and the
        # dashboard) can tail real progress instead of relying on the
        # 5-line snapshots emitted during the health-check loop.
        log_path = container_log_path(backend_name, repo_root)
        _start_container_log_streamer(f"os8-{backend_name}", log_path)
        print(f"  Streaming logs to: {log_path.relative_to(repo_root)}")
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
        log_path = container_log_path(backend_name, repo_root)
        process = _start_pip_process(cmd_string, venv_path, extra_env, log_path=log_path)
        pid = process.pid
        print(f"  Streaming logs to: {log_path.relative_to(repo_root)}")
        print(f"  Process started: PID {pid}")
        check_alive = lambda: is_process_alive(pid)

    elif manifest.install_type == "binary":
        binary_rel = manifest.fields.get("binary")
        if binary_rel and not (repo_root / binary_rel).exists():
            raise BackendError(
                f"Backend '{backend_name}' is not installed.\n"
                f"Run: ./launcher setup {backend_name}"
            )
        binary_env = manifest.fields.get("env") or {}
        log_path = container_log_path(backend_name, repo_root)
        process = _start_binary_process(cmd_string, extra_env=binary_env, log_path=log_path)
        pid = process.pid
        print(f"  Streaming logs to: {log_path.relative_to(repo_root)}")
        print(f"  Process started: PID {pid}")
        check_alive = lambda: is_process_alive(pid)

    else:
        raise BackendError(f"Unknown install_type: {manifest.install_type}")

    health_path = manifest.fields.get("health_path") or "/v1/models"

    # 8. Record state
    set_backend(
        name=backend_name,
        model=model_name,
        port=backend.port,
        install_type=manifest.install_type,
        pid=pid,
        container_id=container_id,
        health_path=health_path,
    )

    # 9. Post-start hook (optional) — runs before the health check.
    # Used by backends like ollama that need to pull a tag once their
    # daemon is up but before they're considered "ready to serve".
    _run_post_start(manifest, variables, repo_root)

    # 10. Health check
    # Manifests can override the default timeout (e.g. NIM cold-start can take
    # 30-60+ minutes while TensorRT-LLM engine files download from NGC).
    health_timeout = int(manifest.fields.get("health_timeout") or 900)
    container_name_for_logs = f"os8-{backend_name}" if container_id else None
    # Every backend now writes its logs to disk (containers via streamer,
    # binary/pip via direct redirection), so the wait loop can tail real
    # progress regardless of install type.
    log_path_for_wait = container_log_path(backend_name, repo_root)
    try:
        _wait_for_healthy(
            backend.port,
            check_alive,
            timeout=health_timeout,
            container_name=container_name_for_logs,
            log_path=log_path_for_wait,
            health_path=health_path,
        )
    except BackendError as e:
        print("\n  Cleaning up...")
        record_failure(model_name, backend_name, str(e))
        stop_backend()
        raise
    except KeyboardInterrupt:
        print("\n  Cleaning up...")
        stop_backend()
        raise

    # 10. Record verification + report success
    runtime_id = (
        model.nim_image
        or (resolve_image(manifest.fields) if manifest.fields else None)
        or manifest.install_type
    )
    record_success(model_name, backend_name, runtime=runtime_id)
    print(f"\nBackend ready: {model_name} via {backend_name}")
    print(f"  URL:      http://localhost:{backend.port}")
    print(f"  API base: http://localhost:{backend.port}/v1")


def stop_backend():
    """Stop the running backend."""
    state = validate_state()
    backend = state.get("backend")

    if not backend:
        print("No backend running.")
        return

    name = backend["name"]
    log_start("backend", name)
    try:
        _stop_backend_inner(backend)
    except Exception as e:
        log_fail("backend", name, e)
        raise
    log_stopped("backend", name)


def _stop_backend_inner(backend: dict):
    name = backend["name"]
    install_type = backend.get("install_type")
    container_id = backend.get("container_id")
    pid = backend.get("pid")

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
    clients = state.get("clients", {})

    if not clients and "backend" not in state:
        print("Nothing running.")
        return

    log_group_start("Stopping all services")
    try:
        # Stop clients first
        for client_name, entry in list(clients.items()):
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

        # Stop backend
        if "backend" in state:
            stop_backend()
    except Exception as e:
        log_fail("group", "all services", e)
        raise
    log_group_done("All services stopped")


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


def _check_health(port: int, health_path: str = "/v1/models") -> str:
    """Quick one-shot health check. Returns 'healthy' or 'unhealthy'."""
    try:
        url = f"http://localhost:{port}{health_path}"
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
        health = _check_health(backend["port"], backend.get("health_path") or "/v1/models")
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
            "ready": entry.get("ready", True),  # legacy entries assumed ready
        }

    return result


def get_status() -> str:
    """Get a formatted status string for all running services.

    Thin formatter over get_status_data() — that function is the single
    source of truth for what's running. Output format is preserved
    byte-for-byte for any scripts parsing it.
    """
    data = get_status_data()
    backend = data["backend"]
    clients = data["clients"]

    if not backend and not clients:
        return "Nothing running."

    lines = []

    if backend:
        lines.append(f"Backend: {backend['name']} serving {backend['model']}")
        lines.append(f"  URL:    http://localhost:{backend['port']}")
        lines.append(f"  API:    http://localhost:{backend['port']}/v1")
        lines.append(f"  Health: {backend['health']}")
        lines.append(f"  Uptime: {backend['uptime']}")

    if clients:
        lines.append("")
        lines.append("Clients:")
        for name, entry in clients.items():
            port_str = f"http://localhost:{entry['port']}" if entry.get("port") else ""
            lines.append(f"  {name:<15} {port_str:<30} running  {entry['uptime']}")

    return "\n".join(lines)
