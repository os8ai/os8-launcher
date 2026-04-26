"""Backend lifecycle — start, stop, health-check, and status."""

import os
import re
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


# Default output reserve returned alongside max_model_len in
# /api/serve/ensure responses. vLLM's effective output cap depends on
# runtime params (--max-num-seqs, batch composition) we don't model here;
# 4K is a safe default for any client that wants to budget input vs. output
# against the model's input window.
_DEFAULT_OUTPUT_RESERVE = 4096


_TOTAL_GPU_MEMORY_GB_CACHE: float | None = None


def _total_gpu_memory_gb() -> float:
    """Total GPU memory in GB, cached for the process lifetime.

    On unified-memory hardware like the DGX Spark, nvidia-smi reports
    memory.total as `[N/A]` (no dedicated VRAM). In that case we fall
    back to /proc/meminfo's MemTotal — accurate, since the GPU is
    sharing that exact pool with the CPU.

    Falls back to 0.0 only when both lookups fail.
    """
    global _TOTAL_GPU_MEMORY_GB_CACHE
    if _TOTAL_GPU_MEMORY_GB_CACHE is not None:
        return _TOTAL_GPU_MEMORY_GB_CACHE
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0].strip()
            if line and line != "[N/A]":
                try:
                    _TOTAL_GPU_MEMORY_GB_CACHE = float(line) / 1024.0
                    return _TOTAL_GPU_MEMORY_GB_CACHE
                except ValueError:
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # Unified memory fallback — applies to Grace-Blackwell, Apple Silicon,
    # and any system where nvidia-smi can't report dedicated VRAM.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    _TOTAL_GPU_MEMORY_GB_CACHE = kb / (1024.0 * 1024.0)
                    return _TOTAL_GPU_MEMORY_GB_CACHE
    except (OSError, ValueError):
        pass

    _TOTAL_GPU_MEMORY_GB_CACHE = 0.0
    return _TOTAL_GPU_MEMORY_GB_CACHE


def _parse_gpu_memory_utilization(backend_args: str) -> float | None:
    """Extract the float value of --gpu-memory-utilization from backend_args.

    vLLM accepts both `--gpu-memory-utilization 0.6` and the `=` form.
    Returns None when the flag is absent.
    """
    if not backend_args:
        return None
    m = re.search(r"--gpu-memory-utilization[\s=]([0-9]*\.?[0-9]+)", backend_args)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def compute_effective_size_gb(model) -> float:
    """Admission-accounting weight for a model.

    Priority:
      1. Explicit `effective_size_gb` on the model — user override.
      2. `--gpu-memory-utilization X` in backend_args, times total GPU
         memory. vLLM pre-allocates this fraction up front for KV cache,
         so on a unified-memory box that's also RAM the OS won't get back.
         We take max(weights, fraction*total) — the larger of "what the
         model declares" and "what the backend will actually grab".
      3. Fall back to model.size_gb.

    Returns 0.0 only when both size_gb and the derivation fail (no
    nvidia-smi, no flag) — admission still works against the budget but
    is effectively unguarded for that model.
    """
    if getattr(model, "effective_size_gb", None):
        return float(model.effective_size_gb)
    declared = float(getattr(model, "size_gb", None) or 0)
    util = _parse_gpu_memory_utilization(getattr(model, "backend_args", "") or "")
    if util is not None:
        total_gb = _total_gpu_memory_gb()
        if total_gb > 0:
            return max(declared, util * total_gb)
    return declared


def _parse_max_model_len(backend_args: str) -> int | None:
    """Extract the integer value of --max-model-len from a backend_args string.

    Returns None when the flag is absent. Tolerates both `--max-model-len 65536`
    and `--max-model-len=65536` forms. Backends that don't accept this flag
    (e.g. ollama, llama.cpp without explicit window override) yield None and
    callers should fall back to whatever the runtime defaults to.
    """
    if not backend_args:
        return None
    m = re.search(r"--max-model-len[\s=](\d+)", backend_args)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


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
from src.runtime import (
    allocate_port,
    expand_template,
    build_env_for_venv,
    parse_command,
    PortAllocationError,
)
from src.state import (
    validate_state,
    set_backend,
    clear_backend,
    clear_client,
    is_process_alive,
    is_container_running,
    get_primary_backend,
    compute_instance_id,
)

import threading
from src.verification import record_success, record_failure


class BackendError(Exception):
    """Raised when a backend operation fails.

    `code` is an optional machine-readable tag (e.g. BUDGET_EXCEEDED,
    START_FAILED) that API handlers surface to callers so OS8 can show
    differentiated toasts without parsing the human-readable message.
    """

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


# Coarse guard around the "check state + kick off start" critical section
# in ensure_backend. FastAPI dispatches sync handlers on a threadpool, so
# two ensures for the same instance landing at the same time would both
# see 'missing from state' and both kick off starts. The lock serializes
# the decision; actual loads run in a background task after the state
# write, so the critical section is ~microseconds.
_ensure_lock = threading.Lock()


def _is_configured_resident(config, model_name: str, backend_name: str) -> bool:
    """True if this (model, backend) pair matches any option of any role in
    `config.resident`. Used by ensure_backend to auto-promote user-driven
    ensures into resident pins when the user declared them resident in
    config.yaml — so a direct /api/serve/ensure for a configured resident
    doesn't accidentally become an eviction target for the next ensure.

    Matches every option of every resident role (not just the currently-
    selected one). That way the inactive options of a multi-option role
    (e.g. AEON-7 when chat is set to Qwen) still get resident pinning when
    a user spins them up via /api/serve/ensure — important because
    /api/triplet/role/apply transiently loads the *new* selection while
    the *old* one is being torn down.

    Each option's backend is matched literally; an option with no explicit
    `backend:` matches any backend for that model — same default-backend
    semantics as the rest of the launcher.
    """
    resident_roles = config.resident or []
    roles = config.roles or {}
    for role_name in resident_roles:
        role = roles.get(role_name)
        if not role:
            continue
        for opt in role.options:
            if opt.model != model_name:
                continue
            if opt.backend is None or opt.backend == backend_name:
                return True
    return False


# Per-model → OS8-task eligibility. Drives /api/status/capabilities. The
# ensure flow doesn't read this table — OS8 asks for a specific model by
# name — so missing an entry just means a model is invisible to OS8's
# capability-based discovery, not that it can't be served.
_MODEL_ELIGIBILITY: dict[str, list[str]] = {
    "gemma-4-31B-it-nvfp4": ["conversation", "summary", "planning"],
    "gemma-4-E2B-it": ["conversation", "summary"],
    "qwen3-coder-30b": ["coding", "jobs"],
    "qwen3-coder-next": ["coding", "jobs"],
    "kokoro-v1": ["tts"],
    "fish-s2-pro": ["tts"],
    "flux1-schnell": ["image-gen"],
    "flux1-dev": ["image-gen"],
    "flux1-kontext-dev": ["image-edit"],
    "qwen3-6-35b-a3b": ["conversation", "vision"],
    "aeon-7-gemma-4-26b": ["conversation", "summary", "planning", "coding", "jobs"],
}


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
        "vllm_image": model.vllm_image or "",
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


def _build_variables(model, backend, config, repo_root: Path, port: int | None = None) -> dict:
    """Build the template variables dict for a backend run command.

    `port` overrides backend.port when the allocator has assigned a
    non-default port to a sibling instance. Leaves backend.port un-
    mutated so subsequent allocations against the same BackendConfig
    keep seeing the configured default.
    """
    assigned_port = port if port is not None else backend.port
    variables = {
        "port": str(assigned_port),
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

    # Container image from manifest (arch-aware: prefers image_{machine}).
    # If the manifest declares download.image_field, a per-model override of
    # that name takes precedence — lets one model on vLLM run under a custom
    # image (e.g. a nightly with a new arch) while other models keep the
    # manifest default. The same hook gates auto-pull in start_backend().
    if backend.manifest and "image" in backend.manifest.fields:
        image_field = (backend.manifest.fields.get("download") or {}).get("image_field")
        per_model = getattr(model, image_field, None) if image_field else None
        variables["image"] = per_model or resolve_image(backend.manifest.fields)

    # Binary-specific
    if backend.manifest and "binary" in backend.manifest.fields:
        variables["binary"] = str(repo_root / backend.manifest.fields["binary"])

    # Per-model env vars become "-e KEY=VAL" flags injected into the run template.
    env_flags = " ".join(
        f"-e {k}={v}" for k, v in (model.backend_env or {}).items()
    )
    variables["backend_env_flags"] = env_flags
    variables["backend_args"] = model.backend_args or ""

    # For GGUF models, resolve the weights dir to the canonical entry-point
    # file. llama.cpp's -m flag needs a specific .gguf file (given the first
    # shard, it auto-loads siblings); a directory won't work. Non-GGUF
    # backends simply don't reference {model_file} in their run template.
    weights_dir = repo_root / model.path
    if weights_dir.exists():
        ggufs = sorted(weights_dir.rglob("*.gguf"))
        if ggufs:
            first_shard = next(
                (p for p in ggufs if "-00001-of-" in p.name), ggufs[0]
            )
            variables["model_file"] = str(first_shard)

    return variables


def _inject_docker_flags(cmd: list[str], name: str) -> list[str]:
    """Insert -d --rm --name os8-{name} after 'docker run' in the command."""
    if len(cmd) >= 2 and cmd[0] == "docker" and cmd[1] == "run":
        return cmd[:2] + ["-d", "--rm", "--name", f"os8-{name}"] + cmd[2:]
    return cmd


def _docker_rm_force(container_name: str):
    """Force-remove a (possibly-stopped) Docker container by name, ignoring
    "no such container" errors.

    Used as pre-start cleanup so a host crash that left a stopped container
    (with --rm not having fired) doesn't block the next start with a
    "container name already in use" error.
    """
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True, timeout=30,
    )


def _image_present(image: str) -> bool:
    """Return True if the image is already cached locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0


def _build_local_image(image: str, build_info: dict, manifest, repo_root: Path):
    """Build a local image from a Dockerfile declared in the manifest's
    image_builds block. Paths in build_info are relative to the manifest dir.
    """
    dockerfile = build_info.get("dockerfile")
    if not dockerfile:
        raise BackendError(
            f"image_builds entry for {image} missing 'dockerfile'."
        )
    manifest_dir = Path(manifest.path).parent
    dockerfile_path = manifest_dir / dockerfile
    context = manifest_dir / (build_info.get("context") or ".")

    if not dockerfile_path.exists():
        raise BackendError(f"Dockerfile not found: {dockerfile_path}")

    print(f"Building image: {image}")
    print(f"  Dockerfile: {dockerfile_path.relative_to(repo_root)}")
    print(f"  Context:    {context.relative_to(repo_root)}")
    print(f"This is a one-time build and may take several minutes.")
    build = subprocess.run(
        ["docker", "build", "-f", str(dockerfile_path), "-t", image, str(context)],
    )
    if build.returncode != 0:
        raise BackendError(f"Failed to build image: {image}")
    print(f"Image built successfully.")


def _ensure_image_present(
    image: str,
    ngc_key: str | None = None,
    manifest=None,
    repo_root: Path | None = None,
):
    """If the image isn't cached, build it (if declared in the manifest's
    image_builds) or pull it from the registry. Streams progress to stdout so
    the user can see what's happening. No timeout — large NIM images can
    legitimately take 30+ min.
    """
    if _image_present(image):
        return

    # Prefer local build if the manifest declares one — avoids a confusing
    # "Failed to pull" error for images that only exist as Dockerfiles in-tree.
    if manifest and repo_root:
        builds = (manifest.fields.get("image_builds") or {}) if manifest else {}
        if image in builds:
            _build_local_image(image, builds[image], manifest, repo_root)
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
        hint = ""
        if manifest and (manifest.fields.get("image_builds") or {}):
            hint = (
                f"\nIf {image} is a local build target, declare it in the "
                f"backend manifest's image_builds: block so the launcher "
                f"builds it instead of pulling."
            )
        raise BackendError(f"Failed to pull image: {image}{hint}")
    print(f"Image pulled successfully.")


def container_log_path(instance_id: str, repo_root: Path) -> Path:
    """Return the on-disk path where an instance's logs are streamed.

    Keyed by instance_id (not backend kind) so two instances of the same
    backend — say, two vLLM processes serving different models — don't
    collide into the same log file.
    """
    return repo_root / "var" / "backends" / f"{instance_id}.log"


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
    cwd: Path | None = None,
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
        cwd=str(cwd) if cwd else None,
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


class LeftoversFound(BackendError):
    """Raised when survey_leftovers() finds state from a previous run.

    Carries the structured `findings` list so callers (CLI / API) can
    decide how to surface it: CLI prompts, API returns 409 + JSON.
    """
    def __init__(self, findings: list, message: str | None = None):
        super().__init__(message or "Leftover state detected", code="LEFTOVERS_FOUND")
        self.findings = findings


def act_on_leftover(action: dict) -> dict:
    """Execute one leftover action returned by survey_leftovers().

    Returns {"ok": bool, "detail": str}. Never raises — the caller is
    typically iterating over multiple actions and wants to surface
    per-item results.

    Action types:
      - clear_state: remove the named instance_id from state.yaml
      - remove_container: `docker rm -f <id_or_name>`
      - kill_pid: SIGTERM, then SIGKILL after a 2s grace period
    """
    kind = action.get("type")
    try:
        if kind == "clear_state":
            instance_id = action.get("instance_id")
            if not instance_id:
                return {"ok": False, "detail": "missing instance_id"}
            clear_backend(instance_id=instance_id)
            return {"ok": True, "detail": f"cleared state entry {instance_id}"}

        if kind == "remove_container":
            target = action.get("name") or action.get("id")
            if not target:
                return {"ok": False, "detail": "missing name/id"}
            r = subprocess.run(
                ["docker", "rm", "-f", target],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0:
                return {"ok": True, "detail": f"removed container {target}"}
            return {"ok": False, "detail": (r.stderr or r.stdout).strip()}

        if kind == "kill_pid":
            pid = int(action.get("pid", 0))
            if pid <= 1:
                return {"ok": False, "detail": "refusing to signal pid <= 1"}
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return {"ok": True, "detail": f"pid {pid} already gone"}
            except PermissionError:
                return {"ok": False, "detail": f"no permission to signal pid {pid}"}
            for _ in range(20):  # 2s grace
                time.sleep(0.1)
                if not is_process_alive(pid):
                    return {"ok": True, "detail": f"sent SIGTERM, pid {pid} exited"}
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return {"ok": True, "detail": f"pid {pid} exited just before SIGKILL"}
            except PermissionError:
                return {"ok": False, "detail": f"no permission to SIGKILL pid {pid}"}
            return {"ok": True, "detail": f"sent SIGKILL to pid {pid}"}

        return {"ok": False, "detail": f"unknown action type: {kind}"}
    except Exception as e:  # any unexpected failure surfaces back to the caller
        return {"ok": False, "detail": f"{type(e).__name__}: {e}"}


def survey_for_start(
    config: Config,
    target_ports: list[int] | None = None,
) -> list[dict]:
    """Convenience wrapper around preflight.survey_leftovers that knows
    how to derive target ports from config when none are supplied.

    With `target_ports=None`, the survey skips the port-conflict check —
    use this when surveying generically (e.g. status endpoint). The
    start paths pass the specific port(s) they're about to bind.
    """
    from src.preflight import survey_leftovers
    return survey_leftovers(target_ports=target_ports or [])


def start_backend(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
    resident: bool = False,
    skip_leftover_check: bool = False,
):
    """Start a serving backend for the given model.

    `resident=True` persists a flag on the state entry so the instance
    is exempt from LRU eviction. Used by the resident-set auto-start.

    `skip_leftover_check=True` bypasses the survey_for_start gate. The
    triplet auto-start uses this — it runs ONE survey at the top and
    handles findings itself, rather than re-running per role.
    """
    if not skip_leftover_check:
        try:
            backend = config.get_backend(backend_name or config.get_model(model_name).default_backend)
            target_port = backend.port
        except Exception:
            target_port = None
        findings = survey_for_start(config, target_ports=[target_port] if target_port else None)
        if findings:
            raise LeftoversFound(findings)

    log_start("backend", f"{backend_name or '?'} for {model_name}")
    try:
        _start_backend_inner(model_name, backend_name, config, repo_root, resident=resident)
    except Exception as e:
        log_fail("backend", model_name, e)
        raise
    log_ready("backend", model_name)


def _start_backend_inner(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
    resident: bool = False,
):
    # 1. Resolve model and backend
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

    instance_id = compute_instance_id(backend_name, model_name)

    # 2. Validate state — refuse only if *this exact instance* is already
    # running. Two different (backend, model) pairs can coexist under the
    # Phase 2 port allocator.
    state = validate_state()
    existing_instances = state.get("backends") or {}
    if instance_id in existing_instances:
        b = existing_instances[instance_id]
        raise BackendError(
            f"{b['name']} is already serving {b['model']} on port {b['port']}.\n"
            f"Stop it with: ./launcher stop (stops everything) or wait for idempotent "
            f"/api/serve/ensure to no-op."
        )

    # 3. Check model is downloaded. Backends that manage their own model store
    #    (download.type: daemon-pull / none) or that download on first run from
    #    a container (install_type: container) skip the on-disk check.
    model_path = repo_root / model.path
    download_type = (manifest.fields.get("download") or {}).get("type")
    skip_disk_check = (
        manifest.install_type == "container"
        or download_type in ("daemon-pull", "none")
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

    # 5. Allocate a port. Priority: per-instance override > configured
    # default > first free in a +10 scan window. The ports other running
    # instances already hold are implicitly honored via check_port — they
    # bind their sockets, so the scan skips them.
    try:
        assigned_port = allocate_port(
            backend_name, model_name, backend.port,
        )
    except PortAllocationError as e:
        raise BackendError(str(e))
    if assigned_port != backend.port:
        print(f"  Port {backend.port} in use; allocator picked {assigned_port}.")

    # 6. Build variables and expand template
    run_template = manifest.fields.get("run")
    if not run_template:
        raise BackendError(f"Manifest for '{backend_name}' has no 'run' field.")

    variables = _build_variables(model, backend, config, repo_root, port=assigned_port)

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
            _ensure_image_present(
                image_to_check,
                ngc_key=ngc_key_for_pull,
                manifest=manifest,
                repo_root=repo_root,
            )
        # Clear any stale container left over from a previous crash. With
        # --rm containers are usually auto-removed, but a host crash or
        # OOM kill can leave a stopped container with this name and block
        # the next start. Per-instance name (os8-{instance_id}) keeps this
        # from colliding with a sibling that's legitimately running.
        _docker_rm_force(f"os8-{instance_id}")
        container_id = _start_container(cmd_string, instance_id)
        print(f"  Container started: {container_id[:12]}")
        # Stream the container's full logs to disk so the user (and the
        # dashboard) can tail real progress instead of relying on the
        # 5-line snapshots emitted during the health-check loop.
        log_path = container_log_path(instance_id, repo_root)
        _start_container_log_streamer(f"os8-{instance_id}", log_path)
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
        raw_env = manifest.fields.get("env") or {}
        extra_env = {k: expand_template(v, variables) for k, v in raw_env.items()} if raw_env else None
        log_path = container_log_path(instance_id, repo_root)
        cwd_rel = manifest.fields.get("cwd")
        cwd = repo_root / cwd_rel if cwd_rel else None
        process = _start_pip_process(cmd_string, venv_path, extra_env, log_path=log_path, cwd=cwd)
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
        log_path = container_log_path(instance_id, repo_root)
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
        port=assigned_port,
        install_type=manifest.install_type,
        pid=pid,
        container_id=container_id,
        health_path=health_path,
        instance_id=instance_id,
        size_gb=model.size_gb,
        effective_size_gb=compute_effective_size_gb(model),
        resident=resident,
    )

    # 9. Post-start hook (optional) — runs before the health check.
    # Used by backends like ollama that need to pull a tag once their
    # daemon is up but before they're considered "ready to serve".
    _run_post_start(manifest, variables, repo_root)

    # 10. Health check
    # Manifests can override the default timeout (e.g. NIM cold-start can take
    # 30-60+ minutes while TensorRT-LLM engine files download from NGC).
    health_timeout = int(manifest.fields.get("health_timeout") or 900)
    container_name_for_logs = f"os8-{instance_id}" if container_id else None
    # Every backend now writes its logs to disk (containers via streamer,
    # binary/pip via direct redirection), so the wait loop can tail real
    # progress regardless of install type.
    log_path_for_wait = container_log_path(instance_id, repo_root)
    try:
        _wait_for_healthy(
            assigned_port,
            check_alive,
            timeout=health_timeout,
            container_name=container_name_for_logs,
            log_path=log_path_for_wait,
            health_path=health_path,
        )
    except BackendError as e:
        print("\n  Cleaning up...")
        record_failure(model_name, backend_name, str(e))
        stop_backend(instance_id=instance_id)
        raise
    except KeyboardInterrupt:
        print("\n  Cleaning up...")
        stop_backend(instance_id=instance_id)
        raise

    # 10. Record verification + report success
    runtime_id = (
        model.nim_image
        or (resolve_image(manifest.fields) if manifest.fields else None)
        or manifest.install_type
    )
    record_success(model_name, backend_name, runtime=runtime_id)
    print(f"\nBackend ready: {model_name} via {backend_name}")
    print(f"  URL:      http://localhost:{assigned_port}")
    print(f"  API base: http://localhost:{assigned_port}/v1")


def stop_backend(instance_id: str | None = None):
    """Stop a running backend instance.

    With `instance_id`, stops that specific instance. Without, stops the
    primary (most-recently-started) instance — back-compat for single-
    backend callers. Raises BackendError if `instance_id` doesn't match
    any known instance.
    """
    state = validate_state()
    backends = state.get("backends") or {}

    if instance_id is not None:
        entry = backends.get(instance_id)
        if not entry:
            print(f"Instance '{instance_id}' is not running.")
            return
        target = entry
    else:
        target = get_primary_backend(state)
        if not target:
            print("No backend running.")
            return

    name = target["name"]
    log_start("backend", name)
    try:
        _stop_backend_inner(target)
    except Exception as e:
        log_fail("backend", name, e)
        raise
    log_stopped("backend", name)


def _stop_backend_inner(backend: dict):
    name = backend["name"]
    instance_id = backend.get("instance_id") or compute_instance_id(name, backend.get("model", ""))
    install_type = backend.get("install_type")
    container_id = backend.get("container_id")
    pid = backend.get("pid")

    if install_type == "container" and container_id:
        # Per-instance container name — a sibling backend on the same
        # kind (e.g. a second vllm) won't be affected.
        subprocess.run(
            ["docker", "stop", f"os8-{instance_id}"],
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

    clear_backend(instance_id)


def stop_all():
    """Stop every backend and all tracked clients."""
    state = validate_state()
    clients = state.get("clients", {})
    backends = state.get("backends") or {}

    if not clients and not backends:
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

        # Stop every backend. launcher-1 only ever has ≤1 entry here,
        # but iterating keeps the call correct once launcher-2 lifts the
        # single-backend guard.
        for entry in list(backends.values()):
            name = entry["name"]
            log_start("backend", name)
            try:
                _stop_backend_inner(entry)
            except Exception as e:
                log_fail("backend", name, e)
                raise
            log_stopped("backend", name)
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
    """Quick one-shot health check. Returns 'healthy' or 'unhealthy'.

    Timeout 5s because vLLM's /v1/models can stall briefly during
    warmup-adjacent moments on large models (qwen3-6-35b a real example) —
    3s gave sporadic false-negatives where curl answered 200 immediately
    but the status poll reported unhealthy.
    """
    try:
        url = f"http://localhost:{port}{health_path}"
        with urllib.request.urlopen(url, timeout=5):
            return "healthy"
    except Exception:
        return "unhealthy"


# Instances whose start has been scheduled but haven't yet called
# set_backend — guards against two concurrent ensures both kicking off a
# start because neither has persisted state yet. Keyed by instance_id;
# value is the declared model.size_gb so admission accounting can count
# pending starts against the budget without waiting for set_backend.
_starting_instances: dict[str, float] = {}


def _entry_accounting_size(entry: dict) -> float:
    """Pull the right size field off a state entry for admission accounting.

    Prefers the persisted `effective_size_gb` (set by start_backend after
    deriving from `--gpu-memory-utilization`). Falls back to declared
    `size_gb` for entries written by an older launcher.
    """
    eff = entry.get("effective_size_gb")
    if eff is not None:
        return float(eff)
    return float(entry.get("size_gb") or 0)


def _reserved_gb(state_data: dict, kv_margin: float) -> float:
    """Sum (effective_size_gb + kv_margin) over every running AND pending
    instance.

    An instance is added to `_starting_instances` at admission time and
    removed only when the start fully completes (~90s for vLLM). Between
    `set_backend` (which writes to state.yaml) and that completion, an
    instance exists in BOTH places. Skip the _starting_instances entry
    when state already has it — otherwise admission double-counts
    in-flight starts and rejects subsequent ensures under a budget that
    would otherwise fit.
    """
    backends = state_data.get("backends") or {}
    total = 0.0
    for entry in backends.values():
        total += _entry_accounting_size(entry) + kv_margin
    for instance_id, size in _starting_instances.items():
        if instance_id in backends:
            continue  # already counted via state
        total += float(size) + kv_margin
    return total


def _live_mem_available_gb() -> float:
    """Live MemAvailable from /proc/meminfo, in GB. 0.0 if unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except (OSError, ValueError):
        pass
    return 0.0


def _admit_or_evict(
    instance_id: str,
    new_size_gb: float,
    resources,
) -> list[str]:
    """Run under _ensure_lock. Evict LRU non-resident instances until
    the new instance fits, or raise BUDGET_EXCEEDED. Returns the list
    of evicted instance_ids (empty if the admission fit without
    eviction) — the caller can surface it to the user.

    Two gates run in order:

      1. Static budget — sum(effective_size_gb + kv_margin) over all
         running + pending instances must stay under
         resources.memory_budget_gb. Eviction can free room here.

      2. Live RAM floor — `MemAvailable` from /proc/meminfo must remain
         at least `ram_safety_floor_gb` above the new instance's
         effective size after admission. This catches the case where
         the static math says yes but the OS is already squeezed
         (other apps, kernel caches, GPU memory we didn't track).
         Eviction CAN help here too — freeing a sibling drops their
         held memory back to the OS.

    Candidates for eviction: running non-resident instances that aren't
    already mid-stop. A pending `_starting_instances` entry is never
    evicted (nothing to stop yet) and contributes its size to the
    reserved total.
    """
    budget = resources.memory_budget_gb
    kv_margin = resources.kv_margin_gb
    floor = float(getattr(resources, "ram_safety_floor_gb", 4.0))
    needed = new_size_gb + kv_margin
    evicted: list[str] = []

    def _try_evict(reason: str, detail: str) -> bool:
        """Pick the LRU non-resident victim and stop it. Returns True if
        we evicted something, False if there's no candidate (caller
        should raise BUDGET_EXCEEDED)."""
        running = state.get("backends") or {}
        candidates = {
            k: v for k, v in running.items()
            if not v.get("resident") and k != instance_id
        }
        if not candidates:
            raise BackendError(
                f"{reason}: {detail} and all running instances are "
                f"resident-pinned (nothing to evict).",
                code="BUDGET_EXCEEDED",
            )
        def _lru_key(k: str) -> str:
            v = candidates[k]
            return v.get("last_used") or v.get("start_time") or ""
        victim = min(candidates.keys(), key=_lru_key)
        print(f"  LRU eviction ({reason}): stopping {victim} to make room for {instance_id}")
        stop_backend(instance_id=victim)
        evicted.append(victim)
        return True

    # Gate 1: static budget — loop with eviction.
    while True:
        state = validate_state()
        reserved = _reserved_gb(state, kv_margin)
        if reserved + needed <= budget:
            break
        _try_evict(
            "Budget exceeded",
            f"would reserve {reserved + needed:.1f} GB but limit is {budget:.1f} GB",
        )

    # Gate 2: live RAM — also loops with eviction in case freeing a
    # sibling actually returns memory to the OS. Skip when needed is 0
    # (couldn't size the model — we'd false-fail every admission).
    if needed > 0 and floor > 0:
        while True:
            avail = _live_mem_available_gb()
            if avail == 0.0:
                # Couldn't read meminfo. Don't block on a measurement
                # we can't trust; admission falls back to static budget.
                break
            # The new instance will eat `new_size_gb` of MemAvailable at
            # load time; we need `floor` left over after that.
            if avail >= new_size_gb + floor:
                break
            try:
                _try_evict(
                    "RAM floor",
                    f"only {avail:.1f} GB MemAvailable; need {new_size_gb + floor:.1f} GB "
                    f"({new_size_gb:.1f} for the new instance + {floor:.1f} GB safety floor)",
                )
            except BackendError as e:
                # Re-raise with a code the caller can render distinctly.
                raise BackendError(str(e), code="RAM_FLOOR")
            state = validate_state()  # refresh after eviction

    return evicted


def ensure_backend(
    model_name: str,
    backend_name: str | None,
    config: Config,
    repo_root: Path,
    wait: bool = False,
    schedule_start: "callable | None" = None,
    resident: bool | None = None,
    skip_leftover_check: bool = False,
) -> dict:
    """Make sure the (backend, model) instance is running. Idempotent.

    Return shape:
        {"status": "ready"|"loading",
         "instance_id": ..., "port": int|None, "base_url": str|None,
         "model": ..., "backend": ..., "evicted": [instance_id, ...],
         "max_model_len": int|None, "output_reserve": int}

    `max_model_len` is the model's input window in tokens, parsed from
    `--max-model-len` in the model's `backend_args`. None when the backend
    doesn't declare one (e.g. ollama, llama.cpp without an override) — the
    client should fall back to a conservative default.

    `output_reserve` is a flat per-request reservation for output tokens
    (default 4096); clients use it to compute "how much input fits" as
    `max_model_len - output_reserve - client_overhead`.

    `ready` means a state entry exists AND its health endpoint responds
    200. `loading` means either a state entry exists but isn't yet
    healthy, or a start has just been scheduled.

    Before scheduling/starting, runs admission against
    config.resources.memory_budget_gb. When over budget, evicts LRU
    non-resident siblings (via stop_backend) until the new instance
    fits. Raises BackendError(code=BUDGET_EXCEEDED) when the resident
    set already exceeds the budget and no eviction candidates remain.

    `resident` controls LRU protection:
      - `True`  — explicitly pin this instance; it becomes ineligible
                  for eviction.
      - `False` — explicitly non-resident; can be evicted by a later
                  ensure under memory pressure.
      - `None`  (default) — auto-detect from `config.resident`: if the
                  (model, backend) pair matches one of the configured
                  resident roles, this is pinned. Otherwise non-resident.
                  This lets direct /api/serve/ensure calls for the
                  user's declared resident triplet inherit protection
                  without every caller having to pass resident=True.

    With `wait=True`, runs the start synchronously. With `wait=False`
    (default), schedules via `schedule_start` (typically FastAPI's
    background_tasks.add_task wrapped in _run_with_log_capture).
    """
    # Resolve up front so we fail fast on bad input, before any lock.
    model = config.get_model(model_name)
    resolved_backend = backend_name or model.default_backend
    if resolved_backend not in model.backends:
        raise BackendError(
            f"Backend '{resolved_backend}' is not compatible with "
            f"model '{model_name}'. Compatible: {', '.join(model.backends)}"
        )
    backend = config.get_backend(resolved_backend)
    instance_id = compute_instance_id(resolved_backend, model_name)
    size_gb = compute_effective_size_gb(model)

    # Survey for leftover state from a previous run BEFORE the lock — this
    # raises before we account against the budget, so a stale entry doesn't
    # also trip eviction. Skip when the caller has already handled it
    # (e.g. _auto_start_resident_set runs one survey at the top).
    if not skip_leftover_check:
        target_port = backend.port
        from src.preflight import survey_leftovers
        findings = survey_leftovers(target_ports=[target_port] if target_port else None)
        # Filter out a stale entry for this exact instance — start_backend
        # already calls stop_backend for that, and the user shouldn't be
        # asked to confirm before idempotently retrying their own start.
        findings = [f for f in findings
                    if not (f["kind"] == "stale_state"
                            and f.get("action", {}).get("instance_id") == instance_id)]
        if findings:
            raise LeftoversFound(findings)
    # Surface the model's input window so clients can size their requests
    # accordingly (e.g. opencode's `limit.context` field). None when the
    # backend doesn't declare one — clients should fall back to a conservative
    # default rather than assuming a window.
    max_model_len = _parse_max_model_len(model.backend_args or "")
    output_reserve = _DEFAULT_OUTPUT_RESERVE

    # Resolve resident flag: explicit wins; None → look at config.resident.
    if resident is None:
        resident = _is_configured_resident(config, model_name, resolved_backend)

    evicted: list[str] = []

    with _ensure_lock:
        state = validate_state()
        backends_running = state.get("backends") or {}
        existing = backends_running.get(instance_id)
        if existing:
            port = existing.get("port")
            health_path = existing.get("health_path") or "/v1/models"
            status = "ready" if _check_health(port, health_path) == "healthy" else "loading"
            return {
                "status": status,
                "instance_id": instance_id,
                "port": port,
                "base_url": f"http://localhost:{port}" if port else None,
                "model": model_name,
                "backend": resolved_backend,
                "evicted": [],
                "max_model_len": max_model_len,
                "output_reserve": output_reserve,
            }

        if instance_id in _starting_instances:
            # Another ensure already scheduled this one; don't double-start.
            return {
                "status": "loading",
                "instance_id": instance_id,
                "port": None,
                "base_url": None,
                "model": model_name,
                "backend": resolved_backend,
                "evicted": [],
                "max_model_len": max_model_len,
                "output_reserve": output_reserve,
            }

        # Admission: evict LRU non-residents until we fit, or raise.
        evicted = _admit_or_evict(instance_id, size_gb, config.resources)

        if not wait:
            if schedule_start is None:
                raise BackendError(
                    "ensure_backend with wait=False requires schedule_start."
                )
            _starting_instances[instance_id] = size_gb

    # Released lock — actual start runs here (sync if wait=True, else
    # dispatched via schedule_start).
    def _run_and_unmark():
        try:
            start_backend(
                model_name, resolved_backend, config, repo_root,
                resident=resident,
                skip_leftover_check=True,
            )
        finally:
            with _ensure_lock:
                _starting_instances.pop(instance_id, None)

    if wait:
        with _ensure_lock:
            _starting_instances[instance_id] = size_gb
        try:
            start_backend(
                model_name, resolved_backend, config, repo_root,
                resident=resident,
                skip_leftover_check=True,
            )
        finally:
            with _ensure_lock:
                _starting_instances.pop(instance_id, None)
        state = validate_state()
        entry = (state.get("backends") or {}).get(instance_id)
        if not entry:
            raise BackendError(
                f"ensure(wait=True): {instance_id} vanished after start."
            )
        port = entry.get("port")
        return {
            "status": "ready",
            "instance_id": instance_id,
            "port": port,
            "base_url": f"http://localhost:{port}",
            "model": model_name,
            "backend": resolved_backend,
            "evicted": evicted,
            "max_model_len": max_model_len,
            "output_reserve": output_reserve,
        }

    # wait=False: schedule the start and return 'loading' immediately.
    schedule_start(_run_and_unmark)
    return {
        "status": "loading",
        "instance_id": instance_id,
        "port": None,
        "base_url": None,
        "model": model_name,
        "backend": resolved_backend,
        "evicted": evicted,
        "max_model_len": max_model_len,
        "output_reserve": output_reserve,
    }


def get_capabilities_data() -> dict:
    """Return {task_type: [entry, ...]} for all currently-running
    backends, derived from the _MODEL_ELIGIBILITY table.

    Each entry carries enough for OS8 to bypass `/api/status` entirely
    for task-based routing: {instance_id, model, base_url, model_id,
    priority}. Launcher-3 keeps `priority: 0` for everything; launcher-4
    (or a follow-up) may use the resident set to bias priority.

    Shape change from Phase-1: values are arrays, not objects, because
    with N running backends multiple instances can cover a given task.
    """
    state = validate_state()
    backends = state.get("backends") or {}
    per_task: dict[str, list[dict]] = {}
    for entry in backends.values():
        model = entry.get("model")
        port = entry.get("port")
        if not model or not port:
            continue
        eligible = _MODEL_ELIGIBILITY.get(model, [])
        for task in eligible:
            per_task.setdefault(task, []).append({
                "instance_id": entry.get("instance_id"),
                "model": model,
                "base_url": f"http://localhost:{port}",
                "model_id": model,
                "priority": 0,
            })
    return per_task


def get_status_data() -> dict:
    """Return structured status for JSON serialization.

    Wire shape (Phase 2):
        {
          "backends": [ {instance_id, name, model, port, health, ...}, ... ],
          "backend": {...} | null,   # back-compat shim: the primary instance
          "clients": {name: {...}}
        }

    The singular `backend` key preserves the pre-Phase-2 contract for
    clients like Phase-1 OS8 that still consume it. It mirrors the most-
    recently-started backend (or null when nothing is running). Callers
    with multi-instance awareness should read `backends`; `backend` will
    be removed after OS8 migrates off the shim (Phase 3).
    """
    state = validate_state()
    backends = state.get("backends") or {}
    clients = state.get("clients", {})

    def _describe(entry: dict) -> dict:
        port = entry.get("port")
        health = _check_health(port, entry.get("health_path") or "/v1/models")
        uptime = _format_uptime(entry.get("start_time", ""))
        return {
            "instance_id": entry.get("instance_id"),
            "name": entry["name"],
            "model": entry["model"],
            "port": port,
            "base_url": f"http://localhost:{port}" if port else None,
            "health": health,
            "uptime": uptime,
            "start_time": entry.get("start_time"),
            "install_type": entry.get("install_type"),
            "resident": bool(entry.get("resident")),
            "size_gb": entry.get("size_gb"),
        }

    backends_list = [_describe(e) for e in backends.values()]
    # Sort so the "primary" (most-recently-started) is last-in in a stable way.
    backends_list.sort(key=lambda b: b.get("start_time") or "")

    # Back-compat shim — single-backend callers read the most-recently-
    # started entry via get_status_data()["backend"].
    primary = backends_list[-1] if backends_list else None

    result = {
        "backends": backends_list,
        "backend": primary,
        "clients": {},
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
