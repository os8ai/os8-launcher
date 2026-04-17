"""Model weight management — download, list, and remove."""

import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path

from src.config import Config, ConfigError, ModelConfig
from src.credentials import get_ngc_key, get_hf_token, prompt_ngc_key, prompt_hf_token
from src.preflight import check_docker, check_disk_space, run_checks
from src.verification import get_for_model


DEFAULT_REQUIRED_GB = 5
CLI_SERVER_ID = "cli"


class ModelError(Exception):
    """Raised when a model operation fails."""


def _get_dir_size_bytes(path: Path) -> int:
    """Get total size of a directory in bytes."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


DOWNLOADING_MARKER = ".downloading"

# --- download-in-flight registry -----------------------------------------
# An in-process set of model names whose download is actively running in
# this Python process.  Used together with the on-disk marker to classify
# downloads as "downloading" (live, belonging to us) vs "interrupted" (marker
# present but the owner is gone — either a dead dashboard process or a
# task that already returned).  See get_models_data() for the decision.

_active_downloads: set[str] = set()
_active_lock = threading.Lock()


def _mark_active(name: str) -> None:
    with _active_lock:
        _active_downloads.add(name)


def _mark_inactive(name: str) -> None:
    with _active_lock:
        _active_downloads.discard(name)


def is_download_active(name: str) -> bool:
    with _active_lock:
        return name in _active_downloads


# --- structured marker file ----------------------------------------------
# The .downloading marker is a JSON blob pinned to a specific server_id
# (dashboard process identity) so a restart can tell its own live downloads
# apart from orphaned ones.  An empty / legacy marker left over from an
# earlier launcher version is parsed as {} — always classified interrupted,
# which is the safe side (Resume button shows, HF resumes skipping files).

def _write_marker(weights_path: Path, server_id: str, error: str | None = None) -> None:
    weights_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "server_id": server_id,
        "started_at": datetime.now().isoformat(),
    }
    if error is not None:
        payload["last_error"] = error
    (weights_path / DOWNLOADING_MARKER).write_text(json.dumps(payload))


def _read_marker(weights_path: Path) -> dict | None:
    p = weights_path / DOWNLOADING_MARKER
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text() or "{}")
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _is_downloaded(model: ModelConfig, repo_root: Path) -> bool:
    """Check if model weights are present on disk (not trusting config field)."""
    weights_path = repo_root / model.path
    if not weights_path.exists():
        return False
    # An in-progress or interrupted download leaves partial files but the
    # marker is still present — do not treat those as downloaded.
    if (weights_path / DOWNLOADING_MARKER).exists():
        return False
    # Has at least one file in it (other than the marker)
    return any(p.name != DOWNLOADING_MARKER for p in weights_path.rglob("*"))


# --- data ---

def get_models_data(config: Config, repo_root: Path, current_server_id: str | None = None) -> list[dict]:
    """Return model info as a list of dicts for JSON serialization.

    ``current_server_id`` is the dashboard's per-process identity.  If
    provided, it's used to decide whether an on-disk marker was written
    by *this* dashboard (and therefore is a live download owned by an
    active BackgroundTask) or by a previous one (orphan — "interrupted").
    When called without a server_id (e.g. from the CLI), we fall back
    to the in-process active set, so CLI `list` still reports accurately
    for downloads running in the same process.
    """
    from src.hf_sizes import resolve_model_expected_bytes
    result = []
    for name, model in config.models.items():
        weights_path = repo_root / model.path
        marker = _read_marker(weights_path)
        size_bytes = _get_dir_size_bytes(weights_path)
        # Prefer the live HF-resolved total (cached after first call);
        # fall back to the declared size_gb hint only if HF is
        # unreachable or the source isn't a HuggingFace repo.
        expected_bytes = resolve_model_expected_bytes(model)
        if expected_bytes is None and model.size_gb:
            expected_bytes = int(model.size_gb * 1024**3)

        last_error: str | None = None
        if marker is not None:
            # A marker exists — decide live vs orphan.
            active = is_download_active(name)
            same_server = (
                current_server_id is not None
                and marker.get("server_id") == current_server_id
            )
            if active and same_server:
                state = "downloading"
            elif active and current_server_id is None:
                # CLI path: no server_id to compare, but our in-process
                # registry says this download is ours and live.
                state = "downloading"
            else:
                state = "interrupted"
                last_error = marker.get("last_error")
        elif _is_downloaded(model, repo_root):
            state = "downloaded"
        else:
            state = "not_downloaded"

        result.append({
            "name": name,
            "source": model.source,
            "format": model.format,
            "backends": model.backends,
            "default_backend": model.default_backend,
            "downloaded": state == "downloaded",
            "state": state,
            "last_error": last_error,
            "size_bytes": size_bytes,
            "size_human": _format_size(size_bytes),
            "expected_bytes": expected_bytes,
            "expected_human": _format_size(expected_bytes) if expected_bytes else None,
            "nim_image": model.nim_image,
            "verification": get_for_model(name),
        })
    return result


# --- list ---

def list_models(config: Config, repo_root: Path):
    """Print a table of all configured models with download status and disk usage."""
    if not config.models:
        print("No models configured.")
        return

    print(f"{'Model':<30} {'Format':<8} {'Status':<15} {'Size':<10} {'Backends'}")
    print("-" * 85)

    for name, model in config.models.items():
        weights_path = repo_root / model.path
        marker = _read_marker(weights_path)
        if marker is not None:
            status = "downloading" if is_download_active(name) else "interrupted"
        elif _is_downloaded(model, repo_root):
            status = "downloaded"
        else:
            status = "not downloaded"
        size = _format_size(_get_dir_size_bytes(weights_path))
        backends = ", ".join(model.backends)
        print(f"{name:<30} {model.format:<8} {status:<15} {size:<10} {backends}")


# --- download ---

def download_model(
    name: str,
    config: Config,
    repo_root: Path,
    backend: str | None = None,
    server_id: str = CLI_SERVER_ID,
):
    """Download model weights using the strategy declared in the backend manifest.

    ``server_id`` identifies which dashboard process owns this download —
    so a later restart can tell its own in-flight downloads apart from
    orphaned ones left behind by a previous dashboard.  CLI callers pass
    the default sentinel; BackgroundTasks from the dashboard pass
    ``app.state.server_id``.
    """
    model = config.get_model(name)
    target_backend = backend or model.default_backend

    if target_backend not in model.backends:
        raise ModelError(
            f"Backend '{target_backend}' is not compatible with model '{name}'.\n"
            f"Compatible backends: {', '.join(model.backends)}"
        )

    backend_cfg = config.get_backend(target_backend)
    manifest = backend_cfg.manifest
    download_cfg = (manifest.fields.get("download") if manifest else None) or {}
    download_type = download_cfg.get("type") or "hf-snapshot"

    weights_path = repo_root / model.path

    if download_type == "image-pull":
        _download_image(model, download_cfg, weights_path)
    elif download_type == "daemon-pull":
        _download_via_daemon(model, backend_cfg, download_cfg, repo_root, weights_path)
    elif download_type == "hf-snapshot":
        _download_huggingface(model, weights_path, server_id=server_id)
    elif download_type == "none":
        # Create the weights dir so _is_downloaded() returns True and the
        # model appears in the dashboard launch controls.
        weights_path.mkdir(parents=True, exist_ok=True)
        (weights_path / ".ready").touch()
        print(f"No download needed — {target_backend} fetches model weights automatically on first serve.")
    else:
        raise ModelError(
            f"Unknown download type '{download_type}' in manifest for '{target_backend}'."
        )


def _download_via_daemon(
    model: ModelConfig,
    backend,
    download_cfg: dict,
    repo_root: Path,
    weights_path: Path,
):
    """Pull a model into a backend's own store by briefly running its daemon.

    Driven by the manifest:
      download:
        type: daemon-pull
        tag_field: ollama_tag   # which model field holds the tag
    The backend's `run` template is reused to start the daemon, and a
    `<binary> pull <tag>` is issued once the daemon is up.
    """
    import time
    import urllib.request
    import urllib.error

    manifest = backend.manifest
    if not manifest:
        raise ModelError(f"Backend '{backend.name}' has no manifest.")

    tag_field = download_cfg.get("tag_field")
    tag = getattr(model, tag_field, None) if tag_field else None
    tag = tag or model.name

    binary_rel = manifest.fields.get("binary")
    if not binary_rel:
        raise ModelError(
            f"Backend '{backend.name}' is not a binary install — daemon-pull "
            f"requires a 'binary' field in the manifest."
        )
    binary_abs = str(repo_root / binary_rel)
    daemon_env = manifest.fields.get("env") or {}

    weights_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting {backend.name} daemon...")
    proc = subprocess.Popen(
        [binary_abs, "serve"],
        env={**os.environ, **daemon_env},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    health_path = manifest.fields.get("health_path") or "/api/version"
    health_url = f"http://localhost:{backend.port}{health_path}"
    for _ in range(30):
        try:
            with urllib.request.urlopen(health_url, timeout=2):
                break
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(1)
    else:
        proc.terminate()
        raise ModelError(f"{backend.name} daemon did not start in time.")

    print(f"Pulling {backend.name} model: {tag}")
    pull = subprocess.run([binary_abs, "pull", tag])
    if pull.returncode != 0:
        proc.terminate()
        raise ModelError(f"Failed to pull {backend.name} model: {tag}")

    # Marker so the launcher can tell the model is present even though the
    # actual bytes live in the backend's own store, not under weights_path.
    (weights_path / f".{backend.name}-managed").write_text(f"tag: {tag}\n")
    proc.terminate()
    print(f"Pulled {tag} into {backend.name} store.")


def _download_image(model: ModelConfig, download_cfg: dict, weights_path: Path):
    """Download a backend's model by pulling a container image.

    Driven by the manifest:
      download:
        type: image-pull
        image_field: nim_image  # which model field holds the image
    """
    image_field = download_cfg.get("image_field")
    image = getattr(model, image_field, None) if image_field else None
    if not image:
        raise ModelError(
            f"Model '{model.name}' has no '{image_field}' configured.\n"
            f"Add it to the model entry in config.yaml."
        )

    # Preflight — check disk space at an existing ancestor of weights_path
    disk_check_path = weights_path
    while not disk_check_path.exists() and disk_check_path != disk_check_path.parent:
        disk_check_path = disk_check_path.parent

    required_gb = (model.size_gb or DEFAULT_REQUIRED_GB) * 1.2
    checks = [
        ("Docker", check_docker()),
        ("Disk space", check_disk_space(str(disk_check_path), required_gb=required_gb)),
    ]
    if not run_checks(checks):
        raise ModelError("Prerequisites not met.")

    # NGC auth only when pulling from nvcr.io
    if "nvcr.io" in image:
        ngc_key = get_ngc_key()
        if not ngc_key:
            ngc_key = prompt_ngc_key()
        if not ngc_key:
            raise ModelError("NGC API key is required to pull this image.")

        print(f"Logging in to nvcr.io...")
        result = subprocess.run(
            ["docker", "login", "nvcr.io",
             "--username", "$oauthtoken",
             "--password-stdin"],
            input=ngc_key,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise ModelError(
                "Failed to log in to nvcr.io. Check your NGC API key.\n"
                f"  {result.stderr.strip()}"
            )

    print(f"Pulling image: {image}...")
    print(f"This may take a while depending on your network speed.")
    pull_result = subprocess.run(
        ["docker", "pull", image],
        timeout=3600,  # 1 hour timeout for large images
    )
    if pull_result.returncode != 0:
        raise ModelError(f"Failed to pull image: {image}")

    # Cache dir for the runtime to mount.
    weights_path.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Image pulled successfully.")
    print(f"Cache directory created at: {weights_path}")
    print(f"Model weights will download on first serve.")
    print(f"Run: ./launcher serve {model.name}")


def _download_huggingface(model: ModelConfig, weights_path: Path, server_id: str = CLI_SERVER_ID):
    """Download model weights from HuggingFace.

    Resumable by construction: HF's snapshot_download skips files that
    already exist with the expected size, so re-running after an
    interruption picks up where it left off.  We wrap that with a
    structured on-disk marker + an in-process active-set so the
    dashboard can distinguish "running", "interrupted", and "done" and
    surface a Resume button when appropriate.
    """
    from huggingface_hub import snapshot_download
    from src.hf_sizes import resolve_model_expected_bytes

    # Check HF auth first so the preflight size-lookup can use the token
    # for gated repos (metadata usually works without, but not always).
    hf_token = get_hf_token()

    # Preflight — check disk space at an existing ancestor path.  Prefer
    # the HF-resolved total over the declared size_gb hint; the hint is
    # hand-typed and known to drift (see fastwan22-ti2v-5b 14 vs 23).
    disk_check_path = weights_path
    while not disk_check_path.exists() and disk_check_path != disk_check_path.parent:
        disk_check_path = disk_check_path.parent

    resolved_bytes = resolve_model_expected_bytes(model, token=hf_token)
    if resolved_bytes:
        required_gb = (resolved_bytes / 1024**3) * 1.2
    else:
        required_gb = (model.size_gb or DEFAULT_REQUIRED_GB) * 1.2
    checks = [
        ("Disk space", check_disk_space(str(disk_check_path), required_gb=required_gb)),
    ]
    if not run_checks(checks):
        raise ModelError("Prerequisites not met.")

    # Inform the user if we're resuming a prior interrupted attempt.
    prior = _read_marker(weights_path)
    if prior is not None:
        prior_err = prior.get("last_error") if isinstance(prior, dict) else None
        if prior_err:
            print(f"Resuming {model.name} — prior attempt failed: {prior_err}")
        else:
            print(f"Resuming {model.name} — prior attempt was interrupted.")
    else:
        print(f"Downloading {model.name} from HuggingFace: {model.source}")
    print(f"Destination: {weights_path}")
    print(f"This download is resumable — safely interrupt and re-run.")
    print()

    _write_marker(weights_path, server_id)
    _mark_active(model.name)

    # Optional subfolder / glob filter — lets a model entry pull only one
    # quant from a multi-quant HF repo (e.g. ["UD-Q3_K_XL/*"]). Forwarded
    # verbatim to huggingface_hub's snapshot_download.
    allow_patterns = model.allow_patterns or None

    # Build the list of (repo_id, allow_patterns) pairs to download. The main
    # source is always first; extra_sources are pulled afterwards into the
    # same weights_path so a single model dir can aggregate pieces from
    # multiple repos (e.g. Flux Kontext's transformer + text encoders + VAE).
    sources: list[tuple[str, list[str] | None]] = [(model.source, allow_patterns)]
    for extra in model.extra_sources:
        extra_src = extra.get("source")
        if not extra_src:
            raise ModelError(
                f"Model '{model.name}' has an extra_sources entry with no 'source' field."
            )
        sources.append((extra_src, extra.get("allow_patterns")))

    def _pull(repo_id: str, patterns: list[str] | None):
        nonlocal hf_token
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(weights_path),
                token=hf_token,
                allow_patterns=patterns,
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
                if not hf_token:
                    hf_token = prompt_hf_token()
                    if hf_token:
                        print("Retrying with token...")
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=str(weights_path),
                            token=hf_token,
                            allow_patterns=patterns,
                        )
                        return
                raise ModelError(
                    f"Access denied for {repo_id}.\n"
                    f"This model may be gated. Accept the license at:\n"
                    f"  https://huggingface.co/{repo_id}\n"
                    f"Then set your HF_TOKEN environment variable."
                )
            raise ModelError(f"Download failed for {repo_id}: {e}")

    marker_path = weights_path / DOWNLOADING_MARKER
    try:
        for i, (repo_id, patterns) in enumerate(sources):
            if len(sources) > 1:
                label = "main source" if i == 0 else f"extra source {i}/{len(sources)-1}"
                print(f"[{label}] Fetching {repo_id}" + (f"  ({patterns})" if patterns else ""))
            _pull(repo_id, patterns)
    except BaseException as e:
        # Leave the marker in place, annotated with the error, so the
        # dashboard/CLI can show "interrupted" and offer Resume.  Catches
        # BaseException so SystemExit / KeyboardInterrupt also record
        # state before propagating — otherwise a Ctrl-C from the CLI
        # would leave an unannotated marker behind.
        _write_marker(weights_path, server_id, error=str(e) or type(e).__name__)
        raise
    else:
        # Clean success — marker's sole purpose is signalling in-progress
        # or failed state, so it goes away when we're done.
        if marker_path.exists():
            marker_path.unlink()
        print(f"\n{model.name} downloaded successfully.")
    finally:
        _mark_inactive(model.name)


# --- remove ---

def remove_model(name: str, config: Config, repo_root: Path):
    """Remove downloaded model weights.

    Also handles the "interrupted" state: a partial download with a
    leftover `.downloading` marker is removable so the user can discard
    it without first having to complete the download.  Refuses only
    when nothing is on disk or when a download is actively running in
    this process (we don't want to rm out from under a live writer).
    """
    model = config.get_model(name)
    weights_path = repo_root / model.path

    if is_download_active(name):
        raise ModelError(
            f"Cannot remove {name}: a download is currently running. "
            f"Stop the dashboard (or wait for it to finish) and try again."
        )

    if not weights_path.exists() or _get_dir_size_bytes(weights_path) == 0:
        print(f"{name} is not downloaded.")
        return

    marker = _read_marker(weights_path)
    if marker is not None:
        print(f"Discarding partial download of {name}...")
    else:
        size = _format_size(_get_dir_size_bytes(weights_path))
        print(f"Removing {name} ({size})...")

    shutil.rmtree(weights_path)
    print(f"Weights removed.")

    # If any of this model's backends used image-pull, surface the lingering
    # image so the user can decide whether to free the disk space.
    for backend_name in model.backends:
        backend = config.backends.get(backend_name)
        if not backend or not backend.manifest:
            continue
        download_cfg = backend.manifest.fields.get("download") or {}
        if download_cfg.get("type") != "image-pull":
            continue
        image_field = download_cfg.get("image_field")
        image = getattr(model, image_field, None) if image_field else None
        if not image:
            continue
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"Image {image} is still present.")
            print(f"Remove it with: docker rmi {image}")
