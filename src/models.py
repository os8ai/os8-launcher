"""Model weight management — download, list, and remove."""

import os
import shutil
import subprocess
from pathlib import Path

from src.config import Config, ConfigError, ModelConfig
from src.credentials import get_ngc_key, get_hf_token, prompt_ngc_key, prompt_hf_token
from src.preflight import check_docker, check_disk_space, run_checks
from src.verification import get_for_model


DEFAULT_REQUIRED_GB = 5


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


def _is_downloaded(model: ModelConfig, repo_root: Path) -> bool:
    """Check if model weights are present on disk (not trusting config field)."""
    weights_path = repo_root / model.path
    if not weights_path.exists():
        return False
    # An in-progress download leaves partial files but the marker is still present.
    if (weights_path / DOWNLOADING_MARKER).exists():
        return False
    # Has at least one file in it (other than the marker)
    return any(p.name != DOWNLOADING_MARKER for p in weights_path.rglob("*"))


# --- data ---

def get_models_data(config: Config, repo_root: Path) -> list[dict]:
    """Return model info as a list of dicts for JSON serialization."""
    result = []
    for name, model in config.models.items():
        weights_path = repo_root / model.path
        downloading = (weights_path / DOWNLOADING_MARKER).exists()
        downloaded = _is_downloaded(model, repo_root)
        size_bytes = _get_dir_size_bytes(weights_path)
        # Expected total in bytes, from the declared model.size_gb (GiB).
        expected_bytes = int(model.size_gb * 1024**3) if model.size_gb else None
        if downloading:
            state = "downloading"
        elif downloaded:
            state = "downloaded"
        else:
            state = "not_downloaded"
        result.append({
            "name": name,
            "source": model.source,
            "format": model.format,
            "backends": model.backends,
            "default_backend": model.default_backend,
            "downloaded": downloaded,
            "state": state,
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
        downloaded = _is_downloaded(model, repo_root)
        status = "downloaded" if downloaded else "not downloaded"
        size = _format_size(_get_dir_size_bytes(repo_root / model.path))
        backends = ", ".join(model.backends)
        print(f"{name:<30} {model.format:<8} {status:<15} {size:<10} {backends}")


# --- download ---

def download_model(
    name: str,
    config: Config,
    repo_root: Path,
    backend: str | None = None,
):
    """Download model weights using the strategy declared in the backend manifest."""
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
        _download_huggingface(model, weights_path)
    elif download_type == "none":
        print(f"Backend '{target_backend}' does not require a model download step.")
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


def _download_huggingface(model: ModelConfig, weights_path: Path):
    """Download model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    # Preflight — check disk space at an existing ancestor path
    disk_check_path = weights_path
    while not disk_check_path.exists() and disk_check_path != disk_check_path.parent:
        disk_check_path = disk_check_path.parent

    required_gb = (model.size_gb or DEFAULT_REQUIRED_GB) * 1.2
    checks = [
        ("Disk space", check_disk_space(str(disk_check_path), required_gb=required_gb)),
    ]
    if not run_checks(checks):
        raise ModelError("Prerequisites not met.")

    # Check HF auth (some models are gated)
    hf_token = get_hf_token()

    print(f"Downloading {model.name} from HuggingFace: {model.source}")
    print(f"Destination: {weights_path}")
    print(f"This will download tens of GB. The download is resumable — you can")
    print(f"safely interrupt and re-run to continue where you left off.")
    print()

    weights_path.mkdir(parents=True, exist_ok=True)
    marker = weights_path / DOWNLOADING_MARKER
    marker.touch()

    try:
        try:
            snapshot_download(
                repo_id=model.source,
                local_dir=str(weights_path),
                token=hf_token,
            )
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
                if not hf_token:
                    hf_token = prompt_hf_token()
                    if hf_token:
                        print("Retrying with token...")
                        snapshot_download(
                            repo_id=model.source,
                            local_dir=str(weights_path),
                            token=hf_token,
                        )
                        print(f"\n{model.name} downloaded successfully.")
                        return
                raise ModelError(
                    f"Access denied for {model.source}.\n"
                    f"This model may be gated. Accept the license at:\n"
                    f"  https://huggingface.co/{model.source}\n"
                    f"Then set your HF_TOKEN environment variable."
                )
            raise ModelError(f"Download failed: {e}")
    finally:
        if marker.exists():
            marker.unlink()

    print(f"\n{model.name} downloaded successfully.")


# --- remove ---

def remove_model(name: str, config: Config, repo_root: Path):
    """Remove downloaded model weights."""
    model = config.get_model(name)
    weights_path = repo_root / model.path

    if not _is_downloaded(model, repo_root):
        print(f"{name} is not downloaded.")
        return

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
