"""Model weight management — download, list, and remove."""

import shutil
import subprocess
from pathlib import Path

from src.config import Config, ConfigError, ModelConfig
from src.credentials import get_ngc_key, get_hf_token, prompt_ngc_key, prompt_hf_token
from src.preflight import check_docker, check_disk_space, run_checks


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


def _is_downloaded(model: ModelConfig, repo_root: Path) -> bool:
    """Check if model weights are present on disk (not trusting config field)."""
    weights_path = repo_root / model.path
    if not weights_path.exists():
        return False
    # Has at least one file in it
    return any(weights_path.rglob("*"))


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
    """Download model weights using the appropriate strategy for the backend."""
    model = config.get_model(name)
    target_backend = backend or model.default_backend

    if target_backend not in model.backends:
        raise ModelError(
            f"Backend '{target_backend}' is not compatible with model '{name}'.\n"
            f"Compatible backends: {', '.join(model.backends)}"
        )

    weights_path = repo_root / model.path

    if target_backend == "nim":
        _download_nim(model, weights_path)
    else:
        _download_huggingface(model, weights_path)


def _download_nim(model: ModelConfig, weights_path: Path):
    """Download for NIM: pull the model-specific container image and prepare cache dir."""
    if not model.nim_image:
        raise ModelError(
            f"Model '{model.name}' has no nim_image configured.\n"
            f"Add nim_image to the model entry in config.yaml."
        )

    # Preflight
    checks = [
        ("Docker", check_docker()),
        ("Disk space", check_disk_space(str(weights_path.parent), required_gb=100)),
    ]
    if not run_checks(checks):
        raise ModelError("Prerequisites not met.")

    # NGC auth
    ngc_key = get_ngc_key()
    if not ngc_key:
        ngc_key = prompt_ngc_key()
    if not ngc_key:
        raise ModelError("NGC API key is required to pull NIM images.")

    # Docker login
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

    # Pull the image
    print(f"Pulling NIM image: {model.nim_image}...")
    print(f"This may take a while depending on your network speed.")
    pull_result = subprocess.run(
        ["docker", "pull", model.nim_image],
        timeout=3600,  # 1 hour timeout for large images
    )
    if pull_result.returncode != 0:
        raise ModelError(f"Failed to pull NIM image: {model.nim_image}")

    # Create cache directory
    weights_path.mkdir(parents=True, exist_ok=True)

    print()
    print(f"NIM image pulled successfully.")
    print(f"Cache directory created at: {weights_path}")
    print(f"Model weights will download on first serve (~60-75 GB).")
    print(f"Run: ./launcher serve {model.name}")


def _download_huggingface(model: ModelConfig, weights_path: Path):
    """Download model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    # Preflight — estimate ~75GB for a 120B model
    checks = [
        ("Disk space", check_disk_space(str(weights_path.parent), required_gb=75)),
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

    # Offer to remove NIM image too
    if model.nim_image:
        result = subprocess.run(
            ["docker", "image", "inspect", model.nim_image],
            capture_output=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"NIM image {model.nim_image} is still present.")
            print(f"Remove it with: docker rmi {model.nim_image}")
