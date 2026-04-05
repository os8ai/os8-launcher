"""System prerequisite checks for os8-launcher operations.

Each check function returns (ok, message). On success, message is empty.
On failure, message contains an actionable description of what's missing and how to fix it.
"""

import shutil
import subprocess


def check_docker() -> tuple[bool, str]:
    """Check that Docker is installed and the daemon is running."""
    if not shutil.which("docker"):
        return False, "Docker is not installed. Install it with: sudo apt install docker.io"

    result = subprocess.run(
        ["docker", "info"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        if "permission denied" in result.stderr.lower():
            return False, (
                "Docker is installed but your user can't access it.\n"
                "Fix with: sudo usermod -aG docker $USER && newgrp docker"
            )
        return False, (
            "Docker is installed but the daemon is not running.\n"
            "Start it with: sudo systemctl start docker"
        )

    return True, ""


def check_nvidia_gpu() -> tuple[bool, str]:
    """Check that nvidia-smi can see a GPU."""
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found. NVIDIA drivers may not be installed."

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return False, "nvidia-smi failed. GPU may not be accessible."

    gpu_name = result.stdout.strip()
    if not gpu_name:
        return False, "nvidia-smi found no GPUs."

    return True, ""


def check_nvidia_container_toolkit() -> tuple[bool, str]:
    """Check that the NVIDIA Container Toolkit is available in Docker."""
    # First verify Docker works
    ok, msg = check_docker()
    if not ok:
        return False, msg

    result = subprocess.run(
        ["docker", "info"],
        capture_output=True, text=True, timeout=10,
    )
    if "nvidia" not in result.stdout.lower():
        return False, (
            "NVIDIA Container Toolkit is not configured for Docker.\n"
            "Install it with: sudo apt install nvidia-container-toolkit\n"
            "Then restart Docker: sudo systemctl restart docker"
        )

    return True, ""


def check_python() -> tuple[bool, str]:
    """Check that python3 is available for venv creation."""
    if not shutil.which("python3"):
        return False, "python3 not found. Install it with: sudo apt install python3"

    # Check venv module is available
    result = subprocess.run(
        ["python3", "-m", "venv", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return False, (
            "python3 is installed but the venv module is missing.\n"
            "Install it with: sudo apt install python3-venv"
        )

    return True, ""


def check_disk_space(path: str, required_gb: float) -> tuple[bool, str]:
    """Check that there's enough free disk space at the given path."""
    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        return False, f"Could not check disk space at {path}: {e}"

    free_gb = usage.free / (1024 ** 3)
    if free_gb < required_gb:
        return False, (
            f"Not enough disk space at {path}.\n"
            f"Required: {required_gb:.0f} GB, Available: {free_gb:.0f} GB"
        )

    return True, ""


def check_ngc_auth(ngc_api_key: str | None) -> tuple[bool, str]:
    """Check that an NGC API key is configured and valid."""
    if not ngc_api_key:
        return False, (
            "No NGC API key configured.\n"
            "Get one at: https://ngc.nvidia.com\n"
            "Then run: ./launcher setup --ngc-key <your-key>\n"
            "Or set the NGC_API_KEY environment variable."
        )

    # Validate by attempting a docker login (dry-run style check)
    result = subprocess.run(
        ["docker", "login", "nvcr.io",
         "--username", "$oauthtoken",
         "--password-stdin"],
        input=ngc_api_key,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return False, (
            "NGC API key is invalid or expired.\n"
            "Get a new one at: https://ngc.nvidia.com"
        )

    return True, ""


def check_hf_auth(hf_token: str | None) -> tuple[bool, str]:
    """Check that a HuggingFace token is configured."""
    if not hf_token:
        return False, (
            "No HuggingFace token configured.\n"
            "Get one at: https://huggingface.co/settings/tokens\n"
            "Then set the HF_TOKEN environment variable."
        )

    # Basic format check — HF tokens start with "hf_"
    if not hf_token.startswith("hf_"):
        return False, "HuggingFace token appears invalid (should start with 'hf_')."

    return True, ""


def run_checks(checks: list[tuple[str, tuple[bool, str]]]) -> bool:
    """Run a list of named checks. Prints failures and returns True if all pass."""
    all_ok = True
    for name, (ok, message) in checks:
        if not ok:
            print(f"✗ {name}")
            for line in message.split("\n"):
                print(f"  {line}")
            all_ok = False

    return all_ok
