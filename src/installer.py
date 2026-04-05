"""Install, update, and check status of backends and clients from manifests."""

import subprocess
import sys
from pathlib import Path

from src.config import Config, ManifestConfig, ConfigError
from src.credentials import get_ngc_key, prompt_ngc_key
from src.preflight import (
    check_docker,
    check_nvidia_container_toolkit,
    check_python,
    run_checks,
)


class InstallError(Exception):
    """Raised when a tool installation fails."""


def _resolve_tool(name: str, config: Config) -> tuple[str, ManifestConfig]:
    """Find a tool by name in backends or clients. Returns (kind, manifest)."""
    if name in config.backends:
        backend = config.backends[name]
        if backend.manifest is None:
            raise ConfigError(f"Backend '{name}' has no manifest")
        return "backend", backend.manifest

    if name in config.clients:
        client = config.clients[name]
        if client.type == "bridge":
            raise InstallError(f"Client '{name}' is a bridge — nothing to install.")
        if client.manifest is None:
            raise ConfigError(f"Client '{name}' has no manifest")
        return "client", client.manifest

    available = sorted(
        list(config.backends.keys())
        + [n for n, c in config.clients.items() if c.type != "bridge"]
    )
    raise ConfigError(
        f"Unknown tool '{name}'. Available: {', '.join(available)}"
    )


def _run_command(cmd: list[str], label: str, env: dict | None = None) -> bool:
    """Run a command, streaming output. Returns True on success."""
    print(f"  Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            env={**dict(__import__("os").environ), **(env or {})},
            timeout=600,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  {label} timed out after 10 minutes.")
        return False
    except FileNotFoundError:
        print(f"  Command not found: {cmd[0]}")
        return False


# --- pip install type ---

def _install_pip(manifest: ManifestConfig, repo_root: Path):
    """Install a pip-based tool into its own venv."""
    if not run_checks([("Python 3", check_python())]):
        raise InstallError("Prerequisites not met.")

    package = manifest.fields.get("package")
    venv_rel = manifest.fields.get("venv")
    if not package or not venv_rel:
        raise InstallError(
            f"Manifest for '{manifest.name}' missing 'package' or 'venv' field."
        )

    venv_path = repo_root / venv_rel
    pip_path = venv_path / "bin" / "pip"

    # Create venv if it doesn't exist
    if not venv_path.exists():
        print(f"  Creating virtual environment at {venv_rel}...")
        if not _run_command(["python3", "-m", "venv", str(venv_path)], "venv creation"):
            raise InstallError("Failed to create virtual environment.")

    # Install package
    print(f"  Installing {package}...")
    if not _run_command([str(pip_path), "install", package], f"{package} install"):
        raise InstallError(
            f"Failed to install {package}.\n"
            f"This may be an aarch64 compatibility issue — check if {package} "
            f"publishes ARM wheels."
        )

    print(f"  {manifest.name} installed successfully.")


def _update_pip(manifest: ManifestConfig, repo_root: Path):
    """Update a pip-based tool in its venv."""
    package = manifest.fields.get("package")
    venv_rel = manifest.fields.get("venv")
    if not package or not venv_rel:
        raise InstallError(
            f"Manifest for '{manifest.name}' missing 'package' or 'venv' field."
        )

    venv_path = repo_root / venv_rel
    pip_path = venv_path / "bin" / "pip"

    if not venv_path.exists():
        print(f"  {manifest.name} is not installed. Run setup first.")
        raise InstallError(f"{manifest.name} not installed.")

    print(f"  Upgrading {package}...")
    if not _run_command(
        [str(pip_path), "install", "--upgrade", package],
        f"{package} upgrade",
    ):
        raise InstallError(f"Failed to upgrade {package}.")

    print(f"  {manifest.name} updated successfully.")


# --- container install type ---

def _install_container(manifest: ManifestConfig, config: Config):
    """Pull a Docker container image."""
    checks = [
        ("Docker", check_docker()),
    ]

    image = manifest.fields.get("image", "")

    # NGC images need auth and nvidia runtime
    if "nvcr.io" in image:
        checks.append(("NVIDIA Container Toolkit", check_nvidia_container_toolkit()))

    if not run_checks(checks):
        raise InstallError("Prerequisites not met.")

    # For NIM, the actual image comes from the model config, not the manifest.
    # The manifest image field is a template. We pull all model-specific NIM images.
    if manifest.name == "nim":
        _install_nim_images(config)
        return

    # For non-NIM containers (e.g., Open WebUI)
    if not image or image.startswith("{"):
        raise InstallError(
            f"Manifest for '{manifest.name}' has no valid image to pull."
        )

    print(f"  Pulling {image}...")
    if not _run_command(["docker", "pull", image], "docker pull"):
        raise InstallError(f"Failed to pull {image}.")

    print(f"  {manifest.name} installed successfully.")


def _install_nim_images(config: Config):
    """Pull NIM container images for all models that support NIM."""
    # Ensure NGC auth
    ngc_key = get_ngc_key()
    if not ngc_key:
        ngc_key = prompt_ngc_key()
    if not ngc_key:
        raise InstallError("NGC API key is required to pull NIM images.")

    # Docker login to nvcr.io
    print("  Logging in to nvcr.io...")
    result = subprocess.run(
        ["docker", "login", "nvcr.io",
         "--username", "$oauthtoken",
         "--password-stdin"],
        input=ngc_key,
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise InstallError(
            "Failed to log in to nvcr.io. Check your NGC API key.\n"
            f"  {result.stderr.strip()}"
        )

    # Pull each model's NIM image
    pulled = 0
    for name, model in config.models.items():
        if "nim" not in model.backends or not model.nim_image:
            continue
        print(f"  Pulling NIM image for {name}: {model.nim_image}...")
        if not _run_command(["docker", "pull", model.nim_image], "docker pull"):
            raise InstallError(f"Failed to pull NIM image: {model.nim_image}")
        pulled += 1

    if pulled == 0:
        print("  No models configured for NIM.")
    else:
        print(f"  NIM setup complete — pulled {pulled} image(s).")


def _update_container(manifest: ManifestConfig, config: Config):
    """Update a container image (same as install — Docker handles the delta)."""
    _install_container(manifest, config)


# --- binary install type ---

def _install_binary(manifest: ManifestConfig, repo_root: Path):
    """Download a binary release."""
    binary_rel = manifest.fields.get("binary")
    source = manifest.fields.get("source")

    if not binary_rel or not source:
        raise InstallError(
            f"Manifest for '{manifest.name}' missing 'binary' or 'source' field."
        )

    binary_path = repo_root / binary_rel
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    # For now, print instructions. Automating GitHub release downloads for the
    # correct platform (linux-aarch64) requires parsing the GitHub API, which
    # is straightforward but specific to each project's release naming.
    print(f"  Binary install for {manifest.name}:")
    print(f"    Download from: {source}")
    print(f"    Platform: linux-aarch64")
    print(f"    Place binary at: {binary_rel}")
    print(f"    Make executable: chmod +x {binary_rel}")
    print()
    print(f"  Automated binary download for {manifest.name} is not yet implemented.")
    print(f"  Please download manually from the URL above.")


def _update_binary(manifest: ManifestConfig, repo_root: Path):
    """Update a binary (same as install)."""
    _install_binary(manifest, repo_root)


# --- public API ---

def setup_tool(name: str, config: Config, repo_root: Path):
    """Install a tool (backend or client) from its manifest."""
    kind, manifest = _resolve_tool(name, config)

    print(f"Setting up {manifest.name} ({manifest.install_type})...")

    match manifest.install_type:
        case "pip":
            _install_pip(manifest, repo_root)
        case "container":
            _install_container(manifest, config)
        case "binary":
            _install_binary(manifest, repo_root)
        case "bridge":
            print(f"  {manifest.name} is a bridge — nothing to install.")
        case _:
            raise InstallError(
                f"Unknown install_type '{manifest.install_type}' "
                f"in manifest for '{manifest.name}'."
            )


def setup_all(config: Config, repo_root: Path):
    """Install all backends and clients."""
    tools = []

    for name in config.backends:
        tools.append(name)
    for name, client in config.clients.items():
        if client.type != "bridge":
            tools.append(name)

    for tool_name in tools:
        try:
            setup_tool(tool_name, config, repo_root)
        except InstallError as e:
            print(f"  Warning: {tool_name} setup failed: {e}")
        print()


def update_tool(name: str, config: Config, repo_root: Path):
    """Update a tool (backend or client) from its manifest."""
    kind, manifest = _resolve_tool(name, config)

    print(f"Updating {manifest.name} ({manifest.install_type})...")

    match manifest.install_type:
        case "pip":
            _update_pip(manifest, repo_root)
        case "container":
            _update_container(manifest, config)
        case "binary":
            _update_binary(manifest, repo_root)
        case "bridge":
            print(f"  {manifest.name} is a bridge — nothing to update.")
        case _:
            raise InstallError(
                f"Unknown install_type '{manifest.install_type}' "
                f"in manifest for '{manifest.name}'."
            )


def update_all(config: Config, repo_root: Path):
    """Update all backends and clients."""
    tools = []

    for name in config.backends:
        tools.append(name)
    for name, client in config.clients.items():
        if client.type != "bridge":
            tools.append(name)

    for tool_name in tools:
        try:
            update_tool(tool_name, config, repo_root)
        except InstallError as e:
            print(f"  Warning: {tool_name} update failed: {e}")
        print()


def get_all_tools_status(config: Config, repo_root: Path) -> list[dict]:
    """Return install status for all backends and clients."""
    tools = []
    for name in config.backends:
        status = get_tool_status(name, config, repo_root)
        install_type = config.backends[name].manifest.install_type if config.backends[name].manifest else "unknown"
        tools.append({"name": name, "kind": "backend", "status": status, "install_type": install_type})
    for name, client in config.clients.items():
        if client.type == "bridge":
            tools.append({"name": name, "kind": "client", "status": "bridge", "install_type": "bridge"})
        else:
            status = get_tool_status(name, config, repo_root)
            install_type = client.manifest.install_type if client.manifest else "unknown"
            tools.append({"name": name, "kind": "client", "status": status, "install_type": install_type})
    return tools


def get_tool_status(name: str, config: Config, repo_root: Path) -> str:
    """Check if a tool is installed. Returns a status string."""
    kind, manifest = _resolve_tool(name, config)

    match manifest.install_type:
        case "pip":
            venv_rel = manifest.fields.get("venv")
            if venv_rel and (repo_root / venv_rel).exists():
                return "installed"
            return "not installed"
        case "container":
            image = manifest.fields.get("image", "")
            if manifest.name == "nim":
                # Check if any NIM model image is pulled
                for model in config.models.values():
                    if model.nim_image:
                        result = subprocess.run(
                            ["docker", "image", "inspect", model.nim_image],
                            capture_output=True, timeout=10,
                        )
                        if result.returncode == 0:
                            return "installed"
                return "not installed"
            if not image or image.startswith("{"):
                return "unknown"
            result = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, timeout=10,
            )
            return "installed" if result.returncode == 0 else "not installed"
        case "binary":
            binary_rel = manifest.fields.get("binary")
            if binary_rel and (repo_root / binary_rel).exists():
                return "installed"
            return "not installed"
        case _:
            return "unknown"
