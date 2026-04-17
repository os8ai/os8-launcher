"""Install, update, and check status of backends and clients from manifests."""

import os
import subprocess
import sys
from pathlib import Path

from src.config import Config, ManifestConfig, ConfigError
from src.credentials import get_ngc_key, prompt_ngc_key
from src.preflight import (
    check_docker,
    check_nvidia_container_toolkit,
    check_python,
    check_python_dev,
    detect_arch,
    resolve_image,
    run_checks,
)


# Maps `requires:` entries in a manifest to the preflight check that
# enforces them.  Add new entries here as backends grow new prerequisites
# (e.g. ffmpeg, git, build-essential variants).
_CHECK_REGISTRY = {
    "python_dev": ("Python development headers", check_python_dev),
}


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
    checks = [("Python 3", check_python())]
    for req in manifest.fields.get("requires") or []:
        entry = _CHECK_REGISTRY.get(req)
        if entry is None:
            raise InstallError(
                f"Manifest for '{manifest.name}' references unknown check "
                f"'{req}'. Known checks: {', '.join(sorted(_CHECK_REGISTRY))}"
            )
        label, check_fn = entry
        checks.append((label, check_fn()))

    if not run_checks(checks):
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

    # Install package — use install_cmd if provided (for multi-step or
    # platform-specific installs), otherwise fall back to pip install.
    install_cmd = manifest.fields.get("install_cmd")
    if install_cmd:
        print(f"  Running install command for {manifest.name}...")
        arch = detect_arch()
        env = {
            **os.environ,
            "ARCH": arch["machine"],
            "ARCH_DOCKER": arch["docker"],
            "VIRTUAL_ENV": str(venv_path),
            "PATH": f"{venv_path / 'bin'}:{os.environ.get('PATH', '')}",
        }
        result = subprocess.run(
            install_cmd, shell=True, cwd=str(repo_root), timeout=1200, env=env,
        )
        if result.returncode != 0:
            raise InstallError(f"Install command for {manifest.name} failed.")
    else:
        print(f"  Installing {package}...")
        if not _run_command([str(pip_path), "install", package], f"{package} install"):
            arch = detect_arch()
            raise InstallError(
                f"Failed to install {package}.\n"
                f"This may be a compatibility issue — check if {package} "
                f"publishes {arch['machine']} wheels."
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
    """Pull a Docker container image.

    For backends whose manifest declares `download.type: image-pull`, the
    actual image is per-model and we pull every image referenced by any
    model that lists this backend. Otherwise we pull the static `image:`
    field from the manifest itself.
    """
    checks = [
        ("Docker", check_docker()),
    ]

    image = resolve_image(manifest.fields)
    download_cfg = manifest.fields.get("download") or {}
    is_per_model_image = download_cfg.get("type") == "image-pull"

    # NGC images need auth and nvidia runtime
    if "nvcr.io" in image or is_per_model_image:
        checks.append(("NVIDIA Container Toolkit", check_nvidia_container_toolkit()))

    if not run_checks(checks):
        raise InstallError("Prerequisites not met.")

    if is_per_model_image:
        _install_per_model_images(manifest, config)
        return

    if not image or image.startswith("{"):
        raise InstallError(
            f"Manifest for '{manifest.name}' has no valid image to pull."
        )

    # Build any images declared in the manifest's image_builds block. These
    # are local-only tags (e.g. os8-vllm:qwen36) that cannot be pulled from
    # a registry. We build them all so `./launcher setup <backend>` leaves
    # the system ready to serve any compatible model.
    builds = manifest.fields.get("image_builds") or {}
    if builds:
        manifest_dir = Path(manifest.path).parent
        for tag, info in builds.items():
            dockerfile = info.get("dockerfile")
            if not dockerfile:
                raise InstallError(
                    f"image_builds entry for {tag} missing 'dockerfile'."
                )
            dockerfile_path = manifest_dir / dockerfile
            context = manifest_dir / (info.get("context") or ".")
            if not dockerfile_path.exists():
                raise InstallError(f"Dockerfile not found: {dockerfile_path}")
            print(f"  Building {tag} from {dockerfile}...")
            cmd = ["docker", "build", "-f", str(dockerfile_path),
                   "-t", tag, str(context)]
            if not _run_command(cmd, "docker build"):
                raise InstallError(f"Failed to build {tag}.")

    # If the default manifest image is in image_builds it's already handled.
    # Otherwise pull it from the registry as before.
    if image not in builds:
        print(f"  Pulling {image}...")
        if not _run_command(["docker", "pull", image], "docker pull"):
            raise InstallError(f"Failed to pull {image}.")

    print(f"  {manifest.name} installed successfully.")


def _install_per_model_images(manifest: ManifestConfig, config: Config):
    """Pull a per-model image for every model that lists this backend.

    Driven by the manifest's `download` block:
        download:
          type: image-pull
          image_field: nim_image
    """
    download_cfg = manifest.fields.get("download") or {}
    image_field = download_cfg.get("image_field")
    if not image_field:
        raise InstallError(
            f"Manifest for '{manifest.name}' has download.type=image-pull "
            f"but no image_field."
        )

    backend_name = manifest.name
    candidates = [
        (name, getattr(model, image_field, None))
        for name, model in config.models.items()
        if backend_name in model.backends
    ]
    candidates = [(n, img) for n, img in candidates if img]

    if not candidates:
        print(f"  No models configured for {backend_name}.")
        return

    # If any image is from nvcr.io, log in once up front.
    if any("nvcr.io" in img for _, img in candidates):
        ngc_key = get_ngc_key()
        if not ngc_key:
            ngc_key = prompt_ngc_key()
        if not ngc_key:
            raise InstallError("NGC API key is required to pull these images.")

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

    pulled = 0
    for name, image in candidates:
        print(f"  Pulling image for {name}: {image}...")
        if not _run_command(["docker", "pull", image], "docker pull"):
            raise InstallError(f"Failed to pull image: {image}")
        pulled += 1

    print(f"  {backend_name} setup complete — pulled {pulled} image(s).")


def _update_container(manifest: ManifestConfig, config: Config):
    """Update a container image (same as install — Docker handles the delta)."""
    _install_container(manifest, config)


# --- binary install type ---

def _install_binary(manifest: ManifestConfig, repo_root: Path):
    """Install a binary tool. If the manifest provides an `install_cmd`, run
    it via the shell (so pipes/redirects work). Otherwise fall back to
    printing manual download instructions.
    """
    binary_rel = manifest.fields.get("binary")
    source = manifest.fields.get("source")
    install_cmd = manifest.fields.get("install_cmd")

    if not binary_rel:
        raise InstallError(
            f"Manifest for '{manifest.name}' missing 'binary' field."
        )

    binary_path = repo_root / binary_rel
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    if install_cmd:
        print(f"  Running install command for {manifest.name}...")
        print(f"  $ {install_cmd}")
        arch = detect_arch()
        env = {**os.environ, "ARCH": arch["machine"], "ARCH_DOCKER": arch["docker"]}
        result = subprocess.run(
            install_cmd, shell=True, cwd=str(repo_root), timeout=600, env=env,
        )
        if result.returncode != 0:
            raise InstallError(f"Install command for {manifest.name} failed.")
        if not binary_path.exists():
            raise InstallError(
                f"Install command for {manifest.name} succeeded but "
                f"{binary_rel} is not present."
            )
        print(f"  {manifest.name} installed successfully.")
        return

    arch = detect_arch()
    print(f"  Binary install for {manifest.name}:")
    if source:
        print(f"    Download from: {source}")
    print(f"    Platform: linux-{arch['machine']}")
    print(f"    Place binary at: {binary_rel}")
    print(f"    Make executable: chmod +x {binary_rel}")
    print()
    print(f"  Automated binary download for {manifest.name} is not yet implemented.")
    print(f"  Please download manually from the URL above.")


def _update_binary(manifest: ManifestConfig, repo_root: Path):
    """Update a binary. Uses `update_cmd` if present, else `install_cmd`."""
    update_cmd = manifest.fields.get("update_cmd") or manifest.fields.get("install_cmd")
    if update_cmd:
        # Temporarily swap install_cmd so _install_binary runs the right thing.
        original = manifest.fields.get("install_cmd")
        manifest.fields["install_cmd"] = update_cmd
        try:
            _install_binary(manifest, repo_root)
        finally:
            if original is None:
                manifest.fields.pop("install_cmd", None)
            else:
                manifest.fields["install_cmd"] = original
        return
    _install_binary(manifest, repo_root)


# --- public API ---

INSTALLING_MARKER = ".installing"
INSTALLED_MARKER = ".installed"


def setup_tool(name: str, config: Config, repo_root: Path):
    """Install a tool (backend or client) from its manifest."""
    kind, manifest = _resolve_tool(name, config)

    print(f"Setting up {manifest.name} ({manifest.install_type})...")

    # Write a marker so get_tool_status can report "installing" while the
    # background task runs.  The marker lives inside the tool's own directory
    # (venv for pip, manifest dir for others) and is removed on completion.
    marker_dir = _marker_dir(manifest, repo_root)
    if marker_dir:
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / INSTALLING_MARKER).touch()
        # Remove stale completion marker — a re-setup should prove success.
        (marker_dir / INSTALLED_MARKER).unlink(missing_ok=True)

    try:
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
        # Mark successful completion.
        if marker_dir:
            (marker_dir / INSTALLED_MARKER).touch()
    finally:
        if marker_dir:
            (marker_dir / INSTALLING_MARKER).unlink(missing_ok=True)


def _marker_dir(manifest: ManifestConfig, repo_root: Path):
    """Return the directory where install markers live for a tool.

    Must NOT be inside the venv itself — creating the marker dir would
    cause _install_pip to skip venv creation (``if not venv_path.exists()``).
    We use the manifest's own directory (e.g. ``serving/kokoro/``) instead.
    """
    if not manifest.path:
        return None
    p = manifest.path.parent
    return repo_root / p if not p.is_absolute() else p


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
            mdir = _marker_dir(manifest, repo_root)
            if mdir and (mdir / INSTALLING_MARKER).exists():
                return "installing"
            if mdir and (mdir / INSTALLED_MARKER).exists():
                return "installed"
            return "not installed"
        case "container":
            image = resolve_image(manifest.fields)
            download_cfg = manifest.fields.get("download") or {}
            if download_cfg.get("type") == "image-pull":
                # Per-model images — installed if any model's image is cached.
                image_field = download_cfg.get("image_field")
                for model in config.models.values():
                    if name not in model.backends:
                        continue
                    img = getattr(model, image_field, None) if image_field else None
                    if not img:
                        continue
                    result = subprocess.run(
                        ["docker", "image", "inspect", img],
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
            mdir = _marker_dir(manifest, repo_root)
            if mdir and (mdir / INSTALLING_MARKER).exists():
                return "installing"
            if mdir and (mdir / INSTALLED_MARKER).exists():
                return "installed"
            return "not installed"
        case _:
            return "unknown"
