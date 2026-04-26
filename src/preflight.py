"""System prerequisite checks for os8-launcher operations.

Each check function returns (ok, message). On success, message is empty.
On failure, message contains an actionable description of what's missing and how to fix it.
"""

import platform
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_ARCH_MAP = {
    "aarch64": {"machine": "aarch64", "docker": "arm64"},
    "x86_64":  {"machine": "x86_64",  "docker": "amd64"},
}


def detect_arch() -> dict[str, str]:
    """Return a dict with 'machine' and 'docker' keys for the current CPU architecture.

    Raises RuntimeError on unsupported architectures.
    """
    machine = platform.machine()
    if machine not in _ARCH_MAP:
        supported = ", ".join(_ARCH_MAP)
        raise RuntimeError(
            f"Unsupported architecture: {machine}. Supported: {supported}"
        )
    return _ARCH_MAP[machine]


def resolve_image(manifest_fields: dict) -> str:
    """Return the arch-appropriate Docker image from manifest fields.

    Checks for image_{machine} (e.g. image_aarch64, image_x86_64) first,
    then falls back to the plain 'image' field.
    """
    arch = detect_arch()
    arch_key = f"image_{arch['machine']}"
    return manifest_fields.get(arch_key, manifest_fields.get("image", ""))


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


def check_python_dev() -> tuple[bool, str]:
    """Check that the Python development headers and gcc are available.

    Required by pip backends that build C extensions at runtime — most
    notably Triton, which compiles a small CUDA shim on first import.
    Without Python.h the build fails with an opaque gcc traceback deep
    inside the backend, so we surface it up front at setup time.
    """
    if not shutil.which("gcc"):
        return False, (
            "gcc not found.\n"
            "Install it with: sudo apt install build-essential"
        )

    # Use sysconfig so we don't hardcode the Python version into the path.
    result = subprocess.run(
        ["python3", "-c",
         "import os, sysconfig; "
         "p = os.path.join(sysconfig.get_path('include'), 'Python.h'); "
         "print(p if os.path.exists(p) else '')"],
        capture_output=True, text=True, timeout=10,
    )
    if not result.stdout.strip():
        return False, (
            "Python development headers (Python.h) not found.\n"
            "Required for backends that build C extensions (e.g. Triton).\n"
            "Install with: sudo apt install python3-dev"
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


def check_cuda_version() -> tuple[bool, str]:
    """Check that CUDA is available and report its version."""
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found. NVIDIA drivers may not be installed."

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return False, "Could not query NVIDIA driver version."

    # CUDA version is shown in the nvidia-smi header, query it directly.
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True, text=True, timeout=10,
    )
    for line in result.stdout.splitlines():
        if "CUDA Version" in line:
            # Line looks like: "| NVIDIA-SMI 580.142  Driver Version: 580.142  CUDA Version: 13.0  |"
            for part in line.split():
                try:
                    # The token right after "Version:" that looks like a number
                    idx = line.index("CUDA Version:")
                    version = line[idx:].split()[2].rstrip("|").strip()
                    return True, version
                except (ValueError, IndexError):
                    pass

    return False, "Could not determine CUDA version from nvidia-smi output."


def check_available_memory(min_gb: float = 8.0) -> tuple[bool, str]:
    """Check total and available system RAM from /proc/meminfo."""
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1])  # kB

        total_gb = meminfo.get("MemTotal", 0) / (1024 * 1024)
        avail_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)

        if total_gb < min_gb:
            return False, (
                f"Insufficient RAM: {total_gb:.0f} GB total, {min_gb:.0f} GB minimum recommended."
            )
        return True, f"{total_gb:.0f} GB total, {avail_gb:.0f} GB available"

    except OSError as e:
        return False, f"Could not read /proc/meminfo: {e}"


def get_gpu_info() -> dict | None:
    """Return GPU name and memory, or None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None

    line = result.stdout.strip().split("\n")[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None

    name = parts[0]
    # memory.total comes back like "131072 MiB"
    try:
        memory_gb = float(parts[1].split()[0]) / 1024
    except (ValueError, IndexError):
        memory_gb = 0.0

    return {"name": name, "memory_gb": memory_gb}


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


# ---------------------------------------------------------------------------
# Leftover survey — detects state from a previous run that could fight with
# a new backend start. Read-only: never kills anything itself.
# ---------------------------------------------------------------------------

def _process_command(pid: int) -> str:
    """Best-effort `cmdline` for a PID; returns a short description."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        return raw or f"pid {pid}"
    except (FileNotFoundError, PermissionError, OSError):
        return f"pid {pid}"


def _port_holder(port: int) -> tuple[int, str] | None:
    """Return (pid, command) of whatever process holds `port`, or None.

    Tries `lsof` first (best info), falls back to `ss -tlnp`. Both are
    standard on Ubuntu — the one-or-the-other path keeps the survey
    working in stripped containers too.
    """
    if shutil.which("lsof"):
        try:
            r = subprocess.run(
                ["lsof", "-iTCP:" + str(port), "-sTCP:LISTEN", "-Fpc"],
                capture_output=True, text=True, timeout=5,
            )
            pid = None
            cmd = ""
            for line in r.stdout.splitlines():
                if line.startswith("p") and line[1:].isdigit():
                    pid = int(line[1:])
                elif line.startswith("c"):
                    cmd = line[1:]
            if pid:
                return pid, cmd or _process_command(pid)
        except (subprocess.TimeoutExpired, OSError):
            pass

    if shutil.which("ss"):
        try:
            r = subprocess.run(
                ["ss", "-tlnpH", f"sport = :{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                # users:(("name",pid=NNN,fd=M))
                idx = line.find("pid=")
                if idx == -1:
                    continue
                tail = line[idx + 4:]
                pid_str = ""
                for ch in tail:
                    if ch.isdigit():
                        pid_str += ch
                    else:
                        break
                if pid_str:
                    pid = int(pid_str)
                    return pid, _process_command(pid)
        except (subprocess.TimeoutExpired, OSError):
            pass

    return None


def _list_os8_containers() -> list[dict]:
    """All `os8-*` Docker containers (running or stopped)."""
    if not shutil.which("docker"):
        return []
    try:
        r = subprocess.run(
            ["docker", "ps", "-a",
             "--filter", "name=os8-",
             "--format", "{{.Names}}\t{{.Status}}\t{{.ID}}"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    out: list[dict] = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            out.append({"name": parts[0], "status": parts[1], "id": parts[2]})
    return out


def _gpu_processes() -> list[dict]:
    """All processes with non-zero GPU memory, via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    out: list[dict] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            mib = int(parts[1])
        except ValueError:
            continue
        out.append({"pid": pid, "memory_mib": mib, "process_name": parts[2]})
    return out


def survey_leftovers(
    target_ports: list[int] | None = None,
    state_data: dict | None = None,
    gpu_min_mib: int = 1024,
) -> list[dict]:
    """Read-only sweep for state from a previous launcher run that could
    interfere with a new backend start.

    Returns a list of finding dicts. Each finding has:
      - kind: "stale_state" | "orphan_container" | "port_conflict" | "gpu_squatter"
      - origin: "ours" (we tracked it) | "foreign" (something else)
      - summary: short human-readable line
      - action: structured payload the /api/leftovers/stop route can act on

    Categorization rules:
      - stale_state: an entry in state.yaml whose pid/container is dead or
        a zombie. Always "ours". Safe to clear.
      - orphan_container: an `os8-*` Docker container not referenced by
        state. "Ours" by name, but not tracked.
      - port_conflict: target_port is bound by some process. "Ours" iff
        the holding pid matches a state entry; "foreign" otherwise.
      - gpu_squatter: nvidia-smi reports a process holding >= gpu_min_mib
        of GPU memory. "Ours" if it matches a state entry. "Foreign"
        otherwise — never auto-stop, requires user confirm.
    """
    from src.state import (  # local import: state imports preflight indirectly
        is_container_running, is_process_alive, validate_state,
    )

    if state_data is None:
        state_data = validate_state()

    findings: list[dict] = []
    backends = (state_data.get("backends") or {})
    tracked_pids = {int(b["pid"]) for b in backends.values() if b.get("pid")}
    tracked_containers = {b.get("container_id") for b in backends.values() if b.get("container_id")}
    tracked_container_names = {f"os8-{b['instance_id']}" for b in backends.values() if b.get("instance_id")}

    # 1. stale_state — entries whose process or container has died but we
    #    haven't reaped from state.yaml yet. validate_state() above will
    #    have already removed cleanly-dead ones, but this catches anything
    #    still hanging around (e.g. mid-write race).
    for instance_id, entry in backends.items():
        alive = False
        if entry.get("container_id"):
            alive = is_container_running(entry["container_id"])
        elif entry.get("pid"):
            alive = is_process_alive(int(entry["pid"]))
        if not alive:
            findings.append({
                "kind": "stale_state",
                "origin": "ours",
                "summary": f"{instance_id} is in state.yaml but its process/container is dead",
                "action": {"type": "clear_state", "instance_id": instance_id},
            })

    # 2. orphan_container — `os8-*` containers we didn't launch (or that
    #    survived a launcher crash). Both stopped and running variants
    #    count: a stopped one with the same name will block the next start.
    for c in _list_os8_containers():
        if c["name"] in tracked_container_names:
            continue
        if c["id"] in tracked_containers:
            continue
        findings.append({
            "kind": "orphan_container",
            "origin": "ours",
            "summary": f"orphan container {c['name']} ({c['status']})",
            "action": {"type": "remove_container", "name": c["name"], "id": c["id"]},
        })

    # 3. port_conflict — for each port we plan to bind, who has it?
    for port in target_ports or []:
        holder = _port_holder(port)
        if holder is None:
            continue
        pid, cmd = holder
        is_ours = pid in tracked_pids
        findings.append({
            "kind": "port_conflict",
            "origin": "ours" if is_ours else "foreign",
            "summary": (
                f"port {port} held by {'our' if is_ours else 'foreign'} pid {pid} "
                f"({cmd[:80]})"
            ),
            "action": {"type": "kill_pid", "pid": pid, "port": port, "cmd": cmd},
        })

    # 4. gpu_squatter — anything sitting on real GPU memory we don't track.
    for proc in _gpu_processes():
        if proc["memory_mib"] < gpu_min_mib:
            continue
        if proc["pid"] in tracked_pids:
            continue
        # Container processes show up under their root pid here; if it
        # belongs to one of our tracked containers we want to skip.
        # Best-effort: read /proc/<pid>/cgroup and look for our container ids.
        ours_via_container = False
        try:
            with open(f"/proc/{proc['pid']}/cgroup") as f:
                cg = f.read()
            for cid in tracked_containers:
                if cid and cid[:12] in cg:
                    ours_via_container = True
                    break
        except (FileNotFoundError, PermissionError, OSError):
            pass
        if ours_via_container:
            continue
        findings.append({
            "kind": "gpu_squatter",
            "origin": "foreign",
            "summary": (
                f"foreign pid {proc['pid']} ({proc['process_name']}) holds "
                f"{proc['memory_mib']} MiB GPU memory"
            ),
            "action": {
                "type": "kill_pid",
                "pid": proc["pid"],
                "memory_mib": proc["memory_mib"],
                "cmd": proc["process_name"],
            },
        })

    return findings


def format_findings(findings: list[dict]) -> str:
    """Pretty-print findings for CLI output. One line per finding, grouped."""
    if not findings:
        return ""
    lines = []
    by_kind: dict[str, list[dict]] = {}
    for f in findings:
        by_kind.setdefault(f["kind"], []).append(f)
    order = ["stale_state", "orphan_container", "port_conflict", "gpu_squatter"]
    headers = {
        "stale_state": "Stale state entries (ours):",
        "orphan_container": "Orphan os8-* containers (ours):",
        "port_conflict": "Port conflicts:",
        "gpu_squatter": "Foreign GPU memory holders:",
    }
    for kind in order:
        if kind not in by_kind:
            continue
        lines.append(headers[kind])
        for f in by_kind[kind]:
            tag = "" if f["origin"] == "ours" else " [FOREIGN]"
            lines.append(f"  - {f['summary']}{tag}")
    return "\n".join(lines)
