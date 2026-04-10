"""System diagnostics for os8-launcher.

Runs all prerequisite checks and prints a formatted report so the user
can verify their machine is ready before downloading models or serving.
"""

import shutil
import subprocess
from pathlib import Path

from src.preflight import (
    check_available_memory,
    check_cuda_version,
    check_disk_space,
    check_docker,
    check_nvidia_container_toolkit,
    check_nvidia_gpu,
    check_python,
    detect_arch,
    get_gpu_info,
)


def run_doctor(repo_root: Path):
    """Run all system checks and print a summary."""
    print("os8-launcher doctor")
    print("=" * 40)
    print()

    passed = 0
    failed = 0

    def report(label: str, ok: bool, detail: str):
        nonlocal passed, failed
        status = "ok" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        # Right-align the status marker
        line = f"  {label + ':':<22} {detail}"
        print(f"{line:<54} [{status}]")

    # 1. Architecture
    try:
        arch = detect_arch()
        report("Architecture", True, f"{arch['machine']} (docker: {arch['docker']})")
    except RuntimeError as e:
        report("Architecture", False, str(e))

    # 2. GPU
    gpu = get_gpu_info()
    if gpu:
        mem = f"{gpu['memory_gb']:.0f} GB" if gpu["memory_gb"] > 0 else "shared"
        report("GPU", True, f"{gpu['name']} ({mem})")
    else:
        ok, msg = check_nvidia_gpu()
        report("GPU", ok, msg if msg else "detected")

    # 3. CUDA
    ok, detail = check_cuda_version()
    if ok:
        report("CUDA", True, detail)
    else:
        report("CUDA", False, detail)

    # 4. RAM
    ok, detail = check_available_memory()
    if ok:
        report("RAM", True, detail)
    else:
        report("RAM", False, detail)

    # 5. Disk
    ok, detail = check_disk_space(str(repo_root), 10)
    if ok:
        import shutil as _sh
        free_gb = _sh.disk_usage(str(repo_root)).free / (1024 ** 3)
        report("Disk", True, f"{free_gb:.0f} GB free")
    else:
        report("Disk", False, detail)

    # 6. Docker
    ok, detail = check_docker()
    if ok:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip().split(",")[0].replace("Docker version ", "") if result.returncode == 0 else "installed"
        report("Docker", True, version)
    else:
        report("Docker", False, detail.split("\n")[0])

    # 7. NVIDIA Container Toolkit
    ok, detail = check_nvidia_container_toolkit()
    if ok:
        report("NVIDIA Toolkit", True, "configured")
    else:
        report("NVIDIA Toolkit", False, detail.split("\n")[0])

    # 8. Python
    ok, detail = check_python()
    if ok:
        result = subprocess.run(
            ["python3", "--version"], capture_output=True, text=True, timeout=10,
        )
        version = result.stdout.strip().replace("Python ", "") if result.returncode == 0 else "installed"
        report("Python", True, version)
    else:
        report("Python", False, detail.split("\n")[0])

    # Summary
    total = passed + failed
    print()
    if failed == 0:
        print(f"All {total} checks passed. Ready to launch.")
    else:
        print(f"{passed}/{total} checks passed, {failed} failed.")
        print("Fix the issues above, then re-run: ./launcher doctor")
