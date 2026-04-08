"""Persistent verification log for (model, backend) pairs.

Records which (model, backend) combinations have been observed to start
successfully on this machine, and the most recent failure if any. The launcher
auto-updates these records on every serve attempt — there is no manual editing.

Stored at ~/.config/os8-launcher/verification.yaml, separate from state.yaml so
that runtime stop/start operations never touch verification history.
"""

from datetime import datetime
from pathlib import Path

import yaml

VERIFICATION_DIR = Path.home() / ".config" / "os8-launcher"
VERIFICATION_FILE = VERIFICATION_DIR / "verification.yaml"


def load_verification() -> dict:
    """Load the full verification map, or return an empty dict."""
    if not VERIFICATION_FILE.exists():
        return {}
    with open(VERIFICATION_FILE) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save(data: dict):
    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(VERIFICATION_FILE, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=True)


def get_for_model(model_name: str) -> dict:
    """Return verification entries keyed by backend name for a given model."""
    return load_verification().get(model_name, {})


def record_success(model_name: str, backend_name: str, runtime: str | None = None):
    """Mark (model, backend) as verified-working as of now.

    `runtime` is a free-form identifier of what served it (e.g. a docker image
    tag). Stored so we know whether a verification predates a backend upgrade.
    Clears any previous failure on this pair.
    """
    data = load_verification()
    entry = data.setdefault(model_name, {}).setdefault(backend_name, {})
    entry["verified"] = True
    entry["verified_on"] = datetime.now().isoformat(timespec="seconds")
    entry["verified_runtime"] = runtime or "unknown"
    entry["last_failure"] = None
    entry["last_failure_on"] = None
    _save(data)


def record_failure(model_name: str, backend_name: str, error: str):
    """Record the most recent failure for a (model, backend) pair.

    Does NOT clear an existing `verified: true` — that remains as historical
    evidence; the dashboard will show both ("verified previously, failed last
    attempt"), which is the most useful signal during a regression.
    """
    data = load_verification()
    entry = data.setdefault(model_name, {}).setdefault(backend_name, {})
    # Truncate long error messages so the file stays readable.
    msg = error.strip().splitlines()[0][:300] if error else "unknown error"
    entry["last_failure"] = msg
    entry["last_failure_on"] = datetime.now().isoformat(timespec="seconds")
    entry.setdefault("verified", False)
    _save(data)
