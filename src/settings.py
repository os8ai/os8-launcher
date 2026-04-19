"""User-level launcher settings — currently just port overrides.

Stored at ~/.config/os8-launcher/settings.yaml, parallel to credentials.yaml.
Defaults live in config.yaml; this file only records deltas the user has
chosen from the dashboard.
"""

from pathlib import Path

import yaml

SETTINGS_DIR = Path.home() / ".config" / "os8-launcher"
SETTINGS_FILE = SETTINGS_DIR / "settings.yaml"


def _load() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    with open(SETTINGS_FILE) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save(data: dict):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def get_port_overrides() -> dict[str, int]:
    """Return {name: port} for every service with a custom port."""
    data = _load()
    raw = data.get("port_overrides") or {}
    out: dict[str, int] = {}
    for name, port in raw.items():
        try:
            out[str(name)] = int(port)
        except (TypeError, ValueError):
            continue
    return out


def set_port_override(name: str, port: int):
    data = _load()
    overrides = data.get("port_overrides") or {}
    overrides[name] = int(port)
    data["port_overrides"] = overrides
    _save(data)


def clear_port_override(name: str):
    data = _load()
    overrides = data.get("port_overrides") or {}
    if name in overrides:
        del overrides[name]
        data["port_overrides"] = overrides
        _save(data)
