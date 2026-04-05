"""API key management for NGC and HuggingFace.

Credentials are stored in ~/.config/os8-launcher/credentials.yaml.
Environment variables (NGC_API_KEY, HF_TOKEN) take precedence over the file.
"""

import os
from pathlib import Path

import yaml

CREDENTIALS_DIR = Path.home() / ".config" / "os8-launcher"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials.yaml"


def _load_credentials() -> dict:
    """Load credentials from the file, or return empty dict."""
    if not CREDENTIALS_FILE.exists():
        return {}
    with open(CREDENTIALS_FILE) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save_credentials(data: dict):
    """Write credentials to the file."""
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CREDENTIALS_FILE, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    # Restrict permissions — local-only keys, but no reason to be world-readable
    CREDENTIALS_FILE.chmod(0o600)


def get_ngc_key() -> str | None:
    """Get NGC API key from env var or credentials file."""
    env_key = os.environ.get("NGC_API_KEY")
    if env_key:
        return env_key
    return _load_credentials().get("ngc_api_key")


def set_ngc_key(key: str):
    """Store NGC API key in credentials file."""
    data = _load_credentials()
    data["ngc_api_key"] = key
    _save_credentials(data)


def get_hf_token() -> str | None:
    """Get HuggingFace token from env var or credentials file."""
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    return _load_credentials().get("hf_token")


def set_hf_token(token: str):
    """Store HuggingFace token in credentials file."""
    data = _load_credentials()
    data["hf_token"] = token
    _save_credentials(data)


def prompt_ngc_key() -> str | None:
    """Interactively prompt for NGC API key, validate, and store it."""
    print("An NGC API key is required for NVIDIA NIM containers.")
    print("Get one at: https://ngc.nvidia.com")
    print()
    try:
        key = input("NGC API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not key:
        return None

    set_ngc_key(key)
    print("NGC API key saved.")
    return key


def prompt_hf_token() -> str | None:
    """Interactively prompt for HuggingFace token and store it."""
    print("A HuggingFace token is required for this model.")
    print("Get one at: https://huggingface.co/settings/tokens")
    print()
    try:
        token = input("HuggingFace token: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not token:
        return None

    set_hf_token(token)
    print("HuggingFace token saved.")
    return token
