"""Project folders — isolated working directories for client sessions.

A "project" is just a directory under `projects_dir` (default
`~/os8-projects`) with an optional `.os8-project.yaml` metadata file.
Clients launched while a project is active run with that directory as
their cwd, so aider (and friends) edit files inside the project rather
than in the launcher repo.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


PROJECTS_DIR = Path.home() / "os8-projects"
META_FILENAME = ".os8-project.yaml"


class ProjectError(Exception):
    """Raised when a project operation fails."""


@dataclass
class Project:
    name: str
    path: Path
    created_at: str | None = None
    default_model: str | None = None
    default_backend: str | None = None
    default_client: str | None = None
    description: str | None = None


def project_payload(p: "Project") -> dict:
    """Single source of truth for serializing a Project to a JSON-friendly dict.

    Used by both the API (dashboard) and the CLI so the two views never drift.
    """
    return {
        "name": p.name,
        "path": str(p.path),
        "description": p.description,
        "created_at": p.created_at,
        "last_model": p.default_model,
        "last_backend": p.default_backend,
        "last_client": p.default_client,
    }


def projects_dir() -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    return PROJECTS_DIR


def _read_meta(path: Path) -> dict:
    meta_path = path / META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_meta(path: Path, data: dict):
    with open(path / META_FILENAME, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _project_from_dir(path: Path) -> Project:
    meta = _read_meta(path)
    return Project(
        name=path.name,
        path=path,
        created_at=meta.get("created_at"),
        default_model=meta.get("last_model") or meta.get("default_model"),
        default_backend=meta.get("last_backend") or meta.get("default_backend"),
        default_client=meta.get("last_client") or meta.get("default_client"),
        description=meta.get("description"),
    )


def update_last_selection(
    name: str,
    model: str | None = None,
    backend: str | None = None,
    client: str | None = None,
):
    """Persist the most-recent launch choices on the project, so the
    dashboard can prefill them next time."""
    try:
        project = get_project(name)
    except ProjectError:
        return
    meta = _read_meta(project.path)
    if model is not None:
        meta["last_model"] = model
    if backend is not None:
        meta["last_backend"] = backend
    # Client is allowed to be cleared, so we record explicit empty string too.
    if client is not None:
        meta["last_client"] = client
    _write_meta(project.path, meta)


def ensure_active_project() -> Project:
    """Guarantee there's an active project. If none is set, pick the most
    recently used one, or create 'default' if there are none at all.

    This is what makes the dashboard "just work" on a fresh install — the
    user never lands on an empty picker.
    """
    active = get_active_project()
    if active:
        return active

    root = projects_dir()
    existing = [p for p in root.iterdir() if p.is_dir()]
    if existing:
        # Most-recently-modified wins — last opened/edited project.
        chosen = max(existing, key=lambda p: p.stat().st_mtime)
        return set_active_project(chosen.name)

    project = create_project("default", description="Default os8-launcher project")
    set_active_project(project.name)
    return project


def list_projects() -> list[Project]:
    root = projects_dir()
    return sorted(
        (_project_from_dir(p) for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )


def get_project(name: str) -> Project:
    path = projects_dir() / name
    if not path.is_dir():
        raise ProjectError(f"Project '{name}' not found at {path}")
    return _project_from_dir(path)


def create_project(name: str, description: str | None = None) -> Project:
    if not name or "/" in name or name.startswith("."):
        raise ProjectError(f"Invalid project name: '{name}'")
    path = projects_dir() / name
    if path.exists():
        raise ProjectError(f"Project '{name}' already exists at {path}")
    path.mkdir(parents=True)
    meta = {"created_at": datetime.now().isoformat()}
    if description:
        meta["description"] = description
    _write_meta(path, meta)
    return _project_from_dir(path)


def rename_project(old_name: str, new_name: str) -> Project:
    if not new_name or "/" in new_name or new_name.startswith("."):
        raise ProjectError(f"Invalid project name: '{new_name}'")
    if old_name == new_name:
        return get_project(old_name)
    src = projects_dir() / old_name
    if not src.is_dir():
        raise ProjectError(f"Project '{old_name}' not found")
    dst = projects_dir() / new_name
    if dst.exists():
        raise ProjectError(f"Project '{new_name}' already exists")
    src.rename(dst)
    # If the renamed project was active, update the pointer in state.
    from src.state import load_state, save_state
    data = load_state()
    if data.get("active_project") == old_name:
        data["active_project"] = new_name
        save_state(data)
    return _project_from_dir(dst)


def update_project_defaults(
    name: str,
    default_model: str | None = None,
    default_backend: str | None = None,
    default_client: str | None = None,
):
    project = get_project(name)
    meta = _read_meta(project.path)
    if default_model is not None:
        meta["default_model"] = default_model
    if default_backend is not None:
        meta["default_backend"] = default_backend
    if default_client is not None:
        meta["default_client"] = default_client
    _write_meta(project.path, meta)


def get_active_project() -> Project | None:
    """Return the currently-active project, or None if none is set."""
    from src.state import load_state
    name = load_state().get("active_project")
    if not name:
        return None
    try:
        return get_project(name)
    except ProjectError:
        return None


def set_active_project(name: str) -> Project:
    from src.state import load_state, save_state
    project = get_project(name)  # validates existence
    data = load_state()
    data["active_project"] = name
    save_state(data)
    return project


def clear_active_project():
    from src.state import load_state, save_state
    data = load_state()
    if "active_project" in data:
        data.pop("active_project")
        save_state(data)
