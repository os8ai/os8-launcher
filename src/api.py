"""FastAPI server — exposes all launcher operations as HTTP endpoints and serves the dashboard."""

import collections
import contextlib
import io
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import load_config, config_to_dict, ConfigError
from src.backends import (
    start_backend, stop_backend, stop_all,
    get_status_data, BackendError,
)
from src.clients import start_client, stop_client, ClientError
from src.runtime import serve_combo
from src.credentials import get_ngc_key, set_ngc_key, get_hf_token, set_hf_token
from src.settings import set_port_override, clear_port_override
from src.installer import (
    setup_tool, get_all_tools_status, InstallError,
)
from src.models import (
    get_models_data, download_model, remove_model, is_download_active, ModelError,
)
from src.projects import (
    list_projects, create_project, get_active_project,
    set_active_project, clear_active_project, projects_dir,
    rename_project, ensure_active_project, update_last_selection,
    project_payload as _project_payload,
    ProjectError,
)

# --- Log buffer ---

log_buffer: collections.deque[str] = collections.deque(maxlen=500)
_log_lock = threading.Lock()


class _LogCapture(io.TextIOBase):
    """Captures written text into the shared log buffer."""

    def write(self, s: str) -> int:
        if s and s.strip():
            with _log_lock:
                log_buffer.append(s.rstrip())
        return len(s)

    def flush(self):
        pass


def _run_with_log_capture(fn, *args, **kwargs):
    """Run a function, capturing its stdout into the log buffer."""
    capture = _LogCapture()
    try:
        with contextlib.redirect_stdout(capture):
            fn(*args, **kwargs)
    except Exception as e:
        with _log_lock:
            log_buffer.append(f"Error: {e}")


# --- App setup ---

app = FastAPI(title="os8-launcher", version="0.1.0")


@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def startup():
    # Discover repo root from this file's location
    app.state.repo_root = Path(__file__).resolve().parent.parent
    app.state.config = load_config(app.state.repo_root)
    # Track config.yaml mtime so _config() can hot-reload when the file
    # changes (e.g. a new model entry added after the dashboard started).
    # Without this, CLI edits to config.yaml don't appear until the
    # dashboard process is restarted.
    try:
        app.state.config_mtime = (app.state.repo_root / "config.yaml").stat().st_mtime
    except OSError:
        app.state.config_mtime = 0.0
    # Per-process identity, used by /api/health so the dashboard can tell
    # the post-restart process apart from the pre-restart one (the gap
    # between socket close and rebind is too small to detect by polling).
    app.state.server_id = f"{os.getpid()}-{int(time.time() * 1000)}"


# Guards the config-reload critical section against concurrent requests
# (FastAPI dispatches sync handlers into a threadpool).
_config_reload_lock = threading.Lock()


@app.on_event("shutdown")
def shutdown():
    # uvicorn fires this on graceful shutdown — including SIGTERM/SIGINT.
    # Without it, an external `kill <dashboard pid>` would orphan any
    # running backend container/process and leave a stale state file
    # pointing at it. The /api/server/stop and /api/server/restart paths
    # already call stop_all() before triggering shutdown, so this is a
    # no-op for those (stop_all is safe to call when nothing's running).
    try:
        stop_all()
    except Exception as e:
        print(f"  (shutdown cleanup warning: {e})")


def _config():
    """Return the current config, hot-reloading if config.yaml changed.

    The dashboard originally loaded config.yaml once at startup, so adding a
    new model required restarting the process. This checks the file's mtime
    on every request and re-parses only when it has changed. A parse failure
    (e.g. mid-edit YAML) logs a warning and returns the last-known-good
    config — the next successful reload replaces it.
    """
    config_path = app.state.repo_root / "config.yaml"
    try:
        current_mtime = config_path.stat().st_mtime
    except OSError:
        return app.state.config
    if current_mtime == app.state.config_mtime:
        return app.state.config
    with _config_reload_lock:
        # Re-check inside the lock — another thread may have already reloaded.
        if current_mtime == app.state.config_mtime:
            return app.state.config
        try:
            app.state.config = load_config(app.state.repo_root)
            print("  (config.yaml reloaded)", flush=True)
        except (ConfigError, Exception) as e:
            print(f"  (config reload failed, keeping last-known-good: {e})", flush=True)
        # Record the mtime regardless of success so we don't retry on every
        # request while the file is mid-edit with a syntax error.
        app.state.config_mtime = current_mtime
    return app.state.config


def _repo_root() -> Path:
    return app.state.repo_root


# --- Request models ---

class ServeRequest(BaseModel):
    model: str
    backend: str | None = None
    client: str | None = None


class DownloadRequest(BaseModel):
    backend: str | None = None


class ClientStartRequest(BaseModel):
    model: str | None = None
    backend: str | None = None


class ProjectCreateRequest(BaseModel):
    name: str
    description: str | None = None


class ProjectActivateRequest(BaseModel):
    name: str


class ProjectRenameRequest(BaseModel):
    new_name: str


class CredentialsRequest(BaseModel):
    ngc_api_key: str | None = None
    hf_token: str | None = None


class PortsSaveRequest(BaseModel):
    overrides: dict[str, int]


class PortsResetRequest(BaseModel):
    name: str


# Services whose listening port isn't actually governed by config.yaml — for
# these, overriding the port in the Ports tab wouldn't change runtime behavior,
# so hide them instead of showing a control that silently does nothing.
#
# ollama: the manifest's env hardcodes OLLAMA_HOST="0.0.0.0:11434" (binary
# backends don't template-expand env, and the value is a literal). Changing
# the config port would only break the health check.
_PORTS_HIDDEN = {"ollama"}


# --- Credentials endpoints ---

@app.get("/api/credentials")
def api_credentials():
    return {
        "ngc": bool(get_ngc_key()),
        "hf": bool(get_hf_token()),
    }


@app.post("/api/credentials")
def api_credentials_set(req: CredentialsRequest):
    if req.ngc_api_key is not None:
        set_ngc_key(req.ngc_api_key)
    if req.hf_token is not None:
        set_hf_token(req.hf_token)
    return {
        "ngc": bool(get_ngc_key()),
        "hf": bool(get_hf_token()),
    }


# --- Ports endpoints ---

def _ports_payload(config) -> list[dict]:
    """List every overridable service with its default and current port."""
    rows = []
    for name, b in config.backends.items():
        if name in _PORTS_HIDDEN:
            continue
        rows.append({
            "name": name,
            "kind": "backend",
            "default_port": b.default_port or b.port,
            "current_port": b.port,
            "overridden": b.port != (b.default_port or b.port),
        })
    for name, c in config.clients.items():
        # Bridge clients route to some other backend's port — overriding
        # here wouldn't mean anything. Clients with no port declared (CLI
        # clients like aider) also have nothing to configure.
        if name in _PORTS_HIDDEN or c.type == "bridge" or c.port is None:
            continue
        rows.append({
            "name": name,
            "kind": "client",
            "default_port": c.default_port or c.port,
            "current_port": c.port,
            "overridden": c.port != (c.default_port or c.port),
        })
    return rows


def _apply_override_in_memory(config, name: str, port: int | None):
    """Mirror a settings-file write into the live config.

    `port=None` means "revert to default". Keeps the dashboard's in-memory
    view consistent with settings.yaml without a full config reload.
    """
    if name in config.backends:
        entry = config.backends[name]
        entry.port = port if port is not None else (entry.default_port or entry.port)
    elif name in config.clients:
        entry = config.clients[name]
        if entry.default_port is None:
            return
        entry.port = port if port is not None else entry.default_port


@app.get("/api/ports")
def api_ports():
    return _ports_payload(_config())


@app.post("/api/ports")
def api_ports_set(req: PortsSaveRequest):
    config = _config()
    valid_names = {
        n for n in list(config.backends.keys()) + list(config.clients.keys())
        if n not in _PORTS_HIDDEN
    }
    # Validate the whole batch before writing anything, so a bad row doesn't
    # leave settings.yaml half-updated.
    for name, port in req.overrides.items():
        if name not in valid_names:
            raise HTTPException(status_code=400, detail=f"Unknown service: {name}")
        if not isinstance(port, int) or port < 1024 or port > 65535:
            raise HTTPException(
                status_code=400,
                detail=f"Port for '{name}' must be an integer between 1024 and 65535 (got {port}).",
            )
    for name, port in req.overrides.items():
        entry = config.backends.get(name) or config.clients.get(name)
        default_port = getattr(entry, "default_port", None) or getattr(entry, "port", None)
        if default_port is not None and port == default_port:
            # User typed the default back in — treat as a reset so we don't
            # leave a redundant row in settings.yaml.
            clear_port_override(name)
            _apply_override_in_memory(config, name, None)
        else:
            set_port_override(name, port)
            _apply_override_in_memory(config, name, port)
    return _ports_payload(config)


@app.post("/api/ports/reset")
def api_ports_reset(req: PortsResetRequest):
    config = _config()
    clear_port_override(req.name)
    _apply_override_in_memory(config, req.name, None)
    return _ports_payload(config)


# --- Projects endpoints ---

@app.get("/api/projects")
def api_projects_list():
    # Make sure there's always an active project — auto-create "default"
    # on a fresh install, or activate the most-recently-used one.
    active = ensure_active_project()
    return {
        "projects_dir": str(projects_dir()),
        "active": active.name,
        "active_project": _project_payload(active),
        "projects": [_project_payload(p) for p in list_projects()],
    }


@app.post("/api/projects")
def api_projects_create(req: ProjectCreateRequest):
    try:
        p = create_project(req.name, description=req.description)
        set_active_project(p.name)
    except ProjectError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _project_payload(p)


@app.put("/api/projects/active")
def api_projects_activate(req: ProjectActivateRequest):
    try:
        p = set_active_project(req.name)
    except ProjectError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _project_payload(p)


@app.delete("/api/projects/active")
def api_projects_deactivate():
    clear_active_project()
    return {"status": "cleared"}


@app.put("/api/projects/{name}")
def api_projects_rename(name: str, req: ProjectRenameRequest):
    try:
        p = rename_project(name, req.new_name)
    except ProjectError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _project_payload(p)


# --- Read-only endpoints ---

@app.get("/api/config")
def api_config():
    return config_to_dict(_config())


@app.get("/api/status")
def api_status():
    return get_status_data()


@app.get("/api/status/capabilities")
def api_status_capabilities():
    """Report which OS8 task types the currently-running backend can serve.

    Phase-1 contract (OS8 Local Models plan, v1.1): derive capabilities from the
    single running backend rather than the full eventual resident pool. The model
    → task mapping lives in OS8; here we just surface {model, base_url, model_id}
    per task the running model is eligible for. Returns {} when nothing is
    serving.
    """
    data = get_status_data()
    backend = data.get("backend")
    if not backend:
        return {}

    model_name = backend.get("model")
    port = backend.get("port")
    if not model_name or not port:
        return {}

    base_url = f"http://localhost:{port}"
    # vLLM runs with `--served-model-name {model_name}`, ollama uses the tag.
    # For Phase 1 the model name from state matches what /v1/chat/completions
    # expects as the `model` field. If that diverges later, resolve via the
    # backend manifest's model_name_template.
    entry = {
        "model": model_name,
        "base_url": base_url,
        "model_id": model_name,
    }

    # Map known models to the OS8 task types they're eligible for (Phase 1).
    # Anything not in this table contributes no capability entries — the OS8
    # side will ignore unknown models rather than route tasks to them blindly.
    eligibility = {
        "gemma-4-31B-it-nvfp4": ["conversation", "summary", "planning"],
        "gemma-4-E2B-it": ["conversation", "summary"],
    }
    tasks = eligibility.get(model_name, [])
    return {task: entry for task in tasks}


@app.get("/api/models")
def api_models():
    return get_models_data(_config(), _repo_root(), current_server_id=app.state.server_id)


@app.get("/api/tools")
def api_tools():
    return get_all_tools_status(_config(), _repo_root())


@app.get("/api/logs")
def api_logs():
    with _log_lock:
        return list(log_buffer)


# --- Mutation endpoints ---

@app.post("/api/serve")
def api_serve(req: ServeRequest, background_tasks: BackgroundTasks):
    try:
        # Quick validation before background task
        config = _config()
        config.get_model(req.model)
        if req.backend:
            config.get_backend(req.backend)
        if req.client:
            config.get_client(req.client)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    active = get_active_project()
    if active:
        update_last_selection(
            active.name,
            model=req.model,
            backend=req.backend or "",
            client=req.client or "",
        )

    background_tasks.add_task(
        _run_with_log_capture,
        serve_combo, req.model, req.backend, req.client, _config(), _repo_root(),
    )
    return {"status": "starting"}


@app.delete("/api/serve")
def api_serve_stop():
    """Stop everything: clients first, then the backend.

    Used by the main launch-controls Stop button.
    """
    try:
        _run_with_log_capture(stop_all)
    except BackendError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "stopped"}


@app.delete("/api/backend")
def api_backend_stop():
    """Stop only the backend, leaving clients running.

    Used by the per-row Stop button next to the backend in the active session.
    """
    try:
        _run_with_log_capture(stop_backend)
    except BackendError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "stopped"}


@app.post("/api/clients/{name}")
def api_client_start(
    name: str,
    background_tasks: BackgroundTasks,
    req: ClientStartRequest = None,
):
    try:
        config = _config()
        config.get_client(name)
        if req and req.model:
            config.get_model(req.model)
        if req and req.backend:
            config.get_backend(req.backend)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = req.model if req else None
    backend = req.backend if req else None

    active = get_active_project()
    if active:
        update_last_selection(
            active.name,
            model=model,
            backend=backend or "",
            client=name,
        )

    background_tasks.add_task(
        _run_with_log_capture,
        start_client, name, _config(), _repo_root(), model, backend,
    )
    return {"status": "starting"}


@app.delete("/api/clients/{name}")
def api_client_stop(name: str):
    _run_with_log_capture(stop_client, name)
    return {"status": "stopped"}


@app.post("/api/models/{name}/download")
def api_model_download(name: str, req: DownloadRequest = None, background_tasks: BackgroundTasks = None):
    try:
        _config().get_model(name)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Guard against double-starts: the same endpoint handles Resume, which
    # is safe against an interrupted download (marker present, no live
    # task) but must not spawn a second task on top of a running one.
    if is_download_active(name):
        raise HTTPException(status_code=409, detail=f"A download for '{name}' is already in progress.")

    backend = req.backend if req else None
    background_tasks.add_task(
        _run_with_log_capture,
        download_model, name, _config(), _repo_root(), backend, app.state.server_id,
    )
    return {"status": "started"}


@app.delete("/api/models/{name}")
def api_model_remove(name: str):
    try:
        _run_with_log_capture(remove_model, name, _config(), _repo_root())
    except (ConfigError, ModelError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "removed"}


@app.post("/api/tools/{name}/setup")
def api_tool_setup(name: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(
        _run_with_log_capture,
        setup_tool, name, _config(), _repo_root(),
    )
    return {"status": "started"}


@app.delete("/api/stop")
def api_stop_all():
    _run_with_log_capture(stop_all)
    return {"status": "stopped"}


# --- Dashboard server lifecycle (stop / restart from the UI) ---

@app.get("/api/health")
def api_health():
    return {"status": "ok", "server_id": getattr(app.state, "server_id", None)}


def _shutdown_after_response(restart: bool):
    """Run in a background thread so the HTTP response is flushed first."""
    import time
    time.sleep(0.2)
    # Tear down any backend/clients we own before exiting, for both stop and
    # restart. Doing it here (rather than on the next startup) means the new
    # process binds the port quickly and the browser's health-poll window
    # isn't eaten by docker stop draining containers.
    try:
        stop_all()
    except Exception as e:
        print(f"  (cleanup warning: {e})")
    app.state.restart_requested = restart
    server = getattr(app.state, "server", None)
    if server is not None:
        # force_exit bypasses the graceful-shutdown wait for keep-alive connections,
        # which the browser dashboard holds open via its 3s status polling.
        server.should_exit = True
        server.force_exit = True


@app.post("/api/server/stop")
def api_server_stop(background_tasks: BackgroundTasks):
    background_tasks.add_task(_shutdown_after_response, False)
    return {"status": "stopping"}


@app.post("/api/server/restart")
def api_server_restart(background_tasks: BackgroundTasks):
    background_tasks.add_task(_shutdown_after_response, True)
    return {"status": "restarting"}


# --- Static files (must be last) ---

_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="dashboard")
