"""FastAPI server — exposes all launcher operations as HTTP endpoints and serves the dashboard."""

import collections
import contextlib
import io
import threading
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
from src.credentials import get_ngc_key, set_ngc_key, get_hf_token, set_hf_token
from src.installer import (
    setup_tool, get_all_tools_status, InstallError,
)
from src.models import (
    get_models_data, download_model, remove_model, ModelError,
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


@app.on_event("startup")
def startup():
    # Discover repo root from this file's location
    app.state.repo_root = Path(__file__).resolve().parent.parent
    app.state.config = load_config(app.state.repo_root)


def _config():
    return app.state.config


def _repo_root() -> Path:
    return app.state.repo_root


# --- Request models ---

class ServeRequest(BaseModel):
    model: str
    backend: str | None = None


class DownloadRequest(BaseModel):
    backend: str | None = None


class CredentialsRequest(BaseModel):
    ngc_api_key: str | None = None
    hf_token: str | None = None


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


# --- Read-only endpoints ---

@app.get("/api/config")
def api_config():
    return config_to_dict(_config())


@app.get("/api/status")
def api_status():
    return get_status_data()


@app.get("/api/models")
def api_models():
    return get_models_data(_config(), _repo_root())


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
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(
        _run_with_log_capture,
        start_backend, req.model, req.backend, _config(), _repo_root(),
    )
    return {"status": "starting"}


@app.delete("/api/serve")
def api_serve_stop():
    try:
        _run_with_log_capture(stop_backend)
    except BackendError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "stopped"}


@app.post("/api/clients/{name}")
def api_client_start(name: str, background_tasks: BackgroundTasks):
    try:
        _config().get_client(name)
    except ConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))

    background_tasks.add_task(
        _run_with_log_capture,
        start_client, name, _config(), _repo_root(),
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

    backend = req.backend if req else None
    background_tasks.add_task(
        _run_with_log_capture,
        download_model, name, _config(), _repo_root(), backend,
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


# --- Static files (must be last) ---

_web_dir = Path(__file__).resolve().parent.parent / "web"
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=str(_web_dir), html=True), name="dashboard")
