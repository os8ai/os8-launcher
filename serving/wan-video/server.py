"""Wan 2.1 text-to-video serving backend.

Thin FastAPI wrapper around diffusers.WanPipeline.  Loads the model at
startup from the path in $WAN_MODEL_PATH, serves one generation at a time,
and exposes progress via a polling endpoint.
"""

import os
import threading
import time
import uuid
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="os8 wan-video")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipe: WanPipeline | None = None
output_dir = Path(os.environ.get("WAN_OUTPUT_DIR", "var/wan-video-output"))

# Generation state — guarded by _lock
_lock = threading.Lock()
_job: dict | None = None  # {id, prompt, step, total_steps, status, path}


class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = Field(default=81, ge=5, le=257)
    height: int = Field(default=480, ge=256, le=1280)
    width: int = Field(default=848, ge=256, le=1280)
    guidance_scale: float = Field(default=5.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=30, ge=1, le=100)
    seed: int = Field(default=-1)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipe
    model_path = os.environ.get("WAN_MODEL_PATH")
    if not model_path:
        raise RuntimeError("WAN_MODEL_PATH environment variable is not set")
    print(f"[wan-video] Loading model from {model_path} ...")
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # With PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in manifest),
    # the full pipeline (~51 GB in bf16) fits comfortably on the Spark's 130 GB
    # unified GPU memory.  VAE tiling/slicing reduces peak memory during video
    # decode so the whole pipeline stays resident.
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[wan-video] Model loaded, ready to serve.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok" if pipe is not None else "loading"}


@app.get("/status")
def status():
    with _lock:
        if _job is None:
            return {"status": "idle"}
        return {
            "status": _job["status"],
            "id": _job["id"],
            "prompt": _job["prompt"],
            "step": _job["step"],
            "total_steps": _job["total_steps"],
        }


@app.post("/generate")
def generate(req: GenerateRequest):
    global _job
    with _lock:
        if _job is not None and _job["status"] == "generating":
            raise HTTPException(409, "A generation is already in progress")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "prompt": req.prompt,
            "step": 0,
            "total_steps": req.num_inference_steps,
            "status": "generating",
            "path": None,
        }
        _job = job

    thread = threading.Thread(target=_run_generation, args=(job, req), daemon=True)
    thread.start()
    return {"id": job_id, "status": "generating"}


@app.get("/result/{job_id}")
def result(job_id: str):
    with _lock:
        if _job is None or _job["id"] != job_id:
            raise HTTPException(404, "Job not found")
        if _job["status"] != "done":
            raise HTTPException(202, "Job still in progress")
        path = _job["path"]
    if not path or not Path(path).exists():
        raise HTTPException(500, "Output file missing")
    return FileResponse(path, media_type="video/mp4", filename=f"{job_id}.mp4")


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------
def _step_callback(pipeline, step_index, timestep, callback_kwargs):
    """Called by the pipeline after each denoising step."""
    with _lock:
        if _job is not None:
            _job["step"] = step_index + 1
    return callback_kwargs


def _run_generation(job: dict, req: GenerateRequest):
    try:
        generator = None
        if req.seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(req.seed)

        output = pipe(
            prompt=req.prompt,
            num_frames=req.num_frames,
            height=req.height,
            width=req.width,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            generator=generator,
            callback_on_step_end=_step_callback,
        )

        video_path = str(output_dir / f"{job['id']}.mp4")
        export_to_video(output.frames[0], video_path, fps=16)

        with _lock:
            job["status"] = "done"
            job["path"] = video_path

    except Exception as e:
        print(f"[wan-video] Generation failed: {e}")
        with _lock:
            job["status"] = "error"
            job["error"] = str(e)
