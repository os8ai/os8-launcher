"""FastWan 2.2 text-to-video serving backend.

FastWan 2.2 TI2V-5B is a DMD-distilled checkpoint: the transformer was
trained to denoise at three specific timesteps in one inference pass,
not to play nicely with a general-purpose multistep scheduler.  Loading
these weights with stock ``WanPipeline`` and running 3 uniformly-spaced
UniPC steps produces garbage — the transformer is being queried at
timesteps it never saw during training.

The canonical DMD inference lives in fastvideo's ``DmdDenoisingStage``,
but installing ``fastvideo`` downgrades torch off the cu130 wheel (no
Blackwell sm_121 support).  So we keep cu130 torch + stock diffusers
and reimplement the DMD denoising loop here against the pipeline's
components directly.  The math, verbatim from the reference:

    1. Initialize a ``FlowMatchEulerDiscreteScheduler(shift=8.0)`` with
       a dense 1000-step sigma table (the checkpoint's own UniPC
       scheduler is placeholder metadata — fastvideo overrides it).
    2. For t in [1000, 757, 522]:
         - Build timestep tensor (per-token expansion if
           ``expand_timesteps`` is set, which it is for Wan 2.2).
         - pred_velocity = transformer(noise_latents, t, prompt_embeds)
         - pred_video = noise_latents - sigma_t * pred_velocity
         - If not last step: noise_latents =
             sigma_next * fresh_noise + (1 - sigma_next) * pred_video
           where sigma_next is looked up against the dense schedule by
           argmin (the DMD timesteps are off-grid for any reasonable
           shifted FlowMatch schedule, so exact-match index lookups —
           e.g. ``scheduler.scale_noise`` — fail).
         - Else: latents = pred_video
    3. Decode via pipe.vae with the standard Wan latents mean/std
       normalization, then postprocess via ``pipe.video_processor``.

No CFG; no negative prompt; no ``pipe(prompt=...)``.
"""

import os
import threading
import traceback
import uuid
from pathlib import Path

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, WanPipeline
from diffusers.utils import export_to_video
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="os8 fastwan-video")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DMD inference constants — values from fastvideo's FastWan2_2_TI2V_5B_Config
# and DmdDenoisingStage.  Changing these almost certainly ruins output quality.
# ---------------------------------------------------------------------------
DMD_TIMESTEPS = [1000, 757, 522]
DMD_FLOW_SHIFT = 8.0
DMD_SCHEDULER_STEPS = 1000  # dense sigma table for timestep lookup

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
pipe: WanPipeline | None = None
dmd_scheduler: FlowMatchEulerDiscreteScheduler | None = None
output_dir = Path(os.environ.get("FASTWAN_OUTPUT_DIR", "var/fastwan-video-output"))

_lock = threading.Lock()
_job: dict | None = None


class GenerateRequest(BaseModel):
    prompt: str
    num_frames: int = Field(default=81, ge=5, le=257)
    height: int = Field(default=480, ge=256, le=1280)
    width: int = Field(default=848, ge=256, le=1280)
    # Kept for API compatibility with the wan-video backend, but ignored:
    # DMD is fixed at 3 specific timesteps (DMD_TIMESTEPS).  Overriding
    # the count would push the transformer off-training-distribution.
    guidance_scale: float = Field(default=1.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(default=3, ge=1, le=8)
    seed: int = Field(default=-1)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_model():
    import time

    global pipe, dmd_scheduler
    model_path = os.environ.get("FASTWAN_MODEL_PATH")
    if not model_path:
        raise RuntimeError("FASTWAN_MODEL_PATH environment variable is not set")

    t0 = time.perf_counter()
    print(f"[fastwan-video] Loading model from {model_path} ...", flush=True)
    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    print(
        f"[fastwan-video] Pipeline components initialized in "
        f"{time.perf_counter() - t0:.1f}s; transferring ~22 GB to CUDA "
        f"(first time after a reboot can take 60-120s on the Spark) ...",
        flush=True,
    )

    t1 = time.perf_counter()
    pipe.to("cuda")
    print(
        f"[fastwan-video] Model on CUDA in {time.perf_counter() - t1:.1f}s. "
        f"Configuring VAE tiling/slicing ...",
        flush=True,
    )

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the DMD scheduler on its dense 1000-step schedule so
    # _dmd_sample() can look up sigma_t for any of our 3 timesteps via
    # argmin.  The checkpoint's own scheduler (UniPC) is left in place
    # on pipe.scheduler — unused by us, but kept so anyone poking the
    # pipe via its normal API gets a consistent view.
    dmd_scheduler = FlowMatchEulerDiscreteScheduler(shift=DMD_FLOW_SHIFT)
    dmd_scheduler.set_timesteps(DMD_SCHEDULER_STEPS, device="cuda")

    print(
        f"[fastwan-video] Model loaded, ready to serve. "
        f"Total cold-start: {time.perf_counter() - t0:.1f}s.",
        flush=True,
    )


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
            # DMD is always 3 steps regardless of what the client sent.
            "total_steps": len(DMD_TIMESTEPS),
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
# DMD inference
# ---------------------------------------------------------------------------
def _update_step(i: int) -> None:
    with _lock:
        if _job is not None:
            _job["step"] = min(i + 1, _job["total_steps"])


def _sigma_for_timestep(t: torch.Tensor) -> torch.Tensor:
    """Look up the FlowMatchEuler sigma for timestep ``t`` via argmin
    against the dense 1000-step schedule.  Matches fastvideo's
    ``pred_noise_to_pred_video``."""
    assert dmd_scheduler is not None
    ts = dmd_scheduler.timesteps.to(t.device).double()
    sigmas = dmd_scheduler.sigmas.to(t.device).double()
    idx = torch.argmin((ts - t.double()).abs())
    return sigmas[idx]


@torch.no_grad()
def _dmd_sample(req: GenerateRequest, generator: torch.Generator | None):
    """Run 3-step DMD inference using WanPipeline's loaded components.

    Returns a list of numpy frames suitable for ``export_to_video``.
    """
    assert pipe is not None and dmd_scheduler is not None
    device = pipe._execution_device
    transformer_dtype = pipe.transformer.dtype

    # 1. Encode prompt — no CFG, so no negative prompt.
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=req.prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(transformer_dtype)

    # 2. Normalize frame count to the VAE's temporal stride and snap
    # H/W to the transformer's patchification grid — both adjustments
    # WanPipeline.__call__ makes up front (pipeline_wan.py:490-511).
    # Critical for odd latent dims: Wan 2.2's VAE scale_factor_spatial
    # is 16 and patch_size[1:] is (2, 2), so H and W must be multiples
    # of 32 or the transformer's stride-2 conv patchification and our
    # `mask[..., ::2, ::2]` timestep expansion disagree by one row/col.
    vsft = pipe.vae_scale_factor_temporal
    num_frames = req.num_frames
    if num_frames % vsft != 1:
        num_frames = num_frames // vsft * vsft + 1

    patch_size = pipe.transformer.config.patch_size
    h_mult = pipe.vae_scale_factor_spatial * patch_size[1]
    w_mult = pipe.vae_scale_factor_spatial * patch_size[2]
    height = req.height // h_mult * h_mult
    width = req.width // w_mult * w_mult

    # 3. Prepare initial noisy latents.  Kept in fp32 during denoising
    # for numerical headroom; the transformer call re-casts to bf16.
    num_channels_latents = pipe.transformer.config.in_channels
    latents = pipe.prepare_latents(
        1,  # batch_size
        num_channels_latents,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        None,
    )
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    # 4. DMD denoising loop over the three specific timesteps.
    timesteps = torch.tensor(DMD_TIMESTEPS, dtype=torch.long, device=device)
    for i, t in enumerate(timesteps):
        noise_latents = latents
        latent_input = latents.to(transformer_dtype)

        if getattr(pipe.config, "expand_timesteps", False):
            # Per-spatial-token timestep; shape (1, seq_len).  Mirrors
            # WanPipeline.__call__:601 for the Wan 2.2 transformer.
            temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
            timestep_tensor = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
        else:
            timestep_tensor = t.expand(latents.shape[0])

        pred_velocity = pipe.transformer(
            hidden_states=latent_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        sigma_t = _sigma_for_timestep(t).to(pred_velocity.dtype)
        pred_video = noise_latents.to(pred_velocity.dtype) - sigma_t * pred_velocity

        if i < len(timesteps) - 1:
            noise = torch.randn(
                latents.shape,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            # Re-noise at the next DMD timestep.  Inlined from
            # FlowMatchEulerDiscreteScheduler.scale_noise because that
            # method indexes its sigma table by exact-equality match on
            # the timestep, and our DMD timesteps (757, 522) are not
            # present in the shift=8 schedule's float values.
            sigma_next = _sigma_for_timestep(timesteps[i + 1]).to(noise.dtype)
            latents = sigma_next * noise + (1.0 - sigma_next) * pred_video.to(noise.dtype)
        else:
            latents = pred_video

        _update_step(i)

    # 5. Decode.  Replicates WanPipeline.__call__:649-661.
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="np")
    return frames[0]  # first (and only) video in the batch


def _run_generation(job: dict, req: GenerateRequest):
    try:
        generator = None
        if req.seed >= 0:
            generator = torch.Generator(device="cuda").manual_seed(req.seed)

        frames = _dmd_sample(req, generator)

        video_path = str(output_dir / f"{job['id']}.mp4")
        export_to_video(frames, video_path, fps=16)

        with _lock:
            job["status"] = "done"
            job["path"] = video_path

    except Exception as e:
        print(f"[fastwan-video] Generation failed: {e}", flush=True)
        traceback.print_exc()
        with _lock:
            job["status"] = "error"
            job["error"] = str(e)
