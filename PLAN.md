# Milestone 1 Plan: Standalone os8-launcher

**Goal:** Download Nemotron-3-Super-120B-A12B-NVFP4, serve it via NVIDIA NIM, connect aider, and use aider for AI-assisted coding powered by a fully local model.

**Approach:** Build bottom-up in four phases. Each phase produces something testable before moving on. No phase depends on anything outside the previous phase.

---

## Phase 1: Foundation

*Project skeleton, configuration, and the CLI entry point.*

**What gets built:**
- Directory structure (`src/`, `web/`, `models/`, `serving/`, `clients/`)
- Python package setup (`pyproject.toml`, `src/__init__.py`)
- `config.yaml` with the initial model/backend/client definitions
- `launcher` CLI entry point (executable script that dispatches subcommands)
- `src/config.py` — parse, validate, and resolve `config.yaml` + manifests
- Manifest files for NIM, vLLM, llama.cpp, aider, Open WebUI, OS8
- `.gitignore` for runtime artifacts (venvs, binaries, weights, containers)

**Testable outcome:** `./launcher config show` prints the parsed configuration. `./launcher --help` lists available subcommands.

**Big rocks:**
- Get the config schema right the first time. Everything downstream reads `config.yaml` — changing its shape later means changing everything.
- Manifest format needs to be flexible enough for three install types (pip, container, binary) without being over-engineered.

**Steps:** 2 (skeleton + config parser from the build sequence)

---

## Phase 2: Tool & Model Management

*Installing dependencies and downloading model weights.*

**What gets built:**
- `src/installer.py` — reads manifests, creates venvs, pulls container images, downloads binaries. Supports `./launcher setup <tool>` and `./launcher setup --all`.
- `src/models.py` — downloads model weights (HuggingFace/NGC), tracks download progress, reports disk usage. Supports `./launcher models list`, `./launcher models download <model>`, `./launcher models remove <model>`.
- `src/preflight.py` — system prerequisite checks that run before any install or serve operation.

**Testable outcome:** `./launcher setup aider` creates a working aider venv. `./launcher setup nim` pulls the NIM container image. `./launcher models download nemotron-3-super-120b` downloads the weights and `./launcher models list` shows them with disk usage. Running any of these without prerequisites installed produces a clear, actionable error — not a stack trace.

**Big rocks:**

- **Prerequisite detection.** The launcher orchestrates tools but doesn't install system-level dependencies. Before any operation, it needs to verify what's present and give actionable guidance when something is missing:
  - Docker + NVIDIA Container Toolkit (for NIM, Open WebUI)
  - Python 3.x (for pip-based tools)
  - git-lfs (for model weight downloads)
  - Sufficient disk space (check *before* starting a 100GB+ download, not after it fails)
  - GPU visibility (can the system see the GB10? is `nvidia-smi` working?)

  The launcher can't `apt install` these for the user, but it can tell them exactly what to run.

- **NGC authentication.** NIM containers and some model weights require an NVIDIA NGC API key. The installer needs to prompt for it on first use, store it, and validate it before attempting a multi-GB pull that would otherwise fail 20 minutes in with a 401.

- **Failure diagnostics.** Each install type fails differently — a pip install can fail on a missing C library, a container pull can fail on auth or networking, a binary download can fail on architecture mismatch. The installer needs to catch these, identify the root cause where possible, and report it in terms the user can act on. A raw pip traceback or `docker: Error response from daemon` is not sufficient.

- **Download resilience.** Model weights are tens to hundreds of GB. Downloads need: progress reporting with ETA, resumability after interruption (don't re-download 80GB because the last 5GB failed), and pre-flight disk space checks. This is the longest wall-clock step in the entire milestone.

- **aarch64 compatibility.** DGX Spark is ARM. Some pip packages don't publish aarch64 wheels. Some container images are x86-only. These need to be surfaced during `setup`, not discovered during `serve` or `client`. If a tool can't be installed on this architecture, say so explicitly.

**Steps:** 2 (installer + model management from the build sequence), plus preflight checks woven into both

---

## Phase 3: Runtime

*Starting backends, health-checking them, and launching clients.*

**What gets built:**
- `src/backends.py` — starts a serving backend as a subprocess or container, polls its health endpoint until ready, stops it cleanly. Supports `./launcher serve <model> [--backend <name>]` and `./launcher stop`.
- `src/clients.py` — starts a client pointed at the running backend. Supports `./launcher client <name>`.
- `src/status.py` (or folded into backends) — `./launcher status` shows what's running, which model, which backend, health, uptime.

**Testable outcome:** `./launcher serve nemotron-3-super-120b --backend nim` starts the NIM container, waits for the health check to pass, and reports "ready." `./launcher client aider` starts aider pointed at the local API. `./launcher stop` tears everything down with no orphaned processes. `./launcher status` shows the running session.

**Big rocks:**
- **Process lifecycle.** The launcher spawns long-running processes (containers, venvs). It must handle: startup timeouts, health-check polling, clean shutdown (SIGTERM → SIGKILL escalation), and crash detection. If the launcher itself is killed, orphaned processes need to be recoverable on next run.
- **Health-check timing.** Large models take time to load into GPU memory. The health-check loop needs a generous timeout (minutes, not seconds) with clear progress feedback so the user doesn't think it's stuck.
- **Port conflicts.** If something is already on port 8000, the launcher needs to detect it and either fail clearly or pick another port.
- **Clean shutdown.** This is a design principle — no orphaned containers or zombie processes. Needs signal handling, PID tracking, and container cleanup.

**Steps:** 2 (backend lifecycle + client launcher from the build sequence)

---

## Phase 4: Interface

*The FastAPI server, web dashboard, and the end-to-end test.*

**What gets built:**
- `src/api.py` — FastAPI app that exposes all operations as HTTP endpoints and serves the static dashboard. Runs on `localhost:9000`.
- `web/index.html`, `web/style.css`, `web/app.js` — static dashboard with four panels: model library, launch controls, active session, logs.
- End-to-end validation: the full flow from dashboard to running aider session.

**Testable outcome:** Open `localhost:9000`, see downloaded models, click to serve Nemotron via NIM, wait for health check, click to launch aider, use aider. Stop everything from the dashboard. Same flow works entirely from CLI.

**Big rocks:**
- **Live logs.** The dashboard needs to stream backend stdout/stderr in real time. This means Server-Sent Events or WebSocket from FastAPI, which is straightforward but needs to be designed into `backends.py` (capture output, make it streamable).
- **State synchronization.** The CLI and dashboard share the same backend code, but if someone starts a backend from CLI, the dashboard should reflect it. State needs to be authoritative (check the actual running process) rather than cached.
- **Dashboard scope.** This is an operations panel, not a chat UI. Keep it minimal — the risk is over-building the frontend and delaying the end-to-end test.

**Steps:** 3 (FastAPI server + web dashboard + end-to-end test from the build sequence)

---

## Sequencing & Dependencies

```
Phase 1: Foundation          ███░░░░░░░░░░░░░░░░░
Phase 2: Tool & Model Mgmt       ██████░░░░░░░░░░░░░░
Phase 3: Runtime                        █████░░░░░░░░░░░
Phase 4: Interface                            ████████░░
                             ─────────────────────────────
                             Start                    Done
```

Each phase strictly depends on the previous one. Within each phase, some work can be parallelized (e.g., installer and model management are independent once config exists), but the phases themselves are sequential.

**The critical path** runs through model download (Phase 2) and backend startup (Phase 3). These are the steps most likely to surface hardware-specific or dependency-specific issues that force rework.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Nemotron model doesn't run on DGX Spark via NIM | Blocks the entire milestone | Test NIM container manually before building Phase 3. Have vLLM as a fallback backend. |
| aarch64 package incompatibility (pip or container) | Blocks specific tool | Surface in Phase 2. Each tool is independent — a broken tool doesn't block others. |
| NIM container requires specific CUDA/driver version | Blocks NIM backend | Check NVIDIA's compatibility matrix early. DGX Spark has CUDA 13.0 — verify NIM supports it. |
| Model download takes hours and fails partway | Wastes time, blocks Phase 3 | Build resumable downloads into `models.py`. Consider starting the download early while building Phase 3 code. |
| Process lifecycle edge cases (orphans, zombies, port conflicts) | Unreliable runtime | Design PID tracking and cleanup into Phase 3 from the start, not as an afterthought. |

---

## What "Done" Looks Like

Milestone 1 is complete when:

1. `./launcher setup --all` installs NIM and aider from a fresh clone
2. `./launcher models download nemotron-3-super-120b` downloads the model weights
3. `./launcher serve nemotron-3-super-120b` starts NIM and reports healthy
4. `./launcher client aider` starts aider connected to the local model
5. aider can be used for AI-assisted coding (send a prompt, get a response)
6. `./launcher stop` tears everything down cleanly
7. The same workflow works from `localhost:9000` in a browser
8. No orphaned processes after shutdown
