# os8-launcher

A unified system for running local open-source AI models. See `VISION.md` for the full spec.

## Project Info

- **Repo:** git@github.com:os8ai/os8-launcher.git (private)
- **Local path:** `/home/leo/Claude/launcher`
- **Git identity (this repo):** OS8 / leo@os8.ai
- **Language:** Python
- **Web framework:** FastAPI (backend) + static HTML/JS/CSS (frontend, no build step)
- **Dashboard URL:** `localhost:9000`

## Related Projects

- **OS8** — `/home/leo/Claude/os8` — the personal AI operating system that os8-launcher will integrate with as a local model provider (Milestone 2). OS8 is an Electron + Express app with its own backend-adapter system for Claude, Gemini, Codex, and Grok.

## Architecture (quick reference)

Three layers + one application layer:

1. **Models** — weights stored in `models/<name>/weights/`
2. **Serving backends** — NIM, vLLM, llama.cpp — each in `serving/<name>/` with a `manifest.yaml`
3. **Clients** — aider, Open WebUI, OS8 — each in `clients/<name>/` with a `manifest.yaml`
4. **Launcher** — Python code in `src/` that orchestrates all three, exposed via CLI (`./launcher`) and web dashboard (`localhost:9000`)

Dependencies (aider, vLLM, etc.) are not embedded. Each is installed in isolation (venvs, containers, or binaries) managed via manifests. Runtime environments are gitignored; manifests are checked in.

## Hardware

NVIDIA DGX Spark — GB10 GPU, 128GB unified memory, 3.7TB NVMe, Ubuntu 24.04 aarch64, CUDA 13.0.

## Current State

- Phase 1 (Foundation) is complete — skeleton, config parser, CLI with stubs
- Phase 2 (Tool & Model Management) is complete — installer, preflight checks, credentials, model management
- Next step: Phase 3 (Runtime — backend lifecycle, client launcher, process management)

## Build Sequence (Milestone 1)

1. Project skeleton — dirs, config.yaml, launcher entry point, Python setup
2. Config parser — src/config.py
3. Installer — src/installer.py (setup/update tools from manifests)
4. Model management — src/models.py (download, track, disk usage)
5. Backend lifecycle — src/backends.py (start/stop/health-check)
6. Client launcher — src/clients.py
7. FastAPI server — src/api.py
8. Web dashboard — web/ static frontend
9. End-to-end test — Nemotron + NIM + aider
