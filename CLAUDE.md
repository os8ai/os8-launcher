# os8-launcher

A unified system for running local open-source AI models. The user picks a combination of **project folder + model + serving backend + client** and the launcher wires them together — downloading weights if needed, starting the backend, and launching the client pointed at it. See `VISION.md` for the full architecture and `PLAN.md` for the roadmap.

## Project Info

- **Repo:** git@github.com:os8ai/os8-launcher.git (open source)
- **Git identity (this repo):** OS8 / leo@os8.ai
- **Language:** Python
- **Web framework:** FastAPI (backend) + static HTML/JS/CSS (frontend, **no build step — don't introduce npm/React/bundlers**)
- **Dashboard URL:** `localhost:9000`

## Entry points

- `./launcher` — CLI (the engine; all functionality is reachable here)
- `./start` — boots the dashboard
- `./install` / `./uninstall` — system install/removal
- `var/` — runtime state, logs, pids (gitignored; check here when debugging a running session)

## Related Projects

- **OS8** — the personal AI operating system that os8-launcher will integrate with as a local model provider (Milestone 2). OS8 is an Electron + Express app with its own backend-adapter system for Claude, Gemini, Codex, and Grok. See [github.com/os8ai](https://github.com/os8ai).

## Architecture (quick reference)

Three layers + one application layer:

1. **Models** — weights stored in `models/<name>/weights/`
2. **Serving backends** — NIM, vLLM, llama.cpp — each in `serving/<name>/` with a `manifest.yaml`
3. **Clients** — aider, Open WebUI, OS8 — each in `clients/<name>/` with a `manifest.yaml`
4. **Launcher** — Python code in `src/` that orchestrates all three, exposed via CLI (`./launcher`) and web dashboard (`localhost:9000`)

Dependencies (aider, vLLM, etc.) are not embedded. Each is installed in isolation (venvs, containers, or binaries) managed via manifests. Runtime environments (`models/`, venvs, downloaded binaries) are gitignored; manifests are checked in — a missing directory on a fresh clone is expected, not a bug.

**The manifest contract:** to add a new backend or client, drop a directory under `serving/` or `clients/` containing a `manifest.yaml`. In most cases no Python changes are needed — the launcher reads the manifest to know how to install, run, and update the tool.

**Cleanup discipline:** always tear down ad-hoc Docker test containers when done. On the Spark's unified memory, orphaned containers silently break subsequent serves.

## Design rules

- **Explicit over magical.** Weights live where the user can see them, config is one readable YAML file, no hidden caches or autodetection. Resist "smart" behavior.
- **Valid combinations only.** The launcher should never offer a model+backend+client pairing it knows won't work. New models/backends must declare their compatibility.
- **Clean shutdown, always.** When a session ends, every process, container, and port it owned must be released. No orphans.
- **CLI and dashboard share code.** New functionality goes into `src/` modules; both `api.py` and the CLI call into them. Never put logic directly in a CLI command or an API route.
- **Dashboard is operations, not chat.** It shows what's installed, what's running, and lets the user start/stop things. OS8 owns the chat/agent experience — don't add one here.

## Scope

os8-launcher orchestrates existing tools; it never reimplements them, and it never trains, fine-tunes, or deploys to cloud. It is standalone-first — don't couple launcher code to OS8 internals or assume OS8 is running.

## Hardware

NVIDIA DGX Spark — GB10 GPU, 128GB unified memory, 3.7TB NVMe, Ubuntu 24.04 aarch64, CUDA 13.0.

## Current State

Milestone 1 (standalone launcher) shipped. Active work is iterating on backends, clients, and project-folder UX. Use `git log` for what changed recently rather than trusting a hand-maintained status list.
