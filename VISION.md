# os8-launcher

**A unified system for running local open-source AI models.**

**GitHub:** [os8ai/os8-launcher](https://github.com/os8ai/os8-launcher) (open source)

os8-launcher manages the full stack of running AI locally: downloading and organizing model weights, starting serving backends that expose an API, and connecting client tools to that API. It runs on a single machine and treats the entire workflow — from raw model weights to usable AI tool — as one integrated system.

os8-launcher works standalone for anyone who wants to run open-source models locally. It also serves as the local model provider for [OS8](https://github.com/os8ai), a personal operating system for AI-assisted development.

---

## The Problem

Running open-source models locally today means juggling disconnected pieces: downloading weights from HuggingFace or NVIDIA NGC, figuring out which serving backend supports which model format, configuring that backend correctly, then separately configuring a client tool to talk to it. Each step has its own docs, its own conventions, and its own failure modes. If you want to try 10 models across 3 backends with 4 client tools, you're managing dozens of configurations by hand.

For OS8 users, the problem extends further: OS8 already connects to Claude, Gemini, ChatGPT, and Grok via their respective CLIs — but there's no path to use local open-source models alongside those cloud providers.

## The Solution

os8-launcher is a single entry point that knows about all three layers — models, serving backends, and client tools — and how they connect. You open `localhost:9000` in a browser and get a dashboard where you can manage models, start serving backends, and connect clients — all from one place.

The same operations are available from the command line for automation and scripting:

```
$ ./launcher serve nemotron-3-super-120b --backend nim
$ ./launcher client aider
$ ./launcher status
```

But day-to-day, the web dashboard is the primary interface.

---

## Architecture

os8-launcher has three data layers and one application layer that ties them together.

### Layer 1: Models

The actual model weights, downloaded and stored in a consistent, predictable structure. Each model gets its own directory with its weights and a metadata file describing where it came from, what format it's in, and how large it is.

- Weights are stored locally under `models/<model-name>/weights/`
- No reliance on scattered cache directories — os8-launcher owns the storage layout
- Models can be downloaded, listed, and removed through os8-launcher
- Disk usage is surfaced clearly so the user can manage their storage budget

### Layer 2: Serving Backends

Software that loads model weights into GPU memory and exposes an OpenAI-compatible HTTP API. Different backends support different model formats and have different performance characteristics.

Initial backends:
- **NVIDIA NIM** — containerized, optimized for NVIDIA hardware and NVIDIA-published models
- **vLLM** — general-purpose, broad HuggingFace model support
- **llama.cpp** — lightweight, good for GGUF-quantized models

Each backend has its own directory under `serving/<backend-name>/` containing configuration templates and startup scripts. All backends are expected to expose an OpenAI-compatible API on a configurable local port.

### Layer 3: Clients

Applications that consume an OpenAI-compatible API to provide a user-facing experience. These range from coding assistants to chat UIs to full application platforms.

Initial clients:
- **aider** — AI pair programming in the terminal
- **Open WebUI** — browser-based chat interface
- **OS8** — personal AI operating system (connects via the OS8 bridge)

Each client has its own directory under `clients/<client-name>/` containing configuration and startup scripts. All clients are configured to point at `localhost:<port>` where the serving backend is running.

### Application Layer: The Launcher Itself

The core application code that orchestrates everything. This is the software that:

1. **Reads the configuration** — knows which models are available, which backends can serve them, and which clients exist
2. **Provides an interactive chooser** — lets the user select a model, a compatible backend, and a client
3. **Manages lifecycle** — starts the backend, waits for the API to be healthy, starts the client, and tears everything down cleanly on exit
4. **Enforces compatibility** — only offers valid combinations (e.g., a NIM-only model won't show llama.cpp as a backend option)

---

## OS8 Bridge

os8-launcher can run standalone, but it also acts as the local model provider for OS8. The bridge between the two systems is the OpenAI-compatible API that os8-launcher's serving backends expose.

### How it works

OS8 currently connects to cloud AI providers (Anthropic, Google, OpenAI, xAI) through CLI-based backend adapters. Each adapter implements a common interface: build args, spawn a CLI process, parse the response. The provider's models are registered in OS8's database as providers, containers, model families, and individual models.

os8-launcher fits into this existing architecture:

1. **os8-launcher serves a model** — starts a serving backend that exposes an OpenAI-compatible API on `localhost:PORT`
2. **OS8 gets a new backend adapter** — a `launcher` entry in OS8's `backend-adapter.js` that calls the local API instead of spawning a CLI
3. **Local models register as a provider** — os8-launcher's models appear in OS8's `ai_providers` / `ai_containers` / `ai_models` tables under a `local` provider
4. **OS8's routing system includes local models** — tasks can be routed to local models just like they're routed to Claude or Gemini, with the same cascade and fallback logic

### Terminology mapping

The two projects use similar but distinct terminology:

| Concept | In os8-launcher | In OS8 |
|---------|-----------------|--------|
| A tool that serves/wraps a provider | Serving backend | Container |
| A tool that consumes the API | Client | (the app itself) |
| os8-launcher as a whole | The application | A container for the `local` provider |

OS8 is a client of os8-launcher. os8-launcher is a container in OS8's world. This is not a conflict — it's two perspectives on the same connection.

### What the bridge requires

- **In os8-launcher:** An API status endpoint so OS8 can check what model is currently being served and whether it's healthy
- **In OS8:** A new backend adapter, a `local` provider in the database, and routing cascade entries for local models

The bridge is a later milestone. os8-launcher works fully standalone first.

### Key OS8 integration points

OS8 lives at `/home/leo/Claude/os8`. The files a future agent will need to touch for the bridge:

- `src/services/backend-adapter.js` — add a `launcher` backend that calls the local HTTP API instead of spawning a CLI
- `src/services/ai-registry.js` — read-only queries for providers, containers, models (no changes needed, just understand the interface)
- `src/services/routing.js` — add local models to routing cascades so tasks can be routed to them
- `src/services/model-discovery.js` — optionally add a discovery config that queries os8-launcher's status API for available models
- `src/db/seeds.js` — add a `local` provider, a `launcher` container, and model families for local models
- `src/db/schema.js` — no changes expected (the existing `ai_providers` / `ai_containers` / `ai_model_families` / `ai_models` tables are sufficient)

---

## Dependency Management

os8-launcher orchestrates third-party tools (aider, vLLM, NIM, Open WebUI, llama.cpp) without embedding them. Each tool is installed and managed by the launcher in an isolated environment. From the user's perspective it feels like one package. Under the hood, each tool is independently installed, isolated, and updatable.

### Installation strategies

Different tools ship differently, so os8-launcher handles three installation types:

- **Docker containers** — for tools that ship as images (NVIDIA NIM, Open WebUI). Launcher pulls and manages images. Updates are `docker pull`.
- **Python venvs** — for Python tools (aider, vLLM). Each gets its own virtual environment so their dependencies can't conflict with each other or with os8-launcher itself. Updates are `pip install --upgrade` inside the venv.
- **Binaries** — for tools that ship as compiled releases (llama.cpp). Launcher downloads the release binary. Updates are downloading the new release.

### Manifests

Each backend and client has a `manifest.yaml` that tells the launcher how to install, run, and update it:

```yaml
# clients/aider/manifest.yaml
name: aider
install_type: pip
package: aider-chat
venv: clients/aider/venv
run: aider --openai-api-base http://localhost:{port}/v1
update: pip install --upgrade aider-chat
```

```yaml
# serving/nim/manifest.yaml
name: nim
install_type: container
image: nvcr.io/nim/meta/llama-3.1-8b-instruct:latest
port: 8000
run: docker run --gpus all -p {port}:8000 -v {model_path}:/models {image}
update: docker pull {image}
```

```yaml
# serving/llamacpp/manifest.yaml
name: llamacpp
install_type: binary
source: https://github.com/ggerganov/llama.cpp/releases
binary: clients/llamacpp/bin/llama-server
run: "{binary} -m {model_path} --port {port}"
update: download latest release
```

The manifest is the contract between os8-launcher and the tool. The launcher doesn't need to know how aider works internally — it just needs to know how to install it, run it, and update it.

### User experience

```
$ ./launcher setup aider          # creates venv, installs aider-chat
$ ./launcher setup nim            # pulls the NIM container image
$ ./launcher update aider         # upgrades aider in its venv
$ ./launcher update --all         # upgrades everything
```

Or from the web dashboard: a setup/update button next to each tool.

### What this means for the project

os8-launcher does not contain the source code of any tool it orchestrates. It contains:
- **Manifests** — how to install, run, and update each tool
- **Configuration** — how to connect each tool to a serving backend
- **Isolation** — each tool's runtime environment (venv, container, binary dir)

The tools' actual files (venvs, binaries, Docker images) are gitignored. A fresh clone has just the manifests. `./launcher setup --all` installs everything.

---

## Configuration

A single YAML file (`config.yaml`) serves as the source of truth for what is installed and how things connect. The manifests (above) define how to install and run each tool. `config.yaml` defines which models, backends, and clients are available and how they relate.

```yaml
models:
  nemotron-3-super-120b:
    source: nvidia/Nemotron-3-Super-120B-A12B-NVFP4
    path: models/nemotron-3-super-120b/weights
    format: nvfp4
    backends: [nim, vllm]
    default_backend: nim
    downloaded: false

backends:
  nim:
    manifest: serving/nim/manifest.yaml
    port: 8000
  vllm:
    manifest: serving/vllm/manifest.yaml
    port: 8000
  llamacpp:
    manifest: serving/llamacpp/manifest.yaml
    port: 8080

clients:
  aider:
    manifest: clients/aider/manifest.yaml
  open-webui:
    manifest: clients/open-webui/manifest.yaml
    port: 3000
  os8:
    type: bridge
    port: 8000
```

As the project grows, this file may split into separate files per section, but starts as one file for simplicity.

---

## Project Structure

```
os8-launcher/
├── README.md              # Setup instructions and quickstart
├── VISION.md              # This document
├── config.yaml            # Source of truth: models, backends, clients
├── launcher               # CLI entry point
├── src/                   # Application code
│   ├── api.py             # FastAPI app — serves dashboard and management API
│   ├── backends.py        # Backend start/stop/health-check
│   ├── clients.py         # Client start/configure
│   ├── models.py          # Download/list/remove models
│   ├── installer.py       # Setup/update tools from manifests
│   └── config.py          # Parse and validate config.yaml
├── web/                   # Dashboard frontend (static files)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/                # Model weight storage (gitignored)
│   ├── nemotron-3-super-120b/
│   │   └── weights/
│   └── .../
├── serving/               # Per-backend manifests, configs, and runtime
│   ├── nim/
│   │   └── manifest.yaml
│   ├── vllm/
│   │   ├── manifest.yaml
│   │   └── venv/          # (gitignored) isolated Python environment
│   └── llamacpp/
│       ├── manifest.yaml
│       └── bin/           # (gitignored) downloaded binary
└── clients/               # Per-client manifests, configs, and runtime
    ├── aider/
    │   ├── manifest.yaml
    │   └── venv/          # (gitignored) isolated Python environment
    ├── open-webui/
    │   └── manifest.yaml
    └── os8/               # Bridge configuration for OS8 integration
        └── manifest.yaml
```

The root is clean. Application code lives in `src/`. Data (model weights) lives in `models/`. Per-backend and per-client directories contain manifests (checked in) and runtime environments (gitignored). A fresh clone is lightweight — `./launcher setup --all` installs everything.

---

## Target Hardware

os8-launcher is being built for and initially tested on an NVIDIA DGX Spark:

- **GPU:** NVIDIA GB10 (Grace Blackwell), CUDA 13.0
- **Memory:** 128 GB unified (shared CPU/GPU)
- **Storage:** 3.7 TB NVMe SSD
- **OS:** Ubuntu 24.04 (aarch64)

This hardware can comfortably run large models (70B+ parameters in quantized formats). The unified memory architecture is particularly well-suited for large model inference since model weights don't need to transfer across a PCIe bus.

os8-launcher should work on any Linux machine with an NVIDIA GPU and sufficient memory, but the DGX Spark is the primary development and testing target.

---

## Interface

os8-launcher has two interfaces: a web dashboard (primary) and a CLI (engine).

### Web Dashboard

A local web application served on `localhost:9000`. This is what you open in a browser to use os8-launcher day-to-day. The pattern is the same as Docker Desktop over Docker CLI, or Cockpit over systemctl — the CLI is the engine, the web UI is the control panel.

The dashboard shows:
- **Model library** — what's downloaded, what's available, disk usage per model
- **Launch controls** — select a model, pick a compatible backend, choose a client, start/stop
- **Active session** — which model is running on which backend, health status, uptime
- **Logs** — live output from the serving backend (useful for debugging and monitoring)

Tech stack:
- **FastAPI** for the local API server — lightweight, async, auto-generates API docs at `/docs`
- **Static HTML/JS/CSS** for the frontend — no build step, no React, no node_modules. Vanilla JS with a clean CSS framework. The dashboard is an operations panel, not a rich application — it doesn't need a framework.
- The FastAPI server serves both the dashboard UI and the management API that the dashboard calls

The dashboard is not a chat interface. OS8 already has the rich agent/chat/app experience. The launcher dashboard is for operations: seeing what you have, starting what you need, monitoring what's running.

### CLI

The same Python code that powers the dashboard is also available as a command-line tool. The CLI is useful for:
- **Automation** — scripting model downloads, starting backends on boot
- **Headless operation** — running on a server without a browser
- **OS8 bridge** — OS8 talks to the launcher's API, not its UI

```
$ ./launcher models list                     # show downloaded models
$ ./launcher models download nemotron-3      # download a model
$ ./launcher serve nemotron-3 --backend nim  # start serving
$ ./launcher status                          # what's running
$ ./launcher stop                            # shut everything down
```

The CLI and web dashboard share all code — they are two views of the same system, not separate implementations.

---

## Implementation Language

Python. It's the natural fit for the ML ecosystem — most backends have Python APIs or Python-based tooling. FastAPI provides the web server. The CLI uses the same Python modules directly.

---

## Design Principles

- **One command to go from zero to working.** Downloading a model, starting a backend, and connecting a client should be a single workflow, not three separate tasks.
- **Explicit over magical.** Weights are stored where you can see them. Configuration is in one readable YAML file. No hidden state scattered across the filesystem.
- **Valid combinations only.** The system knows which backends support which models. The user never has to guess or debug an incompatible pairing.
- **Clean shutdown.** When the user is done, everything stops. No orphaned containers or zombie processes.
- **Standalone first, bridge second.** os8-launcher is useful on its own. The OS8 integration builds on top of a working standalone tool.
- **Start simple, grow naturally.** Begin with one model, one backend, one client. Adding more is just configuration — the system is the same regardless of scale.

---

## Scope Boundaries

**os8-launcher is:**
- A local-first tool for a single machine
- An orchestrator that connects existing tools (not a replacement for any of them)
- A local web dashboard backed by a CLI engine
- The local model provider for OS8 (optional — works standalone)

**os8-launcher is not:**
- A model training or fine-tuning framework
- A cloud deployment tool
- A replacement for vLLM, NIM, llama.cpp, aider, or any tool it orchestrates
- A fork or component of OS8 — it's a separate project that OS8 can connect to

---

## Milestones

### Milestone 1: Standalone

A working end-to-end flow:

1. Download Nemotron-3-Super-120B-A12B-NVFP4
2. Serve it via NVIDIA NIM
3. Connect aider to it
4. Use aider for AI-assisted coding powered by a fully local model

**Build sequence:**

1. **Project skeleton** — directory structure, `config.yaml`, `launcher` CLI entry point, Python package setup
2. **Config parser** — `src/config.py` reads and validates `config.yaml`, resolves manifests
3. **Installer** — `src/installer.py` reads manifests, creates venvs, pulls containers, downloads binaries (`./launcher setup <tool>`)
4. **Model management** — `src/models.py` downloads weights, tracks download status, reports disk usage (`./launcher models download <model>`)
5. **Backend lifecycle** — `src/backends.py` starts/stops serving backends, health-checks the API (`./launcher serve <model>`)
6. **Client launcher** — `src/clients.py` starts clients pointed at the running backend (`./launcher client <client>`)
7. **FastAPI server** — `src/api.py` exposes all of the above as HTTP endpoints, serves the dashboard
8. **Web dashboard** — `web/` static frontend that calls the API: model library, launch controls, active session, logs
9. **End-to-end test** — download Nemotron, serve via NIM, launch aider, verify it works

### Milestone 2: OS8 Bridge

1. os8-launcher exposes a status/health API alongside the model API
2. OS8 gets a `launcher` backend adapter that talks to the local API
3. Local models appear in OS8's provider registry
4. OS8's routing system can send tasks to local models

Everything after that — more models, more backends, more clients — is additive.
