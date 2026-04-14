# os8-launcher

A unified system for running local open-source AI models, primarily designed and tested on the NVIDIA DGX Spark (Linux arm64). Pick a model, a serving backend, and a client — the launcher wires them together, downloading weights if needed, starting the backend, and launching the client pointed at it.

See [`VISION.md`](VISION.md) for the full architecture.

## Prerequisites

- **Linux** (Ubuntu 22.04+ recommended)
- **NVIDIA GPU** with drivers installed
- **CUDA 12.0+**
- **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **Python 3.10+**

## Supported architectures

- **x86_64** (amd64) — standard desktops and servers
- **aarch64** (arm64) — NVIDIA DGX Spark, Jetson, Grace Hopper

## Install

```bash
git clone https://github.com/os8ai/os8-launcher.git
cd os8-launcher
./install
```

`./install` symlinks `os8-launcher` into `~/.local/bin` so you can run it from any directory. It's idempotent — safe to re-run after pulling updates. If `~/.local/bin` isn't on your `PATH`, the script tells you the one line to add to `~/.bashrc`; it never edits your dotfiles for you.

To remove: `./uninstall` (only removes the symlink if it points back at this repo).

## Quick start

```bash
# Verify your system meets the prerequisites
os8-launcher doctor

# Install backends and download a model
os8-launcher setup --all
os8-launcher download gemma-4-E2B-it

# Start serving and open the dashboard
os8-launcher
```

The dashboard runs at <http://localhost:9000>. From there you can pick a model, start a backend, and connect a client.

## Usage

```bash
os8-launcher             # Open the web dashboard
os8-launcher doctor      # Run system diagnostics
os8-launcher setup       # Install backends and clients
os8-launcher serve       # Start a model backend from the CLI
os8-launcher stop        # Stop the running backend
os8-launcher status      # Show what's running
```

Run `os8-launcher --help` for the full command list.

## Customizing with Claude Code

os8-launcher is designed to be extended by AI. We recommend running [Claude Code](https://claude.ai/claude-code) in the repo to help you add new models, serving backends, and clients, or to customize the experience for your hardware and workflow. The manifest-driven architecture and the `CLAUDE.md` file give Claude full context on how the project fits together — just describe what you want and it can write the manifest, update `config.yaml`, and wire everything in.

## How it works

The launcher is manifest-driven. Each serving backend (`serving/<name>/manifest.yaml`) and client (`clients/<name>/manifest.yaml`) declares how to install, run, and update itself. To add a new backend or client, drop a directory with a `manifest.yaml` — in most cases no Python changes are needed.

Models are configured in `config.yaml` with their source, format, compatible backends, and download size.

## License

MIT
