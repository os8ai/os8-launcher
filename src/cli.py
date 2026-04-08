"""os8-launcher CLI — command-line interface for managing local AI models."""

import argparse
import os
import sys
from pathlib import Path

from src.config import ConfigError, load_config, format_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="launcher",
        description="os8-launcher — manage local open-source AI models",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- config ---
    config_parser = subparsers.add_parser("config", help="View configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_show = config_sub.add_parser("show", help="Print the current configuration")
    config_show.set_defaults(handler="config_show")
    config_parser.set_defaults(handler="config_help", _parser=config_parser)

    # --- setup ---
    setup_parser = subparsers.add_parser("setup", help="Install backends and clients")
    setup_parser.add_argument("tool", nargs="?", help="Tool to install (or --all)")
    setup_parser.add_argument("--all", action="store_true", help="Install everything")
    setup_parser.set_defaults(handler="setup")

    # --- update ---
    update_parser = subparsers.add_parser("update", help="Update backends and clients")
    update_parser.add_argument("tool", nargs="?", help="Tool to update (or --all)")
    update_parser.add_argument("--all", action="store_true", help="Update everything")
    update_parser.set_defaults(handler="update")

    # --- models ---
    models_parser = subparsers.add_parser("models", help="Manage model weights")
    models_sub = models_parser.add_subparsers(dest="models_command")

    models_list = models_sub.add_parser("list", help="List available models")
    models_list.set_defaults(handler="models_list")

    models_download = models_sub.add_parser("download", help="Download model weights")
    models_download.add_argument("model", help="Model to download")
    models_download.add_argument("--backend", help="Backend to download for (determines download strategy)")
    models_download.set_defaults(handler="models_download")

    models_remove = models_sub.add_parser("remove", help="Remove model weights")
    models_remove.add_argument("model", help="Model to remove")
    models_remove.set_defaults(handler="models_remove")

    models_parser.set_defaults(handler="models_help", _parser=models_parser)

    # --- serve ---
    serve_parser = subparsers.add_parser("serve", help="Start a serving backend")
    serve_parser.add_argument("model", help="Model to serve")
    serve_parser.add_argument("--backend", help="Backend to use (default: model's default)")
    serve_parser.set_defaults(handler="serve")

    # --- client ---
    client_parser = subparsers.add_parser("client", help="Start a client")
    client_parser.add_argument("name", help="Client to start")
    client_parser.add_argument("--model", help="Model to auto-start a backend for if none is running")
    client_parser.add_argument("--backend", help="Backend to use when auto-starting (default: model's default)")
    client_parser.set_defaults(handler="client")

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Show what's running")
    status_parser.set_defaults(handler="status")

    # --- stop ---
    stop_parser = subparsers.add_parser("stop", help="Stop all running services")
    stop_parser.add_argument("--client", help="Stop only this client")
    stop_parser.set_defaults(handler="stop")

    # --- project ---
    project_parser = subparsers.add_parser("project", help="Manage project folders")
    project_sub = project_parser.add_subparsers(dest="project_command")

    p_list = project_sub.add_parser("list", help="List projects")
    p_list.set_defaults(handler="project_list")

    p_new = project_sub.add_parser("new", help="Create a new project")
    p_new.add_argument("name")
    p_new.add_argument("--description", default=None)
    p_new.set_defaults(handler="project_new")

    p_use = project_sub.add_parser("use", help="Set the active project")
    p_use.add_argument("name")
    p_use.set_defaults(handler="project_use")

    p_clear = project_sub.add_parser("clear", help="Clear the active project")
    p_clear.set_defaults(handler="project_clear")

    p_show = project_sub.add_parser("show", help="Show the active project")
    p_show.set_defaults(handler="project_show")

    project_parser.set_defaults(handler="project_help", _parser=project_parser)

    # --- server ---
    server_parser = subparsers.add_parser("server", help="Start the web dashboard")
    server_parser.add_argument("--port", type=int, default=9000, help="Port for the dashboard (default: 9000)")
    server_parser.set_defaults(handler="server")

    return parser


def main(repo_root: Path):
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = args.handler

    # --- help handlers for subcommand groups ---
    if handler == "config_help":
        args._parser.print_help()
        return
    if handler == "models_help":
        args._parser.print_help()
        return
    if handler == "project_help":
        args._parser.print_help()
        return

    # --- project commands ---
    if handler and handler.startswith("project_"):
        from src.projects import (
            ProjectError, list_projects, create_project,
            set_active_project, clear_active_project, get_active_project,
            projects_dir,
        )
        try:
            if handler == "project_list":
                active = get_active_project()
                active_name = active.name if active else None
                projects = list_projects()
                if not projects:
                    print(f"No projects yet under {projects_dir()}.")
                    print("Create one with: ./launcher project new <name>")
                    return
                print(f"Projects (in {projects_dir()}):")
                for p in projects:
                    marker = "* " if p.name == active_name else "  "
                    print(f"{marker}{p.name}")
                return
            if handler == "project_new":
                p = create_project(args.name, description=args.description)
                set_active_project(p.name)
                print(f"Created and activated project: {p.name}")
                print(f"  Path: {p.path}")
                return
            if handler == "project_use":
                p = set_active_project(args.name)
                print(f"Active project: {p.name} ({p.path})")
                return
            if handler == "project_clear":
                clear_active_project()
                print("Active project cleared.")
                return
            if handler == "project_show":
                p = get_active_project()
                if not p:
                    print("No active project. Set one with: ./launcher project use <name>")
                    return
                print(f"Active project: {p.name}")
                print(f"  Path: {p.path}")
                if p.description:
                    print(f"  Description: {p.description}")
                return
        except ProjectError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # --- config show ---
    if handler == "config_show":
        try:
            config = load_config(repo_root)
            print(format_config(config))
        except ConfigError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- setup ---
    if handler == "setup":
        from src.installer import setup_tool, setup_all, InstallError
        try:
            config = load_config(repo_root)
            if getattr(args, "all", False):
                setup_all(config, repo_root)
            elif args.tool:
                setup_tool(args.tool, config, repo_root)
            else:
                print("Specify a tool to install, or use --all.")
                print("Available tools:", ", ".join(
                    list(config.backends.keys())
                    + [n for n, c in config.clients.items() if c.type != "bridge"]
                ))
                sys.exit(1)
        except (ConfigError, InstallError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- update ---
    if handler == "update":
        from src.installer import update_tool, update_all, InstallError
        try:
            config = load_config(repo_root)
            if getattr(args, "all", False):
                update_all(config, repo_root)
            elif args.tool:
                update_tool(args.tool, config, repo_root)
            else:
                print("Specify a tool to update, or use --all.")
                sys.exit(1)
        except (ConfigError, InstallError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- models list ---
    if handler == "models_list":
        from src.models import list_models
        try:
            config = load_config(repo_root)
            list_models(config, repo_root)
        except ConfigError as e:
            print(f"Configuration error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- models download ---
    if handler == "models_download":
        from src.models import download_model, ModelError
        try:
            config = load_config(repo_root)
            download_model(args.model, config, repo_root, backend=args.backend)
        except (ConfigError, ModelError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- models remove ---
    if handler == "models_remove":
        from src.models import remove_model, ModelError
        try:
            config = load_config(repo_root)
            remove_model(args.model, config, repo_root)
        except (ConfigError, ModelError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- serve ---
    if handler == "serve":
        from src.backends import start_backend, BackendError
        try:
            config = load_config(repo_root)
            start_backend(args.model, getattr(args, "backend", None), config, repo_root)
        except (ConfigError, BackendError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\nAborted.")
            else:
                print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- client ---
    if handler == "client":
        from src.clients import start_client, ClientError
        try:
            config = load_config(repo_root)
            start_client(
                args.name, config, repo_root,
                model=getattr(args, "model", None),
                backend_name=getattr(args, "backend", None),
            )
        except (ConfigError, ClientError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # --- status ---
    if handler == "status":
        from src.backends import get_status
        print(get_status())
        return

    # --- stop ---
    if handler == "stop":
        from src.backends import stop_all, stop_backend
        from src.clients import stop_client
        client_name = getattr(args, "client", None)
        if client_name:
            stop_client(client_name)
        else:
            stop_all()
        return

    # --- server ---
    if handler == "server":
        import uvicorn
        from src.api import app
        # Crash-recovery safety net: clean up any state left over from a
        # previous launcher run that didn't shut down cleanly. Normal
        # shutdowns/restarts already tear things down before exit, so this
        # is usually a no-op.
        from src.backends import stop_all
        try:
            stop_all()
        except Exception as e:
            print(f"  (cleanup warning: {e})")
        print(f"Starting os8-launcher dashboard on http://localhost:{args.port}")
        app.state.restart_requested = False
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        app.state.server = server
        server.run()  # blocks until should_exit is set
        if app.state.restart_requested:
            # Re-exec the whole process for a clean restart. uvicorn.Server
            # cannot be reliably re-run inside the same Python process —
            # leftover asyncio loop state, signal handlers, and background
            # tasks corrupt the next run(). A fresh interpreter sidesteps
            # all of that, and the browser's /api/health poll picks us back
            # up automatically once we bind the port again.
            print("Restarting dashboard...")
            launcher_path = str(repo_root / "launcher")
            os.execv(launcher_path, [launcher_path, *sys.argv[1:]])
            # execv replaces the process — never returns
        print("Dashboard stopped.")
        return
