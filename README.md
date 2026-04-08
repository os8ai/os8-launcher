# os8-launcher

A unified system for running local open-source AI models. See `VISION.md` for the full spec.

## Install

```bash
git clone git@github.com:os8ai/os8-launcher.git
cd os8-launcher
./install
```

`./install` symlinks `os8-launcher` into `~/.local/bin` so you can run it from any directory. It's idempotent — safe to re-run after pulling updates. If `~/.local/bin` isn't on your `PATH`, the script tells you the one line to add to `~/.bashrc`; it never edits your dotfiles for you.

To remove: `./uninstall` (only removes the symlink if it points back at this repo).

## Usage

```bash
os8-launcher
```

This brings up the web dashboard at <http://localhost:9000>. Behavior:

- If the dashboard is already running, prints the URL and exits.
- If something else is bound to port 9000, shows the holding PID and asks before killing it.
- Otherwise starts the dashboard in the background (detached from your terminal, so closing the shell won't kill it). Logs go to `var/dashboard.log`.

You can also run `./start` from inside the repo directory — `os8-launcher` is just a symlink to it.
