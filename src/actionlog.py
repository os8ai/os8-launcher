"""Uniform action-log helpers for lifecycle functions.

All start_*/stop_* functions in backends.py and clients.py bracket their work
with these so the dashboard log shows a consistent shape for every action:

    > Starting <kind> <name>...
      ...per-action detail lines...
    OK <kind> <name> ready.        (success)
    FAIL <kind> <name>: <reason>   (failure)

The dashboard captures stdout via _run_with_log_capture in api.py, so any
print() inside a lifecycle function lands in /api/logs automatically. Keep
all action-log phrasing in this one file so the format is changed in one
place, not sprinkled across the codebase.
"""


def log_start(kind: str, name: str) -> None:
    print(f"> Starting {kind} {name}...")


def log_ready(kind: str, name: str) -> None:
    print(f"OK {kind} {name} ready.")


def log_stopped(kind: str, name: str) -> None:
    print(f"OK {kind} {name} stopped.")


def log_fail(kind: str, name: str, err: object) -> None:
    print(f"FAIL {kind} {name}: {err}")


def log_group_start(label: str) -> None:
    print(f"> {label}...")


def log_group_done(label: str) -> None:
    print(f"OK {label}.")
