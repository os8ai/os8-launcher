"""Authoritative model-size resolution via the HuggingFace Hub API.

The launcher declares a coarse `size_gb` hint in config.yaml, but that's
hand-typed and drifts from reality (see fastwan22-ti2v-5b: declared 14,
actual 23).  For `hf-snapshot` downloads we can do better — HF's
model_info endpoint returns per-file sizes, which we filter by
allow_patterns and sum to get the exact expected footprint.

Any failure (network, rate limit, auth) returns None so the caller
falls back to the declared `size_gb`.  This module is best-effort; it
must never block the dashboard or the download path.
"""

from __future__ import annotations

import fnmatch
import threading

from src.config import ModelConfig


# (repo_id, tuple(patterns)) -> total bytes.  Populated on first call
# per process; invalidated only by restart.  Repo sizes change rarely
# enough that a longer TTL is unnecessary.
_cache: dict[tuple, int] = {}
_lock = threading.Lock()


def _sources(model: ModelConfig) -> list[tuple[str, tuple[str, ...] | None]]:
    out: list[tuple[str, tuple[str, ...] | None]] = [
        (model.source, tuple(model.allow_patterns) if model.allow_patterns else None)
    ]
    for extra in model.extra_sources:
        src = extra.get("source")
        if not src:
            continue
        patterns = extra.get("allow_patterns")
        out.append((src, tuple(patterns) if patterns else None))
    return out


def _resolve_one(
    repo_id: str,
    patterns: tuple[str, ...] | None,
    token: str | None,
) -> int | None:
    key = (repo_id, patterns)
    with _lock:
        if key in _cache:
            return _cache[key]

    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except Exception:
        return None

    total = 0
    for sib in getattr(info, "siblings", None) or []:
        name = getattr(sib, "rfilename", "") or ""
        size = getattr(sib, "size", None) or 0
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        total += int(size)

    with _lock:
        _cache[key] = total
    return total


def resolve_model_expected_bytes(
    model: ModelConfig,
    token: str | None = None,
) -> int | None:
    """Return the sum of expected download bytes across all of the model's
    sources, filtered by each source's allow_patterns.  Returns None if
    any source cannot be resolved — callers should fall back to
    ``model.size_gb`` in that case.
    """
    # hf_sizes is only meaningful for HF-backed models.  Non-HF sources
    # (ollama://…) are skipped here so get_models_data falls through to
    # the declared size_gb for ollama-managed weights.
    if not model.source or "://" in model.source:
        return None

    total = 0
    for source, patterns in _sources(model):
        sz = _resolve_one(source, patterns, token)
        if sz is None:
            return None
        total += sz
    return total or None
