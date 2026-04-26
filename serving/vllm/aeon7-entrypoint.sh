#!/bin/bash
# Pre-start hook for AEON-7 Gemma-4-26B-A4B-it-Uncensored-NVFP4.
#
# vLLM's stock model_executor/models/gemma4.py weight loader doesn't know
# about the per-expert NVFP4 input_global_scale tensors (KeyError on
# `layers.0.experts.0.down_proj.input_global_scale`). The model author
# ships a patched gemma4.py inside the weights repo as gemma4_patched.py.
# We copy it over the upstream file before invoking vllm. Once the upstream
# vLLM Gemma 4 loader handles these tensors natively this whole script and
# Dockerfile.gemma4-aeon can be deleted; the model entry can drop its
# vllm_image override and use the default os8-vllm:gemma4 image.
set -euo pipefail

PATCHED_SRC="/model/gemma4_patched.py"
TARGET="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma4.py"

if [ -f "$PATCHED_SRC" ]; then
  echo "[aeon7-entrypoint] Applying gemma4_patched.py over $TARGET"
  cp "$PATCHED_SRC" "$TARGET"
else
  echo "[aeon7-entrypoint] WARNING: $PATCHED_SRC not present in mounted weights — proceeding without patch"
fi

exec vllm serve "$@"
