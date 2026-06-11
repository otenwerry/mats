#!/bin/bash
# One-shot bootstrap for a fresh Lambda (or any Ubuntu 22.04 + NVIDIA-driver)
# GPU box, to run the real PostTrainBench rollback rollouts. Idempotent-ish:
# skips the apptainer install and the .sif build if their outputs already
# exist on the persistent filesystem (so re-running after a terminate/relaunch
# that re-attached the filesystem is cheap).
#
# Assumes:
#   - a persistent filesystem mounted at $FS (Lambda: /lambda/nfs/<name>);
#   - the host already has the NVIDIA driver (Lambda Stack / GPU Base images do);
#   - this script + standard.def are reachable (scp'd by the operator).
#
# Usage (on the box):
#   FS=/lambda/nfs/my-filesystem bash bootstrap_box.sh
#
# Verified live 2026-06-09: apptainer 1.5.1 from ppa:apptainer/ppa installs
# clean on Lambda Stack 22.04; H100 PCIe 80GB, driver 580. The .def build is
# heavy (vLLM 0.11.0 + full ML stack + flash_attn compile) -> ~45-60 min.
set -euo pipefail

FS="${FS:-/lambda/nfs/my-filesystem}"
DEF="${DEF:-$FS/standard.def}"
SIF="$FS/containers/standard.sif"
export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export APPTAINER_TMPDIR=/tmp/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$FS/containers" "$FS/hf_cache"

echo "== 1. apptainer =="
if ! command -v apptainer >/dev/null; then
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apptainer
fi
apptainer --version

echo "== 2. build standard.sif (skip if present) =="
if [ -f "$SIF" ]; then
    echo "  $SIF already exists ($(du -h "$SIF" | cut -f1)); skipping build"
else
    [ -f "$DEF" ] || { echo "ERROR: $DEF missing (scp standard.def to the box first)"; exit 1; }
    sudo APPTAINER_CACHEDIR=$APPTAINER_CACHEDIR APPTAINER_TMPDIR=$APPTAINER_TMPDIR \
        apptainer build --force "$SIF" "$DEF"
fi

echo "== 3. pre-stage HF cache (base model) onto the filesystem =="
# MODEL is the base model the chosen trajectory fine-tunes. Done INSIDE the
# container (its huggingface_hub); cache dir is the persistent FS so it
# survives terminate. Models accumulate (plenty of space) — switching
# trajectories reuses any already-cached model. GATED models (e.g. Gemma)
# need HF_TOKEN (from an account that accepted the model's terms) — pass it
# in the environment; ungated models (Qwen) ignore it.
MODEL="${MODEL:-Qwen/Qwen3-4B-Base}"
CACHE_DIRNAME="models--$(echo "$MODEL" | tr '/' '-')"
export HF_HOME="$FS/hf_cache"
if [ ! -d "$FS/hf_cache/hub/$CACHE_DIRNAME" ]; then
    echo "  downloading $MODEL ..."
    apptainer exec --nv \
        --env HF_HOME=/hf_cache --env HF_TOKEN="${HF_TOKEN:-}" \
        --bind "$FS/hf_cache:/hf_cache" \
        "$SIF" python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', token=os.environ.get('HF_TOKEN') or None)
print('$MODEL cached')
" || echo "  (model prestage failed; gated model needs HF_TOKEN w/ accepted terms, or runtime can download)"
else
    echo "  $MODEL already cached"
fi

echo "== bootstrap done =="
echo "  SIF: $SIF"
echo "  HF cache: $FS/hf_cache"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
