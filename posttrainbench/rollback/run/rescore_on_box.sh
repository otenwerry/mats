#!/bin/bash
# Re-run JUST the scoring stage for a finished rollout whose scoring failed
# (e.g. flaky vLLM server startup). Runs ON THE GPU BOX; needs the run's
# final_model still present locally. Writes score.json into the run dir.
#
# NOT exp_-prefixed: it spends GPU time but calls no paid API (it serves the
# trained model locally via vLLM and runs the benchmark's code-exec eval).
#
# Usage (on the box):
#   bash rescore_on_box.sh [WORK_DIR]      # default: newest backward_* run dir
set -uo pipefail
CELL="${1:-$(ls -dt /home/ubuntu/backward_*_cut*__* 2>/dev/null | head -1)}"
[ -d "$CELL/task" ] || { echo "FAIL: no task/ in $CELL"; exit 1; }
FS=""; for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem with a container"; exit 1; }
SIF="$FS/containers/standard.sif"; HF_CACHE="$FS/hf_cache"
SCORE_JSON="$CELL/score.json"; SCORE_OUT="$CELL/rescore_$(date +%s).log"
SCORE_TRIES="${SCORE_TRIES:-3}"; SCORE_BUDGET_MIN="${SCORE_BUDGET_MIN:-45}"

find "$CELL/task/final_model" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | grep -q . \
  || { echo "FAIL: no final_model weights in $CELL/task (can't score without the model)"; exit 1; }
[ -f "$CELL/task/evaluate.py" ] || { echo "FAIL: no task/evaluate.py in $CELL"; exit 1; }
mkdir -p "$CELL/tmp"
echo "rescoring $CELL (up to $SCORE_TRIES tries, cap ${SCORE_BUDGET_MIN}m) -> $SCORE_OUT"

for try in $(seq 1 "$SCORE_TRIES"); do
  [ -f "$SCORE_JSON" ] && break
  echo "----- attempt $try -----" >> "$SCORE_OUT"
  timeout --signal=TERM --kill-after=30s "${SCORE_BUDGET_MIN}m" \
  apptainer exec --nv -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env HF_HOME="/home/ben/hf_cache" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --bind "$CELL/tmp:/tmp" \
    --bind "$HF_CACHE:/home/ben/hf_cache" \
    --home "$CELL:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "$SIF" \
    bash -c 'uv pip install --system --no-cache transformers==4.57.3 tokenizers==0.22.2 2>&1 | tail -2; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 8; python3 evaluate.py --model-path final_model --limit -1 --json-output-file /home/ben/score.json' \
    >> "$SCORE_OUT" 2>&1 || echo "attempt $try exit=$? (124=hit cap)" >> "$SCORE_OUT"
done

if [ -f "$SCORE_JSON" ]; then echo "RESCORE OK -> $(cat "$SCORE_JSON")"
else echo "RESCORE FAIL after $SCORE_TRIES tries; tail:"; tail -30 "$SCORE_OUT"; fi
