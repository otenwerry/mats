#!/bin/bash
# Launch ONE prepared rollback cell inside the apptainer container on the GPU
# box. Models the original PostTrainBench src/run_task.sh solve_task(), trimmed
# to our resume case:
#   - no fuse-overlayfs (that protected a shared cache across concurrent cluster
#     jobs; we run cells sequentially, so bind the HF cache directly);
#   - the agent step is our reconstructed-session resume (agent_solve.sh =
#     solve_intervention_opencode.sh), not a fresh PROMPT launch;
#   - same --nv -c --writable-tmpfs --home/--pwd contract and the cuda checks.
#
# The cell dir (JOB_DIR) is a prepared job home (rsynced from the Mac):
#   task/                              cut-point workspace
#   .config/opencode/opencode.json     global config w/ the openrouter provider
#   .local/share/opencode/storage/     reconstructed truncated session
#   agent_solve.sh                     in-container entrypoint
#   run_config.json                    session_id, policy_model, elapsed, etc.
#
# Secrets come from /home/ubuntu/.ptb_secrets on the box (OPENROUTER_API_KEY,
# OPENAI_API_KEY) — never baked into the image or the repo.
#
# Usage:
#   BUDGET_MIN=20 bash run_rollout_on_box.sh /path/to/cell
# The filesystem is AUTO-DETECTED (the /lambda/nfs/* holding our container), so
# nothing depends on its name or an exported env var.
set -uo pipefail

CELL="${1:?usage: run_rollout_on_box.sh <cell_dir>}"
# auto-detect the filesystem: the /lambda/nfs/* that has our container
if [ -z "${FS:-}" ]; then
  for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
fi
[ -n "${FS:-}" ] || { echo "FATAL: no /lambda/nfs filesystem with a container found"; touch "$CELL/.agent_done"; exit 1; }
SIF="${SIF:-$FS/containers/standard.sif}"
HF_CACHE="${HF_CACHE:-$FS/hf_cache}"
SECRETS="${SECRETS:-/home/ubuntu/.ptb_secrets}"

[ -f "$SIF" ] || { echo "no container at $SIF"; exit 1; }
[ -f "$CELL/run_config.json" ] || { echo "no run_config.json in $CELL"; exit 1; }
[ -f "$SECRETS" ] && source "$SECRETS"

# pull the per-cell parameters the agent_solve.sh expects
read_cfg() { python3 -c "import json,sys;print(json.load(open('$CELL/run_config.json')).get('$1',''))"; }
SESSION_ID="$(read_cfg session_id)"
AGENT_CONFIG="$(read_cfg policy_model)"
ELAPSED_SECONDS="$(read_cfg elapsed_seconds)"
NUM_HOURS="$(read_cfg num_hours)"
RESUME_PROMPT="$(read_cfg resume_prompt)"
CELL_ID="$(read_cfg cell_id)"

# BUDGET_MIN caps THIS launch's wall-clock (distinct from NUM_HOURS, which the
# in-container timer.sh reports to the agent as its faithful remaining budget).
# Default to the full budget; override low for smoke tests.
BUDGET_MIN="${BUDGET_MIN:-$(( NUM_HOURS * 60 + 5 ))}"

mkdir -p "$CELL/tmp"
SOLVE_OUT="$CELL/solve_out_$(date +%s).txt"
echo "launching cell=$CELL_ID session=$SESSION_ID model=$AGENT_CONFIG"
echo "  timer shows NUM_HOURS=$NUM_HOURS (elapsed $ELAPSED_SECONDS s); this launch capped at ${BUDGET_MIN}m"
echo "  solve_out -> $SOLVE_OUT"

timeout --signal=TERM --kill-after=30s "${BUDGET_MIN}m" \
apptainer exec \
    --nv \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env HF_HOME="/home/ben/hf_cache" \
    --env OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --env NUM_GPUS="1" \
    --env SESSION_ID="${SESSION_ID}" \
    --env AGENT_CONFIG="${AGENT_CONFIG}" \
    --env ELAPSED_SECONDS="${ELAPSED_SECONDS}" \
    --env NUM_HOURS="${NUM_HOURS}" \
    --env RESUME_PROMPT="${RESUME_PROMPT}" \
    --bind "$CELL/tmp:/tmp" \
    --bind "$HF_CACHE:/home/ben/hf_cache" \
    --home "$CELL:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "$SIF" \
    bash -c 'python /home/ben/check_cuda.py 2>/dev/null || echo "[warn] check_cuda missing/failed"; bash /home/ben/agent_solve.sh' \
    > "$SOLVE_OUT" 2>&1 || echo "launch exit=$? (124=hit BUDGET_MIN cap)"

# clean done-marker the archiver watches for (robust vs pgrep self-match)
touch "$CELL/.agent_done"
echo "=== agent finished; solve_out tail ==="
tail -15 "$SOLVE_OUT"
