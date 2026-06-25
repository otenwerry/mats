#!/bin/bash
# Pull a rollout's trajectory from the box's filesystem to the Mac. Works while
# the box is up — during a run (partial) or in the 30-min grace window after it
# completes (final). Touching .pulled tells the archiver it may terminate now.
#
#   bash exp_pull_result.sh [IP]      # auto-discovers the single running box
#
# Pulls solve_out + the small workspace (excludes model weights) into
# rollback/results/<run_dir>/ on the Mac. NOT exp_ in spirit (read-only,
# free) but named exp_ for grouping; safe to run anytime.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets
SSH="$PTB_SSH"   # shared opts (single source of truth in ptb_lib.sh)
IP="${1:-}"
[ -z "$IP" ] && IP=$(ptb_active_ip)
[ -n "$IP" ] || { echo "no single running box; pass IP"; exit 1; }
FS=$(ptb_fs_detect_remote "$IP")
# Pull the box's OWN run dir (box-local /home/ubuntu/...), NOT "latest on the
# filesystem": with two boxes sharing one Lambda filesystem, the latter grabs
# whichever run finished most recently (wrong box). The box-local dir is unique
# per box and also holds the freshest score.json.
WORK=$($SSH ubuntu@$IP "ls -td /home/ubuntu/backward_*_cut*__* 2>/dev/null | head -1" | tr -d '\r')
[ -n "$WORK" ] || { echo "no run dir on box"; exit 1; }
NAME=$(basename "$WORK")
# pulled rollouts live in mats-local (our own data, off github); mirrors
# config.ROLLBACK_RESULTS. $HERE = rollback/run -> ../../../../mats-local.
LOCAL_ROLLBACK="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
DEST="$LOCAL_ROLLBACK/results/$NAME"
mkdir -p "$LOCAL_ROLLBACK/results"
echo "pulling $WORK -> $DEST"
# keep the trajectory (solve_out, run_config, score.json, logs) + the agent's
# source; drop regenerable model artifacts (checkpoints, merged model, tokenizers).
# IMPORTANT: source has NO trailing slash, so rsync creates results/<NAME>/ — a
# trailing slash would dump the contents loose into results/.
rsync -az -e "$SSH" \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.gguf' \
  --exclude 'outputs/' --exclude 'final_model/' --exclude '*_merged/' --exclude 'tokenizer.json' \
  "ubuntu@$IP:$WORK" "$LOCAL_ROLLBACK/results/" 2>&1 | tail -2
# signal the archiver (its FS results dir) it may terminate (non-KEEP_ALIVE path)
$SSH ubuntu@$IP "[ -f $FS/results/$NAME/.run_complete ] && touch $FS/results/$NAME/.pulled && echo 'signaled terminate' || echo 'run in progress / keep-alive (pulled partial)'" 2>&1 | tail -1
echo "local copy: $DEST"
echo "solve_out: $(ls "$DEST"/solve_out_*.txt 2>/dev/null | head -1)"

# auto-add the pulled run to the viewer (free, idempotent; needs run_config.json,
# which completed runs have). Skips anything already viewerized.
( cd "$HERE/../.." && uv run python -m rollback.sync_viewer ) 2>&1 | tail -3
echo "judge any unjudged rollback runs with:  uv run python posttrainbench/judging/exp_judge_rollback.py"
