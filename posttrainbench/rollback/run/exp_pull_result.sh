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
SECRETS="$HOME/.config/ptb/secrets.env"; set -a; source "$SECRETS"; set +a
SSH="ssh -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new"
IP="${1:-}"
if [ -z "$IP" ]; then
  IP=$(curl -s -m 20 -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances | python3 -c "import sys,json; d=[i for i in json.load(sys.stdin).get('data',[]) if i.get('status')=='active']; print(d[0]['ip'] if len(d)==1 else '')")
fi
[ -n "$IP" ] || { echo "no single running box; pass IP"; exit 1; }
FS=$($SSH ubuntu@$IP 'for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && echo "${d%/}" && break; done' 2>/dev/null | head -1)
RDIR=$($SSH ubuntu@$IP "ls -td $FS/results/*/ 2>/dev/null | head -1" | tr -d '\r')
[ -n "$RDIR" ] || { echo "no results dir on $FS"; exit 1; }
NAME=$(basename "$RDIR")
# pulled rollouts live in mats-local (our own data, off github); mirrors
# config.ROLLBACK_RESULTS. $HERE = rollback/run -> ../../../../mats-local.
LOCAL_ROLLBACK="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
DEST="$LOCAL_ROLLBACK/results/$NAME"
mkdir -p "$DEST"
echo "pulling $RDIR -> $DEST"
# keep the trajectory (solve_out, run_config, logs) + the agent's source code;
# drop regenerable model artifacts (checkpoints, merged model, 32MB tokenizers)
# — large and reproducible from the code; fetch to mats-local separately if needed.
rsync -az -e "$SSH" \
  --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.gguf' \
  --exclude 'outputs/' --exclude 'final_model/' --exclude '*_merged/' --exclude 'tokenizer.json' \
  "ubuntu@$IP:$RDIR" "$LOCAL_ROLLBACK/results/" 2>&1 | tail -2
# signal the archiver it may terminate (only meaningful once .run_complete exists)
$SSH ubuntu@$IP "[ -f $RDIR/.run_complete ] && touch $RDIR/.pulled && echo 'signaled terminate' || echo 'run still in progress (pulled partial)'" 2>&1 | tail -1
echo "local copy: $DEST"
echo "solve_out: $(ls "$DEST"/solve_out_*.txt 2>/dev/null | head -1)"

# auto-add the pulled run to the viewer (free, idempotent; needs run_config.json,
# which completed runs have). Skips anything already viewerized.
( cd "$HERE/../.." && uv run python -m rollback.sync_viewer ) 2>&1 | tail -3
echo "judge any unjudged rollback runs with:  uv run python posttrainbench/judging/exp_judge_rollback.py"
