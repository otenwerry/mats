#!/bin/bash
# Autonomous single-rollout driver for the GPU box. exp_ = spends money
# (gemini tokens + instance hours). Flow:
#   1. copy a pristine cell to local disk (fast I/O; keeps the FS copy clean),
#   2. run it bounded by BUDGET_MIN (run_rollout_on_box.sh self-times-out),
#   3. archive results (solve_out, task/, session) to the persistent FS so
#      they survive termination,
#   4. TERMINATE this instance via the Lambda API.
# A detached absolute-deadline backstop terminates no matter what after
# HARD_CAP minutes, so a hang can never bill overnight.
#
# Usage (on the box):
#   nohup bash exp_run_and_terminate.sh backward_prompt1_cut49 >/dev/null 2>&1 &
set -uo pipefail

FS="${FS:-/lambda/nfs/my-filesystem}"   # honor the FS env var (filesystem name varies per box)
CELL_NAME="${1:?usage: exp_run_and_terminate.sh <cell_name> [BUDGET_MIN]}"
BUDGET_MIN="${2:-}"
RUN_ID="${CELL_NAME}__$(date +%s)"
WORK="/home/ubuntu/$RUN_ID"
ARCHIVE="$FS/results/$RUN_ID"

source /home/ubuntu/.ptb_lambda          # LAMBDA_API_KEY, INSTANCE_ID
mkdir -p "$FS/results"
LOG="$FS/results/${RUN_ID}.driver.log"
exec >>"$LOG" 2>&1
log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

terminate(){
  log "TERMINATING instance $INSTANCE_ID"
  for i in 1 2 3 4 5; do
    code=$(curl -s -o /tmp/term.out -w "%{http_code}" -m 30 -u "$LAMBDA_API_KEY:" \
      -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate)
    log "terminate attempt $i -> HTTP $code"
    [ "$code" = "200" ] && return 0
    sleep 15
  done
  log "API terminate failed; shutdown fallback"; sudo shutdown -h now
}

# absolute-deadline backstop (detached): fire-and-forget terminate
cp -r "$FS/cells/$CELL_NAME" "$WORK"
if [ -z "$BUDGET_MIN" ]; then
  BUDGET_MIN=$(python3 - <<PY
import json, math
cfg=json.load(open("$WORK/run_config.json"))
if cfg.get("agent_timeout_minutes"):
    print(cfg["agent_timeout_minutes"])
else:
    elapsed=int(cfg.get("elapsed_seconds") or 0)
    hours=int(cfg.get("num_hours") or 10)
    print(math.ceil(max(0, hours * 3600 - elapsed) / 60) + 5)
PY
)
fi
HARD_CAP_MIN=$(( BUDGET_MIN + 30 ))
( sleep $(( HARD_CAP_MIN * 60 )); log "HARD CAP ${HARD_CAP_MIN}m reached"; terminate ) &

log "=== run start: cell=$CELL_NAME budget=${BUDGET_MIN}m hardcap=${HARD_CAP_MIN}m ==="
FS="$FS" BUDGET_MIN="$BUDGET_MIN" bash "$FS/run_rollout_on_box.sh" "$WORK" || log "rollout exited non-zero"

log "archiving results -> $ARCHIVE"
mkdir -p "$ARCHIVE"
cp -r "$WORK"/solve_out_*.txt "$ARCHIVE/" 2>/dev/null || true
cp "$WORK"/score.json "$WORK"/prep_score.json "$WORK"/prep_fidelity.json "$ARCHIVE/" 2>/dev/null || true
cp "$WORK"/.prep_diverged "$WORK"/.prep_unverified "$ARCHIVE/" 2>/dev/null || true
cp -r "$WORK/task" "$ARCHIVE/task" 2>/dev/null || true
cp -r "$WORK/.local" "$ARCHIVE/dot_local" 2>/dev/null || true
cp "$WORK/run_config.json" "$ARCHIVE/" 2>/dev/null || true
du -sh "$ARCHIVE" 2>/dev/null | sed 's/^/archived: /'
log "=== run done; terminating ==="
terminate
