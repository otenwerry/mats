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
#   nohup bash exp_run_and_terminate.sh backward_control_cut49 120 >/dev/null 2>&1 &
set -uo pipefail

FS="${FS:-/lambda/nfs/my-filesystem}"   # honor the FS env var (filesystem name varies per box)
CELL_NAME="${1:?usage: exp_run_and_terminate.sh <cell_name> [BUDGET_MIN]}"
BUDGET_MIN="${2:-120}"
HARD_CAP_MIN=$(( BUDGET_MIN + 30 ))
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
( sleep $(( HARD_CAP_MIN * 60 )); log "HARD CAP ${HARD_CAP_MIN}m reached"; terminate ) &

log "=== run start: cell=$CELL_NAME budget=${BUDGET_MIN}m hardcap=${HARD_CAP_MIN}m ==="
cp -r "$FS/cells/$CELL_NAME" "$WORK"
FS="$FS" BUDGET_MIN="$BUDGET_MIN" bash "$FS/run_rollout_on_box.sh" "$WORK" || log "rollout exited non-zero"

log "archiving results -> $ARCHIVE"
mkdir -p "$ARCHIVE"
cp -r "$WORK"/solve_out_*.txt "$ARCHIVE/" 2>/dev/null || true
cp -r "$WORK/task" "$ARCHIVE/task" 2>/dev/null || true
cp -r "$WORK/.local" "$ARCHIVE/dot_local" 2>/dev/null || true
cp "$WORK/run_config.json" "$ARCHIVE/" 2>/dev/null || true
du -sh "$ARCHIVE" 2>/dev/null | sed 's/^/archived: /'
log "=== run done; terminating ==="
terminate
