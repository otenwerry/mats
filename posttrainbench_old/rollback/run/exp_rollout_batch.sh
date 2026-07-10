#!/bin/bash
# End-to-end trajectory batch runner: one box, one prep/fidelity pass, then
# multiple prompt continuations from the same prepped workspace.
# exp_ = spends money (Lambda GPU, policy API calls, judge calls).
#
# Usage:
#   bash rollback/run/exp_rollout_batch.sh <nick|run_id> [prompt1,prompt2,prompt3] [gpus]
#
# Region / GPU selection is inherited from ptb_lib.sh:
#   PTB_FS_NAME=<filesystem> PTB_H100_VARIANT=sxm5 ...
#   PTB_INSTANCE_TYPE_NAME=gpu_4x_h100_sxm5 ...
# Reuse a previously saved prep workspace:
#   PTB_PREPPED_SOURCE=/lambda/nfs/<fs>/prepped/<dir> ...
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"

TRAJ="${1:?usage: exp_rollout_batch.sh <nick|run_id> [prompt1,prompt2,prompt3] [gpus]}"
CONDS_CSV="${2:-prompt1,prompt2,prompt3}"
GPUS="${3:-1}"
PARALLEL="${PARALLEL:-}"
BUDGET_MIN="${BUDGET_MIN:-}"
PTB_PREPPED_SOURCE="${PTB_PREPPED_SOURCE:-}"
export PTB_TRAJECTORY="$TRAJ"
ptb_load_secrets || exit 1

read RUN_NAME CUT RUN_ID < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT, t.run_id)")
LOCAL_ROLLBACK="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
RUNNING_DIR="$LOCAL_ROLLBACK/running"     # per-(traj,condition) in-flight registry the viewer reads
REGION_TAG="${PTB_H100_VARIANT:-auto}/${PTB_FS_NAME:-auto}"
RUNNING_TS="$(date +%s)"

IFS=, read -r -a CONDS <<< "$CONDS_CSV"
CELLS=()
for cond in "${CONDS[@]}"; do
  if ! ( cd "$HERE/../.." && python3 - "$cond" <<'PY'
import sys
from rollback import contract
sys.exit(0 if contract.valid_condition(sys.argv[1]) else 1)
PY
  ); then
    echo "condition must be one of: prompt1 prompt2 prompt3"; exit 1
  fi
  CELLS+=("backward_${cond}_cut${CUT}")
done
CELLS_CSV="$(IFS=,; echo "${CELLS[*]}")"

if [ -z "$PARALLEL" ]; then
  if [[ "$GPUS" =~ ^[0-9]+$ ]]; then
    PARALLEL="$GPUS"
  else
    PARALLEL=1
  fi
  [ "$PARALLEL" -gt "${#CELLS[@]}" ] && PARALLEL="${#CELLS[@]}"
fi
[ "$PARALLEL" -ge 1 ] 2>/dev/null || { echo "PARALLEL must be a positive integer"; exit 1; }

echo "==== BATCH ROLLOUT [$TRAJ] conditions=$CONDS_CSV cells=$CELLS_CSV gpus=$GPUS parallel=$PARALLEL budget=${BUDGET_MIN:-auto}m ===="

LAUNCHED_ID=""
RUNNING_FILES=()
# Register each (trajectory, condition) as in-flight so the viewer's /rollback
# page shows it as RUNNING NOW (orange, "live this session"). One small json per
# condition, unique filename -> no race across concurrent launchers. Removed on
# exit; also auto-suppressed by the viewer once the condition completes, and
# aged out by its TTL if this process is killed before cleanup.
register_running() {  # register_running <ip>
  mkdir -p "$RUNNING_DIR" 2>/dev/null || return 0
  local ip="$1" cond f
  for cond in "${CONDS[@]}"; do
    f="$RUNNING_DIR/${RUN_NAME}__${cond}__${RUNNING_TS}.json"
    python3 - "$f" "$RUN_ID" "$RUN_NAME" "$cond" "$ip" "$REGION_TAG" "$RUNNING_TS" "$PARALLEL" <<'PY' 2>/dev/null || true
import json, sys
f, rid, rname, cond, ip, region, ts, par = sys.argv[1:9]
# parallel = how many conditions this box runs at once (= its GPU count). On a 1x
# box the conditions run sequentially, so the viewer shows the first `parallel`
# as running and the rest as queued — keeping the running count == GPUs in use.
json.dump({"source_trajectory": rid, "run_id": rid, "run_name": rname,
           "condition": cond, "label": cond, "status": "running",
           "ip": ip, "region": region, "launched_at": int(ts),
           "parallel": int(par)}, open(f, "w"))
PY
    RUNNING_FILES+=("$f")
  done
}
unregister_running() { local f; for f in "${RUNNING_FILES[@]:-}"; do [ -n "$f" ] && rm -f "$f"; done; }
cleanup() {
  unregister_running
  [ -n "$LAUNCHED_ID" ] && { echo "== teardown: terminating $LAUNCHED_ID =="; ptb_terminate "$LAUNCHED_ID" >/dev/null 2>&1; }
}
trap cleanup EXIT

echo "== 1. launch box =="
IP=$(ptb_launch_box_chain "$GPUS" "ptb-batch-$(date +%m%d-%H%M%S)") || { echo "BATCH FAIL: box launch"; exit 1; }
LAUNCHED_ID=$(ptb_instance_id "$IP")
echo "  box $IP (id $LAUNCHED_ID)"
register_running "$IP"

echo "== 2. build cut-point cells =="
( cd "$HERE/../.." && python3 -m rollback.run.orchestrate --force --bash-mode skip --cells "${CELLS[@]}" ) 2>&1 | tail -$(( ${#CELLS[@]} * 5 + 2 ))
for cell in "${CELLS[@]}"; do
  [ -d "$HERE/../builds/$RUN_NAME/$cell" ] || { echo "BATCH FAIL: cell not built: $cell"; exit 1; }
done

echo "== 3. preflight OpenRouter balance =="
REM=$(curl -s -m 15 -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/credits | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',{}); tc=d.get('total_credits'); print((tc-d.get('total_usage',0)) if isinstance(tc,(int,float)) else 99999)" 2>/dev/null || echo 0)
MIN_BALANCE="${MIN_BALANCE:-15}"
echo "  OpenRouter remaining: \$$REM (min required \$$MIN_BALANCE)"
awk "BEGIN{exit !($REM < $MIN_BALANCE)}" && { echo "FAIL: OpenRouter balance too low (\$$REM < \$$MIN_BALANCE)"; exit 1; }

echo "== 4. rsync cells + scripts to the box =="
SSH="$PTB_SSH"
FS=$(ptb_fs_detect_remote "$IP")
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on box"; exit 1; }
echo "  filesystem on box: $FS"
for cell in "${CELLS[@]}"; do
  rsync -az -e "$SSH" --exclude tmp --exclude '.agent_done' "$HERE/../builds/$RUN_NAME/$cell" "ubuntu@$IP:$FS/cells/" 2>&1 | tail -1
done
rsync -az -e "$SSH" "$HERE"/{box_prepare.sh,run_rollout_on_box.sh,run_rollout_batch_on_box.sh,archive_and_terminate.sh} "ubuntu@$IP:$FS/" 2>&1 | tail -1

echo "== 5. deploy secrets+cred, preflight, launch batch =="
SECRETS="$PTB_SECRETS"
SEC_B64=$(grep -E '^(OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN|DASHSCOPE_API_KEY|HF_TOKEN)=' "$SECRETS" | base64)
CRED_B64=$( { grep '^LAMBDA_API_KEY=' "$SECRETS"; echo "INSTANCE_ID=$LAUNCHED_ID"; } | base64)
CODEX_AUTH_LOCAL="${PTB_CODEX_AUTH:-$HOME/.config/ptb/codex_auth.json}"
CODEX_AUTH_B64=""; [ -f "$CODEX_AUTH_LOCAL" ] && CODEX_AUTH_B64=$(base64 < "$CODEX_AUTH_LOCAL") && echo "  deploying codex auth.json from $CODEX_AUTH_LOCAL"
$SSH ubuntu@$IP "bash -s" <<EOF
set -uo pipefail
umask 077; echo '$SEC_B64' | base64 -d > /home/ubuntu/.ptb_secrets; echo '$CRED_B64' | base64 -d > /home/ubuntu/.ptb_lambda
[ -n "$CODEX_AUTH_B64" ] && echo '$CODEX_AUTH_B64' | base64 -d > /home/ubuntu/.ptb_codex_auth.json
umask 022
chmod +x $FS/box_prepare.sh $FS/run_rollout_on_box.sh $FS/run_rollout_batch_on_box.sh $FS/archive_and_terminate.sh
cp $FS/run_rollout_on_box.sh $FS/run_rollout_batch_on_box.sh $FS/archive_and_terminate.sh /home/ubuntu/
chmod +x /home/ubuntu/run_rollout_on_box.sh /home/ubuntu/run_rollout_batch_on_box.sh /home/ubuntu/archive_and_terminate.sh
PREP=\$(bash $FS/box_prepare.sh ${CELLS[0]})
echo "\$PREP"
echo "\$PREP" | grep -q "PREP: READY" || { echo "ABORT: preflight failed"; exit 2; }
nohup env BUDGET_MIN="$BUDGET_MIN" PTB_PREPPED_SOURCE="$PTB_PREPPED_SOURCE" bash /home/ubuntu/run_rollout_batch_on_box.sh "$CELLS_CSV" "$PARALLEL" > $FS/batch_live_$(date +%s).log 2>&1 < /dev/null &
echo "BATCH_PID \$!"
nohup bash -c 'source /home/ubuntu/.ptb_lambda; sleep 43200; curl -s -m30 -u "\$LAMBDA_API_KEY:" -H "Content-Type: application/json" -d "{\"instance_ids\":[\"\$INSTANCE_ID\"]}" https://cloud.lambda.ai/api/v1/instance-operations/terminate' > /dev/null 2>&1 < /dev/null &
echo "BACKSTOP_ARMED 43200s"
echo LAUNCH_OK
EOF
REMOTE_RC=$?
[ "$REMOTE_RC" -eq 0 ] || { echo "BATCH FAIL: remote preflight/launch failed (rc=$REMOTE_RC)"; exit 1; }

echo "== 6. wait for batch completion =="
# Break ONLY on the real .batch_done file via its test exit code — never by
# grepping polled text. The diagnostic tail includes prep_driver.log, which
# prints "PREP: DONE in NNmNNs" when prep finishes; grepping it for DONE used to
# break the loop the moment prep completed (mid-run), then exit 2 below skipped
# the pull + left stale running/ rows + the box up. (rollback-run-bugs 2026-06-17 #1)
BATCH=""
BATCH_DONE=0
# 360 x 120s = 12h, matching the box's own hard backstop (BACKSTOP_ARMED 43200s):
# a full run is prep (~2h) + agent (up to the remaining budget, ~5-6h) + scoring,
# so a shorter cap timed out mid-run and forced a manual pull_all.sh. The box can't
# outlive 12h, so polling that long catches any run's auto-pull without ever
# polling a dead box. (Was 120 = 4h, too short; 2026-06-17.)
for i in $(seq 1 360); do
  BATCH=$(ptb_ssh "$IP" "ls -td /home/ubuntu/batch_${CELLS[0]}__* 2>/dev/null | head -1" | tr -d '\r')
  if [ -n "$BATCH" ]; then
    if ptb_ssh "$IP" "[ -f $BATCH/.batch_done ]"; then BATCH_DONE=1; fi
    OUT=$(ptb_ssh "$IP" "{ echo \"batch=$BATCH\"; tail -5 $BATCH/prep_driver.log 2>/dev/null; cat $BATCH/pids.txt 2>/dev/null | tail -5; }" 2>&1)
    echo "  [$i $(date -u +%H:%M)] done=$BATCH_DONE $OUT"
    [ "$BATCH_DONE" = 1 ] && break
  else
    echo "  [$i $(date -u +%H:%M)] waiting for batch dir"
  fi
  sleep 120
done
# On either timeout path, leave the box up for debug but STILL drop the in-flight
# running/ rows so the viewer doesn't show phantom "running" forever.
# On these timeout paths the box is LEFT UP, so the run is still in-flight (it
# keeps going on the box's own 12h backstop). KEEP the registry marker — removing
# it here let gap-fill relaunch a still-running trajectory as a duplicate (bit
# cut300: its 4h wait loop timed out, unregistered, but the agents were still
# running; 2026-06-17). The marker is cleared on normal completion (the trap) and
# is dedup-safe besides (a finished run also shows in viewer_data). TODO: have
# sync_viewer prune running/ entries whose run_id is now viewerized.
if [ -z "$BATCH" ]; then
  echo "BATCH WARN: no batch dir found; box $IP left up (registry marker kept — run may be in-flight)"
  trap - EXIT
  exit 2
fi
if [ "$BATCH_DONE" != 1 ]; then
  echo "BATCH WARN: timed out; box $IP left up for debug: $BATCH (registry marker kept — run still in-flight)"
  trap - EXIT
  exit 2
fi

echo "== 7. pull child results =="
LOCAL_ROLLBACK="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
mkdir -p "$LOCAL_ROLLBACK/results"
WORKS=$(ptb_ssh "$IP" "cat $BATCH/work_dirs.txt 2>/dev/null; ls -d $BATCH/prep_* 2>/dev/null" | tr -d '\r')
for work in $WORKS; do
  name=$(basename "$work")
  echo "pulling $work -> $LOCAL_ROLLBACK/results/$name"
  rsync -az -e "$SSH" \
    --exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.gguf' \
    --exclude 'outputs/' --exclude 'final_model/' --exclude '*_merged/' --exclude 'tokenizer.json' \
    "ubuntu@$IP:$work" "$LOCAL_ROLLBACK/results/" 2>&1 | tail -2
done

echo "== 8. viewerize + judge =="
( cd "$HERE/../.." && uv run python -m rollback.sync_viewer ) 2>&1 | tail -10
RESULT_NAMES=$(for work in $WORKS; do basename "$work"; done | tr '\n' ' ')
RB_RUN_IDS=$( cd "$HERE/../.." && uv run python - "$RESULT_NAMES" <<'PY'
import json, sys
from rollback import config
names = set(" ".join(sys.argv[1:]).split())
for p in sorted(config.ROLLBACK_VIEWER_DATA.glob("rollback_*.json")):
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError:
        continue
    if (rec.get("meta") or {}).get("result_dir") in names:
        rid = rec.get("index_row", {}).get("run_id")
        if rid:
            print(rid)
PY
)
if [ -n "$RB_RUN_IDS" ]; then
  ( cd "$HERE/../../.." && uv run python posttrainbench/judging/exp_judge_rollback.py $RB_RUN_IDS ) \
    || echo "BATCH WARN: targeted judge failed; saved runs can be judged later with the global script."
fi

echo "==== BATCH DONE [$TRAJ / $CONDS_CSV] ===="
