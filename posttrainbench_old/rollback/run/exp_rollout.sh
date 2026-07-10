#!/bin/bash
# exp_rollout.sh — ONE command, end-to-end rollback run:
#   spin up a box -> build the cut-point cell -> run (prep -> resume -> score)
#   -> pull + viewerize -> judge -> terminate the box (+ verify).
# exp_ = spends money (policy API + GPU + judge LLM calls).
#
#   bash exp_rollout.sh <nick|run_id> [prompt1|prompt2|prompt3] [gpus]
#
# Self-contained: launches its OWN box and terminates it at the end — even on
# failure, via an EXIT trap — so it never leaves a box billing. Long-running
# (box boot + agent run + judge), so run it in your terminal or backgrounded.
# On a suspected hang / timeout it leaves the box UP (disarms the trap) so you
# can debug, and exits non-zero.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
TRAJ="${1:?usage: exp_rollout.sh <nick|run_id> [prompt1|prompt2|prompt3] [gpus]}"
COND="${2:-prompt1}"
GPUS="${3:-1}"
BUDGET_MIN="${BUDGET_MIN:-}"        # empty => box runner uses remaining cut budget + grace
export PTB_TRAJECTORY="$TRAJ"
if ! ( cd "$HERE/../.." && python3 - "$COND" <<'PY'
import sys
from rollback import contract
sys.exit(0 if contract.valid_condition(sys.argv[1]) else 1)
PY
); then
  echo "condition must be one of: prompt1 prompt2 prompt3"; exit 1
fi
ptb_load_secrets || exit 1

read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT)")
CELL="backward_${COND}_cut${CUT}"
echo "==== ROLLOUT [$TRAJ / $COND] cell=$CELL budget=${BUDGET_MIN:-auto}m ===="

LAUNCHED_ID=""
cleanup() { [ -n "$LAUNCHED_ID" ] && { echo "== teardown: terminating $LAUNCHED_ID =="; ptb_terminate "$LAUNCHED_ID" >/dev/null 2>&1; }; }
trap cleanup EXIT

echo "== 1. launch box =="
IP=$(ptb_launch_box "$GPUS" "ptb-${COND}-$(date +%m%d-%H%M%S)") || { echo "ROLLOUT FAIL: box launch"; exit 1; }
LAUNCHED_ID=$(ptb_instance_id "$IP")
echo "  box $IP (id $LAUNCHED_ID)"

echo "== 2. build cut-point cell =="
( cd "$HERE/../.." && python3 -m rollback.run.orchestrate --force --bash-mode skip --cells "$CELL" ) 2>&1 | tail -2
[ -d "$HERE/../builds/$RUN_NAME/$CELL" ] || { echo "ROLLOUT FAIL: cell not built"; exit 1; }

echo "== 3. run (prep -> resume -> score) =="
# KEEP_ALIVE so teardown is OURS (the trap); a backstop still bounds a crash.
KEEP_ALIVE=1 BACKSTOP_SEC=43200 bash "$HERE/exp_run_experiment.sh" "$COND" "$BUDGET_MIN" "$IP" \
  || { echo "ROLLOUT FAIL: run dispatch"; exit 1; }
WORK=$(ptb_ssh "$IP" "ls -td /home/ubuntu/${CELL}__* 2>/dev/null | head -1" | tr -d '\r')
[ -n "$WORK" ] || { echo "ROLLOUT FAIL: no work dir on box"; exit 1; }

echo "== 4. wait for completion (.agent_done; hang = static size + idle GPU ~30min) =="
prev=-1; stall=0
for i in $(seq 1 80); do
  OUT=$(ptb_ssh "$IP" "if [ -f $WORK/.agent_done ]; then echo DONE; else echo \"bytes=\$(du -sb $WORK 2>/dev/null|cut -f1) gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|head -1|tr -dc 0-9)\"; fi" 2>&1)
  echo "  [$i $(date -u +%H:%M)] $OUT"
  echo "$OUT" | grep -q DONE && break
  b=$(echo "$OUT"|grep -o 'bytes=[0-9]*'|cut -d= -f2); g=$(echo "$OUT"|grep -o 'gpu=[0-9]*'|cut -d= -f2)
  if [ "${b:-0}" = "$prev" ] && [ "${g:-0}" -lt 5 ]; then stall=$((stall+1)); else stall=0; fi
  prev="${b:-0}"
  [ "$stall" -ge 3 ] && { echo "ROLLOUT WARN: hang suspected — box $IP left UP for debug"; trap - EXIT; exit 2; }
  sleep 600
done
ptb_ssh "$IP" "[ -f $WORK/.agent_done ]" || { echo "ROLLOUT WARN: timed out — box $IP left UP for debug"; trap - EXIT; exit 2; }

# --- fidelity-gate (NON-FATAL as of 2026-06-17) ---------------------------
# The on-box gate no longer skips the agent on a diverged/unverified prep — it
# runs the agent anyway and records prep_fidelity status as a flag that the pull/
# sync step surfaces in the viewer. So a .prep_diverged/.prep_unverified marker
# now coexists with a real solve_out; do NOT abort here. (Just note it; the
# normal pull below carries the flag through sync_viewer -> meta.prep_fidelity*.)
if ptb_ssh "$IP" "[ -f $WORK/.prep_diverged ] || [ -f $WORK/.prep_unverified ]"; then
  echo "== prep fidelity flagged (diverged/unverified) — agent still ran; flag will be surfaced in the viewer =="
fi

# --- empty-run guard ------------------------------------------------------
# .agent_done is touched at the END of run_rollout_on_box.sh AND on its failure
# paths (no filesystem, prep-fail, or a silent opencode resume that streamed
# nothing) — so it is NOT a success signal. Before pulling + tearing the box
# down (which destroys the on-box stderr/logs), confirm the agent ACTUALLY
# produced output: a non-empty solve_out (>200B, the same bar smoke uses for
# "agent took a step") AND a score.json. If either is missing, LEAVE THE BOX UP
# (disarm the trap) so the evidence survives for diagnosis, and exit non-zero —
# same contract as the hang/timeout path. This catches the cut177 failure mode
# (prep OK, agent produced nothing, reported "done").
GUARD=$(ptb_ssh "$IP" "
  so=\$(ls -S $WORK/solve_out_*.txt 2>/dev/null | head -1)
  echo \"solve_bytes=\$([ -n \"\$so\" ] && wc -c < \"\$so\" || echo 0) score=\$([ -f $WORK/score.json ] && echo yes || echo no)\"
" 2>&1 | tr -d '\r')
echo "  guard: $GUARD"
SOB=$(echo "$GUARD" | grep -o 'solve_bytes=[0-9]*' | cut -d= -f2)
SC=$(echo "$GUARD" | grep -o 'score=[a-z]*' | cut -d= -f2)
# Success signal is the AGENT producing output (non-empty solve_out). score.json
# is now best-effort (the vLLM scorer is flaky; a missing score is flagged in the
# viewer + rescorable), so it no longer gates run success — just warn if absent.
if [ "${SOB:-0}" -lt 200 ]; then
  echo "ROLLOUT WARN: empty/failed run (solve_out=${SOB:-0}B) — agent produced no usable output."
  echo "  box $IP left UP for debug (ssh ubuntu@$IP; inspect $WORK). exit 3."
  trap - EXIT; exit 3
fi
[ "$SC" = yes ] || echo "ROLLOUT NOTE: no score.json (scoring failed/again best-effort) — run is valid; score will be flagged in the viewer + can be rescored."

echo "== 5. pull + viewerize =="
bash "$HERE/exp_pull_result.sh" "$IP" 2>&1 | tail -6
RESULT_DIR=$(basename "$WORK")
RB_RUN_ID=$( cd "$HERE/../.." && uv run python - "$RESULT_DIR" <<'PY'
import json, sys
from rollback import config
result_dir = sys.argv[1]
hits = []
for p in sorted(config.ROLLBACK_VIEWER_DATA.glob("rollback_*.json")):
    try:
        rec = json.loads(p.read_text())
    except json.JSONDecodeError:
        continue
    if (rec.get("meta") or {}).get("result_dir") == result_dir:
        hits.append(rec.get("index_row", {}).get("run_id"))
print(hits[-1] if hits else "")
PY
)
[ -n "$RB_RUN_ID" ] || { echo "ROLLOUT WARN: could not find viewer row for result_dir=$RESULT_DIR; skipping targeted judge"; RB_RUN_ID=""; }

# --- empty-continuation guard ---------------------------------------------
# A run can pass the guard above (non-empty solve_out + a score) yet still have
# produced NOTHING after the resume user-turn — the agent resumed but emitted
# zero events (the healthbench/gemma failure mode). viewerize records this as
# meta.empty_continuation in the run's viewer_data; if true, the "control run"
# is vacuous (nothing to judge), so LEAVE THE BOX UP and exit non-zero so the
# resume can be diagnosed before the evidence is torn down.
NCONT=$( cd "$HERE/../.." && uv run python -c "
import json
from rollback import config as c
hits = []
for p in sorted(c.ROLLBACK_VIEWER_DATA.glob('rollback_*.json')):
    rec = json.load(open(p))
    if (rec.get('meta') or {}).get('result_dir') == '${RESULT_DIR}':
        hits.append(rec)
print((hits[-1].get('meta') or {}).get('n_continuation_events', 0) if hits else -1)
" 2>/dev/null | tail -1 )
echo "  continuation events after resume: ${NCONT}"
if [ "${NCONT:-0}" -le 0 ] 2>/dev/null; then
  echo "ROLLOUT WARN: empty continuation (${NCONT} events after the resume turn) — the agent produced nothing."
  echo "  box $IP left UP for debug (ssh ubuntu@$IP; inspect $WORK). exit 4."
  trap - EXIT; exit 4
fi

echo "== 6. judge =="
if [ -n "$RB_RUN_ID" ]; then
  ( cd "$HERE/../../.." && uv run python posttrainbench/judging/exp_judge_rollback.py "$RB_RUN_ID" ) 2>&1 | tail -4 \
    || echo "ROLLOUT WARN: targeted judge failed for $RB_RUN_ID; saved run can be judged later with the global script."
fi

echo "==== ROLLOUT DONE [$TRAJ / $COND] — terminating box ===="
# cleanup() (EXIT trap) terminates the box.
