#!/bin/bash
# Autonomous end-to-end runner for a set of conditions on ONE box. exp_ = spends
# money (agent tokens via OpenRouter + the Anthropic judge).
#
#   bash exp_autorun.sh <IP> [conditions...]      # default: control treatment
#
# For each condition, sequentially on the same box:
#   launch prep -> agent -> score (via exp_run_experiment, KEEP_ALIVE),
#   wait for it to finish, then pull + viewerize (via exp_pull_result).
# After each condition: judge only the rollback run produced by that condition.
# Does NOT terminate the box — it's left alive (a 12h backstop guards against a
# forgotten box, but terminate manually once you trust the results). Reload the
# viewer when this finishes to see the new, scored, judged trajectories.
#
# Pre-req: run smoke_test.sh first (it rebuilds standard.sif with the pinned
# transformers). This script ABORTS if the container isn't rebuilt, to avoid
# wasting a pair of runs on the broken-eval container.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTB="$(cd "$HERE/../.." && pwd)"
SECRETS="$HOME/.config/ptb/secrets.env"; set -a; source "$SECRETS"; set +a
IP="${1:?usage: exp_autorun.sh <IP> [conditions...]}"; shift || true
CONDS="${*:-control treatment}"
BUDGET_MIN="${BUDGET_MIN:-}"
SSH="ssh -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=/tmp/ptb-ssh-%r@%h:%p -o ControlPersist=300"

read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; print(c.TRAJECTORY.run_name, c.CUT_BEFORE_EVENT)")
echo "== autorun: trajectory=$RUN_NAME cut=$CUT conditions=[$CONDS] box=$IP =="

# guard: refuse to run on a container that still has the eval-breaking transformers
FS=$($SSH ubuntu@$IP 'for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && echo "${d%/}" && break; done' 2>/dev/null | head -1)
[ -n "$FS" ] || { echo "ABORT: no container filesystem on box"; exit 1; }
# a fresh box that mounts the shared filesystem HAS the rebuilt container but no
# apptainer yet — install it so the guard (and the run) can use the container.
echo "  ensuring apptainer on box..."
$SSH ubuntu@$IP 'command -v apptainer >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common && sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apptainer; }' 2>&1 | tail -1
TV=$($SSH ubuntu@$IP "apptainer exec -c $FS/containers/standard.sif python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null" | tr -d '\r')
if [ "$TV" != "4.57.3" ]; then
  echo "ABORT: container transformers=$TV (need 4.57.3). Run smoke_test.sh first to rebuild standard.sif."
  exit 1
fi
echo "  container transformers $TV OK"

poll_done() {  # $1=condition ; succeeds when that run's .agent_done appears
  local cond="$1" tries=0
  while [ "$tries" -lt 200 ]; do          # ~5h cap (90s * 200)
    local s
    s=$($SSH ubuntu@$IP "W=\$(ls -dt /home/ubuntu/backward_${cond}_cut*__* 2>/dev/null | head -1); [ -n \"\$W\" ] && [ -f \"\$W/.agent_done\" ] && echo DONE || echo WAIT" 2>/dev/null | tr -d '\r')
    [ "$s" = "DONE" ] && return 0
    sleep 90; tries=$((tries + 1))
  done
  return 1
}

viewer_run_id_for_result() {  # $1=result_dir
  ( cd "$PTB" && uv run python - "$1" <<'PY'
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
}

for cond in $CONDS; do
  echo; echo "################ $cond: LAUNCH ################"
  KEEP_ALIVE=1 BACKSTOP_SEC="${BACKSTOP_SEC:-43200}" bash "$HERE/exp_run_experiment.sh" "$cond" "$BUDGET_MIN" "$IP" \
    || { echo "ABORT: launch failed for $cond"; exit 1; }
  echo "################ $cond: WAITING (prep -> agent -> score) ################"
  poll_done "$cond" || { echo "ABORT: timed out waiting for $cond (box left alive for inspection)"; exit 1; }
  WORK=$($SSH ubuntu@$IP "ls -dt /home/ubuntu/backward_${cond}_cut*__* 2>/dev/null | head -1" | tr -d '\r')
  RESULT_DIR=$(basename "$WORK")
  echo "################ $cond: COMPLETE — pulling + viewerizing ################"
  bash "$HERE/exp_pull_result.sh" "$IP"
  RB_RUN_ID=$(viewer_run_id_for_result "$RESULT_DIR")
  if [ -n "$RB_RUN_ID" ]; then
    echo "################ $cond: JUDGING $RB_RUN_ID ################"
    ( cd "$PTB/.." && uv run python posttrainbench/judging/exp_judge_rollback.py "$RB_RUN_ID" ) \
      || echo "WARN: targeted judge failed for $RB_RUN_ID; saved run can be judged later with the global script."
  else
    echo "WARN: no viewer row found for $RESULT_DIR; skipping targeted judge"
  fi
done

echo; echo "################ DONE ################"
echo "Box $IP LEFT ALIVE (terminate manually when confident; 12h backstop armed)."
echo "Reload the viewer — the new scored + judged runs are pinned at the top."
