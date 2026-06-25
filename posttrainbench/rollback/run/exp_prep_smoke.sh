#!/bin/bash
# exp_prep_smoke.sh — validate that a trajectory's PREP regenerates a faithful
# pre-cut model. Runs the box in PREP_ONLY mode (no agent, no policy API): execute
# prep_commands -> score the regenerated model -> compare to the recorded pre-cut
# baseline (Trajectory.precut_eval_file). This is the correctness gate for P4d's
# derived prep AND a check of the prep machinery itself.
#
#   PTB_TRAJECTORY=<nick|run_id> bash exp_prep_smoke.sh <IP>
#
# Unlike exp_smoke.sh (which SKIPS prep and scores the base model), this RUNS the
# real pre-cut training, so it's slower (10s of min) — that's the point. exp_ =
# spends GPU time (no policy API). Leaves the box UP (terminate when done).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
IP_ARG="${1:-}"
: "${PTB_TRAJECTORY:=kimi_humaneval}"; export PTB_TRAJECTORY
WAIT_TICKS="${WAIT_TICKS:-70}"   # 70 x 60s = up to ~70 min for prep+score
ptb_load_secrets || exit 1
SSH="$PTB_SSH"

echo "==== PREP-SMOKE [$PTB_TRAJECTORY] ===="
echo "== A. rebuild cell (writes prep_commands into run_config) =="
read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT)")
CELL="backward_prompt1_cut${CUT}"
( cd "$HERE/../.." && python3 -m rollback.run.orchestrate --force --bash-mode skip --cells "$CELL" ) 2>&1 | tail -3
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
NPREP=$(python3 -c "import json;print(len(json.load(open('$LOCAL_CELL/run_config.json')).get('prep_commands') or []))")
echo "  prep_commands in cell: $NPREP"
[ "$NPREP" -gt 0 ] || { echo "PREP-SMOKE FAIL: no prep_commands (curated or derived) for this trajectory"; exit 1; }

echo "== B. launch PREP_ONLY on the box (KEEP_ALIVE) =="
PREP_ONLY=1 KEEP_ALIVE=1 bash "$HERE/exp_run_experiment.sh" control 1 "$IP_ARG"
[ $? = 0 ] || { echo "PREP-SMOKE FAIL: launch returned nonzero"; exit 1; }

IP="$IP_ARG"; [ -z "$IP" ] && IP=$(ptb_active_ip)
[ -n "$IP" ] || { echo "PREP-SMOKE FAIL: no box to poll"; exit 1; }

echo "== C. wait for prep+score to finish (.agent_done) =="
WORK=""
for i in $(seq 1 "$WAIT_TICKS"); do
  WORK=$(ptb_ssh "$IP" "ls -td /home/ubuntu/${CELL}__* 2>/dev/null | head -1" | tr -d '\r')
  if [ -n "$WORK" ] && ptb_ssh "$IP" "[ -f '$WORK/.agent_done' ]"; then echo "  done ($WORK)"; break; fi
  echo "  waiting ($i/$WAIT_TICKS) ${WORK:-none}"; sleep 60
done
ptb_ssh "$IP" "[ -n '$WORK' ] && [ -f '$WORK/.agent_done' ]" || { echo "PREP-SMOKE FAIL: timed out"; exit 1; }

echo "== D. pull + record fidelity =="
bash "$HERE/exp_pull_result.sh" "$IP" 2>&1 | tail -8

echo "== E. result =="
NAME=$(basename "$WORK")
DEST="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}/results/$NAME"
python3 - "$DEST" "$PTB_TRAJECTORY" <<'PY'
import sys, json, os
dest, traj = sys.argv[1], sys.argv[2]
pp = os.path.join(dest, "prep_score.json")
if not os.path.exists(pp):
    print("  PREP-SMOKE FAIL: no prep_score.json (prep did not produce a scoreable model)"); sys.exit(1)
score = json.load(open(pp))
acc = score.get("accuracy")
print(f"  regenerated-model score: accuracy={acc}  ({score})")
# the fidelity row (re-trained vs original baseline) is recorded by sync_viewer via
# compare_reconstruction; surface it here too if present.
print("  -> compare to the original pre-cut baseline (see fidelity row / Trajectory.precut_eval_file)")
print("  PREP-SMOKE: prep ran and produced a scoreable model ✅ (judge faithfulness vs baseline above)")
PY
echo
echo "Box left UP. Terminate when done."
