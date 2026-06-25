#!/bin/bash
# exp_smoke.sh — FAST end-to-end validation of the whole rollback pipeline on an
# already-running Lambda box. Exercises EVERY stage a real run hits — build,
# rsync, box_prepare preflight, agent resume (ONE real step), scoring, archive,
# pull, viewerize — in minutes for pennies, then gates an explicit per-stage
# PASS/FAIL. The point: a green smoke means a real multi-hour run won't die at
# the finish line (which is exactly where past runs failed).
#
#   PTB_TRAJECTORY=bfcl_codexmax bash exp_smoke.sh [prompt1|prompt2|prompt3] [IP]
#
# WHAT IT DOES NOT VALIDATE (by design, to stay fast):
#   - PREP (pre-cut model regeneration): skipped. Use a no-prep fixture
#     (bfcl_codexmax) so prep is a no-op; validate prep separately/once.
#   - SCORING ACCURACY: it scores the cached BASE model (no trained model exists
#     after one step), so the accuracy is meaningless — only the scoring/save
#     MACHINERY is exercised. The run is viewerized under a DEBUG_ label
#     (hidden from the viewer's real pages, kept for eyeballing).
#
# exp_ = may spend money (one real agent step ~ pennies). Leaves the box UP
# (KEEP_ALIVE) so you can iterate; terminate it yourself when done.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
COND="${1:-prompt1}"
IP_ARG="${2:-}"
: "${PTB_TRAJECTORY:=bfcl_codexmax}"; export PTB_TRAJECTORY   # canonical no-prep fixture
SMOKE_BUDGET_MIN="${SMOKE_BUDGET_MIN:-5}"
WAIT_TICKS="${WAIT_TICKS:-60}"           # 60 x 30s = up to 30 min for .agent_done
if ! ( cd "$HERE/../.." && python3 - "$COND" <<'PY'
import sys
from rollback import contract
sys.exit(0 if contract.valid_condition(sys.argv[1]) else 1)
PY
); then
  echo "condition must be one of: prompt1 prompt2 prompt3"; exit 1
fi
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1
SSH="$PTB_SSH"   # shared opts (single source of truth in ptb_lib.sh)

echo "==== SMOKE [$PTB_TRAJECTORY / $COND] ===="

echo "== A. rebuild cell from config (offline, free — validates the build stage + fresh run_config) =="
read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT)")
CELL="backward_${COND}_cut${CUT}"
( cd "$HERE/../.." && python3 -m rollback.run.orchestrate --force --bash-mode skip --cells "$CELL" ) 2>&1 | tail -4
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
[ -d "$LOCAL_CELL" ] || { echo "SMOKE FAIL: cell not built: $LOCAL_CELL"; exit 1; }

echo "== B. launch on the box (SMOKE=1, KEEP_ALIVE=1, budget ${SMOKE_BUDGET_MIN}m) =="
SMOKE=1 KEEP_ALIVE=1 bash "$HERE/exp_run_experiment.sh" "$COND" "$SMOKE_BUDGET_MIN" "$IP_ARG"
RC=$?
[ "$RC" = 0 ] || { echo "SMOKE FAIL: launch (exp_run_experiment.sh) returned $RC — box left up for debugging"; exit 1; }

echo "== C. resolve box ip for polling =="
IP="$IP_ARG"
[ -z "$IP" ] && IP=$(ptb_active_ip)
[ -n "$IP" ] || { echo "SMOKE FAIL: can't resolve a single running box to poll; pass IP"; exit 1; }
echo "  box ip=$IP"

echo "== D. wait for the run to hit its end-condition (.agent_done) =="
WORK=""
for i in $(seq 1 "$WAIT_TICKS"); do
  WORK=$($SSH ubuntu@$IP "ls -td /home/ubuntu/${CELL}__* 2>/dev/null | head -1" | tr -d '\r')
  if [ -n "$WORK" ] && $SSH ubuntu@$IP "[ -f '$WORK/.agent_done' ]"; then
    echo "  .agent_done seen ($WORK)"; break
  fi
  echo "  waiting ($i/$WAIT_TICKS) work=${WORK:-none}"; sleep 30
done
$SSH ubuntu@$IP "[ -n '$WORK' ] && [ -f '$WORK/.agent_done' ]" \
  || { echo "SMOKE FAIL: timed out waiting for .agent_done (box left up; check $FS/run_live_*.log)"; exit 1; }

echo "== E. pull + viewerize (validates the save + viewer stages) =="
bash "$HERE/exp_pull_result.sh" "$IP" 2>&1 | tail -10

echo "== F. verify saved artifacts =="
NAME=$(basename "$WORK")
LOCAL_ROLLBACK="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
DEST="$LOCAL_ROLLBACK/results/$NAME"
VIEWER_DATA="$LOCAL_ROLLBACK/viewer_data"
python3 - "$DEST" "$VIEWER_DATA" "$NAME" <<'PY'
import sys, json, glob, os
dest, vdir, name = sys.argv[1], sys.argv[2], sys.argv[3]
checks, ok = [], True
def chk(label, cond, detail=""):
    global ok
    ok = ok and cond
    checks.append((("PASS" if cond else "FAIL"), label, detail))

rc = os.path.join(dest, "run_config.json")
have_rc = os.path.exists(rc)
cfg = json.load(open(rc)) if have_rc else {}
chk("run_config.json saved", have_rc)
chk("marked smoke", bool(cfg.get("smoke")), "run_config.smoke")

solves = glob.glob(os.path.join(dest, "solve_out_*.txt"))
chk("solve_out present", bool(solves))

sr = os.path.join(dest, "smoke_report.json")
rep = json.load(open(sr)) if os.path.exists(sr) else {}
chk("smoke_report.json saved", bool(rep))
chk("agent took a step", bool(rep.get("agent_took_step")),
    f"{rep.get('agent_out_bytes')} bytes in {rep.get('agent_seconds')}s")
chk("scoring machinery ran", bool(rep.get("score_ran")),
    f"model_scored={rep.get('model_scored')}")

# viewerize stage: a DEBUG_ viewer file for THIS result dir must exist
vd_hit = None
for f in glob.glob(os.path.join(vdir, "rollback_DEBUG_*.json")):
    try: m = json.load(open(f)).get("meta", {})
    except Exception: continue
    if m.get("result_dir") == name:
        vd_hit = os.path.basename(f); break
chk("viewerized (DEBUG_, quarantined)", vd_hit is not None, vd_hit or "no DEBUG file for this run")

print("\n  stage                                  result")
print("  " + "-"*52)
for status, label, detail in checks:
    mark = "✓" if status == "PASS" else "✗"
    print(f"  {mark} {label:<36} {status}" + (f"  ({detail})" if detail else ""))
if rep.get("caveat"):
    print(f"\n  caveat: {rep['caveat']}")
print("\n  ==== SMOKE " + ("PASS ✅" if ok else "FAIL ❌") + " ====")
sys.exit(0 if ok else 1)
PY
SMOKE_RC=$?
echo
[ "$SMOKE_RC" = 0 ] && echo "Pipeline validated end-to-end. Box left UP — terminate it when done iterating." \
  || echo "Smoke failed above. Box left UP for debugging (run_live log on the box: <filesystem>/run_live_*.log)."
exit $SMOKE_RC
