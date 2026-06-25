#!/bin/bash
# box_status.sh — READ-ONLY live status of every rollback GPU box. FREE (no agent
# or API spend; just the Lambda list API + ssh reads). Run anytime, from ANY
# session, to see what every box is doing — the persistent cross-session window
# into in-flight work (background launcher processes do NOT survive a session;
# the boxes + this tool do).
#
#   bash rollback/run/box_status.sh
#
# For each active Lambda instance it shows: ip / region / GPUs, the trajectory +
# conditions (from the running/ registry, matched by ip), the current on-box
# STAGE (prep / agent / scoring / done), and whether results are sitting on the
# filesystem ready to pull (a box whose launcher died finishes + archives but is
# NOT auto-pulled — pull it with exp_pull_result.sh / a sweep before it
# self-terminates at its 12h backstop).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1
RUN_DIR="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}/running"

echo "==== ROLLBACK BOX STATUS  $(date -u '+%Y-%m-%d %H:%M:%SZ') ===="

# registry: ip -> "run_name [conds]" (from the per-entry running/ dir)
reg_for_ip() {
  python3 - "$RUN_DIR" "$1" <<'PY' 2>/dev/null
import json, sys, glob, os
rundir, ip = sys.argv[1], sys.argv[2]
conds, rn = [], ""
for p in glob.glob(os.path.join(rundir, "*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    if d.get("ip") == ip:
        rn = d.get("run_name") or (d.get("source_trajectory","").split("__",1)[-1])
        if d.get("condition"): conds.append(d["condition"])
print(f"{rn}  [{','.join(sorted(set(conds))) or '?'}]" if rn else "(not in registry)")
PY
}

INSTANCES=$(curl -s -m 25 -u "$LAMBDA_API_KEY:" "$LAMBDA_API/instances" \
  | python3 -c "import sys,json;[print(i.get('ip') or '-', (i.get('region') or {}).get('name'), (i.get('instance_type') or {}).get('name'), i.get('name')) for i in json.load(sys.stdin).get('data',[]) if i.get('status')=='active']")
[ -n "$INSTANCES" ] || { echo "no active instances"; exit 0; }

printf '%s\n' "$INSTANCES" | while read -r IP REGION ITYPE NAME; do
  echo
  echo "── $IP  ($REGION  $ITYPE)  name=$NAME"
  echo "   registry: $(reg_for_ip "$IP")"
  # one quick ssh: locate the run dir, report stage + pullable.
  # NOTE: </dev/null is REQUIRED — without it ssh inherits (and slurps) this
  # `while read` loop's piped stdin, so only the FIRST instance is processed and
  # every other box silently vanishes from the report. (Can't fix via -n in
  # ptb_ssh: ptb_rsync reuses the same opts and rsync needs ssh stdin/stdout.)
  ptb_ssh "$IP" '
    FS=""; for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
    W=$(ls -td /home/ubuntu/batch_* /home/ubuntu/backward_* 2>/dev/null | head -1)
    if [ -z "$W" ]; then echo "   stage: no run dir (idle / booting / bootstrap box)"; exit 0; fi
    echo "   work: $W"
    if [ -f "$W/.batch_done" ]; then ST="BATCH DONE"
    elif ls "$W"/*/.agent_done >/dev/null 2>&1 || [ -f "$W/.agent_done" ]; then ST="agent done; scoring/finishing"
    elif pgrep -f "agent_solve.sh|opencode run|codex .* resume|claude .* --resume" >/dev/null 2>&1; then ST="AGENT RUNNING"
    elif pgrep -f "prep_run.sh|train.py|finetune.py|evaluate.py" >/dev/null 2>&1; then ST="PREP/SCORING"
    else ST="active (stage unclear)"; fi
    echo "   stage: $ST"
    # THIS box result ready to pull? (results/ is shared across boxes on the FS,
    # so only report the dir matching this box-local work dir.)
    if [ -n "$FS" ]; then
      n=$(basename "$W"); r="$FS/results/$n/"
      if [ -f "${r}.run_complete" ]; then
        [ -f "${r}.pulled" ] && echo "   result: $n (pulled)" || echo "   result: $n  *** READY TO PULL ***"
      fi
    fi
    # newest live log line
    L=$(ls -t "$W"/*.log "$W"/*/*.log 2>/dev/null | head -1)
    [ -n "$L" ] && echo "   log[$(basename "$L")]: $(tail -1 "$L" 2>/dev/null | cut -c1-160)"
  ' </dev/null 2>/dev/null || echo "   (ssh unreachable — booting or wrapper crashed)"
done
echo
echo "==== registry (running/ entries; persists across sessions) ===="
ls "$RUN_DIR"/*.json 2>/dev/null | xargs -n1 basename 2>/dev/null | sed 's/__[0-9]*\.json//' | sort | uniq -c || echo "  (none)"
