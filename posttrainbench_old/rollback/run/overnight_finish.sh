#!/bin/bash
# Unattended overnight finisher: watch one or more boxes; as each rollout
# completes, pull its data to mats-local, VERIFY the pull landed, and only THEN
# terminate that box. Exits when every box is resolved. NOT exp_-prefixed: spends
# no LLM money (pull is free; terminate just stops billing). Does NOT judge (the
# paid step) — run the judge in the morning.
#
#   bash overnight_finish.sh <ip1> [ip2 ...]
#
# Keep the Mac awake + online while this runs (lid OPEN; closing the lid forces
# sleep regardless):   caffeinate -dimsu bash overnight_finish.sh <ips...>
#
# Safety: terminates a box ONLY after confirming its trajectory landed in
# mats-local; otherwise leaves it UP and warns. A box that goes unreachable is
# marked resolved (not terminated) after a few failed checks so the loop exits.
# Hard runtime cap. Independent backstops regardless: each box archives to the FS
# every ~2 min and self-terminates at its launch backstop.
#
# Written for macOS /bin/bash 3.2 (no associative arrays): per-IP state is kept
# in a space-delimited DONE set + indirect FAILS_<ip> counters.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SECRETS="$HOME/.config/ptb/secrets.env"; set -a; source "$SECRETS"; set +a
SSHO="-o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=/tmp/ptb-ssh-%r@%h:%p -o ControlPersist=300"
[ $# -ge 1 ] || { echo "usage: overnight_finish.sh <ip...>"; exit 1; }
IPS=("$@")
RESULTS_DIR="$(cd "$HERE/../.." && python3 -c "from rollback import config as c; print(c.ROLLBACK_RESULTS)")"
MAX_SECONDS="${MAX_SECONDS:-25200}"   # 7h hard cap on this watcher
START=$(date +%s)
DONE_LIST=" "                          # space-delimited set of resolved IPs
echo "watching ${#IPS[@]} box(es): ${IPS[*]}  (Ctrl-C is safe; runs continue on the boxes)"
echo "results land in: $RESULTS_DIR"

terminate_box() {  # $1 = ip
  local iid
  iid=$(curl -s -m20 -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances \
    | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin)['data'] if i.get('ip')=='$1']" 2>/dev/null)
  [ -n "$iid" ] || { echo "    (no instance id for $1 — already gone?)"; return 0; }
  curl -s -m30 -u "$LAMBDA_API_KEY:" -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$iid\"]}" https://cloud.lambda.ai/api/v1/instance-operations/terminate >/dev/null \
    && echo "    terminated $1"
}

while :; do
  if [ $(( $(date +%s) - START )) -gt "$MAX_SECONDS" ]; then
    echo "WARN: hit ${MAX_SECONDS}s cap; stopping. Unresolved boxes left UP (launch backstop still applies)."
    break
  fi
  remaining=0
  for IP in "${IPS[@]}"; do
    case "$DONE_LIST" in *" $IP "*) continue;; esac   # already resolved
    remaining=1
    fv="FAILS_$(echo "$IP" | tr '.:' '__')"
    out=$(ssh $SSHO ubuntu@"$IP" 'W=$(ls -dt /home/ubuntu/backward_*_cut*__* 2>/dev/null | head -1); n=$(basename "$W" 2>/dev/null)
      if [ -f "$W/.agent_done" ]; then echo "DONE $n"
      elif [ -f "$W/score.json" ] && ! pgrep -f run_rollout_on_box >/dev/null; then echo "DONE $n"
      else echo "WAIT $n"; fi' 2>/dev/null); rc=$?
    st=${out%% *}; wd=${out#* }
    if [ $rc -ne 0 ] || [ -z "$st" ]; then          # unreachable
      eval "f=\${$fv:-0}"; f=$((f + 1)); eval "$fv=$f"
      echo "[$(date +%H:%M)] $IP unreachable ($f/5)"
      [ "$f" -ge 5 ] && { echo "    -> assuming gone; NOT terminating, marking resolved"; DONE_LIST="$DONE_LIST$IP "; }
      continue
    fi
    eval "$fv=0"
    if [ "$st" != DONE ]; then echo "[$(date +%H:%M)] $IP WAIT"; continue; fi

    echo "[$(date +%H:%M)] $IP DONE ($wd) -> pulling"
    bash "$HERE/exp_pull_result.sh" "$IP" 2>&1 | tail -3
    if ls "$RESULTS_DIR/$wd"/solve_out_*.txt "$RESULTS_DIR/$wd"/score.json >/dev/null 2>&1; then
      echo "    pull verified ($wd present in mats-local)"
      terminate_box "$IP"
    else
      echo "    WARN: pull did NOT land $wd in mats-local — leaving box UP for manual handling (data is also on the FS)."
    fi
    DONE_LIST="$DONE_LIST$IP "
  done
  [ "$remaining" = 0 ] && { echo "all boxes resolved."; break; }
  sleep 180
done
echo "morning: run the RH judge ->  uv run python judging/exp_judge_rollback.py"
