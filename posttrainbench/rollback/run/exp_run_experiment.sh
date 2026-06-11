#!/bin/bash
# ONE-COMMAND experiment launcher (run on the Mac). exp_ = spends money.
#
#   bash exp_run_experiment.sh <control|treatment> [BUDGET_MIN] [IP]
#
# Does the whole pipeline robustly, with minimal SSH connections (Lambda
# rate-limits aggressive reconnects):
#   1. resolve the target cell from the configured trajectory (PTB_TRAJECTORY)
#   2. auto-discover the running Lambda box (IP + instance-id) via the API
#      (or use the IP arg)
#   3. PREFLIGHT (Mac side): OpenRouter key balance — abort if too low
#   4. one rsync: push the pristine cell + box scripts to the filesystem
#   5. one bundled SSH: deploy secrets+cred, run box_prepare (installs
#      apptainer if missing + verifies container/model/cell/GPU). Only if it
#      reports READY does it copy the cell to a work dir and launch the run +
#      the archive-and-self-terminate watcher.
#   6. it does NOT start the agent unless every preflight passes (so a
#      misconfig fails loudly in seconds, costing $0, and leaves the box up).
#
# After launch the box runs autonomously (archives to filesystem, terminates
# on completion). Use exp_pull_result.sh later to fetch the trajectory.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"            # .../mats
COND="${1:?usage: exp_run_experiment.sh <control|treatment> [BUDGET_MIN] [IP]}"
BUDGET_MIN="${2:-120}"
IP_ARG="${3:-}"
SECRETS="$HOME/.config/ptb/secrets.env"
MIN_BALANCE="${MIN_BALANCE:-15}"               # abort if OpenRouter remaining < this
KEEP_ALIVE="${KEEP_ALIVE:-0}"                  # 1 = keep box up after run (for more seeds)
BACKSTOP_SEC="${BACKSTOP_SEC:-14400}"          # KEEP_ALIVE overnight-safety terminate (4h)
[ "$COND" = control ] || [ "$COND" = treatment ] || { echo "condition must be control|treatment"; exit 1; }
[ -f "$SECRETS" ] || { echo "no $SECRETS"; exit 1; }
set -a; source "$SECRETS"; set +a

echo "== 1. resolve cell from configured trajectory =="
read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT)")
CELL="backward_${COND}_cut${CUT}"
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
[ -d "$LOCAL_CELL" ] || { echo "FAIL: cell not built locally: $LOCAL_CELL  (run: python -m rollback.run.orchestrate --bash-mode skip)"; exit 1; }
echo "  trajectory=$RUN_NAME cell=$CELL"

echo "== 2. discover the running Lambda box =="
if [ -n "$IP_ARG" ]; then IP="$IP_ARG"; else
  IP=$(curl -s -m 20 -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances | python3 -c "import sys,json; d=[i for i in json.load(sys.stdin).get('data',[]) if i.get('status')=='active']; print(d[0]['ip'] if len(d)==1 else '')")
fi
[ -n "$IP" ] || { echo "FAIL: couldn't auto-pick a single running box; pass IP as 3rd arg"; exit 1; }
IID=$(curl -s -m 20 -u "$LAMBDA_API_KEY:" https://cloud.lambda.ai/api/v1/instances | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin)['data'] if i.get('ip')=='$IP']")
[ -n "$IID" ] || { echo "FAIL: no instance with ip $IP"; exit 1; }
echo "  box ip=$IP id=$IID"

echo "== 3. preflight OpenRouter balance =="
REM=$(curl -s -m 15 -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/credits | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',{}); tc=d.get('total_credits'); print((tc-d.get('total_usage',0)) if isinstance(tc,(int,float)) else 99999)" 2>/dev/null || echo 0)
echo "  OpenRouter remaining: \$$REM (min required \$$MIN_BALANCE)"
awk "BEGIN{exit !($REM < $MIN_BALANCE)}" && { echo "FAIL: OpenRouter balance too low (\$$REM < \$$MIN_BALANCE) — top up or switch key"; exit 1; }

echo "== 4. rsync cell + scripts to the box =="
SSH="ssh -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new"
# detect FS name on the box (one quick read)
FS=$($SSH ubuntu@$IP 'for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && echo "${d%/}" && break; done; [ -z "$FS" ] && for d in /lambda/nfs/*/; do echo "${d%/}"; break; done' 2>/dev/null | head -1)
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on box (region mismatch?)"; exit 1; }
echo "  filesystem on box: $FS"
rsync -az -e "$SSH" --exclude tmp --exclude '.agent_done' "$LOCAL_CELL" "ubuntu@$IP:$FS/cells/" 2>&1 | tail -1
rsync -az -e "$SSH" "$HERE"/{box_prepare.sh,run_rollout_on_box.sh,archive_and_terminate.sh} "ubuntu@$IP:$FS/" 2>&1 | tail -1

echo "== 5. deploy secrets+cred, preflight, launch (one connection) =="
SEC_B64=$(grep -E '^(OPENROUTER_API_KEY|OPENAI_API_KEY|HF_TOKEN)=' "$SECRETS" | base64)
CRED_B64=$( { grep '^LAMBDA_API_KEY=' "$SECRETS"; echo "INSTANCE_ID=$IID"; } | base64)
$SSH ubuntu@$IP "bash -s" <<EOF
set -uo pipefail
umask 077; echo '$SEC_B64' | base64 -d > /home/ubuntu/.ptb_secrets; echo '$CRED_B64' | base64 -d > /home/ubuntu/.ptb_lambda; umask 022
chmod +x $FS/box_prepare.sh $FS/run_rollout_on_box.sh $FS/archive_and_terminate.sh
PREP=\$(bash $FS/box_prepare.sh $CELL)
echo "\$PREP"
echo "\$PREP" | grep -q "PREP: READY" || { echo "ABORT: preflight failed, not launching (box left up for debugging)"; exit 2; }
TS=\$(date +%s); WORK=/home/ubuntu/${CELL}__\$TS
cp -r $FS/cells/$CELL \$WORK
nohup env BUDGET_MIN=$BUDGET_MIN bash $FS/run_rollout_on_box.sh \$WORK > $FS/run_live_\$TS.log 2>&1 < /dev/null &
echo "ROLLOUT_PID \$!"
nohup env KEEP_ALIVE=$KEEP_ALIVE bash $FS/archive_and_terminate.sh \$WORK > /dev/null 2>&1 < /dev/null &
echo "ARCHIVER_PID \$!"
# KEEP_ALIVE: box stays up after the run for more seeds; arm an absolute
# backstop so a forgotten box still can't bill overnight.
if [ "$KEEP_ALIVE" = 1 ]; then
  nohup bash -c 'source /home/ubuntu/.ptb_lambda; sleep '"$BACKSTOP_SEC"'; curl -s -m30 -u "\$LAMBDA_API_KEY:" -H "Content-Type: application/json" -d "{\"instance_ids\":[\"\$INSTANCE_ID\"]}" https://cloud.lambda.ai/api/v1/instance-operations/terminate' > /dev/null 2>&1 < /dev/null &
  echo "BACKSTOP_ARMED ${BACKSTOP_SEC}s"
fi
echo "WORK \$WORK"
echo "LAUNCH_OK"
EOF
echo "== launch dispatched. $( [ "$KEEP_ALIVE" = 1 ] && echo "box KEPT ALIVE for more seeds (backstop ${BACKSTOP_SEC}s; terminate manually when done)" || echo "box self-terminates on completion" ) =="
echo "== pull/monitor: bash rollback/run/exp_pull_result.sh =="
