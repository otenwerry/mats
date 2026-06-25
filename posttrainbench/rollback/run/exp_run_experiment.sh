#!/bin/bash
# ONE-COMMAND experiment launcher (run on the Mac). exp_ = spends money.
#
#   bash exp_run_experiment.sh <prompt1|prompt2|prompt3> [BUDGET_MIN] [IP]
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
COND="${1:?usage: exp_run_experiment.sh <prompt1|prompt2|prompt3> [BUDGET_MIN] [IP]}"
BUDGET_MIN="${2:-}"
IP_ARG="${3:-}"
source "$HERE/ptb_lib.sh"          # shared SSH/Lambda/fs helpers (one source of truth)
SECRETS="$PTB_SECRETS"
MIN_BALANCE="${MIN_BALANCE:-15}"               # abort if OpenRouter remaining < this
KEEP_ALIVE="${KEEP_ALIVE:-0}"                  # 1 = keep box up after run (for more seeds)
PREP_ONLY="${PREP_ONLY:-0}"                    # 1 = prep+score the clean model only (recon-fidelity check; no agent, no paid API)
SMOKE="${SMOKE:-0}"                            # 1 = fast plumbing smoke (1 step -> forced end-condition, base-model scoring); see exp_smoke.sh
PTB_ROLLOUT_SIF="${PTB_ROLLOUT_SIF:-}"         # optional staged-image override, e.g. $FS/containers/codex_next.sif
[ "$SMOKE" = 1 ] && BUDGET_MIN="${2:-5}"        # smoke caps the agent tight by default
BACKSTOP_SEC="${BACKSTOP_SEC:-14400}"          # KEEP_ALIVE overnight-safety terminate (4h)
if ! ( cd "$HERE/../.." && python3 - "$COND" <<'PY'
import sys
from rollback import contract
sys.exit(0 if contract.valid_condition(sys.argv[1]) else 1)
PY
); then
  echo "condition must be one of: prompt1 prompt2 prompt3"; exit 1
fi
ptb_load_secrets || exit 1

echo "== 1. resolve cell from configured trajectory =="
read RUN_NAME CUT < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT)")
CELL="backward_${COND}_cut${CUT}"
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
[ -d "$LOCAL_CELL" ] || { echo "FAIL: cell not built locally: $LOCAL_CELL  (run: python -m rollback.run.orchestrate --bash-mode skip)"; exit 1; }
echo "  trajectory=$RUN_NAME cell=$CELL"

# The resume prompt has ONE source of truth: config.CONTROL_STEM /
# INTERVENTION_REMINDER. A cell bakes a COPY into run_config.json at build time,
# which can go stale if the config changes. So re-materialize it from config AT
# LAUNCH — the cell never carries a stale prompt. OpenCode delivers it on stdin;
# Claude/Codex deliver it as the explicit resume prompt.
cd "$HERE/../.." && python3 - "$LOCAL_CELL/run_config.json" "$COND" "$SMOKE" <<'PY'
import sys, json
from rollback import contract
p = sys.argv[1]
cfg = json.load(open(p))
cfg["condition"] = sys.argv[2]
cfg["intervention"] = contract.intervention_text(sys.argv[2])
cfg["resume_prompt"] = contract.resume_prompt(sys.argv[2])
cfg["prompt_contract"] = contract.prompt_contract(sys.argv[2], cfg.get("scaffold", ""))
# stamp smoke so sync_viewer quarantines the run under a DEBUG_ label
cfg["smoke"] = bool(int(sys.argv[3]))
json.dump(cfg, open(p, "w"), indent=1)
print(f"  resume_prompt refreshed from config ({sys.argv[2]}); smoke={cfg['smoke']}")
PY

echo "== 2. discover the running Lambda box =="
if [ -n "$IP_ARG" ]; then IP="$IP_ARG"; else
  IP=$(ptb_active_ip)
fi
[ -n "$IP" ] || { echo "FAIL: couldn't auto-pick a single running box; pass IP as 3rd arg"; exit 1; }
IID=$(ptb_instance_id "$IP")
[ -n "$IID" ] || { echo "FAIL: no instance with ip $IP"; exit 1; }
echo "  box ip=$IP id=$IID"

echo "== 3. preflight OpenRouter balance =="
REM=$(curl -s -m 15 -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/credits | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',{}); tc=d.get('total_credits'); print((tc-d.get('total_usage',0)) if isinstance(tc,(int,float)) else 99999)" 2>/dev/null || echo 0)
echo "  OpenRouter remaining: \$$REM (min required \$$MIN_BALANCE)"
awk "BEGIN{exit !($REM < $MIN_BALANCE)}" && { echo "FAIL: OpenRouter balance too low (\$$REM < \$$MIN_BALANCE) — top up or switch key"; exit 1; }

echo "== 4. rsync cell + scripts to the box =="
# Multiplex ALL of this launch's ssh/rsync over ONE connection (ControlMaster):
# a launch otherwise opens ~4 connections in seconds (fs-detect + 2 rsyncs +
# launch), and that burst trips Lambda's per-IP SSH rate-limit (banner-exchange
# timeouts). With a shared master socket it's a single handshake.
SSH="$PTB_SSH"   # shared opts (single source of truth in ptb_lib.sh)
# detect FS name on the box (one quick read)
FS=$(ptb_fs_detect_remote "$IP")
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on box (region mismatch?)"; exit 1; }
echo "  filesystem on box: $FS"
[ -n "$PTB_ROLLOUT_SIF" ] && echo "  using staged container override: $PTB_ROLLOUT_SIF"
rsync -az -e "$SSH" --exclude tmp --exclude '.agent_done' "$LOCAL_CELL" "ubuntu@$IP:$FS/cells/" 2>&1 | tail -1
rsync -az -e "$SSH" "$HERE"/{box_prepare.sh,run_rollout_on_box.sh,archive_and_terminate.sh} "ubuntu@$IP:$FS/" 2>&1 | tail -1

echo "== 5. deploy secrets+cred, preflight, launch (one connection) =="
SEC_B64=$(grep -E '^(OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN|DASHSCOPE_API_KEY|HF_TOKEN)=' "$SECRETS" | base64)
CRED_B64=$( { grep '^LAMBDA_API_KEY=' "$SECRETS"; echo "INSTANCE_ID=$IID"; } | base64)
# Codex (codex_non_api) authenticates the policy via a ChatGPT SUBSCRIPTION, not
# an env key — so deploy its auth.json FILE (generate it once with `codex login`;
# default path below). Empty/absent => codex cells will fail the oauth check (we
# don't run codex yet). FILL IN: put your codex auth at ~/.config/ptb/codex_auth.json
CODEX_AUTH_LOCAL="${PTB_CODEX_AUTH:-$HOME/.config/ptb/codex_auth.json}"
CODEX_AUTH_B64=""; [ -f "$CODEX_AUTH_LOCAL" ] && CODEX_AUTH_B64=$(base64 < "$CODEX_AUTH_LOCAL") && echo "  deploying codex auth.json from $CODEX_AUTH_LOCAL"
$SSH ubuntu@$IP "bash -s" <<EOF
set -uo pipefail
umask 077; echo '$SEC_B64' | base64 -d > /home/ubuntu/.ptb_secrets; echo '$CRED_B64' | base64 -d > /home/ubuntu/.ptb_lambda
[ -n "$CODEX_AUTH_B64" ] && echo '$CODEX_AUTH_B64' | base64 -d > /home/ubuntu/.ptb_codex_auth.json
umask 022
chmod +x $FS/box_prepare.sh $FS/run_rollout_on_box.sh $FS/archive_and_terminate.sh
# Run the long-lived scripts from LOCAL disk, not the NFS filesystem: during a
# 100min+ agent step, bash's NFS read of the script's next line goes stale
# ("Stale file handle"), wedging the script right after the agent and skipping
# scoring + the done-marker. (The container/cache stay on \$FS — only the script
# files, which bash reads line-by-line for hours, need to be local.)
cp $FS/run_rollout_on_box.sh $FS/archive_and_terminate.sh /home/ubuntu/
chmod +x /home/ubuntu/run_rollout_on_box.sh /home/ubuntu/archive_and_terminate.sh
PREP=\$(bash $FS/box_prepare.sh $CELL)
echo "\$PREP"
echo "\$PREP" | grep -q "PREP: READY" || { echo "ABORT: preflight failed, not launching (box left up for debugging)"; exit 2; }
TS=\$(date +%s); WORK=/home/ubuntu/${CELL}__\$TS
cp -r $FS/cells/$CELL \$WORK
nohup env BUDGET_MIN="$BUDGET_MIN" PREP_ONLY=$PREP_ONLY SMOKE=$SMOKE SIF="$PTB_ROLLOUT_SIF" bash /home/ubuntu/run_rollout_on_box.sh \$WORK > $FS/run_live_\$TS.log 2>&1 < /dev/null &
echo "ROLLOUT_PID \$!"
nohup env KEEP_ALIVE=$KEEP_ALIVE bash /home/ubuntu/archive_and_terminate.sh \$WORK > /dev/null 2>&1 < /dev/null &
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
