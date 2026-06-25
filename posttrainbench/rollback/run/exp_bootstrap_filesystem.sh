#!/bin/bash
# exp_bootstrap_filesystem.sh — stage a Lambda persistent filesystem so it can
# run rollback rollouts: build standard.sif on it + download every base model
# the trajectories fine-tune. exp_ = SPENDS MONEY (one GPU box for ~75 min).
#
# Run ONCE per filesystem/region (the artifacts persist across box terminates).
# We have two filesystems — west-filesystem (us-west-3, PCIe; already staged) and
# south-filesystem (us-south-2, SXM) — and want both ready so we can launch in
# whichever region has H100 capacity at the moment.
#
#   bash exp_bootstrap_filesystem.sh [MODEL ...]
# Region/filesystem/instance selection comes from ptb_lib env (single source):
#   PTB_FS_NAME=south-filesystem PTB_H100_VARIANT=sxm5 bash exp_bootstrap_filesystem.sh
#
# Defaults to the 4 base models the current RH trajectories use. gemma-3-4b-pt is
# GATED — needs HF_TOKEN (in the secrets file) from an account that accepted the
# terms; the others are ungated.
#
# Robust to a dropped SSH: the build+downloads run under nohup ON THE BOX (writing
# a .bootstrap_all_done marker); we poll for it, then self-terminate the box. The
# filesystem keeps the .sif + hf_cache. A trap terminates the box on any exit.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1

MODELS=("$@")
[ "${#MODELS[@]}" -gt 0 ] || MODELS=(
  "Qwen/Qwen3-1.7B-Base"
  "Qwen/Qwen3-4B-Base"
  "google/gemma-3-4b-pt"
  "HuggingFaceTB/SmolLM3-3B-Base"
)
POLL_MIN="${POLL_MIN:-150}"          # max minutes to wait for the build+downloads

echo "==== BOOTSTRAP FILESYSTEM ===="
echo "  fs=${PTB_FS_NAME:-<auto>} variant=${PTB_H100_VARIANT:-<any>} models=${MODELS[*]}"

LAUNCHED_ID=""
cleanup() { [ -n "$LAUNCHED_ID" ] && { echo "== teardown: terminating $LAUNCHED_ID =="; ptb_terminate "$LAUNCHED_ID" >/dev/null 2>&1; }; }
trap cleanup EXIT

echo "== 1. launch a 1xH100 box (cheapest; build is CPU-bound) =="
IP=$(ptb_launch_box 1 "ptb-bootstrap-$(date +%m%d-%H%M%S)") || { echo "FAIL: box launch"; exit 1; }
LAUNCHED_ID=$(ptb_instance_id "$IP")
echo "  box $IP (id $LAUNCHED_ID)"

SSH="$PTB_SSH"
FS=$(ptb_fs_detect_remote "$IP")
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on box (region mismatch?)"; exit 1; }
echo "  filesystem on box: $FS"

echo "== 2. deploy secrets (HF_TOKEN for the gated model) + cred + scripts =="
SECRETS="$PTB_SECRETS"
SEC_B64=$(grep -E '^(HF_TOKEN|OPENROUTER_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)=' "$SECRETS" | base64)
CRED_B64=$( { grep '^LAMBDA_API_KEY=' "$SECRETS"; echo "INSTANCE_ID=$LAUNCHED_ID"; } | base64)
$SSH "ubuntu@$IP" "umask 077; echo '$SEC_B64' | base64 -d > /home/ubuntu/.ptb_secrets; echo '$CRED_B64' | base64 -d > /home/ubuntu/.ptb_lambda; umask 022"
rsync -az -e "$SSH" "$HERE/bootstrap_box.sh" "$HERE/standard.def" "ubuntu@$IP:$FS/" 2>&1 | tail -1

# Box-side backstop: if THIS Mac process is killed (SIGKILL skips our EXIT trap),
# the box must still not bill forever. Arm an absolute self-terminate well past
# the expected build time. (Normal path: step 5 terminates first, before this fires.)
BACKSTOP_SEC=$(( (POLL_MIN + 45) * 60 ))
$SSH "ubuntu@$IP" "nohup bash -c 'source /home/ubuntu/.ptb_lambda; sleep $BACKSTOP_SEC; curl -s -m30 -u \"\$LAMBDA_API_KEY:\" -H \"Content-Type: application/json\" -d \"{\\\"instance_ids\\\":[\\\"\$INSTANCE_ID\\\"]}\" https://cloud.lambda.ai/api/v1/instance-operations/terminate' > /dev/null 2>&1 < /dev/null & echo BACKSTOP_ARMED ${BACKSTOP_SEC}s"

echo "== 3. build .sif + stage models under nohup on the box =="
# stage_all.sh runs bootstrap_box.sh once per model: the FIRST call builds
# standard.sif (~60 min) and downloads model 1; the rest skip the build (idempotent)
# and just download. A UNIQUE (timestamped) done-marker lets us survive a dropped
# SSH without a stale marker from a prior bootstrap causing a false "done".
MODELS_CSV="$(IFS='|'; echo "${MODELS[*]}")"
MARKER="$FS/.bootstrap_done_$(date +%s)"
$SSH "ubuntu@$IP" "cat > /home/ubuntu/stage_all.sh" <<EOF
#!/bin/bash
set -uo pipefail
FS="$FS"
IFS='|' read -r -a MS <<< "$MODELS_CSV"
for m in "\${MS[@]}"; do
  echo "===== staging \$m  \$(date -u) ====="
  MODEL="\$m" bash "\$FS/bootstrap_box.sh" || echo "stage \$m: rc=\$?"
done
touch "$MARKER"
echo "===== ALL DONE \$(date -u) ====="
EOF
$SSH "ubuntu@$IP" "rm -f $MARKER; chmod +x /home/ubuntu/stage_all.sh; nohup bash /home/ubuntu/stage_all.sh > $FS/bootstrap_all.log 2>&1 < /dev/null & echo STAGE_PID \$!"

echo "== 4. poll for completion (up to ${POLL_MIN}m) =="
sleep 30   # let stage_all start before the first marker check
DONE=0
for i in $(seq 1 "$POLL_MIN"); do
  if $SSH "ubuntu@$IP" "[ -f $MARKER ]" 2>/dev/null; then DONE=1; break; fi
  if [ $(( i % 5 )) -eq 0 ]; then
    echo "  [$i/${POLL_MIN}m $(date -u +%H:%M)] $($SSH "ubuntu@$IP" "tail -1 $FS/bootstrap_all.log 2>/dev/null" | tr -d '\r')"
  fi
  sleep 60
done

echo "== 5. verify staged assets =="
$SSH "ubuntu@$IP" "
  echo -n 'standard.sif: '; [ -f $FS/containers/standard.sif ] && du -h $FS/containers/standard.sif | cut -f1 || echo MISSING
  for m in ${MODELS[*]}; do
    d=models--\$(echo \$m | sed 's#/#--#g')
    echo -n \"\$m: \"; [ -d $FS/hf_cache/hub/\$d ] && echo cached || echo MISSING
  done
" 2>&1

if [ "$DONE" = 1 ]; then
  echo "==== BOOTSTRAP DONE (filesystem $FS staged; terminating box) ===="
else
  echo "==== BOOTSTRAP TIMED OUT after ${POLL_MIN}m — check $FS/bootstrap_all.log; terminating box anyway (FS keeps partial progress) ===="
fi
