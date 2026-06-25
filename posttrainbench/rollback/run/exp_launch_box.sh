#!/bin/bash
# exp_launch_box.sh — programmatically launch a Lambda GPU box for rollback runs,
# attaching our persistent filesystem (so the prebuilt standard.sif + hf_cache
# are already present and bootstrap is cheap). exp_ = SPENDS MONEY.
#
#   bash exp_launch_box.sh [GPUS] [--bootstrap]
#     GPUS         1 (default, matches the originals) | 2 | 4 | 8
#     --bootstrap  after the box is active, rsync + run bootstrap_box.sh
#                  (idempotent; cheap if the filesystem already has the .sif)
#
# Discovery is automatic (filesystem, its region, an SSH key, an H100 instance
# type with capacity IN that region). Override with env PTB_FS_NAME / PTB_SSH_KEY.
# GPU selection:
#   PTB_H100_VARIANT=sxm5        choose SXM H100 capacity, if an attached
#                                filesystem exists in a matching region
#   PTB_INSTANCE_TYPE_NAME=...    require an exact Lambda instance type
# Prints the box IP on success; pass it to exp_run_experiment.sh / exp_smoke.sh,
# or those will auto-discover the single running box.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
GPUS="${1:-1}"; [ "$GPUS" = "--bootstrap" ] && GPUS=1   # allow `exp_launch_box.sh --bootstrap`
BOOTSTRAP=0; for a in "$@"; do [ "$a" = "--bootstrap" ] && BOOTSTRAP=1; done
ptb_load_secrets || exit 1

echo "== launching a ${GPUS}xH100 box (attaching the persistent filesystem) =="
IP=$(ptb_launch_box "$GPUS" "ptb-rollback-${GPUS}x-$(date +%m%d-%H%M%S)") || { echo "launch failed (see reason above)"; exit 1; }
echo "== box active: $IP =="

if [ "$BOOTSTRAP" = 1 ]; then
  echo "== bootstrap (install apptainer if missing; build .sif only if absent) =="
  FS=$(ptb_fs_detect_remote "$IP")
  [ -n "$FS" ] || { echo "WARN: no /lambda/nfs filesystem detected on box yet"; FS=""; }
  ptb_rsync "$HERE/bootstrap_box.sh" "$HERE/standard.def" "ubuntu@$IP:${FS:-/home/ubuntu}/" 2>&1 | tail -1
  # run from the filesystem if detected (so the .sif lands there), else home
  ptb_ssh "$IP" "bash ${FS:-/home/ubuntu}/bootstrap_box.sh" 2>&1 | tail -20
fi

echo
echo "box IP: $IP"
echo "next:  PTB_TRAJECTORY=<nick|run_id> bash $HERE/exp_smoke.sh control $IP"
echo "(terminate when done:  it self-terminates after a non-KEEP_ALIVE run, or"
echo " manually via the Lambda dashboard / API)"
