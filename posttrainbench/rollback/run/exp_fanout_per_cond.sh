#!/bin/bash
# exp_fanout_per_cond.sh — launch ONE single-GPU box per (trajectory, condition),
# instead of one multi-GPU box per trajectory (that's exp_fanout.sh). exp_ = SPENDS MONEY.
#
#   bash exp_fanout_per_cond.sh <selector> [conds=auto]
#
# Why: 1x H100 capacity is ABUNDANT while 4x is scarce. One condition on one 1x box
# is the native PTB single-GPU config, and a box's wall-time is prep+agent+score
# whether it runs 1 condition or 3 in parallel — so splitting conditions onto their
# own 1x boxes finishes the same set in the same time WITHOUT waiting for a 4x box.
# The cost is redundant prep: each condition re-materializes the same pre-cut model
# (3x per trajectory). We accept that to escape the 4x bottleneck.
#
# Dedup, plan, and gap-fill are inherited from rollback.run.targets (same as the
# trajectory-level fanout): conditions already done (viewer_data) or in-flight
# (running/) are skipped, so this is safe to run alongside in-flight trajectory boxes.
#
# env:
#   DRY_RUN=1               print the flat job list and exit (no boxes, no spend)
#   MAX_PARALLEL=20         how many single-condition boxes to run concurrently
#   LAUNCH_STAGGER_SEC=20   gap between starting boxes (avoid a Lambda API/SSH burst)
#   PTB_GPU_FALLBACK="1 2 4"  prefer a 1x; the on-box clamp uses just 1 GPU on a bigger box
#   PTB_FS_DENY / PTB_GPU_FALLBACK / PTB_FALLBACK_RETRIES_PER_SIZE etc. per ptb_lib.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"

SELECTOR="${1:?usage: exp_fanout_per_cond.sh <selector> [conds]}"
CONDS="${2:-auto}"
MAX_PARALLEL="${MAX_PARALLEL:-20}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-20}"
INCLUDE_DONE_FLAG=""; [ "${INCLUDE_DONE:-0}" = 1 ] && INCLUDE_DONE_FLAG="--include-done"

echo "==== PER-CONDITION FAN-OUT [$SELECTOR] conds=$CONDS max_parallel=$MAX_PARALLEL ===="
echo "== plan (trajectory level) =="
( cd "$HERE/../.." && python3 -m rollback.run.targets "$SELECTOR" $INCLUDE_DONE_FLAG )

# Flatten (run_id, missing_csv) -> one job per (run_id, single condition). If an
# explicit CONDS was passed, force those on every trajectory instead of gap-fill.
RIDS=(); JCONDS=()
while IFS=$'\t' read -r rid conds; do
  [ -n "$rid" ] || continue
  use="$conds"; [ "$CONDS" != auto ] && use="$CONDS"
  IFS=, read -r -a cs <<< "$use"
  for c in "${cs[@]}"; do
    [ -n "$c" ] || continue
    RIDS+=("$rid"); JCONDS+=("$c")
  done
done < <(cd "$HERE/../.." && python3 -m rollback.run.targets "$SELECTOR" $INCLUDE_DONE_FLAG --ids --with-conds)

N="${#RIDS[@]}"
[ "$N" -gt 0 ] || { echo "nothing to launch (all complete/blocked/in-flight, or empty selector)"; exit 0; }
echo "== $N single-condition boxes to launch =="
for i in $(seq 0 $((N-1))); do echo "   ${JCONDS[$i]}   ${RIDS[$i]}"; done

if [ "${DRY_RUN:-0}" = 1 ]; then echo; echo "DRY_RUN=1 — not launching."; exit 0; fi

ptb_load_secrets || exit 1
LOGDIR="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}/fanout_logs"
mkdir -p "$LOGDIR"
echo "== launching $N single-condition box(es); logs -> $LOGDIR =="

PIDS=(); LABELS=(); launched=0
for i in $(seq 0 $((N-1))); do
  rid="${RIDS[$i]}"; c="${JCONDS[$i]}"
  log="$LOGDIR/$(date +%Y%m%d_%H%M%S)__${c}__${rid:0:48}.log"
  echo "  [$((launched+1))/$N] $c  $rid"
  echo "        log: $log"
  # one condition, 1 GPU. exp_rollout_batch.sh is self-contained per box (own launch
  # + 12h backstop + self-terminate), so a dropped wrapper can't strand a box.
  ( bash "$HERE/exp_rollout_batch.sh" "$rid" "$c" 1 > "$log" 2>&1 ) &
  PIDS+=("$!"); LABELS+=("$c $rid")
  launched=$((launched+1))
  if [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; then
    wait "${PIDS[0]}" || echo "  ! failed: ${LABELS[0]}"
    PIDS=("${PIDS[@]:1}"); LABELS=("${LABELS[@]:1}")
  fi
  sleep "$LAUNCH_STAGGER_SEC"
done

i=0
for pid in "${PIDS[@]:-}"; do
  [ -n "$pid" ] || continue
  wait "$pid" || echo "  ! failed: ${LABELS[$i]:-?}"
  i=$((i+1))
done
echo "==== PER-CONDITION FAN-OUT DONE [$SELECTOR] — launched $launched; logs -> $LOGDIR ===="
