#!/bin/bash
# exp_fanout.sh — launch the SAME rollback experiment across MANY trajectories,
# one Lambda box per trajectory, in parallel. The single command for goal (a):
# run all three prompt conditions on a set of trajectories. exp_ = SPENDS MONEY.
#
#   bash exp_fanout.sh <selector> [conds=prompt1,prompt2,prompt3] [gpus=1]
#
# selector (see rollback/run/targets.py):
#   runnable                  every credential-ready, not-yet-complete trajectory
#   opencode|claude|codex     by scaffold
#   <nick|run_id>[,...]        explicit list
#
# Each trajectory is handed to the proven exp_rollout_batch.sh (launch box ->
# build cells -> prep ONCE -> fidelity gate -> run the conditions in parallel
# across the box's GPUs -> pull -> viewerize -> judge). This wrapper just fans
# those out with a concurrency cap and a low-risk-first order, logging each.
#
# env:
#   DRY_RUN=1        print the plan and exit (no boxes, no spend)
#   MAX_PARALLEL=4   how many trajectory boxes to run concurrently
#   MAX_TRAJ=N       cap the number of trajectories launched (first N in order)
#   INCLUDE_DONE=1   don't skip trajectories that already have all 3 prompts
#   LAUNCH_STAGGER_SEC=15   gap between starting boxes (avoid a Lambda API burst)
#   PTB_H100_VARIANT=sxm5 / PTB_FS_NAME=...   region/box selection (per ptb_lib)
#   PTB_CODEX_ROLLOUT_VALIDATED=1   allow codex trajectories (after recon validation)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"

SELECTOR="${1:?usage: exp_fanout.sh <selector> [conds] [gpus]}"
# conds: "auto" (default) = gap-fill, i.e. run only the prompts each trajectory
# still lacks (all 3 if untouched). Pass an explicit csv (e.g. prompt1,prompt2,
# prompt3) to force those conditions on every trajectory regardless (fresh seeds).
CONDS="${2:-auto}"
GPUS="${3:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-15}"
INCLUDE_DONE_FLAG=""; [ "${INCLUDE_DONE:-0}" = 1 ] && INCLUDE_DONE_FLAG="--include-done"

echo "==== FAN-OUT [$SELECTOR] conds=$CONDS gpus=$GPUS max_parallel=$MAX_PARALLEL ===="
echo "== plan =="
( cd "$HERE/../.." && python3 -m rollback.run.targets "$SELECTOR" $INCLUDE_DONE_FLAG )

# resolve to concrete run_id + per-trajectory conds (read loop — bash 3.2 has no mapfile).
# targets emits 'run_id<TAB>missing_csv'; if an explicit CONDS was given, override.
IDS=(); TCONDS=()
while IFS=$'\t' read -r rid conds; do
  [ -n "$rid" ] || continue
  IDS+=("$rid")
  if [ "$CONDS" = auto ]; then TCONDS+=("$conds"); else TCONDS+=("$CONDS"); fi
done < <(cd "$HERE/../.." && python3 -m rollback.run.targets "$SELECTOR" $INCLUDE_DONE_FLAG --ids --with-conds)
[ "${#IDS[@]}" -gt 0 ] || { echo "nothing to launch (all complete/blocked, or empty selector)"; exit 0; }
if [ -n "${MAX_TRAJ:-}" ]; then IDS=("${IDS[@]:0:$MAX_TRAJ}"); TCONDS=("${TCONDS[@]:0:$MAX_TRAJ}"); fi

if [ "${DRY_RUN:-0}" = 1 ]; then
  echo; echo "DRY_RUN=1 — would launch ${#IDS[@]} trajectory batch(es) (conds per trajectory):"
  for i in $(seq 0 $((${#IDS[@]}-1))); do echo "  ${TCONDS[$i]}   ${IDS[$i]}"; done
  exit 0
fi

ptb_load_secrets || exit 1
LOGDIR="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}/fanout_logs"
mkdir -p "$LOGDIR"
echo "== launching ${#IDS[@]} trajectory batch(es); logs -> $LOGDIR =="

PIDS=(); LABELS=()
launched=0
for idx in $(seq 0 $((${#IDS[@]}-1))); do
  rid="${IDS[$idx]}"; conds="${TCONDS[$idx]}"
  log="$LOGDIR/$(date +%Y%m%d_%H%M%S)__${rid:0:60}.log"
  echo "  [$((launched+1))/${#IDS[@]}] $rid  conds=$conds"
  echo "        log: $log"
  # exp_rollout_batch.sh is self-contained per box (own launch + 12h backstop +
  # self-terminate), so a dropped wrapper can't strand a box (only the pull/judge,
  # which can be re-run). Each gets its OWN box via ptb_launch_box.
  ( bash "$HERE/exp_rollout_batch.sh" "$rid" "$conds" "$GPUS" > "$log" 2>&1 ) &
  PIDS+=("$!"); LABELS+=("$rid")
  launched=$((launched+1))
  # cap concurrency: when at the limit, block on the OLDEST job (bash 3.2-safe;
  # mirrors run_rollout_batch_on_box.sh — no `wait -n`).
  if [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; then
    wait "${PIDS[0]}" || echo "  ! batch failed: ${LABELS[0]}"
    PIDS=("${PIDS[@]:1}"); LABELS=("${LABELS[@]:1}")
  fi
  sleep "$LAUNCH_STAGGER_SEC"
done

# drain the rest (guard the empty-array case: bash 3.2 + set -u errors on "${PIDS[@]}"
# when PIDS is empty, e.g. a single trajectory already drained by the cap check).
i=0
for pid in "${PIDS[@]:-}"; do
  [ -n "$pid" ] || continue
  wait "$pid" || echo "  ! batch failed: ${LABELS[$i]:-?}"
  i=$((i+1))
done

echo "==== FAN-OUT DONE [$SELECTOR] — launched $launched; see $LOGDIR for per-trajectory logs ===="
