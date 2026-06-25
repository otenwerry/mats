#!/bin/bash
# Run multiple rollback prompt conditions for one trajectory on one Lambda box.
#
# The first cell is used to materialize/score/gate the pre-cut model once. Its
# prepped task/ is then copied into each condition cell, so prompt1/prompt2/
# prompt3 do not retrain the same pre-cut model independently.
#
# Usage:
#   bash run_rollout_batch_on_box.sh <comma-separated-cell-ids> [parallel=1]
set -uo pipefail

CELLS_CSV="${1:?usage: run_rollout_batch_on_box.sh <cells_csv> [parallel]}"
PARALLEL="${2:-1}"

# Clamp PARALLEL to the GPUs ACTUALLY on this box. The Mac-side launcher computes
# PARALLEL from the *requested* GPU count, but the capacity fallback can hand us a
# smaller box (e.g. asked 4x, got 1x). Without clamping, gpu=$((idx % PARALLEL))
# below maps conditions onto GPU indices that don't exist -> CUDA invalid device
# ordinal, silently killing those conditions. Clamping makes a smaller box simply
# run fewer conditions at once (the rest queue) instead of crashing.
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if [ "${GPU_COUNT:-0}" -ge 1 ] && [ "$PARALLEL" -gt "$GPU_COUNT" ]; then
  echo "BATCH: clamping parallel $PARALLEL -> $GPU_COUNT (actual GPUs on this box)"
  PARALLEL="$GPU_COUNT"
fi

if [ -z "${FS:-}" ]; then
  for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
fi
[ -n "${FS:-}" ] || { echo "BATCH: FAIL no /lambda/nfs filesystem with standard.sif"; exit 1; }

IFS=, read -r -a CELLS <<< "$CELLS_CSV"
[ "${#CELLS[@]}" -gt 0 ] || { echo "BATCH: FAIL no cells"; exit 1; }
BASE_CELL="${CELLS[0]}"
TS="$(date +%s)"
BATCH="/home/ubuntu/batch_${BASE_CELL}__${TS}"
mkdir -p "$BATCH"
echo "BATCH: fs=$FS base=$BASE_CELL cells=$CELLS_CSV parallel=$PARALLEL batch=$BATCH"

for c in "${CELLS[@]}"; do
  [ -d "$FS/cells/$c" ] || { echo "BATCH: FAIL missing $FS/cells/$c"; exit 1; }
done

PREP_WORK="$BATCH/prep_${BASE_CELL}__${TS}"
if [ -n "${PTB_PREPPED_SOURCE:-}" ]; then
  echo "BATCH: reusing prepped source=$PTB_PREPPED_SOURCE -> $PREP_WORK"
  [ -d "$PTB_PREPPED_SOURCE" ] || { echo "BATCH: FAIL missing PTB_PREPPED_SOURCE=$PTB_PREPPED_SOURCE"; exit 1; }
  cp -a "$PTB_PREPPED_SOURCE" "$PREP_WORK"
  PREP_RC=0
else
  cp -a "$FS/cells/$BASE_CELL" "$PREP_WORK"
  echo "BATCH: prep work=$PREP_WORK"
  PREP_AND_EXIT=1 PTB_SCORE_LOCK="$BATCH/score.lock" \
    bash /home/ubuntu/run_rollout_on_box.sh "$PREP_WORK" > "$BATCH/prep_driver.log" 2>&1
  PREP_RC=$?
fi
echo "BATCH: prep rc=$PREP_RC"

# Fidelity is NON-FATAL (2026-06-17): a diverged/unverified prep no longer blocks
# the condition runs — we launch anyway and let prep_fidelity.json (copied into
# each condition cell below) carry the status to the viewer. Only a prep that
# never produced a ready workspace (.prep_ready missing) still stops the batch.
if [ -f "$PREP_WORK/.prep_diverged" ]; then
  echo "BATCH: prep DIVERGED from baseline — launching condition runs anyway (flagged prep_fidelity=diverged)"
fi
if [ -f "$PREP_WORK/.prep_unverified" ]; then
  echo "BATCH: prep fidelity UNVERIFIED — launching condition runs anyway (flagged prep_fidelity=unverified)"
fi
if [ ! -f "$PREP_WORK/.prep_ready" ]; then
  echo "BATCH: prep did not produce .prep_ready; not launching condition runs"
  touch "$BATCH/.batch_done"
  exit 1
fi

PREP_N=$(python3 -c "import json;print(len(json.load(open('$PREP_WORK/run_config.json')).get('prep_commands') or []))")
echo "BATCH: prep commands=$PREP_N"

if [ "$PREP_N" -gt 0 ] && [ -z "${PTB_PREPPED_SOURCE:-}" ] && [ "${PTB_SAVE_PREP:-1}" = 1 ]; then
  SAVED_PREP="$FS/prepped/${BASE_CELL}__${TS}"
  echo "BATCH: saving faithful prep workspace -> $SAVED_PREP"
  mkdir -p "$FS/prepped"
  if [ -e "$SAVED_PREP" ]; then
    echo "BATCH: FAIL saved prep destination already exists: $SAVED_PREP"
    touch "$BATCH/.batch_done"
    exit 1
  fi
  cp -a "$PREP_WORK" "$SAVED_PREP"
  touch "$SAVED_PREP/.saved_prep_ready"
  echo "$SAVED_PREP" > "$BATCH/saved_prep_path.txt"
  echo "BATCH: saved prep ready at $SAVED_PREP"
fi

PIDS=()
WORKS=()
idx=0
FAIL=0
for c in "${CELLS[@]}"; do
  WORK="/home/ubuntu/${c}__${TS}"
  cp -a "$FS/cells/$c" "$WORK"
  if [ "$PREP_N" -gt 0 ]; then
    rm -rf "$WORK/task"
    cp -a "$PREP_WORK/task" "$WORK/task"
    cp "$PREP_WORK/prep_score.json" "$WORK/" 2>/dev/null || true
    cp "$PREP_WORK/prep_fidelity.json" "$WORK/" 2>/dev/null || true
  fi
  gpu=$(( idx % PARALLEL ))
  echo "BATCH: launching $c work=$WORK gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" SCORE_PREP=0 PTB_SCORE_LOCK="$BATCH/score.lock" \
    bash /home/ubuntu/run_rollout_on_box.sh "$WORK" > "$BATCH/${c}_driver.log" 2>&1 &
  pid=$!
  PIDS+=("$pid")
  WORKS+=("$WORK")
  echo "$WORK" >> "$BATCH/work_dirs.txt"
  echo "$pid $WORK" >> "$BATCH/pids.txt"

  # Archive each child while it runs; KEEP_ALIVE prevents per-child termination.
  KEEP_ALIVE=1 bash /home/ubuntu/archive_and_terminate.sh "$WORK" > "$BATCH/${c}_archive.log" 2>&1 &

  idx=$((idx + 1))
  if [ "${#PIDS[@]}" -ge "$PARALLEL" ]; then
    wait "${PIDS[0]}" || FAIL=1
    PIDS=("${PIDS[@]:1}")
  fi
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || FAIL=1
done

for w in "${WORKS[@]}"; do
  # Give each archiver a chance to observe .agent_done and do the final archive.
  for _ in $(seq 1 30); do
    name="$(basename "$w")"
    [ -f "$FS/results/$name/.run_complete" ] && break
    sleep 10
  done
done

touch "$BATCH/.batch_done"
echo "BATCH: done fail=$FAIL work_dirs=${WORKS[*]}"
exit "$FAIL"
