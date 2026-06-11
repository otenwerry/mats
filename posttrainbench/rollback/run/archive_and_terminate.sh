#!/bin/bash
# Watch a keep-alive rollout: archive its trajectory to the persistent
# filesystem every 2 min, and when the run's done-marker appears do a final
# archive + self-terminate the instance.
# Usage:  bash archive_and_terminate.sh <WORK_DIR>
#
# Uses the run's "$WORK/.agent_done" marker (written by run_rollout_on_box.sh
# when the agent finishes or is capped) — NOT pgrep — so it can't self-match or
# race the cold start. Filesystem auto-detected. Terminate is the LAST thing,
# only after a final archive, so the trajectory is always safe.
set -uo pipefail
WORK="${1:?work dir}"
source /home/ubuntu/.ptb_lambda 2>/dev/null   # LAMBDA_API_KEY, INSTANCE_ID

for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
TS="$(basename "$WORK")"
DEST="$FS/results/$TS"
mkdir -p "$DEST/task"
# exclude regenerable model artifacts (weights, checkpoints, merged model, the
# 32MB tokenizer.json) — only the trajectory + the agent's source code is kept.
EXCL=(--exclude "*.safetensors" --exclude "*.bin" --exclude "*.pt" --exclude "*.gguf"
      --exclude "outputs/" --exclude "final_model/" --exclude "*_merged/" --exclude "tokenizer.json")
echo "archiver: WORK=$WORK DEST=$DEST" >> "$DEST/archive.log"

archive() {
  cp "$WORK"/solve_out_*.txt "$DEST"/ 2>/dev/null
  cp "$WORK"/run_config.json "$DEST"/ 2>/dev/null
  rsync -a "${EXCL[@]}" "$WORK/task/" "$DEST/task/" 2>/dev/null
}

# archive loop until the done-marker appears (cap at ~3h so a hang can't loop forever)
for _ in $(seq 1 90); do
  archive
  [ -f "$WORK/.agent_done" ] && break
  sleep 120
done
archive
touch "$DEST/.run_complete"
echo "done-marker seen $(date -u); final archive complete" >> "$DEST/archive.log"

# KEEP_ALIVE: leave the box up after the run (capacity is scarce — re-acquiring
# a box can take an hour; a separate backstop watchdog bounds overnight cost).
if [ "${KEEP_ALIVE:-0}" = 1 ]; then
  echo "KEEP_ALIVE set: archived, NOT terminating (terminate manually when done)" >> "$DEST/archive.log"
  exit 0
fi

# grace window: give the Mac up to 30 min to pull the result (exp_pull_result.sh
# touches $DEST/.pulled when done). Terminate as soon as it's pulled, or at the
# cap regardless. Trajectory is already safe on the filesystem either way.
for _ in $(seq 1 60); do [ -f "$DEST/.pulled" ] && break; sleep 30; done
echo "grace ended ($( [ -f "$DEST/.pulled" ] && echo pulled || echo timeout )) $(date -u)" >> "$DEST/archive.log"

# self-terminate (trajectory is safe on the filesystem)
if [ -n "${INSTANCE_ID:-}" ]; then
  for n in 1 2 3 4 5; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -u "$LAMBDA_API_KEY:" \
      -H "Content-Type: application/json" -d "{\"instance_ids\":[\"$INSTANCE_ID\"]}" \
      https://cloud.lambda.ai/api/v1/instance-operations/terminate)
    echo "terminate $n -> $code" >> "$DEST/archive.log"
    [ "$code" = "200" ] && break
    sleep 15
  done
fi
