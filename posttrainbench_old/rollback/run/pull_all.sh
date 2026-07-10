#!/bin/bash
# pull_all.sh — collect EVERY completed-but-unpulled rollback result off the
# Lambda filesystem(s) to mats-local, then viewerize. FREE (rsync + sync_viewer;
# no agent/API spend). This is the cross-session DURABILITY path: a launcher
# process that dies (session ends, wrapper crash) leaves its finished run archived
# on the filesystem but un-pulled; this sweeps them all in. Judging stays a
# separate (paid) step — the command is printed at the end.
#
#   bash rollback/run/pull_all.sh
#
# results/ is shared across all boxes on a given filesystem, so we pull once per
# distinct filesystem (via any one active box that has it mounted). Touching
# .pulled signals the archiver it may self-terminate that box.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1
SSH="$PTB_SSH"
LOCAL="${PTB_ROLLBACK_LOCAL:-$HERE/../../../../mats-local/rollback}"
mkdir -p "$LOCAL/results"
EXCL=(--exclude '*.safetensors' --exclude '*.bin' --exclude '*.pt' --exclude '*.gguf'
      --exclude 'outputs/' --exclude 'final_model/' --exclude '*_merged/' --exclude 'tokenizer.json')

IPS=$(_lam_get instances | python3 -c "import sys,json;[print(i['ip']) for i in json.load(sys.stdin).get('data',[]) if i.get('status')=='active' and i.get('ip')]")
[ -n "$IPS" ] || { echo "no active boxes (nothing mounts the filesystem to pull from)"; exit 0; }

SEEN_FS=""
PULLED=0
for IP in $IPS; do
  FS=$(ptb_fs_detect_remote "$IP"); [ -n "$FS" ] || continue
  case " $SEEN_FS " in *" $FS "*) continue;; esac   # one box per distinct FS
  SEEN_FS="$SEEN_FS $FS"
  echo "== filesystem $FS (via $IP) =="
  DIRS=$($SSH "ubuntu@$IP" "for r in $FS/results/*/; do [ -f \"\${r}.run_complete\" ] && [ ! -f \"\${r}.pulled\" ] && basename \"\$r\"; done" 2>/dev/null | tr -d '\r')
  [ -n "$DIRS" ] || { echo "  nothing unpulled"; continue; }
  for d in $DIRS; do
    echo "  pulling $d"
    if rsync -az -e "$SSH" "${EXCL[@]}" "ubuntu@$IP:$FS/results/$d" "$LOCAL/results/" 2>&1 | tail -1; then
      $SSH "ubuntu@$IP" "touch $FS/results/$d/.pulled" 2>/dev/null
      PULLED=$((PULLED+1))
    fi
  done
done

echo "== viewerize ($PULLED newly pulled) =="
( cd "$HERE/../.." && uv run python -m rollback.sync_viewer ) 2>&1 | tail -5
echo
echo "judge any unjudged rollback runs (PAID) with:"
echo "  cd $HERE/../../.. && uv run python posttrainbench/judging/exp_judge_rollback.py"
