#!/bin/bash
# Cheap pre-flight check that our two fixes work BEFORE committing to a full pair
# of runs:
#   (1) no-stale-data: rebuilding the cut cell leaves NO inherited eval-result
#       files (so the agent can't read the original's score as its own);
#   (2) eval-works: standard.sif rebuilt with pinned transformers serves the
#       model in vLLM and produces a score (the bug that invalidated the last run).
#
# NOT exp_-prefixed: spends GPU time on the box you pass, but calls no paid API.
#
#   bash smoke_test.sh <IP>
#
# Step 2 rebuilds standard.sif on the filesystem (~40 min, one-time — the real
# runs need it anyway), then runs a 5-sample eval on the base model.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTB="$(cd "$HERE/../.." && pwd)"            # .../mats/posttrainbench
IP="${1:?usage: smoke_test.sh <IP>}"
# ServerAlive keepalives so the long (~40 min) rebuild SSH connection survives.
SSH="ssh -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=120 -o ControlMaster=auto -o ControlPath=/tmp/ptb-ssh-%r@%h:%p -o ControlPersist=180"

read RUN_NAME CUT MODEL < <(cd "$HERE/../.." && python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT, t.model_to_train)")
CELL="backward_prompt1_cut${CUT}"
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
echo "trajectory=$RUN_NAME cell=$CELL base_model=$MODEL"

echo "== 1. LOCAL: rebuild cell, assert NO inherited eval-result files =="
( cd "$HERE/../.." && uv run python -m rollback.run.orchestrate --cut "$CUT" --cells "$CELL" --bash-mode skip --force ) 2>&1 | tail -3
STALE=$(find "$LOCAL_CELL/task" -maxdepth 1 \( -name '*eval*.json' -o -name '*eval*.log' -o -name 'baseline*' \) 2>/dev/null)
if [ -n "$STALE" ]; then echo "  FAIL: stale eval files remain:"; echo "$STALE"; exit 1; fi
echo "  PASS: no stale eval-result files in the cut workspace"

echo "== 2. BOX: detect filesystem =="
FS=$($SSH ubuntu@$IP 'for d in /lambda/nfs/*/; do [ -d "${d}containers" ] && echo "${d%/}" && break; done' 2>/dev/null | head -1)
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on box"; exit 1; }
echo "  filesystem: $FS"

echo "== 3. push pinned standard.def + scripts + cell =="
rsync -az -e "$SSH" "$HERE/standard.def" "ubuntu@$IP:$FS/standard.def" 2>&1 | tail -1
rsync -az -e "$SSH" "$HERE"/{bootstrap_box.sh,box_prepare.sh} "ubuntu@$IP:$FS/" 2>&1 | tail -1
rsync -az -e "$SSH" --exclude tmp "$LOCAL_CELL" "ubuntu@$IP:$FS/cells/" 2>&1 | tail -1

echo "== 4. REBUILD standard.sif (pinned transformers) + 5-sample eval on base model =="
$SSH ubuntu@$IP "bash -s" <<EOF
set -uo pipefail
chmod +x $FS/bootstrap_box.sh
echo '--- rebuilding container (this is the slow part, ~40 min) ---'
FS='$FS' REBUILD=1 MODEL='$MODEL' bash $FS/bootstrap_box.sh 2>&1 | tail -8
W=/home/ubuntu/smoke_${CELL}; rm -rf "\$W"; cp -r $FS/cells/$CELL "\$W"; mkdir -p "\$W/tmp"
echo '--- 5-sample eval (confirms vLLM serves + scores) ---'
timeout 20m apptainer exec --nv -c \
  --env PATH="/root/.local/bin:/home/ben/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env HF_HOME="/home/ben/hf_cache" --env VLLM_API_KEY="inspectai" --env PYTHONNOUSERSITE="1" \
  --bind "\$W/tmp:/tmp" --bind "$FS/hf_cache:/home/ben/hf_cache" \
  --home "\$W:/home/ben" --pwd /home/ben/task --writable-tmpfs \
  $FS/containers/standard.sif \
  bash -c 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 5; python3 evaluate.py --model-path "$MODEL" --limit 5 --json-output-file /home/ben/smoke_score.json && echo SMOKE_SCORE: && cat /home/ben/smoke_score.json' 2>&1 | tail -15
echo "--- eval-stack versions in rebuilt container ---"
apptainer exec -c $FS/containers/standard.sif python3 -c "import transformers,openai,huggingface_hub as h; print('transformers',transformers.__version__,'openai',openai.__version__,'hub',h.__version__)" 2>&1 | tail -3
EOF
echo
echo "== INTERPRET =="
echo "  PASS if you see 'SMOKE_SCORE:' followed by an accuracy + transformers 4.57.3."
echo "  FAIL if you see a vLLM 'Failed to start' / TokenizersBackend error."