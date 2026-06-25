#!/bin/bash
# HealthBench eval-GRADER smoke test. exp_ = spends money: the HealthBench scorer
# grades model outputs with the OpenAI gpt-5-mini judge.
#
#   bash exp_healthbench_grader_smoke.sh <IP>
#
# HealthBench is the only LLM-graded benchmark we use, and we've never run its
# grader inside our container. This confirms, cheaply (~5 min, a few cents of
# OpenAI), that the ALREADY-REBUILT standard.sif can run the whole healthbench
# eval end-to-end: vLLM serves the base model AND gpt-5-mini grades 5 samples.
# Run this BEFORE committing a healthbench rollback pair; if it fails we fall back
# to a second bfcl trajectory instead.
#
# Uses the rebuilt container on the shared filesystem (NO 40-min rebuild) and the
# already-built cut cell (for its task/evaluate.py + evaluation_code/grader.py).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
IP="${1:?usage: exp_healthbench_grader_smoke.sh <IP>}"
SECRETS="$HOME/.config/ptb/secrets.env"
[ -f "$SECRETS" ] || { echo "no $SECRETS"; exit 1; }
set -a; source "$SECRETS"; set +a
export PTB_TRAJECTORY=healthbench_kimi17
SSH="ssh -o ConnectTimeout=45 -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=60 -o ServerAliveCountMax=120 -o ControlMaster=auto -o ControlPath=/tmp/ptb-ssh-%r@%h:%p -o ControlPersist=180"

read RUN_NAME CUT MODEL < <(cd "$HERE/../.." && PTB_TRAJECTORY=healthbench_kimi17 python3 -c "from rollback import config as c; t=c.TRAJECTORY; print(t.run_name, c.CUT_BEFORE_EVENT, t.model_to_train)")
CELL="backward_prompt1_cut${CUT}"
LOCAL_CELL="$HERE/../builds/$RUN_NAME/$CELL"
[ -d "$LOCAL_CELL" ] || { echo "FAIL: cell not built: $LOCAL_CELL"; exit 1; }
echo "trajectory=$RUN_NAME cell=$CELL base_model=$MODEL judge=gpt-5-mini"

echo "== preflight OpenRouter-independent: OpenAI key present? =="
[ -n "${OPENAI_API_KEY:-}" ] || { echo "FAIL: OPENAI_API_KEY not in $SECRETS (grader needs it)"; exit 1; }

echo "== detect filesystem + verify rebuilt container =="
FS=$($SSH ubuntu@$IP 'for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && echo "${d%/}" && break; done' 2>/dev/null | head -1)
[ -n "$FS" ] || { echo "FAIL: no container filesystem on box (region mismatch?)"; exit 1; }
$SSH ubuntu@$IP 'command -v apptainer >/dev/null || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common && sudo add-apt-repository -y ppa:apptainer/ppa && sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apptainer; }' 2>&1 | tail -1
TV=$($SSH ubuntu@$IP "apptainer exec -c $FS/containers/standard.sif python3 -c 'import transformers;print(transformers.__version__)' 2>/dev/null" | tr -d '\r')
[ "$TV" = "4.57.3" ] || { echo "ABORT: container transformers=$TV (need 4.57.3). Run smoke_test.sh to rebuild standard.sif first."; exit 1; }
echo "  FS=$FS  container transformers $TV OK"

echo "== push cell + deploy OpenAI key =="
rsync -az -e "$SSH" --exclude tmp "$LOCAL_CELL" "ubuntu@$IP:$FS/cells/" 2>&1 | tail -1
SEC_B64=$(grep -E '^(OPENAI_API_KEY|HF_TOKEN)=' "$SECRETS" | base64)

echo "== 5-sample healthbench eval (vLLM serve + gpt-5-mini grader) =="
$SSH ubuntu@$IP "bash -s" <<EOF
set -uo pipefail
umask 077; echo '$SEC_B64' | base64 -d > /home/ubuntu/.ptb_secrets; umask 022
W=/home/ubuntu/hbsmoke_${CELL}; rm -rf "\$W"; cp -r $FS/cells/$CELL "\$W"; mkdir -p "\$W/tmp"
timeout 25m apptainer exec --nv -c \
  --env PATH="/root/.local/bin:/home/ben/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  --env HF_HOME="/home/ben/hf_cache" \
  --env VLLM_API_KEY="inspectai" --env PYTHONNOUSERSITE="1" \
  --bind /home/ubuntu/.ptb_secrets:/ptb_secrets:ro \
  --bind "\$W/tmp:/tmp" --bind "$FS/hf_cache:/home/ben/hf_cache" \
  --home "\$W:/home/ben" --pwd /home/ben/task --writable-tmpfs \
  $FS/containers/standard.sif \
  bash -c 'set -a; . /ptb_secrets; set +a; uv pip install --system --no-cache transformers==4.57.3 tokenizers==0.22.2 2>&1 | tail -1; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 8; python3 evaluate.py --model-path "$MODEL" --limit 5 --json-output-file /home/ben/hb_smoke.json && echo HB_SMOKE_SCORE: && cat /home/ben/hb_smoke.json' 2>&1 | tail -30
rm -f /home/ubuntu/.ptb_secrets
EOF
echo
echo "== INTERPRET =="
echo "  PASS  -> you see 'HB_SMOKE_SCORE:' + a JSON score (vLLM served AND gpt-5-mini graded)."
echo "  FAIL  -> OpenAI auth / 'gpt-5-mini' model error, a grader import error, or vLLM 'Failed to start'."
echo "          (on FAIL: fall back to a second bfcl trajectory instead of healthbench.)"
