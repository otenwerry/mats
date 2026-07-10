#!/bin/bash
# Build a timestamped Codex-capable candidate container on the Lambda filesystem.
# exp_ = may spend Lambda GPU instance time. This does NOT overwrite standard.sif.
#
# Usage:
#   bash exp_build_codex_candidate.sh [existing_box_ip]
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1

IP="${1:-}"
if [ -z "$IP" ]; then
  IP=$(ptb_launch_box 1 "ptb-build-codex-$(date +%m%d-%H%M%S)") || exit 1
fi
IID=$(ptb_instance_id "$IP")
FS=$(ptb_fs_detect_remote "$IP")
[ -n "$FS" ] || { echo "FAIL: no /lambda/nfs filesystem on $IP"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
CAND_NAME="codex_0140_${STAMP}.sif"
DEF="$FS/standard_codex_${STAMP}.def"
SIF="$FS/containers/$CAND_NAME"
BUILD_LOG="$FS/codex_candidate_build_${STAMP}.log"
LOCAL_STATE="/Users/owenterry/supermats/mats-local/rollback/build_logs/20260616/codex_candidate_current.txt"

mkdir -p "$(dirname "$LOCAL_STATE")"
{
  echo "IP=$IP"
  echo "IID=$IID"
  echo "FS=$FS"
  echo "STAMP=$STAMP"
  echo "SIF=$SIF"
  echo "BUILD_LOG=$BUILD_LOG"
} | tee "$LOCAL_STATE"

ptb_rsync "$HERE/standard.def" "ubuntu@$IP:$DEF"

ptb_ssh "$IP" "bash -s" <<EOF
set -uo pipefail
FS="$FS"
DEF="$DEF"
SIF="$SIF"
BUILD_LOG="$BUILD_LOG"
cat > /home/ubuntu/build_codex_candidate.sh <<'EOS'
#!/bin/bash
set -uo pipefail
FS="$FS"
DEF="$DEF"
SIF="$SIF"
export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export APPTAINER_TMPDIR=/tmp/apptainer-tmp
mkdir -p "\$APPTAINER_CACHEDIR" "\$APPTAINER_TMPDIR" "$FS/containers"
echo "START \$(date -u)"
if ! command -v apptainer >/dev/null 2>&1; then
  echo "installing apptainer"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common
  sudo add-apt-repository -y ppa:apptainer/ppa
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apptainer
fi
apptainer --version
sudo APPTAINER_CACHEDIR="$APPTAINER_CACHEDIR" APPTAINER_TMPDIR="$APPTAINER_TMPDIR" \
  apptainer build "$SIF" "$DEF"
echo "VERIFY"
apptainer exec -c "$SIF" bash -lc "codex --version; python3 -c 'import transformers, openai, huggingface_hub as h; print(\"transformers\", transformers.__version__, \"openai\", openai.__version__, \"hub\", h.__version__)'"
echo "DONE \$(date -u)"
EOS
chmod +x /home/ubuntu/build_codex_candidate.sh
nohup /home/ubuntu/build_codex_candidate.sh > "$BUILD_LOG" 2>&1 < /dev/null &
echo "BUILD_PID \$!"
echo "BUILD_LOG $BUILD_LOG"
echo "SIF $SIF"
EOF

echo "Build started on $IP ($IID)."
echo "Poll: tail -f via SSH, or run: bash $HERE/check_codex_candidate.sh"
