#!/bin/bash
# Run on a Lambda box. Builds one candidate SIF at the caller-provided path.
set -uo pipefail

FS="${FS:?set FS}"
DEF="${DEF:?set DEF}"
SIF="${SIF:?set SIF}"

export APPTAINER_CACHEDIR=/tmp/apptainer-cache
export APPTAINER_TMPDIR=/tmp/apptainer-tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$FS/containers"

echo "START $(date -u)"
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
apptainer exec -c "$SIF" bash -lc \
  "codex --version; python3 -c 'import transformers, openai, huggingface_hub as h; print(\"transformers\", transformers.__version__, \"openai\", openai.__version__, \"hub\", h.__version__)'"
echo "DONE $(date -u)"
