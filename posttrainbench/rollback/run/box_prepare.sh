#!/bin/bash
# Idempotent box-side prepare + PREFLIGHT. Free (no agent / no paid calls).
# Run on a fresh Lambda box; prints "PREP: READY" iff everything needed for a
# rollout is present, else "PREP: FAIL <reason>". Safe to re-run.
#
# Auto-detects the filesystem (the /lambda/nfs/* that holds our container), so
# nothing depends on the filesystem's name or an exported env var.
set -uo pipefail

# 1. locate the persistent filesystem (prefer the one with our container)
FS=""
for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
[ -z "$FS" ] && for d in /lambda/nfs/*/; do FS="${d%/}"; break; done
[ -z "$FS" ] && { echo "PREP: FAIL no /lambda/nfs filesystem mounted (region mismatch?)"; exit 1; }
echo "PREP: filesystem = $FS"
echo "FS=$FS"   # parseable line for callers

# 2. apptainer present? install if not (skips heavy build — container is on FS)
if ! command -v apptainer >/dev/null 2>&1; then
  echo "PREP: installing apptainer..."
  sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq software-properties-common >/dev/null 2>&1
  sudo add-apt-repository -y ppa:apptainer/ppa >/dev/null 2>&1
  sudo apt-get update -qq >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq apptainer >/dev/null 2>&1
fi
command -v apptainer >/dev/null 2>&1 || { echo "PREP: FAIL apptainer install failed"; exit 1; }
echo "PREP: apptainer $(apptainer --version 2>/dev/null | awk '{print $NF}')"

# 3. assets present
[ -f "$FS/containers/standard.sif" ] || { echo "PREP: FAIL no container at $FS/containers/standard.sif"; exit 1; }
CELL="${1:-}"
if [ -n "$CELL" ]; then
  [ -d "$FS/cells/$CELL" ] || { echo "PREP: FAIL cell $CELL not on filesystem ($FS/cells/)"; exit 1; }
  echo "PREP: cell $CELL present"
fi

# 4. GPU visible
nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1 || { echo "PREP: FAIL no GPU (nvidia-smi)"; exit 1; }
echo "PREP: GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# 5. secrets present (deployed separately by the orchestrator)
[ -f /home/ubuntu/.ptb_secrets ] && grep -q OPENROUTER_API_KEY /home/ubuntu/.ptb_secrets \
  && echo "PREP: secrets present" || echo "PREP: WARN secrets not yet deployed"

echo "PREP: READY"
