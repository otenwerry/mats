#!/usr/bin/env bash
# Run a calibration with the SAME library versions the sandbox image ships, on any machine.
#
#   ./envgen/calibrate.sh                  # full report (writes calibration_summary.json)
#   ./envgen/calibrate.sh --sweep 0.15,0.18,0.25
#
# Free: CPU only, no API calls.
#
# Why this wrapper exists: xgboost on macOS/arm64 needs an arm64 libomp, and Homebrew here
# is the Intel build (its libomp is x86_64 and cannot load into arm64 Python). Rather than
# install a second Homebrew, this borrows the arm64 libomp that the torch wheel already
# bundles and points dyld at it. On Linux (and inside the sandbox image) the extra bits are
# harmless no-ops. Without this, calibrate_fraud.py silently falls back to sklearn's
# booster and marks the result "definitive": false.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENTS="$(dirname "$HERE")"
MATS="$(dirname "$ENVIRONMENTS")"

# Pinned to match environments/sandbox/ml/Dockerfile.
PINS=(--with 'xgboost==2.1.1' --with 'scikit-learn==1.5.2' --with 'numpy==2.1.3' --with torch)

cd "$MATS"
OMP_DIR="$(uv run --with torch python -c 'import torch,pathlib;print(pathlib.Path(torch.__file__).parent/"lib")' | tail -1)"
echo "using libomp from: $OMP_DIR"
DYLD_FALLBACK_LIBRARY_PATH="$OMP_DIR" LD_LIBRARY_PATH="${OMP_DIR}:${LD_LIBRARY_PATH:-}" \
  uv run "${PINS[@]}" python "$ENVIRONMENTS/envgen/calibrate_fraud.py" "$@"
