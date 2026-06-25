#!/usr/bin/env bash
# Generic single-node GPU sanity check.
#
# Use for RunPod, Vast.ai, Lambda VMs, cloud GPU VMs, or any SSH-accessible
# GPU machine.
#
# This script checks:
#   - GPU visibility
#   - expected GPU count
#   - optional expected GPU name
#   - driver / CUDA visibility
#   - existing GPU processes
#   - persistent storage path
#   - disk write speed
#   - network ingress
#   - PyTorch CUDA availability
#   - short fp16 matmul throughput
#
# Usage:
#   chmod +x gpu-sanity.sh
#   EXPECTED_GPUS=1 ./gpu-sanity.sh
#
# Optional:
#   EXPECTED_GPU_NAME=H100 ./gpu-sanity.sh
#   PERSISTENT_PATH=/workspace ./gpu-sanity.sh
#   MIN_DOWNLOAD_MBPS=500 ./gpu-sanity.sh
#   MIN_DISK_MBPS=200 ./gpu-sanity.sh
#   MIN_TFLOPS=400 ./gpu-sanity.sh
#
# Exit code:
#   0 = passed required checks
#   1 = failed required check
#   2 = warning-level issue only, if STRICT_WARNINGS=1

set -euo pipefail

EXPECTED_GPUS="${EXPECTED_GPUS:-1}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-}"
PERSISTENT_PATH="${PERSISTENT_PATH:-}"
MIN_DOWNLOAD_MBPS="${MIN_DOWNLOAD_MBPS:-100}"
MIN_DISK_MBPS="${MIN_DISK_MBPS:-100}"
MIN_TFLOPS="${MIN_TFLOPS:-0}"
STRICT_WARNINGS="${STRICT_WARNINGS:-0}"

WARNINGS=0

warn() {
    echo "  WARNING: $*"
    WARNINGS=$((WARNINGS + 1))
}

fail() {
    echo "  FAIL: $*"
    exit 1
}

section() {
    echo
    echo "=== $* ==="
}

echo "=== Generic GPU Sanity Check ==="
echo "Host:      $(hostname)"
echo "Time UTC:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "User:      $(whoami)"
echo "Workdir:   $(pwd)"
echo

section "1. GPU visibility"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. NVIDIA driver/runtime is missing or not exposed."
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version,pstate,temperature.gpu,power.draw,power.limit --format=csv,noheader || {
    fail "nvidia-smi failed. GPU is not usable."
}

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"

if [[ "$GPU_COUNT" != "$EXPECTED_GPUS" ]]; then
    fail "Expected ${EXPECTED_GPUS} GPU(s), found ${GPU_COUNT}."
fi

echo "  OK: Found expected GPU count: ${GPU_COUNT}"

if [[ -n "$EXPECTED_GPU_NAME" ]]; then
    if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "$EXPECTED_GPU_NAME"; then
        fail "Expected GPU name containing '${EXPECTED_GPU_NAME}', but got: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ',')"
    fi
    echo "  OK: GPU name matches expected pattern: ${EXPECTED_GPU_NAME}"
fi

section "2. Driver, CUDA, and topology"

nvidia-smi | head -15

echo
echo "GPU topology:"
if nvidia-smi topo -m >/tmp/gpu_topology.txt 2>/dev/null; then
    cat /tmp/gpu_topology.txt
else
    warn "Could not read GPU topology with nvidia-smi topo -m."
fi

echo
echo "MIG mode:"
if nvidia-smi -q | grep -i "MIG Mode" | head -10; then
    if nvidia-smi -q | grep -i "MIG Mode" | grep -qi "Enabled"; then
        warn "MIG appears to be enabled. This may be intended, but verify that your job expects MIG slices rather than full GPUs."
    fi
else
    echo "  MIG information unavailable."
fi

section "3. Existing GPU processes"

if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -q .; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
    warn "There are already GPU processes running. Make sure they are yours."
else
    echo "  OK: No existing compute processes detected."
fi

section "4. Persistent storage"

if [[ -n "$PERSISTENT_PATH" ]]; then
    if [[ ! -d "$PERSISTENT_PATH" ]]; then
        fail "PERSISTENT_PATH does not exist: ${PERSISTENT_PATH}"
    fi

    echo "Persistent path: ${PERSISTENT_PATH}"
    df -h "$PERSISTENT_PATH" || warn "Could not run df on ${PERSISTENT_PATH}."

    if mountpoint -q "$PERSISTENT_PATH" 2>/dev/null; then
        echo "  OK: ${PERSISTENT_PATH} is a mount point."
    else
        warn "${PERSISTENT_PATH} is not a mount point. This may be okay, but verify it survives instance stop/restart."
    fi
else
    echo "No PERSISTENT_PATH set."
    echo "Set PERSISTENT_PATH to your durable storage path, for example:"
    echo "  PERSISTENT_PATH=/workspace"
    echo "  PERSISTENT_PATH=/mnt/nw/home/\$USER"
    echo "  PERSISTENT_PATH=/home/ubuntu/shared"
    warn "Persistent storage was not verified."
fi

section "5. Disk write speed"

DISK_TEST_DIR="${PERSISTENT_PATH:-/tmp}"

if [[ ! -d "$DISK_TEST_DIR" ]]; then
    DISK_TEST_DIR="/tmp"
fi

DISK_TEST_FILE="${DISK_TEST_DIR}/.gpu_sanity_disktest_$$"

echo "Testing write speed in: ${DISK_TEST_DIR}"

set +e
DD_OUTPUT="$(dd if=/dev/zero of="$DISK_TEST_FILE" bs=64M count=16 oflag=dsync 2>&1)"
DD_STATUS=$?
set -e

rm -f "$DISK_TEST_FILE" || true

echo "$DD_OUTPUT" | tail -1

if [[ "$DD_STATUS" -ne 0 ]]; then
    warn "Disk write test failed."
else
    DISK_MBPS="$(echo "$DD_OUTPUT" | awk -F, '/copied/ {gsub(/^ /,"",$NF); print $NF}' | awk '
        /GB\/s/ {print $1 * 1000}
        /MB\/s/ {print $1}
        /kB\/s/ {print $1 / 1000}
    ')"

    if [[ -n "${DISK_MBPS:-}" ]]; then
        DISK_INT="$(awk "BEGIN {printf \"%d\", ${DISK_MBPS}}")"
        echo "Approx disk write speed: ${DISK_INT} MB/s"
        if (( DISK_INT < MIN_DISK_MBPS )); then
            warn "Disk write speed is below MIN_DISK_MBPS=${MIN_DISK_MBPS}. Large datasets/checkpoints may be painful."
        else
            echo "  OK: Disk write speed >= ${MIN_DISK_MBPS} MB/s"
        fi
    else
        warn "Could not parse disk write speed."
    fi
fi

section "6. Network ingress"

if command -v curl >/dev/null 2>&1; then
    SPEED_BYTES="$(curl -L -o /dev/null -s -w "%{speed_download}" "https://speed.cloudflare.com/__down?bytes=104857600" || echo 0)"
    DOWNLOAD_MBPS="$(awk "BEGIN {printf \"%.1f\", ${SPEED_BYTES} * 8 / 1000000}")"
    echo "Download speed: ${SPEED_BYTES} B/s (${DOWNLOAD_MBPS} Mbps)"

    DOWNLOAD_INT="$(awk "BEGIN {printf \"%d\", ${DOWNLOAD_MBPS}}")"
    if (( DOWNLOAD_INT < MIN_DOWNLOAD_MBPS )); then
        warn "Download speed is below MIN_DOWNLOAD_MBPS=${MIN_DOWNLOAD_MBPS}. Pulling model weights or datasets may be slow."
    else
        echo "  OK: Download speed >= ${MIN_DOWNLOAD_MBPS} Mbps"
    fi
else
    warn "curl not installed; skipping network ingress test."
fi

section "7. Python and PyTorch CUDA"

if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 not found."
fi

python3 - <<'PY' || fail "PyTorch CUDA check failed. Install a CUDA-enabled PyTorch build or switch image."
import sys

try:
    import torch
except Exception as e:
    print(f"Could not import torch: {e}")
    raise SystemExit(1)

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
print(f"torch.cuda.device_count: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit(1)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    total_gb = props.total_memory / 1024**3
    print(f"GPU {i}: {props.name}, {total_gb:.1f} GiB")
PY

section "8. Short compute sanity test"

python3 - <<PY || fail "Compute sanity test failed."
import os
import time
import torch

min_tflops = float(os.environ.get("MIN_TFLOPS", "0"))

if not torch.cuda.is_available():
    print("CUDA unavailable.")
    raise SystemExit(1)

device = "cuda"
gpu = torch.cuda.get_device_name(0)

# Use a smaller matrix if VRAM is limited.
total_mem = torch.cuda.get_device_properties(0).total_memory
if total_mem >= 35 * 1024**3:
    n = 8192
elif total_mem >= 18 * 1024**3:
    n = 6144
else:
    n = 4096

print(f"GPU: {gpu}")
print(f"Matrix size: {n} x {n}")
print("Running fp16 matmul benchmark...")

a = torch.randn(n, n, device=device, dtype=torch.float16)
b = torch.randn(n, n, device=device, dtype=torch.float16)

for _ in range(3):
    c = a @ b

torch.cuda.synchronize()
t0 = time.time()

iters = 10
for _ in range(iters):
    c = a @ b

torch.cuda.synchronize()
elapsed = time.time() - t0

tflops = 2 * (n ** 3) * iters / elapsed / 1e12

print(f"Observed fp16 matmul throughput: {tflops:.1f} TFLOPS")

rough_floors = {
    "T4": 20,
    "A10": 40,
    "A10G": 40,
    "L4": 40,
    "L40": 60,
    "L40S": 80,
    "A40": 50,
    "A100": 120,
    "H100": 300,
    "H200": 350,
    "B200": 500,
    "RTX 3090": 30,
    "RTX 4090": 60,
}

floor = 0
for key, value in rough_floors.items():
    if key.lower() in gpu.lower():
        floor = value
        break

if min_tflops > 0:
    floor = min_tflops

if floor:
    print(f"Rough sanity floor: {floor:.1f} TFLOPS")
    if tflops < floor:
        print("Below sanity floor. This host may be degraded, throttled, oversubscribed, or using an unexpected precision path.")
        raise SystemExit(1)
else:
    print("No default floor for this GPU type. Compare against a known-good run.")

print("Compute sanity passed.")
PY

section "9. Result"

if (( WARNINGS > 0 )); then
    echo "Completed with ${WARNINGS} warning(s)."
    echo "Review warnings before starting an expensive job."

    if [[ "$STRICT_WARNINGS" == "1" ]]; then
        exit 2
    fi
else
    echo "All checks passed."
fi

echo
echo "Recommendation:"
echo "  - If this is a short/dev job: proceed."
echo "  - If this is a long training job: verify checkpointing before launch."
echo "  - If this is a marketplace/interruptible host and warnings look serious: relaunch."
