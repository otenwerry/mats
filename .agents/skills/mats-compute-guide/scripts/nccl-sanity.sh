#!/usr/bin/env bash
# Generic multi-node NCCL sanity check.
#
# Use before distributed training on Lambda clusters, cloud clusters,
# Slurm allocations, or any multi-node GPU setup.
#
# Checks:
#   - nvidia-smi exists
#   - hostfile exists
#   - MPI exists
#   - nccl-tests can be built
#   - single-node all-reduce works
#   - multi-node all-reduce works
#   - multi-node bandwidth clears a configurable threshold
#
# Usage:
#   NUM_NODES=2 GPUS_PER_NODE=8 HOSTFILE=./hostfile ./nccl-sanity.sh
#
# Optional:
#   MIN_MULTI_NODE_BUSBW_GBPS=50 ./nccl-sanity.sh
#   NCCL_TESTS_DIR=/path/to/nccl-tests ./nccl-sanity.sh
#
# Notes:
#   - The right bandwidth threshold depends heavily on hardware.
#   - For real InfiniBand H100 clusters, <50 GB/s bus bandwidth is suspicious.
#   - For Ethernet-only clusters, lower bandwidth may be expected.

set -euo pipefail

NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HOSTFILE="${HOSTFILE:-./hostfile}"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-./nccl-tests}"
MIN_MULTI_NODE_BUSBW_GBPS="${MIN_MULTI_NODE_BUSBW_GBPS:-50}"
BUILD_DIR="${BUILD_DIR:-./build-nccl-sanity}"

fail() {
    echo "FAIL: $*"
    exit 1
}

warn() {
    echo "WARNING: $*"
}

section() {
    echo
    echo "=== $* ==="
}

echo "=== Generic NCCL Multi-node Sanity Check ==="
echo "Head host:   $(hostname)"
echo "Time UTC:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NUM_NODES:   ${NUM_NODES}"
echo "GPUS/node:   ${GPUS_PER_NODE}"
echo "HOSTFILE:    ${HOSTFILE}"
echo

section "1. Local prerequisites"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found."
command -v git >/dev/null 2>&1 || fail "git not found."
command -v make >/dev/null 2>&1 || fail "make not found."
command -v mpirun >/dev/null 2>&1 || fail "mpirun not found. Install OpenMPI or use your cluster's MPI module."

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

LOCAL_GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
if [[ "$LOCAL_GPU_COUNT" != "$GPUS_PER_NODE" ]]; then
    warn "Local GPU count (${LOCAL_GPU_COUNT}) does not match GPUS_PER_NODE (${GPUS_PER_NODE}). This may be okay if this node is special, but usually it is wrong."
fi

section "2. Hostfile"

[[ -f "$HOSTFILE" ]] || fail "Hostfile not found: ${HOSTFILE}"

echo "Hostfile contents:"
cat "$HOSTFILE"

HOST_LINES="$(grep -v '^[[:space:]]*$' "$HOSTFILE" | grep -v '^[[:space:]]*#' | wc -l | tr -d ' ')"
if (( HOST_LINES < NUM_NODES )); then
    fail "Hostfile has fewer non-empty hosts (${HOST_LINES}) than NUM_NODES (${NUM_NODES})."
fi

section "3. SSH/MPI reachability"

echo "Testing mpirun hostname across nodes..."

mpirun --hostfile "$HOSTFILE" -np "$NUM_NODES" hostname || {
    fail "mpirun hostname failed. Check SSH keys, hostfile, MPI installation, and firewall/security group rules."
}

section "4. Build nccl-tests"

mkdir -p "$BUILD_DIR"

if [[ ! -d "$NCCL_TESTS_DIR" ]]; then
    git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_DIR"
fi

pushd "$NCCL_TESTS_DIR" >/dev/null
make MPI=1 -j"$(nproc)" || fail "Failed to build nccl-tests with MPI=1."
popd >/dev/null

ALL_REDUCE_BIN="${NCCL_TESTS_DIR}/build/all_reduce_perf"

[[ -x "$ALL_REDUCE_BIN" ]] || fail "all_reduce_perf binary not found after build."

section "5. Single-node all-reduce baseline"

SINGLE_LOG="${BUILD_DIR}/single_node_all_reduce.log"

"$ALL_REDUCE_BIN" \
    -b 1G \
    -e 4G \
    -f 2 \
    -g "$GPUS_PER_NODE" \
    | tee "$SINGLE_LOG"

SINGLE_BUSBW="$(awk 'NF >= 12 && $1 !~ /^#/ {val=$12} END {print val+0}' "$SINGLE_LOG")"

if [[ -z "$SINGLE_BUSBW" || "$SINGLE_BUSBW" == "0" ]]; then
    warn "Could not parse single-node bus bandwidth."
else
    echo "Parsed single-node bus bandwidth: ${SINGLE_BUSBW} GB/s"
fi

section "6. Multi-node all-reduce"

MULTI_LOG="${BUILD_DIR}/multi_node_all_reduce.log"
TOTAL_PROCS=$((NUM_NODES * GPUS_PER_NODE))

echo "Running ${TOTAL_PROCS} total MPI processes..."

mpirun \
    --hostfile "$HOSTFILE" \
    -np "$TOTAL_PROCS" \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}" \
    -x NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
    -x NCCL_IB_HCA="${NCCL_IB_HCA:-}" \
    "$ALL_REDUCE_BIN" \
        -b 1G \
        -e 4G \
        -f 2 \
        -g 1 \
    | tee "$MULTI_LOG"

MULTI_BUSBW="$(awk 'NF >= 12 && $1 !~ /^#/ {val=$12} END {print val+0}' "$MULTI_LOG")"

if [[ -z "$MULTI_BUSBW" || "$MULTI_BUSBW" == "0" ]]; then
    fail "Could not parse multi-node bus bandwidth."
fi

echo "Parsed multi-node bus bandwidth: ${MULTI_BUSBW} GB/s"

section "7. Interpretation"

awk -v bw="$MULTI_BUSBW" -v min="$MIN_MULTI_NODE_BUSBW_GBPS" 'BEGIN {
    if (bw < min) {
        printf("FAIL: Multi-node bus bandwidth %.1f GB/s is below threshold %.1f GB/s.\n", bw, min)
        exit 1
    } else {
        printf("OK: Multi-node bus bandwidth %.1f GB/s is above threshold %.1f GB/s.\n", bw, min)
    }
}' || {
    echo
    echo "Likely causes:"
    echo "  - NCCL is using Ethernet instead of InfiniBand."
    echo "  - NCCL_SOCKET_IFNAME points to the wrong interface."
    echo "  - NCCL_IB_HCA is unset or wrong."
    echo "  - Security groups/firewalls block node-to-node traffic."
    echo "  - MPI is launching processes on the wrong nodes."
    echo
    echo "Useful debugging commands:"
    echo "  ibstat"
    echo "  ibv_devinfo"
    echo "  ip addr"
    echo "  NCCL_DEBUG=INFO"
    echo
    exit 1
}

if [[ -n "${SINGLE_BUSBW:-}" && "$SINGLE_BUSBW" != "0" ]]; then
    RATIO="$(awk -v m="$MULTI_BUSBW" -v s="$SINGLE_BUSBW" 'BEGIN {printf "%.2f", m / s}')"
    echo "Multi-node / single-node bus bandwidth ratio: ${RATIO}"
    echo "A much lower multi-node ratio is expected, but an extremely low ratio suggests networking problems."
fi

echo
echo "NCCL sanity check passed."
echo
echo "Before launching training, also verify:"
echo "  - checkpoint path is persistent/shared"
echo "  - all nodes see the same code and data"
echo "  - torchrun/deepspeed config matches NUM_NODES and GPUS_PER_NODE"
echo "  - failure/restart behavior is understood"
