# Sanity checks before paid or long runs

The useful checks are nearly the same across GPU providers:

- Is the expected number of GPUs visible?
- Is the GPU model what you requested?
- Is CUDA usable from PyTorch?
- Is there already a process using the GPU?
- Is checkpoint storage actually present?
- Is disk write speed terrible?
- Is network ingress terrible for model/dataset pulls?
- Is a short matmul benchmark wildly below the expected class?

Three scripts in `scripts/` (generic templates — adapt as needed):

| Script | Use for | Do not use for |
| --- | --- | --- |
| `gpu-sanity.sh` | Single-node MATS, RunPod, Vast, Lambda, cloud VMs | Modal standard use; multi-node NCCL |
| `nccl-sanity.sh` | Multi-node distributed training | Single-GPU pods; Modal function sweeps |
| `modal-smoke-test.py` | One tiny Modal invocation before a big sweep | Host-level checks |

## Single-node examples

```bash
# Basic smoke test. Assumes 1 GPU and PyTorch should work.
bash gpu-sanity.sh

# A 2-GPU pod or VM.
EXPECTED_GPUS=2 bash gpu-sanity.sh

# Make sure you got an H100/H200/B200-class host.
EXPECTED_GPUS=8 EXPECTED_GPU_NAME='H100|H200|B200' bash gpu-sanity.sh

# Make sure every visible GPU has enough VRAM for the job.
EXPECTED_GPUS=1 MIN_VRAM_GB=40 bash gpu-sanity.sh

# Fail if your expected checkpoint path is not present.
REQUIRE_PERSISTENT=1 PERSISTENT_DIR=/workspace bash gpu-sanity.sh

# Treat warnings as failures for expensive jobs.
FAIL_ON_WARN=1 EXPECTED_GPUS=8 bash gpu-sanity.sh
```

## Multi-node NCCL examples

```bash
# hostfile example:
# node-a slots=8
# node-b slots=8

NUM_NODES=2 GPUS_PER_NODE=8 HOSTFILE=./hostfile bash nccl-sanity.sh

# If you know the cluster should exceed a stronger bandwidth threshold,
# set it explicitly.
MIN_MULTI_BW_GBS=<your-threshold> \
NUM_NODES=2 GPUS_PER_NODE=8 HOSTFILE=./hostfile bash nccl-sanity.sh
```

## How to interpret results

- A **GPU count**, **CUDA**, or **PyTorch** failure usually means the
  environment is wrong. Stop and fix it before launching.
- A **persistent storage** failure means you are about to lose checkpoints.
  Relaunch with the correct volume or change the checkpoint path.
- A **network** warning means model/data pulls may dominate runtime.
  Pre-stage weights or choose another host.
- A **disk** warning means checkpointing or dataset loading may be slow.
  Use another volume, another data center, or another host.
- A **compute** warning is not proof the host is bad, but it is a reason to
  inspect thermals, GPU sharing, MIG, power limits, or host quality.
- A **multi-node NCCL** failure means you should not start distributed
  training. Fix networking/NCCL first.
