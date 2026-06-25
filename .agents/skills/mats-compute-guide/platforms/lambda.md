# Lambda

## Use when

- You need full VM control with sudo.
- You need Docker, Docker-in-Docker, or privileged containers (this includes
  container-spawning agent harnesses, apptainer, ControlArena, etc.).
- You need reliable GPU instances.
- You need large, reserved, multi-node training through 1-Click Clusters.

## Do not use when

- The MATS cluster can run the job.
- You only need a quick, cheap single-GPU test.
- You have not confirmed reimbursement or funding path.

## Access

Lambda is usually **reimbursement or approval-based** rather than direct MATS
billing. Ask in `#support-mats-cluster` before signing up or committing spend.

## Pricing and availability

Use GPU Compass for current relative pricing and Lambda's dashboard for exact
launch-time cost. For clusters or reservations, confirm with the Compute Team
before committing. Known caveat: Lambda has availability issues for popular
GPUs — check before planning a deadline around it.

- Usually the right tool when you need full VM control.
- Usually the right tool for serious multi-node training with approved budget.
- Usually not the first choice for a short, simple experiment that fits on the
  cluster or a pod.

## Storage

- Local NVMe scratch per instance.
- Persistent/shared storage options depend on instance or cluster type (Lambda
  tends to have better linked-storage support than pod providers).
- For clusters, confirm the shared path before launching training.

## Gotchas

- Cluster tiers can have meaningful minimums and approval requirements.
- Confirm reimbursement before committing.
- A full VM gives you power, but also more responsibility for cleanup and security.
- For multi-node work, run `scripts/nccl-sanity.sh` before training.

## Cheatsheet

```bash
# SSH into a Lambda VM
ssh ubuntu@<vm-ip>

# Docker and NVIDIA drivers are often preinstalled on Lambda GPU VMs.
nvidia-smi
docker --version
```

```bash
# Example torchrun pattern for a multi-node cluster.
# Fill these variables from the cluster environment/docs.
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$HEAD_NODE_IP:29500 \
  train.py
```
