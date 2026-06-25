---
name: mats-compute-guide
description: How to choose and use compute for MATS work — routing between the MATS cluster, RunPod, Modal, Vast, Lambda, and cloud VMs; billing/access rules (Compute Request Form, team accounts, no personal cards); storage and reliability gotchas; pre-run GPU sanity checks. Use whenever renting GPUs, picking a platform, setting up a GPU box, or debugging provider issues.
---

# MATS compute guide

Per-platform details live in `platforms/` (read the one you need):
`mats-cluster.md` · `runpod.md` · `modal.md` · `vast.md` · `lambda.md`.
Pre-run checks: `sanity-checks.md` + runnable scripts in `scripts/`.

## Route by workload, not by platform

Before choosing a platform, identify what the job actually needs: GPU type and
VRAM · expected runtime · storage size and persistence · Docker/root/privileged
requirements · single-node vs multi-node · interactive vs batch vs serverless ·
tolerance for interruption · deadline/queue tolerance.

**Practical routing rule:** pick the platform that minimizes
**completed-job cost** — runtime, retries, storage, data transfer, debugging
time, and risk of lost work included:

```
completed_job_cost = hourly_rate * runtime_hours + storage + bandwidth + retry_overhead
                     (+ your time debugging bad hosts and recovering lost work)
```

A GPU with a higher hourly rate can still be cheaper if it finishes much
faster or fails less often. Benchmark a small slice for `runtime_hours`; do
not guess from the GPU name alone.

The MATS cluster is a good fit for some batch jobs, especially when L40-class
GPUs, shared storage, and the cluster runtime limits are sufficient. Other
platforms are on MATS billing when you need different GPUs, more VRAM,
Docker/root, multi-node, serverless scaling, large storage, urgent
availability, or stronger reliability — reach out to the Compute Team via the
**Compute Request Form**. Never add personal payment information for MATS
work.

## The three questions that usually decide the platform

**1. What is the unit of work?**

| Workload shape | Usually choose |
| --- | --- |
| One experiment, one machine, hours to days | MATS cluster, RunPod Pod, Vast instance, or Lambda instance |
| Many short GPU calls with bursty parallelism | Modal or RunPod Serverless |
| Multi-node distributed training | Lambda 1-Click Cluster, RunPod Instant Cluster, Modal clustered functions, or another approved cluster |
| Hosted API model evals | Anthropic, OpenAI, Gemini, or Vertex AI API, OpenRouter. No GPU needed. |

**2. What is the cost of interruption?**

| If interruption means… | Prefer |
| --- | --- |
| Losing hours of work | MATS cluster, RunPod Secure Cloud, Lambda, or a high-reliability Vast on-demand host |
| Losing seconds or minutes because you checkpoint | Vast interruptible, RunPod Community Cloud, or preemptible/serverless work |
| Losing only one idempotent function call | Modal or RunPod Serverless |

**3. Does the workload need Docker-in-Docker, root, or `--privileged`?**

| Need | Usually choose |
| --- | --- |
| Full VM control, sudo, Docker daemon, or privileged containers | Lambda VM, GCP `a3-*`, AWS `p5`, or another approved full VM |
| Plain Python, PyTorch, Jupyter, SSH, or a custom image | MATS cluster, RunPod, Modal, Vast, or Lambda |

> Container-spawning workloads (ControlArena, agent eval harnesses, anything
> requiring nested containers) generally don't fit the MATS cluster, RunPod
> Pods, Vast containers, or standard Modal functions — none expose
> `--privileged`. A VM-based platform (Lambda, GCP `a3-highgpu-*`, AWS `p5`)
> is usually the right home. Modal's Docker-in-Sandbox is still alpha — check
> with the Compute Team before relying on it for a deadline.

## 30-second decision guide

| Situation | Go here first |
| --- | --- |
| Calling Claude, GPT, Gemini, or hosted models for evals | API provider. Check AI Down if requests fail. |
| Single-GPU experiment, 1-24 hours | MATS cluster first. If blocked, RunPod or Vast. |
| Multi-GPU, single-node fine-tune | MATS cluster if it fits; otherwise RunPod Secure, RunPod Community, Vast, or Lambda depending on reliability needs. |
| Workload needs root, Docker daemon, or privileged containers | Lambda VM, GCP `a3-*`, AWS `p5`, or another approved full VM. |
| Serverless inference or bursty parallel sweeps | Modal or RunPod Serverless. |
| Cheapest checkpointed runs where interruptions are acceptable | Vast interruptible, with reliability and DLPerf filtering. |
| Multi-node training without a long reservation | RunPod Instant Clusters or Modal clustered functions, if approved. |
| Large, reserved, production-style multi-node training | Lambda 1-Click Cluster or another approved reserved cluster, after confirming funding. |

## Live pricing, availability, and status

Use live references instead of prices copied into this doc:

1. **GPU Compass**: https://gpus.skypilot.co/ (and https://vast.ai/pricing) —
   price/capacity comparison across providers; first stop when choosing.
2. **AI Down**: https://aidown.io/ — API-provider status, latency, outages.

Relative cost guidance (qualitative; check live prices before acting):
MATS cluster is cheapest when the job fits · Vast interruptible is often the
cheapest raw GPU time but completed-job cost can be worse · RunPod Community
< Secure on price, > on risk · Modal wins for bursty fan-out, loses for one
long saturated job · Lambda is chosen for full-VM control and multi-node, not
raw price, and has availability issues · newer/faster GPUs can be cheaper per
completed job for training, while cheaper GPUs are often better for dev.

## Picking a GPU type

| Workload | Good first choice | Why |
| --- | --- | --- |
| Dev iteration, notebooks, small inference | RTX 4090, L40, L40S, or L4 | Cheap; high-end GPUs are often wasted here |
| Mech interp on GPT-2, Pythia, Llama <= 7B | L40 48 GB or 4090 24 GB | Enough VRAM for many workflows |
| Multiple checkpoints or hooks in memory | A100 40 GB or 80 GB | VRAM headroom matters more than peak FLOPs |
| SAE or dictionary learning at scale | A100 80 GB, H100, or newer | Activations and bandwidth become bottlenecks |
| LoRA/QLoRA fine-tune <= 13B | A100 40 GB, L40S, or 4090 for QLoRA | Single-GPU can be enough |
| Full fine-tune >= 30B | Multi-GPU A100/H100/newer with high-speed interconnect | Scheduling and memory are the hard parts |
| Batch inference/evals | L40S, L4, A100, or H100 with vLLM | Pick by completed-job cost, not hourly price |
| 70B+ training or serious serving | H100 or newer | Throughput and interconnect matter |
| Multi-node training | Lambda, RunPod Instant Cluster, Modal clustered functions, or another approved cluster | Needs coordinated networking and scheduling |

Two rules of thumb: **VRAM gates feasibility; bandwidth gates speed.**
NVLink/interconnect matters only for multi-GPU training, not single-GPU dev.

## Checkpoint discipline

You will eventually lose a run; make every long job resumable. On
interruptible/preemptible/marketplace hosts, checkpoints on the server's local
disk are not enough — the training script should automatically transfer
checkpoints off-server to reliable storage after each save, and verify the
upload exists before continuing.

## Common mistakes

1. Treating temporary pod/container disk as persistent.
2. Filling `/tmp` on the cluster login node.
3. Asking for 4 GPUs when the job only uses 1.
4. Launching a large Modal sweep without checking concurrency caps.
5. Picking Vast by raw hourly price without checking reliability, DLPerf, bandwidth, and max duration.
6. Running unpinned installs.
7. Skipping sanity checks (`sanity-checks.md`).
