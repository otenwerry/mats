# RunPod

## Use when

- You need a self-contained Jupyter or SSH pod.
- You need one machine with multiple GPUs.
- You need serverless inference or scale-to-zero workers.
- You want on-demand multi-node clusters without a long reservation.

Typical shapes:

- **Single-GPU, 1–24h experiment** → **Community Cloud** (A100 80GB is the usual default).
- **2–8 GPU fine-tune in one node** → **Secure Cloud** for reliability.
- **Serverless / scale-to-zero inference endpoint** → **Serverless**.

## Do not use when

- Your workload needs privileged Docker or a full VM. Container-spawning
  workloads (ControlArena, agent eval harnesses, anything needing
  Docker-in-Docker) don't fit — RunPod doesn't expose `--privileged`.
  Use a VM platform (Lambda, GCP `a3-*`, AWS `p5`) instead.
- You require a specific enterprise support or reservation model without confirming it first.
- You cannot tolerate pod storage mistakes or interruptions and have not designed checkpointing.

## Access

Use the Compute Request Form. You should be added to the relevant MATS team account.

> Do not add personal payment information for MATS work. If RunPod asks for a
> card, stop and ask the Compute Team.

## Pricing and availability

Use GPU Compass (https://gpus.skypilot.co/) to compare current RunPod GPU
listings against other providers. Use the RunPod dashboard for final
availability and exact launch-time billing.

- **Community Cloud**: cheaper, on shared/lower-reliability hosts. Good for
  low-stakes, short jobs you're OK restarting.
- **Secure Cloud**: dedicated hosts, more reliable. Use for multi-GPU work or
  anything where a mid-run interruption is costly.
- Serverless is a different cost shape from Pods: good for bursty work, less
  obvious for one long job.
- Network Volume placement can constrain which GPUs and data centers are usable.

## Storage

| Storage type | Persistence | Use for |
| --- | --- | --- |
| Container disk | Temporary | OS, temporary files, caches |
| Pod/volume disk | Tied to the Pod or volume lifecycle | Checkpoints or data tied to that Pod |
| Network Volume | Persistent but data-center-specific | Datasets, shared data, portable checkpoints within that data center |

A pod's default disk is **ephemeral**: everything you write to it is wiped
when the pod stops. Save anything you want to keep to a **Network Volume**:

- Volumes persist across pods but are **locked to a specific data center** —
  pick the DC where you plan to launch pods; pods in any other DC can't see
  them. If a GPU is unavailable in that DC, your storage choice becomes your
  capacity bottleneck.
- They're slower than the pod's local disk. Do active I/O on local disk and
  checkpoint out to the volume.
- Data is not automatically synced between Network Volumes in different DCs.

## Best practices

- Stop pods when not in use (closing the browser tab does not stop billing).
- Use spot instances for cost savings where interruption is tolerable.
- Set up automatic shutdowns.
- Regularly back your work up off the pod.
- Run `scripts/gpu-sanity.sh` before starting a long run.

## How to check your balance

1. Go to the Billing section in your RunPod account. Ensure you're in the team
   account (click the icon in the upper right corner).
2. Current spend and estimated remaining funding are at the top of the page.
3. Billing Explorer at the bottom shows a breakdown of recent spend.

## Common issues

- **Pod won't start.** Check the team account balance first. If funded, the
  GPU type/region may be at capacity — try a different GPU or region, or
  switch tiers (Community ↔ Secure). If everything is unavailable, check
  status.runpod.io for a maintenance window.
- **Lost data after stopping a pod.** That's the ephemeral disk being wiped.
  Use a Network Volume next time.
- **Can't see my Network Volume from a new pod.** The volume is in a different
  data center than the pod. Launch the pod in the volume's DC, or create a new
  volume in the desired DC and copy data over before stopping the old pod.

## Cheatsheet

```bash
# SSH into a Pod
ssh root@<pod-ip> -p <pod-ssh-port>

# Check mounted storage
df -h

# Put Hugging Face cache on persistent workspace storage, if that path is durable
mkdir -p /workspace/hf_cache
ln -s /workspace/hf_cache ~/.cache/huggingface

# Stop via dashboard or CLI. Do not just close the browser tab.
runpodctl stop <pod-id>
```
