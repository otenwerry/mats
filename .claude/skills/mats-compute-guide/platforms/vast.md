# Vast.ai

## Use when

- You need very cheap GPU time.
- Your job checkpoints frequently.
- You can tolerate interruptions, host variability, and manual filtering.
- You are running exploration sweeps where losing one run is acceptable.

## Do not use when

- The job is expensive to restart.
- You need guaranteed uptime.
- You need sensitive-data or compliance guarantees.
- You cannot babysit host quality, storage, and bandwidth.

## Access

Use the Compute Request Form. Make sure you are using the **MATS Vast team
account**, not a personal Vast account. Volumes, SSH keys, instances, and
credits do not carry over between accounts.

## Pricing and availability

Use GPU Compass for current market comparison, then verify the specific Vast
listing in the Vast UI before launch.

- Vast interruptible is often the cheapest raw GPU time.
- Vast on-demand is usually less risky than interruptible but still host-specific.
- Sort by effective value, not raw hourly rate — a very cheap unreliable host
  is often more expensive per completed job than a less cheap reliable host.

## Host filters (practical defaults)

| Filter | Recommendation |
| --- | --- |
| Reliability score | High for important on-demand work |
| Interruptible reliability | High enough that restarts will not dominate the job |
| DLPerf | Compare against the expected GPU class |
| Network speed | Check before downloading large weights |
| Disk speed and disk size | Check before dataset-heavy jobs |
| Max duration | Make sure the host can run long enough |
| Sort key | Prefer value per useful performance over raw hourly price |

## Storage

Vast volumes are local to the physical machine:

- A volume is tied to the machine where it was created, can only attach to
  instances on that same machine, and cannot be moved.
- Copy important outputs to S3, GCS, the MATS cluster, or another durable store.
- Stopping an instance does not necessarily stop storage charges. Destroy
  instances and delete unneeded volumes when done.

## Launch checklist

1. Use a recommended template unless you know you need a custom one.
2. Filter for high reliability for important work.
3. Check DLPerf, network bandwidth, disk size, and max duration.
4. Sort by value per useful performance, not only raw hourly price.
5. Start with on-demand for important jobs; use interruptible only with checkpoints.
6. Run `scripts/gpu-sanity.sh` before a long run.

```bash
# Example: push checkpoints out of the host as you train
aws s3 cp checkpoint.pt s3://my-bucket/run-id/checkpoint.pt
```
