# Modal

## Use when

- Your workload can be expressed as Python functions.
- You want fan-out parallelism over prompts, seeds, configs, or data shards.
- You want an inference endpoint that scales to zero.
- You want simple deployment without managing a VM.

## Do not use when

- You need a persistent interactive machine.
- You need guaranteed non-preemptible GPU execution.
- You need Docker-in-Docker for deadline-critical production work (Modal's
  Docker-in-Sandbox support is still alpha — check with the Compute Team
  before relying on it for a deadline).
- You will exceed the Team plan's GPU concurrency cap without planning around it.

## Access

Use the Compute Request Form. MATS is on a Modal Team plan unless the Compute
Team says otherwise.

## Pricing and availability

Use GPU Compass and the Modal dashboard/pricing estimator for live rates. The
important part is the cost shape, not the sticker price:

- Modal is attractive for bursty GPU work because idle functions do not hold a GPU.
- Modal can be expensive for long, saturated, always-on jobs compared with a
  plain pod or VM. If a deadline crunch is more your issue than budget, consider it.
- Region, image build time, volume use, cold starts, CPU, memory, and
  concurrency caps can matter as much as the GPU line item.
- GPU work should be made idempotent and checkpointed because GPU functions
  can be interrupted.

## Storage

- Function-local filesystem: temporary.
- Modal Volume: persistent across invocations.
- External object storage: best for large datasets, artifacts, and
  cross-platform results.

## Sanity checking

Modal abstracts host choice away, so the generic `gpu-sanity.sh` is the wrong
interface. Do a **workload smoke test** instead (`scripts/modal-smoke-test.py`):

- Run one tiny invocation before launching a large `.map()` sweep.
- Confirm the returned GPU type and CUDA availability if your code depends on it.
- Confirm model weights load from the intended cache or volume.
- Cap concurrency before fanning out.
- Confirm the app stops when the sweep is done.

## Cheatsheet

```python
import modal

app = modal.App("my-eval")

@app.function(
    gpu="H100",
    image=modal.Image.debian_slim().pip_install("transformers", "torch"),
    timeout=600,
)
def run_one(prompt: str) -> str:
    # Put your inference or eval code here.
    return "result"

@app.local_entrypoint()
def main():
    prompts = ["prompt 1", "prompt 2"]
    results = list(run_one.map(prompts, return_exceptions=True))
    print(results)
```

```bash
# Run locally against Modal
modal run my_eval.py

# Create a persistent volume
modal volume create my-results

# Stop an app when done
modal app stop <app-name>
```
