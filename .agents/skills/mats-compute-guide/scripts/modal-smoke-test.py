"""
Generic Modal/serverless GPU smoke test.

Use this when creating a new app, changing GPU type, changing image
dependencies, changing region, or before launching a large fan-out sweep.

This is not a host benchmark. It checks that your serverless GPU function
can start, see CUDA, run a small tensor operation, and return results.

Run:
    modal run modal_smoke_test.py
"""

from __future__ import annotations

import time
import modal

GPU_TYPE = "H100"

app = modal.App("gpu-smoke-test")

image = (
    modal.Image.debian_slim()
    .pip_install("torch")
)

@app.function(
    gpu=GPU_TYPE,
    image=image,
    timeout=300,
)
def smoke_test() -> dict:
    import os
    import platform
    import torch

    started = time.time()

    result = {
        "gpu_requested": GPU_TYPE,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "modal_region": os.environ.get("MODAL_REGION", "unset"),
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Modal function.")

    result["device_name"] = torch.cuda.get_device_name(0)

    # Tiny compute check. Keep this small: the point is a smoke test,
    # not a benchmark.
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)

    for _ in range(2):
        c = a @ b

    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    tflops = 2 * (n ** 3) * 5 / elapsed / 1e12

    result["smoke_tflops_fp16"] = round(tflops, 1)
    result["function_runtime_seconds"] = round(time.time() - started, 2)

    return result

@app.local_entrypoint()
def main():
    t0 = time.time()
    result = smoke_test.remote()
    wall_time = time.time() - t0

    print("=== Modal/serverless GPU smoke test ===")
    print(f"End-to-end wall time: {wall_time:.1f} seconds")
    print()

    for key, value in result.items():
        print(f"{key}: {value}")

    print()
    print("Interpretation:")
    print("- If CUDA is unavailable, fix the image or GPU config.")
    print("- If startup is too slow, reduce image size or pre-build dependencies.")
    print("- If this works, run a tiny app-specific batch before launching a large sweep.")
    print("- Cap concurrency deliberately; do not accidentally fan out hundreds of GPU calls.")
