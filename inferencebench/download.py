"""Download the full InferenceBench trajectories dataset from HuggingFace.

The whole dataset is small (~0.38 GB, 269 runs), so we mirror all of it into
mats-local (kept off github) rather than fetching runs piecemeal:

  mats-local/inferencebench/data/
    manifest.json
    runs/run_NNNN/{trace.jsonl, run_meta.json, metrics.json, logs/...}

Free (plain HTTP to the HF CDN, no API keys). Re-running is cheap: files
already present are skipped via etag checks.

Usage:  uv run python mats/inferencebench/download.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "aisa-group/InferenceBench-Trajectories"
DEST = Path(__file__).resolve().parents[2] / "mats-local" / "inferencebench" / "data"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"downloading {REPO_ID} -> {DEST}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=DEST)
    n_runs = len(list((DEST / "runs").iterdir()))
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"done: {n_runs} runs, {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
