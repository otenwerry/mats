"""Print raw-dir paths of runs flagged for cheating that have NO judge output.

"Flagged" = the original benchmark judges flagged it (contamination detected /
disallowed use detected). "No judge output" = the raw task dir lacks a
non-empty judge_output.json (the judge agent's investigation trajectory) — for
these, only the one-line verdict file exists. One absolute path per line, no
other output, so it's copy-pasteable.

Usage: uv run python posttrainbench/list_flagged_missing_judge.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)

import json

import paths


def main():
    runs = json.load(open(paths.VIEWER_DATA / "index.json"))["runs"]
    for r in runs:
        contam = (r.get("contamination") or "").strip()
        disallowed = (r.get("disallowed_model") or "").strip()
        if not (contam.startswith("contamination detected")
                or disallowed == "disallowed use detected"):
            continue
        d = paths.raw_dir(r["run_id"])
        j = d / "judge_output.json"
        if not j.exists() or j.stat().st_size == 0:
            print(d.relative_to(paths.RAW))


if __name__ == "__main__":
    main()
