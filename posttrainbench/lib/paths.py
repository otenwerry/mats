"""Shared path resolution for the PostTrainBench tooling.

Code is in this repo (git); the data is in the sibling `mats-local` tree.
A `run_id` is `{experiment}__{run_name}`; the raw per-task dir (with the agent
trace, judge_output.json, workspace files, verdicts) is
`{RAW}/{experiment}/{run_name}/`, and the pre-parsed viewer JSON is
`{VIEWER_DATA}/{run_id}.json`.
"""
from __future__ import annotations

import os
from pathlib import Path

def _find_repo_root() -> Path:
    """Nearest ancestor holding the sibling `mats-local` data tree — robust to
    running from a git worktree (see lib/runs.py:_find_root)."""
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "mats-local").is_dir():
            return p
    return here.parents[3]


_REPO = _find_repo_root()
RAW = Path(os.environ.get("PTB_RAW", _REPO / "mats-local" / "posttrainbench"))
VIEWER_DATA = Path(os.environ.get("PTB_DATA", RAW / "viewer_data"))
REFERENCES = _REPO / "references"  # read-only clones of relevant external repos
HIGHLIGHTS = Path(__file__).resolve().parents[1] / "highlights"


def raw_dir(run_id: str) -> Path:
    experiment, run_name = run_id.split("__", 1)
    return RAW / experiment / run_name
