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

_REPO = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("PTB_RAW", _REPO / "mats-local" / "posttrainbench"))
VIEWER_DATA = Path(os.environ.get("PTB_DATA", RAW / "viewer_data"))
HIGHLIGHTS = Path(__file__).resolve().parents[1] / "highlights"


def raw_dir(run_id: str) -> Path:
    experiment, run_name = run_id.split("__", 1)
    return RAW / experiment / run_name
