"""Regenerate the prompts the PTB harness gave the agents.

The initial task prompt was passed to the agent as a CLI argument and is not
echoed in any stream, so we regenerate it with the harness's own
src/eval/general/get_prompt.py. The harness clone may be newer than the one
that produced a given run (observed drift: timestamp prefixes, reprompt
loops), so every regenerated prompt carries a fidelity flag rather than being
treated as verbatim.

Relaunch ("reprompt") continuation prompts come from the loop in
agents/claude_reprompt/solve.sh:
    You still have <H>h <M>m remaining. Please continue improving your result
    and maximize performance.
The exact remaining time per relaunch was never recorded for runs without
line timestamps, so we estimate it and flag the estimate.
"""
from __future__ import annotations

import subprocess

from .runs import PTB_REPO, Trajectory

_cache: dict[tuple, str] = {}

CONTINUATION_TEMPLATE = ("You still have {h}h {m}m remaining. Please continue "
                         "improving your result and maximize performance.")


def task_prompt(traj: Trajectory) -> str:
    key = (traj.model_to_train, traj.benchmark_id, traj.num_hours, traj.agent)
    if key not in _cache:
        out = subprocess.run(
            ["python3", "src/eval/general/get_prompt.py",
             "--model-to-train", traj.model_to_train,
             "--benchmark-id", traj.benchmark_id,
             "--num-hours", str(traj.num_hours),
             "--num-gpus", "1",
             "--agent", traj.agent],
            cwd=PTB_REPO, capture_output=True, text=True, check=True)
        _cache[key] = out.stdout
    return _cache[key]


def continuation_prompt(frac_elapsed: float, total_hours: int) -> tuple[str, str]:
    """(prompt text, how the numbers were estimated)."""
    remaining_min = max(0, int(round(total_hours * 60 * (1 - frac_elapsed))))
    h, m = divmod(remaining_min, 60)
    basis = (f"remaining time estimated by linear interpolation over stream "
             f"position (frac_elapsed={frac_elapsed:.3f})")
    return CONTINUATION_TEMPLATE.format(h=h, m=m), basis
