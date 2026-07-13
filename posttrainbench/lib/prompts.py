"""Regenerate the prompts the PTB harness gave the agents.

The initial task prompt was passed to the agent as a CLI argument and is not
echoed in any stream, so we regenerate it with the harness's own
src/eval/general/get_prompt.py — with era matching: harness commit aebea2f
(2026-04-04) is the ONLY prompt.txt change in the window our runs span, and
the same commit added timestamp line-prefixes to trajectory files. So a
trajectory with no line timestamps ran on the pre-April harness, whose prompt
said "equiped" (typo) where the current template says "equipped" — the entire
measured difference (both prompt.txt and get_prompt.py diffed, 2026-07-13).
task_prompt() restores the era wording, making regeneration byte-exact for
the public-harness window; the residual risk (run-time harness matching no
public commit) stays flagged.

Relaunch ("reprompt") continuation prompts come from the loop in
agents/claude_reprompt/solve.sh:
    You still have <H>h <M>m remaining. Please continue improving your result
    and maximize performance.
The exact remaining time per relaunch was never recorded for runs without
line timestamps, so we estimate it and flag the estimate. NB: for the
non-reprompt agents (claude / claude_non_api / qwen3max) there is evidence
this template is the wrong content entirely (see posttrainbench/ISSUES.md,
"unexplained restarts").
"""
from __future__ import annotations

import re
import subprocess

from .runs import PTB_REPO, Trajectory

_cache: dict[tuple, str] = {}
_era_cache: dict[str, bool] = {}

CONTINUATION_TEMPLATE = ("You still have {h}h {m}m remaining. Please continue "
                         "improving your result and maximize performance.")

# The one line that differs between the pre-2026-04-04 prompt template and the
# current one (num_gpus=1 fill).
_GPU_LINE_CURRENT = "- The machine is equipped with an Nvidia H100 GPU."
_GPU_LINE_PRE_APRIL = "- The machine is equiped with an Nvidia H100 GPU."
_TS_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def pre_april_harness(traj: Trajectory) -> bool:
    """True iff the run predates harness commit aebea2f (2026-04-04), detected
    by the absence of timestamp line-prefixes (added in that same commit)."""
    if traj.run_id not in _era_cache:
        has_ts = False
        try:
            with open(traj.traj_file, errors="replace") as f:
                for i, line in enumerate(f):
                    if _TS_PREFIX.match(line):
                        has_ts = True
                        break
                    if i > 500:
                        break
        except OSError:
            pass
        _era_cache[traj.run_id] = not has_ts
    return _era_cache[traj.run_id]


def task_prompt(traj: Trajectory) -> str:
    """The initial task prompt, regenerated with era-matched wording."""
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
    text = _cache[key]
    if pre_april_harness(traj):
        if _GPU_LINE_CURRENT not in text:
            raise RuntimeError(
                f"era matching failed for {traj.run_id}: expected GPU line not found "
                f"in the regenerated prompt (get_prompt.py output changed?)")
        text = text.replace(_GPU_LINE_CURRENT, _GPU_LINE_PRE_APRIL)
    return text


def prompt_provenance(traj: Trajectory) -> str:
    """Fidelity-flag detail describing how the task prompt was regenerated."""
    if pre_april_harness(traj):
        return ("initial task prompt regenerated via the harness's get_prompt.py with "
                "era-matched wording (no line timestamps -> pre-2026-04-04 harness -> "
                "original 'equiped' GPU line restored); byte-exact for the public-harness "
                "window, residual risk only if the run-time harness matched no public commit")
    return ("initial task prompt regenerated via the harness's get_prompt.py; this run's "
            "line timestamps date it to the current prompt era (no template change since)")


def continuation_prompt(frac_elapsed: float, total_hours: int) -> tuple[str, str]:
    """(prompt text, how the numbers were estimated)."""
    remaining_min = max(0, int(round(total_hours * 60 * (1 - frac_elapsed))))
    h, m = divmod(remaining_min, 60)
    basis = (f"remaining time estimated by linear interpolation over stream "
             f"position (frac_elapsed={frac_elapsed:.3f})")
    return CONTINUATION_TEMPLATE.format(h=h, m=m), basis
