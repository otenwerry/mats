"""Assemble the t=0 task/ directory the agent originally started from.

Mirrors what `src/run_task.sh` copies into the job dir before launching the
agent: evaluate.py, evaluation_code/ (incl. the TEST data the run archive
strips), templates/, task_context/, timer.sh, and the task PROMPT. Both rebuild
strategies (forward replay and backward roll-back) start from this baseline.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from . import config


def generate_prompt(traj: config.Trajectory, num_gpus: int = 1) -> str:
    """Regenerate the exact agent PROMPT via the repo's get_prompt.py (faithful
    to how the original run built it). Falls back to the recorded prompt."""
    try:
        out = subprocess.run(
            ["python3", "src/eval/general/get_prompt.py",
             "--model-to-train", traj.model_to_train,
             "--benchmark-id", traj.benchmark_id,
             "--num-hours", str(traj.num_hours),
             "--num-gpus", str(num_gpus),
             "--agent", traj.agent],
            cwd=config.PTB_REPO, capture_output=True, text=True, check=True,
        )
        return out.stdout
    except Exception as e:  # pragma: no cover - exercised on the GPU box
        from . import ptbio
        rec = ptbio.recorded_prompt(ptbio.load_events(traj))
        if rec is None:
            raise RuntimeError(f"get_prompt.py failed and no recorded prompt: {e}")
        return rec


def opencode_config_text(traj: config.Trajectory) -> str:
    """The opencode.json the original agents/opencode/solve.sh writes into the
    task dir at launch (extracted from its heredoc, so we stay byte-faithful)."""
    solve = (config.PTB_REPO / "agents" / traj.agent / "solve.sh").read_text()
    m = re.search(r"cat > opencode\.json << 'EOF'\n(.*?)\nEOF\n", solve, re.S)
    if not m:
        raise RuntimeError(f"opencode.json heredoc not found in {traj.agent}/solve.sh")
    return m.group(1) + "\n"


def write_timer(dest_task: Path, num_hours: int, creation_epoch: int) -> None:
    """Write a timer.sh whose CREATION_DATE is back-dated so timer.sh reports
    the reconstructed time-remaining at resume. (See timing.reconstruct.)"""
    (dest_task / "timer.sh").write_text(
        "#!/bin/bash\n\n"
        f"NUM_HOURS={num_hours}\n"
        f"CREATION_DATE={creation_epoch}\n\n"
        "DEADLINE=$((CREATION_DATE + NUM_HOURS * 3600))\n"
        "NOW=$(date +%s)\n"
        "REMAINING=$((DEADLINE - NOW))\n\n"
        'if [ $REMAINING -le 0 ]; then\n'
        '    echo "Timer expired!"\n'
        "else\n"
        '    echo "Remaining time (hours:minutes)":\n'
        "    HOURS=$((REMAINING / 3600))\n"
        "    MINUTES=$(((REMAINING % 3600) / 60))\n"
        '    printf "%d:%02d\\n" $HOURS $MINUTES\n'
        "fi\n"
    )
    (dest_task / "timer.sh").chmod(0o755)


def assemble(traj: config.Trajectory, dest_task: Path, *,
             creation_epoch: int, num_gpus: int = 1,
             write_prompt_file: bool = True) -> dict:
    """Build the initial task/ scaffold at dest_task. Returns a manifest."""
    dest_task.mkdir(parents=True, exist_ok=True)
    td = traj.task_def_dir
    manifest: dict = {"copied": [], "missing": []}

    def copy(src: Path, dst: Path):
        if src.exists():
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            manifest["copied"].append(str(dst.relative_to(dest_task)))
        else:
            manifest["missing"].append(str(src))

    copy(td / "evaluate.py", dest_task / "evaluate.py")
    copy(td / "evaluation_code", dest_task / "evaluation_code")
    copy(config.PTB_REPO / "src" / "eval" / "templates", dest_task / "templates")
    # task_context/* is copied flat into task/ by run_task.sh
    tc = td / "task_context"
    if tc.exists():
        for child in tc.iterdir():
            copy(child, dest_task / child.name)

    write_timer(dest_task, traj.num_hours, creation_epoch)
    manifest["copied"].append("timer.sh")

    if traj.scaffold == "opencode":
        (dest_task / "opencode.json").write_text(opencode_config_text(traj))
        manifest["copied"].append("opencode.json")

    if write_prompt_file:
        prompt = generate_prompt(traj, num_gpus)
        (dest_task / ".rollback_prompt.txt").write_text(prompt)
        manifest["prompt_chars"] = len(prompt)

    return manifest
