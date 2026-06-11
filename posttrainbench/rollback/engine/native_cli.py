"""Native Claude Code engine: prepare a job home and run the real CLI in the
apptainer image, resuming the reconstructed session. Faithful to the original
runs (same image, same scaffold, same tools executing against a real FS).

`prepare()` is cheap and offline (Mac-runnable): it builds task/, the session,
and writes the launch config. `launch_command()` returns the apptainer command
(executed only on the Linux/GPU box). The job home IS spec.build_dir:
    spec.build_dir/
        task/                       cut-point workspace (forward|backward)
        .claude/projects/.../*.jsonl  reconstructed session
        agent_solve.sh              copy of run/solve_intervention.sh
        run_config.json             session id, model, elapsed, resume mode
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import config
from ..session import recon
from ..workspace import forward, backward
from .base import Engine

_SOLVE_SRC = Path(__file__).resolve().parent.parent / "run" / "solve_intervention.sh"


class NativeClaudeCodeEngine(Engine):
    def __init__(self, bash_mode: str = "execute", resume_mode: str = "resume",
                 container_sif: str | None = None):
        self.bash_mode = bash_mode          # forward-rebuild bash execution
        self.resume_mode = resume_mode      # "resume" | "continue_prompt"
        self.container_sif = container_sif or "containers/standard.sif"

    def prepare(self, spec: config.ExperimentSpec, job_home: Path,
                elapsed_seconds: int) -> dict:
        # job_home is spec.build_dir by convention
        if spec.rebuild == "forward":
            ws = forward.build(spec, creation_epoch=0, bash_mode=self.bash_mode)
        else:
            ws = backward.build(spec, creation_epoch=0)

        sess = recon.build_session(spec, job_home)
        shutil.copy2(_SOLVE_SRC, job_home / "agent_solve.sh")
        (job_home / "agent_solve.sh").chmod(0o755)

        cfg = {
            "run_id": spec.trajectory.run_id,
            "cell_id": spec.cell_id,
            "rebuild": spec.rebuild,
            "condition": spec.condition,
            "cut_before_event": spec.cut_before_event,
            "policy_model": spec.policy_model,
            "elapsed_seconds": elapsed_seconds,
            "num_hours": spec.trajectory.num_hours,
            "resume_mode": self.resume_mode,
            "session_id": sess["session_id"],
            "session_path": sess["session_path"],
            "intervention": spec.intervention,
        }
        (job_home / "run_config.json").write_text(json.dumps(cfg, indent=1))
        return {"workspace": ws, "session": sess, "config": cfg}

    def launch_command(self, spec: config.ExperimentSpec, job_home: Path) -> list[str]:
        cfg = json.loads((job_home / "run_config.json").read_text())
        budget_min = spec.trajectory.num_hours * 60 + 5
        # treatment reminder is already embedded in the session for resume mode;
        # only used as a prompt under continue_prompt mode.
        resume_prompt = (cfg["intervention"] or "Please continue.") \
            if spec.condition == "treatment" else "Please continue improving your result."
        return [
            "timeout", "--signal=TERM", "--kill-after=30s", f"{budget_min}m",
            "apptainer", "exec", "--nv", "-c",
            "--env", "PATH=/root/.local/bin:/home/ben/.local/bin:$PATH",
            "--env", "HF_HOME=${HF_HOME_NEW}",
            "--env", "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}",
            # HealthBench/ArenaHard graders call OpenAI; harmless otherwise:
            "--env", "OPENAI_API_KEY=${OPENAI_API_KEY}",
            "--env", "VLLM_API_KEY=inspectai",
            "--env", "PYTHONNOUSERSITE=1",
            "--env", f"SESSION_ID={cfg['session_id']}",
            "--env", f"AGENT_CONFIG={cfg['policy_model']}",
            "--env", f"ELAPSED_SECONDS={cfg['elapsed_seconds']}",
            "--env", f"NUM_HOURS={cfg['num_hours']}",
            "--env", f"RESUME_MODE={cfg['resume_mode']}",
            "--env", f"RESUME_PROMPT={resume_prompt}",
            "--bind", f"{job_home}/tmp:/tmp",
            "--bind", "${HF_MERGED}:${HF_HOME_NEW}",
            "--home", f"{job_home}:/home/ben",
            "--pwd", "/home/ben/task",
            "--writable-tmpfs",
            "${POST_TRAIN_BENCH_CONTAINERS_DIR}/" + Path(self.container_sif).name,
            "bash", "/home/ben/agent_solve.sh",
        ]
