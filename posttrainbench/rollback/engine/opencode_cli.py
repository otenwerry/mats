"""Native OpenCode engine: prepare a job home and run the real OpenCode CLI in
the apptainer image, resuming the reconstructed session. Faithful to the
original runs (same image, same opencode-ai@1.1.59, same tools on a real FS).

`prepare()` is cheap and offline (Mac-runnable): it builds task/, the session
storage, and writes the launch config. `launch_command()` returns the apptainer
command (executed only on the Linux/GPU box). The job home IS spec.build_dir:
    spec.build_dir/
        task/                            cut-point workspace (forward|backward)
        .local/share/opencode/storage/   reconstructed session storage
        agent_solve.sh                   copy of run/solve_intervention_opencode.sh
        run_config.json                  session id, model, elapsed, resume prompt

OpenCode cannot resume without sending a new user turn, so BOTH conditions
send one: control a neutral nudge, treatment the rolled-you-back reminder —
the conditions differ in that prompt text only (storage is identical).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import config, contract
from ..session import recon_opencode
from ..workspace import forward, backward
from .base import Engine

_SOLVE_SRC = (Path(__file__).resolve().parent.parent
              / "run" / "solve_intervention_opencode.sh")


class OpenCodeEngine(Engine):
    def __init__(self, bash_mode: str = "execute",
                 container_sif: str | None = None):
        self.bash_mode = bash_mode          # forward-rebuild bash execution
        self.container_sif = container_sif or "containers/standard.sif"

    def prepare(self, spec: config.ExperimentSpec, job_home: Path,
                elapsed_seconds: int) -> dict:
        # job_home is spec.build_dir by convention
        if spec.rebuild == "forward":
            ws = forward.build(spec, creation_epoch=0, bash_mode=self.bash_mode)
        else:
            ws = backward.build(spec, creation_epoch=0)

        sess = recon_opencode.build_session(spec, job_home)
        shutil.copy2(_SOLVE_SRC, job_home / "agent_solve.sh")
        (job_home / "agent_solve.sh").chmod(0o755)
        prep_cmds, prep_source = config.effective_prep_commands(spec.trajectory)
        timeout = contract.timeout_contract(spec.trajectory.num_hours, elapsed_seconds)

        # The original gateway model (opencode/minimax-m2.5-free) left the
        # OpenCode Zen catalog; the default continuation route is OpenRouter's
        # minimax-m2.5 (same weights, different gateway — recorded as drift in
        # run_config). The provider goes in the GLOBAL config (merged by
        # opencode) so the task-dir opencode.json the agent can read stays
        # byte-identical to the original run's.
        if spec.policy_model.startswith("openrouter/"):
            gc = job_home / ".config" / "opencode" / "opencode.json"
            gc.parent.mkdir(parents=True, exist_ok=True)
            gc.write_text(json.dumps({
                "$schema": "https://opencode.ai/config.json",
                "provider": {"openrouter": {
                    "options": {"apiKey": "{env:OPENROUTER_API_KEY}"},
                }},
            }, indent=2))

        cfg = {
            "run_id": spec.trajectory.run_id,
            "cell_id": spec.cell_id,
            "rebuild": spec.rebuild,
            "condition": spec.condition,
            "cut_before_event": spec.cut_before_event,
            "policy_model": spec.policy_model,
            "policy_model_recorded": spec.trajectory.policy_model_recorded,
            "agent": spec.trajectory.agent,
            # scaffold + auth so the box injects the right policy credentials
            # (opencode authenticates the policy via the OpenRouter API key).
            "scaffold": spec.trajectory.scaffold,
            "auth": spec.trajectory.auth,
            "container": spec.trajectory.container,  # apptainer image basename
            # the HF base model the agent fine-tunes — the box uses this to
            # locate the cached weights when SMOKE scoring needs a model present
            # (after a 1-step smoke there's no trained final_model yet).
            "model_to_train": spec.trajectory.model_to_train,
            "elapsed_seconds": elapsed_seconds,
            "num_hours": spec.trajectory.num_hours,
            "prompt_contract": contract.prompt_contract(spec.condition, spec.trajectory.scaffold),
            "timeout_contract": timeout,
            "remaining_seconds": timeout["remaining_seconds_at_cut"],
            "agent_timeout_minutes": timeout["agent_timeout_minutes"],
            "session_id": sess["session_id"],
            "storage_dir": sess["storage_dir"],
            "opencode_version": sess["opencode_version"],
            "intervention": spec.intervention,
            # control = shared stem; treatment = stem + reminder (differ by
            # exactly the reminder). Sent via stdin by the solve script.
            "resume_prompt": spec.resume_prompt,
            # bash run on the GPU box in task/ BEFORE the agent, to materialize
            # model weights the stripped archive lacks (e.g. rerun pre-cut
            # training). curated-if-present-else-derived (P4d); prep_source marks
            # provenance. Empty for cuts before any training. See config.py.
            "prep_commands": prep_cmds,
            "prep_source": prep_source,
            "require_prep_fidelity": bool(prep_cmds) and contract.REQUIRE_PREP_FIDELITY,
            # dir (under task/) holding the servable model to score — "final_model"
            # for full-model trainers, "final_model_merged" for LoRA+merge runs.
            "eval_model_dir": spec.trajectory.eval_model_dir,
            # --limit for our eval; bounded for slow LLM-graded benchmarks.
            "eval_limit": spec.trajectory.eval_limit,
            # original pre-cut score; the on-box fidelity gate aborts the run
            # before the agent if the re-trained prep diverges from this.
            "precut_baseline": config.precut_baseline(spec.trajectory),
            "workspace_audit": _workspace_audit(ws),
            "session_caveats": _session_caveats(sess),
        }
        (job_home / "run_config.json").write_text(json.dumps(cfg, indent=1))
        return {"workspace": ws, "session": sess, "config": cfg}

    def launch_command(self, spec: config.ExperimentSpec, job_home: Path) -> list[str]:
        cfg = json.loads((job_home / "run_config.json").read_text())
        budget_min = cfg.get("agent_timeout_minutes") or contract.agent_timeout_minutes(
            spec.trajectory.num_hours, cfg.get("elapsed_seconds", 0))
        return [
            "timeout", "--signal=TERM", "--kill-after=30s", f"{budget_min}m",
            "apptainer", "exec", "--nv", "-c",
            "--env", "PATH=/root/.local/bin:/home/ben/.local/bin:$PATH",
            "--env", "HF_HOME=${HF_HOME_NEW}",
            # the only provider the original opencode.json resolves for this
            # run; the other {env:...} placeholders harmlessly resolve empty:
            "--env", "OPENCODE_API_KEY=${OPENCODE_API_KEY}",
            "--env", "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}",
            # HealthBench/ArenaHard eval graders call OpenAI; harmless otherwise:
            "--env", "OPENAI_API_KEY=${OPENAI_API_KEY}",
            "--env", "VLLM_API_KEY=inspectai",
            "--env", "PYTHONNOUSERSITE=1",
            "--env", f"SESSION_ID={cfg['session_id']}",
            "--env", f"AGENT_CONFIG={cfg['policy_model']}",
            "--env", f"ELAPSED_SECONDS={cfg['elapsed_seconds']}",
            "--env", f"NUM_HOURS={cfg['num_hours']}",
            "--env", f"RESUME_PROMPT={cfg['resume_prompt']}",
            "--bind", f"{job_home}/tmp:/tmp",
            "--bind", "${HF_MERGED}:${HF_HOME_NEW}",
            "--home", f"{job_home}:/home/ben",
            "--pwd", "/home/ben/task",
            "--writable-tmpfs",
            "${POST_TRAIN_BENCH_CONTAINERS_DIR}/" + Path(self.container_sif).name,
            "bash", "/home/ben/agent_solve.sh",
        ]


def _workspace_audit(ws: dict) -> dict:
    return {
        "strategy": ws.get("strategy"),
        "stripped_stale_eval": ws.get("stripped_stale_eval", []),
        "flagged_modified_after_cut": ws.get("flagged_modified_after_cut", []),
        "kept_by_name_timestamp": ws.get("kept_by_name_timestamp", []),
        "removed_after_cut_count": len(ws.get("removed_after_cut", []) or []),
    }


def _session_caveats(sess: dict) -> list[str]:
    return []
