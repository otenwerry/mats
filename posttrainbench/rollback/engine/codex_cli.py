"""Native Codex CLI engine: prepare a job home and resume the reconstructed
session with `codex exec resume` inside the codex apptainer image. Mirrors
opencode_cli / native_cli. Faithful to the original codex_non_api runs (same
@openai/codex@0.79.0 image, ChatGPT-subscription auth).

prepare() is offline. NOTE: recon_codex.build_session refuses until the rollout
schema is validated against a captured rollout (env PTB_CODEX_ROLLOUT_VALIDATED=1)
— so preparing a codex cell is intentionally blocked until then. Everything else
(routing, run_config, launch command, the box auth plumbing) is in place so that
finishing codex is just validating the one rollout serializer.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import config, contract
from ..session import recon_codex
from ..workspace import forward, backward
from .base import Engine

_SOLVE_SRC = (Path(__file__).resolve().parent.parent
              / "run" / "solve_intervention_codex.sh")


class CodexEngine(Engine):
    def __init__(self, bash_mode: str = "execute", resume_mode: str | None = None,
                 container_sif: str | None = None):
        self.bash_mode = bash_mode
        self.resume_mode = resume_mode or contract.resume_mode_for_scaffold("codex")
        # our standard.def bundles the codex CLI too, so codex runs in standard.sif
        self.container_sif = container_sif or "containers/standard.sif"

    def prepare(self, spec: config.ExperimentSpec, job_home: Path,
                elapsed_seconds: int) -> dict:
        if spec.rebuild == "forward":
            ws = forward.build(spec, creation_epoch=0, bash_mode=self.bash_mode)
        else:
            ws = backward.build(spec, creation_epoch=0)

        sess = recon_codex.build_session(spec, job_home)
        shutil.copy2(_SOLVE_SRC, job_home / "agent_solve.sh")
        (job_home / "agent_solve.sh").chmod(0o755)
        prep_cmds, prep_source = config.effective_prep_commands(spec.trajectory)
        timeout = contract.timeout_contract(spec.trajectory.num_hours, elapsed_seconds)

        cfg = {
            "run_id": spec.trajectory.run_id,
            "cell_id": spec.cell_id,
            "rebuild": spec.rebuild,
            "condition": spec.condition,
            "cut_before_event": spec.cut_before_event,
            "policy_model": spec.policy_model,
            "policy_model_recorded": spec.trajectory.policy_model_recorded,
            "agent": spec.trajectory.agent,
            "elapsed_seconds": elapsed_seconds,
            "num_hours": spec.trajectory.num_hours,
            "resume_mode": self.resume_mode,
            "prompt_contract": contract.prompt_contract(spec.condition, spec.trajectory.scaffold),
            "timeout_contract": timeout,
            "remaining_seconds": timeout["remaining_seconds_at_cut"],
            "agent_timeout_minutes": timeout["agent_timeout_minutes"],
            "session_id": sess["session_id"],          # codex thread_id
            "rollout_path": sess["rollout_path"],
            "intervention": spec.intervention,
            "resume_prompt": spec.resume_prompt,
            "scaffold": spec.trajectory.scaffold,
            "auth": spec.trajectory.auth,              # "oauth" (ChatGPT subscription)
            "container": spec.trajectory.container,    # "codex"
            "model_to_train": spec.trajectory.model_to_train,
            "prep_commands": prep_cmds,
            "prep_source": prep_source,
            "require_prep_fidelity": bool(prep_cmds) and contract.REQUIRE_PREP_FIDELITY,
            "eval_model_dir": spec.trajectory.eval_model_dir,
            "eval_limit": spec.trajectory.eval_limit,
            "precut_baseline": config.precut_baseline(spec.trajectory),  # on-box fidelity-gate baseline
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
            # codex_non_api authenticates via the ChatGPT subscription (auth.json
            # in the job home's .codex); OPENAI_API_KEY cleared by the solve script.
            "--env", "OPENAI_API_KEY=${OPENAI_API_KEY}",
            "--env", "VLLM_API_KEY=inspectai",
            "--env", "PYTHONNOUSERSITE=1",
            "--env", f"SESSION_ID={cfg['session_id']}",
            "--env", f"AGENT_CONFIG={cfg['policy_model']}",
            "--env", f"ELAPSED_SECONDS={cfg['elapsed_seconds']}",
            "--env", f"NUM_HOURS={cfg['num_hours']}",
            "--env", f"RESUME_PROMPT={cfg['resume_prompt']}",
            "--env", f"RESUME_MODE={cfg['resume_mode']}",
            "--env", f"AUTH={cfg['auth']}",
            "--env", f"AGENT_FAMILY={cfg.get('agent', '')}",
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
    out = []
    if sess.get("apply_patch_lossy"):
        out.append(f"codex_apply_patch_history_lossy:{sess.get('n_apply_patch', 0)}")
    return out
