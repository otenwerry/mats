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

from .. import config, contract
from ..session import recon, recon_claude_stream
from ..workspace import forward, backward
from .base import Engine

_SOLVE_SRC = Path(__file__).resolve().parent.parent / "run" / "solve_intervention.sh"


class NativeClaudeCodeEngine(Engine):
    def __init__(self, bash_mode: str = "execute", resume_mode: str | None = None,
                 container_sif: str | None = None):
        self.bash_mode = bash_mode          # forward-rebuild bash execution
        self.resume_mode = resume_mode or contract.resume_mode_for_scaffold("claude")
        self.container_sif = container_sif or "containers/standard.sif"

    def prepare(self, spec: config.ExperimentSpec, job_home: Path,
                elapsed_seconds: int) -> dict:
        # job_home is spec.build_dir by convention
        if spec.rebuild == "forward":
            ws = forward.build(spec, creation_epoch=0, bash_mode=self.bash_mode)
        else:
            ws = backward.build(spec, creation_epoch=0)

        session_caveats: list[str] = []
        try:
            sess = recon_claude_stream.build_session(spec, job_home)
        except NotImplementedError as e:
            session_caveats.append(
                "claude_stream_reconstruction_unavailable_multi_session;"
                "fell_back_to_viewer_events")
            sess = recon.build_session(spec, job_home)
        except Exception as e:
            session_caveats.append(
                f"claude_stream_reconstruction_failed:{type(e).__name__};"
                "fell_back_to_viewer_events")
            sess = recon.build_session(spec, job_home)
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
            "session_id": sess["session_id"],
            "session_path": sess["session_path"],
            "intervention": spec.intervention,
            # the box reads RESUME_PROMPT for continue_prompt mode (control stem
            # / treatment reminder).
            "resume_prompt": spec.resume_prompt,
            # scaffold + auth so the box injects the right policy credentials
            # (claude->ANTHROPIC_API_KEY; claude_non_api->CLAUDE_CODE_OAUTH_TOKEN).
            "scaffold": spec.trajectory.scaffold,
            "auth": spec.trajectory.auth,
            "container": spec.trajectory.container,  # apptainer image basename
            # --- fields the shared box runner (run_rollout_on_box.sh) needs for
            # prep + scoring, identical contract to the opencode engine ---
            "model_to_train": spec.trajectory.model_to_train,
            "prep_commands": prep_cmds,
            "prep_source": prep_source,
            "require_prep_fidelity": bool(prep_cmds) and contract.REQUIRE_PREP_FIDELITY,
            "eval_model_dir": spec.trajectory.eval_model_dir,
            "eval_limit": spec.trajectory.eval_limit,
            "precut_baseline": config.precut_baseline(spec.trajectory),  # on-box fidelity-gate baseline
            "workspace_audit": _workspace_audit(ws),
            "session_caveats": session_caveats + _session_caveats(sess),
        }
        (job_home / "run_config.json").write_text(json.dumps(cfg, indent=1))
        return {"workspace": ws, "session": sess, "config": cfg}

    def launch_command(self, spec: config.ExperimentSpec, job_home: Path) -> list[str]:
        cfg = json.loads((job_home / "run_config.json").read_text())
        budget_min = cfg.get("agent_timeout_minutes") or contract.agent_timeout_minutes(
            spec.trajectory.num_hours, cfg.get("elapsed_seconds", 0))
        resume_prompt = cfg["resume_prompt"]
        return [
            "timeout", "--signal=TERM", "--kill-after=30s", f"{budget_min}m",
            "apptainer", "exec", "--nv", "-c",
            "--env", "PATH=/root/.local/bin:/home/ben/.local/bin:$PATH",
            "--env", "HF_HOME=${HF_HOME_NEW}",
            "--env", "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}",
            "--env", "DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}",
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
    if sess.get("source") != "solve_out.txt stream-json (faithful)":
        out.append("claude_session_reconstructed_from_viewer_events")
    return out
