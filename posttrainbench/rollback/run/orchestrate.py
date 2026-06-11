"""Orchestrate the 2x2 derisking matrix: {forward,backward} x {control,treatment}.

For each cell: reconstruct the timer, prepare the job home (workspace + session
+ solve script), and either print the launch command (default / dry-run on
macOS) or execute it (--launch, on the Linux/GPU box).

Resumability: each cell writes state.json with the phases it has completed
({prepared, launched, completed}); re-running skips finished phases. The forward
workspace builder additionally journals each executed op, and Claude Code
persists the session as it runs — so a Ctrl-C'd rollout resumes via the same
`--resume <session_id>` (the partial turns are already on disk).

Usage (macOS, offline prep + show commands):
    python -m rollback.run.orchestrate
    python -m rollback.run.orchestrate --cells backward_control backward_treatment
Usage (GPU box, actually run):
    python -m rollback.run.orchestrate --launch --bash-mode execute
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from .. import config, timing
from .. import engine as engine_pkg


def _state_path(spec: config.ExperimentSpec) -> Path:
    return spec.build_dir / "state.json"


def _load_state(spec) -> dict:
    p = _state_path(spec)
    return json.loads(p.read_text()) if p.exists() else {"phases": {}}


def _save_state(spec, state) -> None:
    _state_path(spec).parent.mkdir(parents=True, exist_ok=True)
    _state_path(spec).write_text(json.dumps(state, indent=1))


def run_cell(spec: config.ExperimentSpec, engine,
             launch: bool, force: bool) -> dict:
    state = _load_state(spec)
    timer = timing.reconstruct(spec.trajectory, spec.cut_before_event)
    job_home = spec.build_dir

    if force or "prepared" not in state["phases"]:
        manifest = engine.prepare(spec, job_home, timer["elapsed_seconds"])
        state["phases"]["prepared"] = {
            "at": int(time.time()),
            "session_id": manifest["session"]["session_id"],
            "workspace_files": manifest["workspace"]["result_file_count"],
            "intervention_injected": manifest["session"]["intervention_injected"],
        }
        _save_state(spec, state)

    cmd = engine.launch_command(spec, job_home)
    state["phases"]["launch_command"] = cmd
    _save_state(spec, state)

    if launch:
        if "completed" in state["phases"] and not force:
            return {"cell": spec.cell_id, "skipped": "already completed"}
        (job_home / "tmp").mkdir(exist_ok=True)
        rc = subprocess.call(cmd)
        state["phases"]["launched"] = {"at": int(time.time()), "returncode": rc}
        if rc == 0:
            state["phases"]["completed"] = {"at": int(time.time())}
        _save_state(spec, state)
        return {"cell": spec.cell_id, "returncode": rc, "timer": timer}

    return {"cell": spec.cell_id, "prepared": True, "timer": timer,
            "launch_command": " ".join(cmd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=config.CUT_BEFORE_EVENT)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="subset of cell ids, e.g. backward_control_cut50")
    ap.add_argument("--launch", action="store_true",
                    help="actually run (GPU box only); default just prepares + prints")
    ap.add_argument("--bash-mode", default="execute",
                    choices=["skip", "safe", "execute"],
                    help="forward-rebuild bash execution (use skip on macOS)")
    ap.add_argument("--resume-mode", default="resume",
                    choices=["resume", "continue_prompt"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    engine = engine_pkg.for_trajectory(config.TRAJECTORY,
                                       bash_mode=args.bash_mode,
                                       resume_mode=args.resume_mode)
    specs = config.all_cells(cut=args.cut)
    if args.cells:
        specs = [s for s in specs if s.cell_id in args.cells]

    for spec in specs:
        res = run_cell(spec, engine, args.launch, args.force)
        print(f"\n### {spec.cell_id}")
        if "launch_command" in res:
            t = res["timer"]
            print(f"  prepared; timer ~{t['elapsed_seconds']}s elapsed / "
                  f"{t['remaining_seconds']/3600:.2f}h remaining")
            print(f"  build dir: {spec.build_dir}")
            print(f"  launch:\n    {res['launch_command']}")
        else:
            print(f"  {res}")


if __name__ == "__main__":
    main()
