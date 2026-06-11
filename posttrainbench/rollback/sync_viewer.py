"""Add any pulled rollback rollouts to the viewer (idempotent, FREE — no API).

Scans the pulled rollouts (config.ROLLBACK_RESULTS, in mats-local) for
completed/partial runs that carry a run_config.json + solve_out, and viewerizes
any that aren't already in config.ROLLBACK_VIEWER_DATA. A run is recognized as
"already added" by its opencode
session_id (or its results-dir name), recorded in each viewer_data file's meta —
so re-running this never duplicates a run.

Labels are assigned automatically as <condition>_run{N}, N counting per condition
(treatment_run1, treatment_run2, ...). The two original control runs keep their
hand-given labels (control_completed / control_partial); they predate
run_config.json and have no results dir to scan, so they're left untouched.

This is wired into exp_pull_result.sh so pulled runs appear in the viewer with no
extra step, but it's safe to run standalone anytime:

    uv run python -m rollback.sync_viewer            # add any new runs
    uv run python -m rollback.sync_viewer --dry-run  # show what would be added
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config, viewerize

RESULTS = config.ROLLBACK_RESULTS
VIEWER_DATA = viewerize.OUT_DIR
TRAJ_BY_RUN_ID = {t.run_id: t for t in config.TRAJECTORIES.values()}


def _existing() -> tuple[set[str], set[str], dict[str, int]]:
    """(session_ids, result_dirs, per-condition run counts) already in viewer_data."""
    sids, rdirs, counts = set(), set(), {}
    for f in sorted(VIEWER_DATA.glob("rollback_*.json")):
        m = json.loads(f.read_text()).get("meta", {})
        if m.get("session_id"):
            sids.add(m["session_id"])
        if m.get("result_dir"):
            rdirs.add(m["result_dir"])
        cond = m.get("condition") or m.get("label", "").split("_")[0]
        counts[cond] = counts.get(cond, 0) + 1
    return sids, rdirs, counts


def _scan() -> list[tuple[Path, dict, Path]]:
    """(result_dir, run_config, solve_out) for every scannable rollout, oldest first."""
    out = []
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        cfg = d / "run_config.json"
        if not (d.is_dir() and cfg.exists()):
            continue
        solves = sorted(d.glob("solve_out_*.txt"))
        if not solves:
            continue
        out.append((d, json.loads(cfg.read_text()), solves[-1]))
    return out


def sync(dry_run: bool = False) -> int:
    sids, rdirs, counts = _existing()
    added = 0
    for d, cfg, solve_out in _scan():
        sid = cfg.get("session_id")
        if d.name in rdirs or (sid and sid in sids):
            continue  # already viewerized
        cond = cfg.get("condition", "control")
        traj = TRAJ_BY_RUN_ID.get(cfg.get("run_id"), config.TRAJECTORY)
        cut = cfg.get("cut_before_event", traj.default_cut)
        resume = cfg.get("resume_prompt") or config.CONTROL_STEM
        # next free per-condition index (guard against any label collision)
        n = counts.get(cond, 0) + 1
        while (VIEWER_DATA / f"rollback_{cond}_run{n}__{traj.run_name}.json").exists():
            n += 1
        label = f"{cond}_run{n}"
        print(f"+ {d.name}  ->  rollback_{label}  (cut {cut}, session {sid})")
        if not dry_run:
            viewerize.build(solve_out, label, cut, resume, traj=traj,
                            session_id=sid, result_dir=d.name, condition=cond)
        counts[cond] = n
        if sid:
            sids.add(sid)
        rdirs.add(d.name)
        added += 1
    print(f"sync: {added} run(s) {'would be ' if dry_run else ''}added; "
          f"{len(rdirs)} total tracked. reload the viewer to see them.")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be added without writing")
    sync(ap.parse_args().dry_run)


if __name__ == "__main__":
    main()
