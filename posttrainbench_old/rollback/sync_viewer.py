"""Add any pulled rollback rollouts to the viewer (idempotent, FREE — no API).

Scans the pulled rollouts (config.ROLLBACK_RESULTS, in mats-local) for
completed/partial runs that carry a run_config.json + solve_out, and viewerizes
any that aren't already in config.ROLLBACK_VIEWER_DATA. A run is recognized as
"already added" by its opencode
session_id (or its results-dir name), recorded in each viewer_data file's meta —
so re-running this never duplicates a run.

Labels are assigned automatically as <condition>_run{N}, N counting per condition
(prompt1_run1, prompt1_run2, ...), where condition is one of prompt1/prompt2/prompt3.
DEBUG_-labelled smoke runs are quarantined (hidden in the viewer) and don't count
toward run numbering.

This is wired into exp_pull_result.sh so pulled runs appear in the viewer with no
extra step, but it's safe to run standalone anytime:

    uv run python -m rollback.sync_viewer            # add any new runs
    uv run python -m rollback.sync_viewer --dry-run  # show what would be added
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import compare_reconstruction, config, viewerize

RESULTS = config.ROLLBACK_RESULTS
VIEWER_DATA = viewerize.OUT_DIR
TRAJ_BY_RUN_ID = config.ALL_TRAJECTORIES  # run_id -> Trajectory, all ~30 (curated override auto)
RUNNING = config.ROLLBACK_LOCAL / "running_rollouts.json"


def _existing() -> tuple[set[str], dict[str, int]]:
    """(result_dirs, per-(condition,trajectory) run counts) already in viewer_data.

    Dedup is by RESULT DIR only — the result-dir name is unique per launch
    (timestamped). NOT by session_id: control, treatment, AND every re-seed of a
    trajectory all reconstruct the SAME original session, so a session-based key
    would wrongly collapse distinct rollouts. DEBUG_-labelled runs don't count
    toward good-run numbering (so the first real run is run1, not run2)."""
    rdirs, counts = set(), {}
    for f in sorted(VIEWER_DATA.glob("rollback_*.json")):
        try:
            m = json.loads(f.read_text()).get("meta", {})
        except (json.JSONDecodeError, OSError):
            # An empty/partial file (e.g. a viewerize write interrupted mid-flight)
            # must not wedge the entire sync — skip it. It gets rewritten from its
            # result dir on this same pass if the pulled results are present.
            print(f"sync_viewer: skipping unreadable {f.name} (empty/partial write)")
            continue
        if m.get("result_dir"):
            rdirs.add(m["result_dir"])
        label = m.get("label", "")
        if label.startswith("DEBUG_"):
            continue
        cond = m.get("condition") or label.split("_")[0]
        rn = f.stem.split("__", 1)[1] if "__" in f.stem else ""   # trajectory run_name
        counts[(cond, rn)] = counts.get((cond, rn), 0) + 1
    return rdirs, counts


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


def _scan_prep_only() -> list[tuple[Path, dict]]:
    """(result_dir, run_config) for prep-only checks that have no agent transcript."""
    out = []
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        cfg = d / "run_config.json"
        prep = d / "prep_score.json"
        if not (d.is_dir() and cfg.exists() and prep.exists()):
            continue
        if sorted(d.glob("solve_out_*.txt")):
            continue
        out.append((d, json.loads(cfg.read_text())))
    return out


def _record_prep_fidelity(d: Path, cfg: dict, traj: config.Trajectory,
                          run_label: str | None, dry_run: bool) -> dict | None:
    pp = d / "prep_score.json"
    if not pp.exists():
        return None
    try:
        prep = json.loads(pp.read_text())
    except json.JSONDecodeError:
        return None
    if dry_run:
        return None
    row = compare_reconstruction.record_fidelity(
        prep,
        traj,
        run=run_label,
        prep_source=cfg.get("prep_source"),
        baseline=cfg.get("precut_baseline"),
    )
    if row:
        print(f"    fidelity: re-trained {row['retrained']:.3f} vs "
              f"original {row['original']:.3f}  ->  "
              f"{'faithful' if row['faithful'] else 'DIVERGED'}")
    return row


def sync(dry_run: bool = False) -> int:
    rdirs, counts = _existing()
    added = 0
    for d, cfg, solve_out in _scan():
        sid = cfg.get("session_id")
        cond = cfg.get("condition", "prompt1")
        if d.name in rdirs:
            continue  # already viewerized (result dir is the unique per-launch key)
        traj = TRAJ_BY_RUN_ID.get(cfg.get("run_id"), config.TRAJECTORY)
        cut = cfg.get("cut_before_event", traj.default_cut)
        resume = cfg.get("resume_prompt") or config.CONTROL_STEM
        # SMOKE runs are viewerized (so that stage gets validated) but quarantined
        # under a DEBUG_ label: the viewer hides __rollback__DEBUG_ rows, and
        # _existing() doesn't count DEBUG_ labels, so real-run numbering is unaffected.
        prefix = "DEBUG_" if cfg.get("smoke") else ""
        # next free index for THIS (condition, trajectory) (guard against collision)
        n = counts.get((cond, traj.run_name), 0) + 1
        while (VIEWER_DATA / f"rollback_{prefix}{cond}_run{n}__{traj.run_name}.json").exists():
            n += 1
        label = f"{prefix}{cond}_run{n}"
        # benchmark score from the box scoring stage (if it ran)
        score = None
        sp = d / "score.json"
        if sp.exists():
            try:
                score = json.loads(sp.read_text())
            except json.JSONDecodeError:
                score = None
        # score_status distinguishes "scored" from a SILENT score=None: a scorer
        # crash leaves a score_out_*.log but no score.json (the flaky vLLM bug),
        # vs a run that never attempted scoring (no servable model / evaluate.py).
        if isinstance(score, dict) and score:
            score_status = "scored"
        elif list(d.glob("score_out_*.log")):
            score_status = "failed"
        else:
            score_status = "not_attempted"
        sc = f", score {score.get('accuracy')}" if isinstance(score, dict) else f", score {score_status}"
        print(f"+ {d.name}  ->  rollback_{label}  (cut {cut}, session {sid}{sc})")
        # reconstruction fidelity: if this run scored its CLEAN re-trained prep
        # model (before the agent touched it), record the row automatically — same
        # path that adds the index row, no separate command. Keyed by trajectory,
        # so control & treatment of a pair update the one row in place.
        prep = None
        pp = d / "prep_score.json"
        if pp.exists():
            try:
                prep = json.loads(pp.read_text())
            except json.JSONDecodeError:
                prep = None
        # prep_fidelity.json carries the non-fatal-gate verdict (verified /
        # diverged / unverified_*) written on the box; surface it in the viewer.
        prep_fidelity = None
        fp = d / "prep_fidelity.json"
        if fp.exists():
            try:
                prep_fidelity = json.loads(fp.read_text())
            except json.JSONDecodeError:
                prep_fidelity = None
        if isinstance(prep_fidelity, dict) and prep_fidelity.get("status") not in (None, "verified"):
            print(f"    CAVEAT prep_fidelity={prep_fidelity.get('status')}")
        if score_status != "scored":
            print(f"    CAVEAT score_status={score_status}")
        if not dry_run:
            viewerize.build(solve_out, label, cut, resume, traj=traj,
                            session_id=sid, result_dir=d.name, condition=cond,
                            score=score, run_config=cfg,
                            prep_fidelity=prep_fidelity, score_status=score_status)
            if isinstance(prep, dict):
                _record_prep_fidelity(d, cfg, traj, label, dry_run=False)
        counts[(cond, traj.run_name)] = n
        rdirs.add(d.name)
        added += 1
    prep_rows = 0
    for d, cfg in _scan_prep_only():
        traj = TRAJ_BY_RUN_ID.get(cfg.get("run_id"), config.TRAJECTORY)
        run_label = f"prep_only__{d.name}"
        marker = config.ROLLBACK_LOCAL / "reconstruction_fidelity.json"
        already = False
        if marker.exists():
            try:
                already = any(
                    r.get("trajectory") == traj.run_name and r.get("run") == run_label
                    for r in json.loads(marker.read_text())
                )
            except json.JSONDecodeError:
                already = False
        if already:
            continue
        print(f"+ {d.name}  ->  fidelity row only (prep-only, no agent transcript)")
        _record_prep_fidelity(d, cfg, traj, run_label, dry_run=dry_run)
        prep_rows += 1
    print(f"sync: {added} run(s) {'would be ' if dry_run else ''}added; "
          f"{prep_rows} prep-only row(s) {'would be ' if dry_run else ''}recorded; "
          f"{len(rdirs)} total tracked. reload the viewer to see them.")
    if not dry_run and RUNNING.exists():
        try:
            running = json.loads(RUNNING.read_text())
        except json.JSONDecodeError:
            running = []
        if isinstance(running, list):
            kept = [r for r in running if r.get("result_dir") not in rdirs]
            if len(kept) != len(running):
                RUNNING.write_text(json.dumps(kept, indent=1) + "\n")
                print(f"sync: removed {len(running) - len(kept)} completed running placeholder(s).")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be added without writing")
    sync(ap.parse_args().dry_run)


if __name__ == "__main__":
    main()
