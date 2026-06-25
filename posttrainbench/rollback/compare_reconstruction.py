"""Reconstruction-fidelity check (FREE): compare a RE-TRAINED prep model's
benchmark score against the ORIGINAL trajectory's recorded pre-cut score.

This is the evidence for choosing real retraining over LLM-as-environment: if our
re-trained model lands within the eval's noise of the original's score, the
reconstruction faithfully recreates the lost model — whereas the simulator
fabricates scores (see simenv validation).

Baseline = the original's own `final_eval.json` in its archived task dir, which is
the pre-cut model's score WHEN the agent's last action before the cut was the
final eval (true for kimi/humaneval). Falls back to the dataset's recorded
accuracy. NOTE: for trajectories where the original eval'd again AFTER the cut,
the apples-to-apples baseline is the last eval BEFORE first_hack_event — extract
that instead (flagged below).

Usage:
    uv run python -m rollback.compare_reconstruction <prep_score.json> [--traj KEY]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config


def _baseline(traj: config.Trajectory) -> tuple:
    """(accuracy, stderr, source) for the original's pre-cut score."""
    fe = traj.final_task_dir / traj.precut_eval_file
    if fe.exists():
        try:
            d = json.loads(fe.read_text())
            return d.get("accuracy"), d.get("stderr"), f"original {traj.precut_eval_file} (pre-cut)"
        except json.JSONDecodeError:
            pass
    # fallback: dataset-recorded accuracy
    idx = json.loads((config.VIEWER_DATA / "index.json").read_text())
    rows = idx["runs"] if isinstance(idx, dict) else idx
    for r in rows:
        if r.get("run_id") == traj.run_id:
            return r.get("accuracy"), r.get("stderr"), "dataset index accuracy"
    return None, None, "NOT FOUND"


def record_fidelity(prep: dict, traj: config.Trajectory,
                    run: str = None, prep_source: str = None,
                    baseline: dict | None = None) -> dict | None:
    """Compute the reconstruction-fidelity row for `traj` from a re-trained prep
    model's score dict, and write it to reconstruction_fidelity.json. Keyed by
    (trajectory, run) so repeated runs of the same trajectory EACH keep their own
    row (no overwrite) — pass run=<label> (e.g. "control_run2"). prep_source
    defaults to the trajectory's current recipe source, but callers should pass
    the RUN's recorded prep_source (from its run_config) so historical rows keep
    their true 'curated'/'derived' provenance. Returns the row, or None if a score
    is missing. Read by the rollback page's fidelity table; called by sync_viewer
    on every pulled run that carries a prep_score.json."""
    pa, ps = prep.get("accuracy"), prep.get("stderr")
    if baseline:
        ba, bs = baseline.get("accuracy"), baseline.get("stderr")
    else:
        ba, bs, _ = _baseline(traj)
    if ba is None or pa is None:
        return None
    delta = pa - ba
    tol = (ps or 0) + (bs or 0)
    # prep_source records HOW the pre-cut model was reconstructed for this run:
    # "curated" = a validated/hand-or-agent-written recipe (a legitimate datapoint
    # for reconstruction accuracy); "derived" = the un-validated heuristic guess
    # (may have grabbed the wrong training -> an ERRONEOUS reconstruction, excluded
    # from the accuracy measure). Stamped at record time so re-runs update it.
    if prep_source is None:
        prep_source = config.effective_prep_commands(traj)[1]
    row = {"trajectory": traj.run_name, "run": run, "benchmark": traj.benchmark_id,
           "retrained": pa, "retrained_stderr": ps,
           "original": ba, "original_stderr": bs,
           "delta": round(delta, 4), "faithful": bool(abs(delta) <= tol),
           "prep_source": prep_source}
    fid = config.ROLLBACK_LOCAL / "reconstruction_fidelity.json"
    recs = []
    if fid.exists():
        try:
            recs = json.loads(fid.read_text())
        except json.JSONDecodeError:
            recs = []
    # replace only the SAME (trajectory, run) row — DISTINCT runs accumulate
    # (so repeats/seeds are preserved, not overwritten).
    recs = [r for r in recs
            if not (r.get("trajectory") == traj.run_name and r.get("run") == run)]
    recs.append(row)
    fid.parent.mkdir(parents=True, exist_ok=True)
    fid.write_text(json.dumps(recs, indent=1) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_score", help="path to the re-trained prep model's score.json")
    ap.add_argument("--traj", default=None, help="trajectory key (default: configured)")
    args = ap.parse_args()
    traj = config.TRAJECTORIES[args.traj] if args.traj else config.TRAJECTORY

    prep = json.loads(Path(args.prep_score).read_text())
    pa, ps = prep.get("accuracy"), prep.get("stderr")
    ba, bs, src = _baseline(traj)

    print(f"trajectory: {traj.run_name}")
    print(f"  ORIGINAL pre-cut score : {ba}  (±{bs})   [{src}]")
    print(f"  RE-TRAINED prep score  : {pa}  (±{ps})")
    if ba is None or pa is None:
        print("  -> cannot compare (missing a score)"); return
    delta = pa - ba
    tol = (ps or 0) + (bs or 0)        # combined 1-sigma envelope
    verdict = "FAITHFUL (within combined stderr)" if abs(delta) <= tol else \
              "DIVERGED (beyond combined stderr — investigate)"
    print(f"  delta (prep - original): {delta:+.4f}   | combined stderr ±{tol:.4f}")
    print(f"  -> {verdict}")

    record_fidelity(prep, traj)
    print(f"  recorded -> {config.ROLLBACK_LOCAL / 'reconstruction_fidelity.json'}")


if __name__ == "__main__":
    main()
