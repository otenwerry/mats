"""End-to-end begin-resampling pipeline: generate faithful resamples -> (regular judge,
inline) -> deviation judge -> rebuild viewer, in one command.

For each requested ORIGINAL trajectory, this re-runs the SAME scenario from the very start
("begin"), under identical conditions, with NOTHING inserted (CONTROL only). The resample's
AUDITOR is primed with the ENTIRE original run -- the target's transcript with the auditor's
own private reasoning interleaved -- and asked to reproduce it faithfully (same target
action -> same environment response). The question: on a fresh sample, does the TARGET's
behavior change?

Two judges run:
  1. the REGULAR reward-hacking judge, inline via audit() (same dims as a normal audit);
  2. a SECOND "deviation from original" judge that compares each resample to its original
     and scores how much the AUDITOR deviated where the target behaved the same -- the
     faithfulness confounder check (lib/exp_resample.run_deviation_for_dir).

Results fold into the original's row on the viewer, in a distinct "Resampling" section.

All config (target + reasoning pin, auditor + thinking, judge, max_turns, turn_counter,
fixed_sp + system prompt) is INHERITED from each original's stamped metadata, so a resample
uses the same setup as the run it resamples. The core machinery lives in lib/exp_resample.py
and is written to be reused by the forthcoming continuation experiment.

Usage:
  uv run exp_resample_begin_pipeline.py --trajectories=451,452 --n=5
  uv run exp_resample_begin_pipeline.py --trajectories=451 --n=3 --concurrency=50
  uv run exp_resample_begin_pipeline.py --trajectories=451,452 --n=5 --dry-run   # FREE: plan only
  uv run exp_resample_begin_pipeline.py --trajectories=451 --n=5 --skip-judge    # generate only

Flags:
  --trajectories=<ids>  REQUIRED. comma-separated original trajectory ids to resample.
  --n=<int>             REQUIRED. resamples (epochs) per trajectory.
  --concurrency=<int>   one knob -> generation max_samples/max_connections AND deviation
                        judging parallelism (default 50).
  --deviation-model=<m> Anthropic model for the deviation (faithfulness) judge
                        (default claude-opus-4-8).
  --dry-run             load + build + validate the plan and print it; no run, no cost.
  --skip-judge          generate only; skip the deviation judge (regular judging is inline).
  --skip-viewer         don't rebuild the viewer at the end.
  --force-judge         re-run (re-spend on) deviation judgments already in the sidecar.

Costs money (Anthropic + the target provider) unless --dry-run.
"""

import asyncio
import json
import pathlib
import sys
from datetime import datetime

# core modules live in lib/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

from inspect_ai import eval_set

import make_viewer
import exp_resample as R
from exp_rh_audit import DIMENSIONS

DEFAULT_CONCURRENCY = 50
DEFAULT_DEVIATION_MODEL = "claude-opus-4-8"


def _arg(name, default=None):
    for a in sys.argv[1:]:
        if a == name:
            return True
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _parse_args() -> dict:
    traj = _arg("--trajectories")
    if not traj or traj is True:
        raise SystemExit("--trajectories is required (comma-separated original trajectory ids)")
    try:
        ids = [int(x) for x in str(traj).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--trajectories must be integers, got {traj!r}")
    if not ids:
        raise SystemExit("--trajectories had no usable ids")

    n = _arg("--n")
    if n is None or n is True:
        raise SystemExit("--n is required (resamples / epochs per trajectory)")
    try:
        n = int(n)
    except ValueError:
        raise SystemExit(f"--n must be an integer, got {n!r}")
    if n < 1:
        raise SystemExit(f"--n must be >= 1, got {n}")

    def _posint(flag, default):
        v = _arg(flag, default)
        try:
            v = int(v)
        except (TypeError, ValueError):
            raise SystemExit(f"{flag} must be an integer, got {v!r}")
        if v < 1:
            raise SystemExit(f"{flag} must be >= 1, got {v}")
        return v

    return dict(
        ids=ids,
        n=n,
        concurrency=_posint("--concurrency", DEFAULT_CONCURRENCY),
        deviation_model=_arg("--deviation-model", DEFAULT_DEVIATION_MODEL),
        dry_run="--dry-run" in sys.argv,
        skip_judge="--skip-judge" in sys.argv,
        skip_viewer="--skip-viewer" in sys.argv,
        force_judge="--force-judge" in sys.argv,
    )


async def main() -> None:
    cfg = _parse_args()
    print("=" * 72)
    print("  BEGIN-RESAMPLE PIPELINE  (faithful re-run from the start; control only)")
    print("=" * 72)
    print(f"  trajectories: {cfg['ids']}")
    print(f"  n (resamples/trajectory): {cfg['n']}    concurrency: {cfg['concurrency']}")
    print(f"  deviation judge: {cfg['deviation_model']}\n")

    print("[load] reading the requested originals (transcript + scratchpad + config) ...")
    refs = await R.load_original_refs(cfg["ids"])
    tasks = []
    for tid in cfg["ids"]:
        ref = refs[tid]
        tasks.append(R.build_resample_task(ref, DIMENSIONS))
        thinking = "" if ref.auditor_reasoning_effort is None else " +thinking"
        print(f"  #{tid}: target={ref.target_name or ref.target_model.split('/')[-1]}  "
              f"seed={ref.seed}  auditor={ref.auditor_model.split('/')[-1]}{thinking}  "
              f"max_turns={ref.max_turns}  fixed_sp={ref.fixed_sp}  "
              f"reference={len(R.build_reference_blob(ref)):,} chars")
    expected = len(tasks) * cfg["n"]
    print(f"\n  {len(tasks)} trajectory task(s) x {cfg['n']} resample(s) = {expected} resample(s) to generate")

    if cfg["dry_run"]:
        print("\n[dry-run] plan validated; no generation, no judging, no cost. Exiting.")
        return

    run_dir = make_viewer.LOGS / f"resample-{cfg['n']}x-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[generate] eval_set -> {run_dir}")
    print("           (this spends on the target provider; the REGULAR judge runs inline) ...")
    success, _logs = eval_set(
        tasks,
        epochs=cfg["n"],
        max_tasks=min(len(tasks), cfg["concurrency"]),
        max_samples=cfg["concurrency"],
        max_connections=cfg["concurrency"],
        log_dir=str(run_dir),
    )
    print(f"[generate] eval_set finished, success={success}")

    # provenance: what was resampled, under what inherited config (self-describing run dir)
    (run_dir / "resample_meta.json").write_text(json.dumps({
        "experiment": "begin_resample",
        "n": cfg["n"],
        "trajectories": cfg["ids"],
        "config_by_traj": {
            str(tid): {
                "target_model": refs[tid].target_model,
                "target_name": refs[tid].target_name,
                "auditor": refs[tid].auditor_model,
                "auditor_reasoning_effort": refs[tid].auditor_reasoning_effort,
                "judge": refs[tid].judge_model,
                "max_turns": refs[tid].max_turns,
                "turn_counter": refs[tid].turn_counter,
                "fixed_sp": refs[tid].fixed_sp,
                "seed": refs[tid].seed,
                "original_run_dir": refs[tid].run_dir,
            }
            for tid in cfg["ids"]
        },
    }, indent=2))

    if cfg["skip_judge"]:
        print("\n[deviation] skipped (--skip-judge). Run later with the deviation judge.")
    else:
        print("\n[deviation] auditor-faithfulness (deviation-from-original) judge ...")
        await R.run_deviation_for_dir(
            run_dir, refs, model=cfg["deviation_model"],
            concurrency=cfg["concurrency"], force=cfg["force_judge"])

    if cfg["skip_viewer"]:
        print("\n[viewer] skipped (--skip-viewer). Rebuild with: uv run make_viewer.py")
    else:
        print("\n[viewer] rebuilding ...")
        await make_viewer.main()

    print(f"\nDone. Resamples are in {run_dir.name}/ and folded into their originals' rows "
          "on the viewer (Resampling section).")


if __name__ == "__main__":
    asyncio.run(main())
