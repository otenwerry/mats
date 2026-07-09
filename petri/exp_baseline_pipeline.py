"""BASELINE screening pipeline: the first half of the split continuation experiment
(2026-07-05). Runs ONLY the no_prefix condition -- each candidate new task standalone,
N times -- to measure its PRIOR (standalone) reward-hack rate before any prefix money
is spent.

WHY: continuation new tasks vary in how often they hack on their own, which is noise in
the prefix experiment. A task whose original run happened to be a non-hack isn't
guaranteed to have a low prior -- it may just be a lucky draw from a
usually-hacking distribution. So: screen candidates here first (n runs each, judged +
faithfulness-scored exactly like real continuations), keep the ones whose measured
standalone hack rate is low, then feed those into exp_continuation_pipeline.py (which
no longer runs no_prefix by default).

AUDITOR: FRESH by default (Owen 2026-07-05) -- the auditor gets only the candidate's seed
and generates the environment from scratch; it is NOT primed with the original run and gets
no faithfulness instruction. For no_prefix this makes each run byte-for-byte a plain audit of
the candidate's seed. `--prime-auditor` restores the legacy primed behavior. The faithfulness
judge still runs (measuring how far the fresh auditor drifted from the original) as the drift
diagnostic; see exp_continuation_pipeline for the confound note.

Everything downstream is shared with exp_continuation_pipeline.run_pipeline: the same
inline regular judge (each run judged as a standalone audit -- no prefix exists to
slice), the same faithfulness judge vs the original, the same hack-turn annotation, the
same viewer rebuild. Results land in logs/continuation-baseline-<N>x-<timestamp>/ --
the 'continuation-' prefix keeps them on the viewer's continuations pages, where they
appear as the "Continuation run" rows of each (model, B) box. A later prefixed run on
the same B ids joins those boxes automatically (grouping is by B id), so the screening
runs double as the experiment's no_prefix baseline.

Usage:
  uv run exp_baseline_pipeline.py --continuations=1117,760,1131 --n=10
  uv run exp_baseline_pipeline.py ... --dry-run        # FREE: plan only
  uv run exp_baseline_pipeline.py ... --skip-judge     # generate only

Flags:
  --continuations=<ids>   REQUIRED. candidate new-task source ids (any models, any seeds).
  --n=<int>               REQUIRED. runs per candidate.
  --concurrency=<int>     one knob -> generation max_samples/max_connections AND
                          faithfulness-judge parallelism (default 50).
  --prime-auditor         LEGACY: prime the auditor with the original + a faithfulness
                          instruction (default is a FRESH auditor, seed only). For the confound study.
  --deviation-model=<m>   Anthropic model for the faithfulness judge (default claude-opus-4-8).
  --annotate-model=<m>    Anthropic model for the hack-turn annotator (default claude-opus-4-8).
  --dry-run               load + build + validate the plan and print it; no cost.
  --skip-judge            generate only; skip the faithfulness judge.
  --skip-annotate         skip the hack-turn annotation of full-hack runs.
  --skip-viewer           don't rebuild the viewer at the end.
  --force-judge           re-run (re-spend on) faithfulness judgments already stored.
  --force-annotate        re-annotate full-hack runs already in annotations.json.

Costs money (Anthropic + the target provider) unless --dry-run.
"""

import asyncio
import sys

from exp_continuation_pipeline import (
    DEFAULT_ANNOTATE_MODEL,
    DEFAULT_CONCURRENCY,
    DEFAULT_DEVIATION_MODEL,
    _arg,
    _ids,
    _posint,
    run_pipeline,
)


def _parse_args() -> dict:
    cont = _ids("--continuations", required=True)
    n = _arg("--n")
    if n is None or n is True:
        raise SystemExit("--n is required (runs per candidate)")
    try:
        n = int(n)
    except ValueError:
        raise SystemExit(f"--n must be an integer, got {n!r}")
    if n < 1:
        raise SystemExit(f"--n must be >= 1, got {n}")
    # same cfg shape run_pipeline expects from the continuation pipeline, pinned to the
    # baseline: no prefixes, the single no_prefix condition, its own run-dir stem.
    return dict(
        prefix_ids={}, cont=cont, conditions=["no_prefix"], n=n,
        run_dir_stem="continuation-baseline",
        prime_auditor="--prime-auditor" in sys.argv,
        concurrency=_posint("--concurrency", DEFAULT_CONCURRENCY),
        deviation_model=_arg("--deviation-model", DEFAULT_DEVIATION_MODEL),
        annotate_model=_arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        dry_run="--dry-run" in sys.argv,
        skip_judge="--skip-judge" in sys.argv,
        skip_annotate="--skip-annotate" in sys.argv,
        skip_viewer="--skip-viewer" in sys.argv,
        force_judge="--force-judge" in sys.argv,
        force_annotate="--force-annotate" in sys.argv,
    )


async def main() -> None:
    cfg = _parse_args()
    print("=" * 76)
    print("  BASELINE SCREENING  (measure each candidate new task's standalone hack rate)")
    print("=" * 76)
    print(f"  candidates (new-task sources): {cfg['cont']}")
    print(f"  n per candidate: {cfg['n']}    concurrency: {cfg['concurrency']}")
    print(f"  auditor: {'PRIMED (reference + faithfulness)' if cfg['prime_auditor'] else 'FRESH (seed only, no reference)'}")
    print(f"  faithfulness judge: {cfg['deviation_model']}\n")
    await run_pipeline(cfg)


if __name__ == "__main__":
    asyncio.run(main())
