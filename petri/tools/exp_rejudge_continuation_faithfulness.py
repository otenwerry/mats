"""Re-run ONLY the faithfulness (deviation-from-B) judge on an existing continuation run dir.

WHY THIS EXISTS
The continuation pipeline (exp_continuation_pipeline.py) only judges as part of a full
generate->judge run, so re-running it would regenerate the continuations (real target-provider
spend). After editing the faithfulness-judge prompt, you usually just want to re-score the
continuations you already have. This re-runs the faithfulness judge in place: it reads the
existing .eval continuations from the dir and re-judges them against their B original. NO
continuations are regenerated; the ONLY cost is the Anthropic judge calls.

It can be scoped to specific conditions (e.g. --conditions=no_prefix), leaving the other
conditions' stored verdicts untouched. force is ON by default (re-judging is the whole point).

LOSSY PROCESSING: none -- full transcripts are sent uncut, exactly as in the pipeline.

Usage:
  uv run tools/exp_rejudge_continuation_faithfulness.py --dir=continuation-1x-20260630-104750
  uv run tools/exp_rejudge_continuation_faithfulness.py --dir=... --conditions=no_prefix
  uv run tools/exp_rejudge_continuation_faithfulness.py --dir=... --conditions=no_prefix,clean_prefix
  uv run tools/exp_rejudge_continuation_faithfulness.py --dir=... --concurrency=50
  uv run tools/exp_rejudge_continuation_faithfulness.py --dir=... --model=claude-opus-4-8

--dir accepts a bare run-dir name (resolved under logs/) or an absolute path.

Costs money (Anthropic judge only). Then regenerate the viewer (free):
  uv run make_viewer.py
"""

import asyncio
import pathlib
import sys

_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

import exp_continuation as C
from exp_continuation import parse_continuation_task, run_faithfulness_for_dir
from make_viewer import load_mode
from petri_paths import LOGS

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_CONCURRENCY = 50


def _arg(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


async def main() -> None:
    dir_arg = _arg("--dir")
    if not dir_arg:
        raise SystemExit("--dir is required (a continuation run-dir name under logs/, or an abs path)")
    run_dir = pathlib.Path(dir_arg)
    if not run_dir.is_absolute():
        run_dir = LOGS / dir_arg
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    model = _arg("--model", DEFAULT_MODEL)
    concurrency = int(_arg("--concurrency", str(DEFAULT_CONCURRENCY)))
    cond_raw = _arg("--conditions")
    conditions = None
    if cond_raw:
        conditions = {c.strip() for c in cond_raw.split(",") if c.strip()}
        unknown = conditions - C.KNOWN_CONDITIONS
        if unknown:
            raise SystemExit(f"unknown --conditions {sorted(unknown)}; "
                             f"choices: {sorted(C.KNOWN_CONDITIONS)}")

    # build {B traj id -> OriginalRef} from the B ids referenced by this dir's task names.
    audits = await load_mode(run_dir)
    b_ids = sorted({p[2] for a in audits if (p := parse_continuation_task(a["task"]))})
    if not b_ids:
        raise SystemExit(f"no continuation tasks found in {run_dir.name} (is this a continuation dir?)")
    print(f"[setup] dir={run_dir.name}  B id(s)={b_ids}  "
          f"conditions={'all' if conditions is None else sorted(conditions)}  model={model}")
    refs = await C.load_original_refs(b_ids)
    b_refs_by_id = {bid: refs[bid] for bid in b_ids if bid in refs}
    missing = [bid for bid in b_ids if bid not in b_refs_by_id]
    if missing:
        print(f"  WARNING: could not load B original(s) {missing}; their continuations will be skipped")

    await run_faithfulness_for_dir(
        run_dir, b_refs_by_id, model=model, concurrency=concurrency,
        force=True, conditions=conditions)

    print("\nDone. Regenerate the viewer (free) to surface the updated verdicts:  uv run make_viewer.py")


if __name__ == "__main__":
    asyncio.run(main())
