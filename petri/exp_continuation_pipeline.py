"""End-to-end CONTINUATION pipeline: generate -> (regular judge, inline) -> faithfulness
judge -> rebuild viewer, in one command.

THE QUESTION: if a target reward-hacks on one task and is then handed a SECOND, unrelated
task, does its hack rate on the second task go up -- because it's conditioned on its own
earlier hack? For each target MODEL we pick a (hack prefix A, clean prefix A from the same
seed, task source B from a different seed) triple, and run three conditions N times each:

  hack_prefix  : target carries A's HACK transcript,  then B's task (after a fixed pivot)
  clean_prefix : target carries A's CLEAN transcript, then B's task (after the same pivot)
  no_prefix    : target carries no prefix,            then B's task (no pivot) -- baseline

The auditor reproduces B's environment in ALL three conditions (primed with B's full record),
and never sees the prefix. The clean contrasts are hack_prefix-vs-no_prefix and
clean_prefix-vs-no_prefix ("any prior history, with/without a hack"); hack-vs-clean is
confounded by every way a hack run and a clean run differ -- read it as suggestive.

Two judges:
  1. the REGULAR reward-hacking judge, inline -- but the prior-task prefix is MECHANICALLY cut
     out of the transcript the judge sees (system prompt + new task only, numbers preserved), so
     it can't see or infer the prior task (no_prefix is judged as a standalone B run);
  2. a FAITHFULNESS judge that compares each continuation to its B original and scores how
     faithfully the AUDITOR reproduced B (lib/exp_continuation.run_faithfulness_for_dir) --
     stored as a queryable per-run flag. Low faithfulness is a finding about validity.

All config (target + reasoning pin, auditor + thinking, judge, max_turns, turn_counter, the
ONE fixed system prompt) is inherited from B's stamped metadata. The core lives in
lib/exp_continuation.py.

Usage:
  uv run exp_continuation_pipeline.py --hack-trajectories=455 --clean-trajectories=460 \\
      --continuation-trajectories=458 --n=5
  # sweep several models (one hack/clean/B id per model in each list):
  uv run exp_continuation_pipeline.py --hack-trajectories=455,471 --clean-trajectories=460,473 \\
      --continuation-trajectories=458,470 --n=5 --concurrency=50
  uv run exp_continuation_pipeline.py ... --conditions=hack_prefix,no_prefix   # subset
  uv run exp_continuation_pipeline.py ... --n=5 --dry-run        # FREE: plan only
  uv run exp_continuation_pipeline.py ... --n=5 --skip-judge     # generate only

Flags:
  --hack-trajectories=<ids>         REQUIRED. hack prefix (A) ids; one per target model.
  --clean-trajectories=<ids>        REQUIRED. clean prefix (A) ids; same seed as the hack
                                    prefix, one per target model.
  --continuation-trajectories=<ids> REQUIRED. task-source (B) ids; a DIFFERENT seed, one per
                                    target model.
  --models=<names>                  restrict to these target names (default: every model that
                                    appears across the three lists).
  --conditions=<a,b|all>            subset of {hack_prefix,clean_prefix,no_prefix} (default all).
  --n=<int>                         REQUIRED. continuations (epochs) per (triple, condition).
  --concurrency=<int>               one knob -> generation max_samples/max_connections AND
                                    faithfulness-judge parallelism (default 50).
  --deviation-model=<m>             Anthropic model for the faithfulness judge (default
                                    claude-opus-4-8).
  --annotate-model=<m>              Anthropic model for the hack-turn annotator (default
                                    claude-opus-4-8).
  --dry-run                         load + build + validate the plan and print it; no cost.
  --skip-judge                      generate only; skip the faithfulness judge (regular
                                    judging is inline).
  --skip-annotate                   skip the hack-turn annotation of full-hack continuations.
  --skip-viewer                     don't rebuild the viewer at the end.
  --force-judge                     re-run (re-spend on) faithfulness judgments already stored.
  --force-annotate                  re-annotate full-hack continuations already in annotations.json.

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
import exp_continuation as C
from exp_rh_audit import DIMENSIONS

DEFAULT_CONCURRENCY = 50
DEFAULT_DEVIATION_MODEL = "claude-opus-4-8"
DEFAULT_ANNOTATE_MODEL = "claude-opus-4-8"


def _arg(name, default=None):
    for a in sys.argv[1:]:
        if a == name:
            return True
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def _ids(flag: str) -> list[int]:
    raw = _arg(flag)
    if not raw or raw is True:
        raise SystemExit(f"{flag} is required (comma-separated original trajectory ids)")
    try:
        ids = [int(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"{flag} must be integers, got {raw!r}")
    if not ids:
        raise SystemExit(f"{flag} had no usable ids")
    return ids


def _parse_args() -> dict:
    hack = _ids("--hack-trajectories")
    clean = _ids("--clean-trajectories")
    cont = _ids("--continuation-trajectories")

    models_raw = _arg("--models")
    models = ([m.strip() for m in str(models_raw).split(",") if m.strip()]
              if models_raw and models_raw is not True else None)

    cond_raw = _arg("--conditions")
    if cond_raw is None or cond_raw is True or str(cond_raw).strip() == "all":
        conditions = list(C.CONDITIONS)
    else:
        conditions = [c.strip() for c in str(cond_raw).split(",") if c.strip()]
        unknown = [c for c in conditions if c not in C.CONDITIONS]
        if unknown:
            raise SystemExit(f"unknown --conditions {unknown}; choices: {C.CONDITIONS} (or `all`)")
        # canonical order
        conditions = [c for c in C.CONDITIONS if c in conditions]
    if not conditions:
        raise SystemExit(f"--conditions had no usable names; choices: {C.CONDITIONS}")

    n = _arg("--n")
    if n is None or n is True:
        raise SystemExit("--n is required (continuations / epochs per (triple, condition))")
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
        hack=hack, clean=clean, cont=cont, models=models, conditions=conditions, n=n,
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
    print("  CONTINUATION PIPELINE  (condition a target on a prior task, then hand it a new one)")
    print("=" * 76)
    print(f"  hack prefixes (A):   {cfg['hack']}")
    print(f"  clean prefixes (A):  {cfg['clean']}")
    print(f"  task sources (B):    {cfg['cont']}")
    print(f"  conditions: {cfg['conditions']}    n: {cfg['n']}    concurrency: {cfg['concurrency']}")
    print(f"  faithfulness judge: {cfg['deviation_model']}\n")

    print("[load] reading originals, reconstructing prefixes, validating triples ...")
    triples = await C.build_triples(cfg["hack"], cfg["clean"], cfg["cont"], cfg["models"])
    if not triples:
        raise SystemExit("no (model) triples built -- check --models against the provided ids.")

    tasks = []
    for model, t in triples.items():
        synth = []
        if t.hack_synth:
            synth.append("hack-prefix")
        if t.clean_synth:
            synth.append("clean-prefix")
        synth_note = (f"  [synthetic tool-result closer added to: {', '.join(synth)}]"
                      if synth else "")
        print(f"  {model}: hack#{t.hack_ref.traj_id} clean#{t.clean_ref.traj_id} "
              f"(seed {t.hack_ref.seed}) -> B#{t.b_ref.traj_id} (seed {t.b_ref.seed})  "
              f"max_turns={t.b_ref.max_turns}{synth_note}")
        print(f"      prefix sizes: hack={len(t.hack_prefix)} msgs, clean={len(t.clean_prefix)} msgs"
              f"  |  B reference={len(C.build_reference_blob(t.b_ref)):,} chars")
        for cond in cfg["conditions"]:
            tasks.append(C.build_continuation_task(t, cond, DIMENSIONS))

    expected = len(tasks) * cfg["n"]
    print(f"\n  {len(triples)} model(s) x {len(cfg['conditions'])} condition(s) = {len(tasks)} cell(s) "
          f"x n={cfg['n']} = {expected} continuation(s) to generate")

    if cfg["dry_run"]:
        print("\n[dry-run] plan validated (triples + prefixes + conditions). No generation, no cost.")
        return

    run_dir = make_viewer.LOGS / f"continuation-{cfg['n']}x-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[generate] eval_set -> {run_dir}")
    print("           (spends on the target provider; the REGULAR judge runs inline) ...")
    success, _logs = eval_set(
        tasks,
        epochs=cfg["n"],
        max_tasks=min(len(tasks), cfg["concurrency"]),
        max_samples=cfg["concurrency"],
        max_connections=cfg["concurrency"],
        log_dir=str(run_dir),
    )
    print(f"[generate] eval_set finished, success={success}")

    # provenance: what was run, under what inherited config (self-describing run dir).
    (run_dir / "continuation_meta.json").write_text(json.dumps({
        "experiment": "continuation",
        "n": cfg["n"],
        "conditions": cfg["conditions"],
        "preamble": C.PREAMBLE,
        "triples": {
            model: {
                "target_model": t.target_model,
                "hack_prefix_traj_id": t.hack_ref.traj_id,
                "clean_prefix_traj_id": t.clean_ref.traj_id,
                "prefix_seed": t.hack_ref.seed,
                "continuation_traj_id": t.b_ref.traj_id,
                "b_seed": t.b_ref.seed,
                "auditor": t.b_ref.auditor_model,
                "judge": t.b_ref.judge_model,
                "max_turns": t.b_ref.max_turns,
                "fixed_system_prompt_chars": len(t.fixed_system_prompt),
                "hack_synthesized_closer": t.hack_synth,
                "clean_synthesized_closer": t.clean_synth,
            }
            for model, t in triples.items()
        },
    }, indent=2))

    # GUARD: the viewer loader must actually see what we just generated. load_mode keys on the
    # "audit_judge" score and silently SKIPS any sample lacking it -- so a judge score-key mismatch
    # (or a bad task name) makes faithfulness, annotation, AND the Continuations page come up empty
    # with no obvious cause. Fail loudly here instead. (This is exactly the bug that made a whole
    # run invisible once.) A partial drop is surfaced as a warning, not silently tolerated.
    loaded = await make_viewer.load_mode(run_dir)
    if not loaded:
        raise SystemExit(
            f"[fatal] eval_set wrote ~{expected} sample(s) to {run_dir.name}, but the viewer "
            f"loader (make_viewer.load_mode) sees 0 of them. This is almost always a judge "
            f"score-key mismatch: load_mode keys on the 'audit_judge' score, so the continuation "
            f"judge must register under name='audit_judge'. Nothing downstream will populate. "
            f"Re-inspect with MAKE_VIEWER_NO_CACHE=1 and do NOT trust this run.")
    if len(loaded) < expected:
        print(f"  WARNING: viewer loader sees {len(loaded)}/{expected} continuation(s) -- "
              f"{expected - len(loaded)} were dropped (judge error / unscored?) and won't appear "
              f"downstream.")
    else:
        print(f"[generate] viewer loader sees all {len(loaded)} continuation(s) -- OK")

    if cfg["skip_judge"]:
        print("\n[faithfulness] skipped (--skip-judge). Run later with the faithfulness judge.")
    else:
        print("\n[faithfulness] auditor-faithfulness (deviation-from-B) judge ...")
        b_refs_by_id = {t.b_ref.traj_id: t.b_ref for t in triples.values()}
        await C.run_faithfulness_for_dir(
            run_dir, b_refs_by_id, model=cfg["deviation_model"],
            concurrency=cfg["concurrency"], force=cfg["force_judge"])

    # hack-turn annotation of the FULL-hack continuations (same stage the original audit
    # pipeline runs); scoped to the new task, written to annotations.json, picked up by the viewer.
    if cfg["skip_annotate"]:
        print("\n[annotate] skipped (--skip-annotate).")
    else:
        print("\n[annotate] localizing reward-hack turns in full-hack continuations (new task only) ...")
        await C.run_continuation_annotation(
            run_dir, model=cfg["annotate_model"],
            concurrency=cfg["concurrency"], force=cfg["force_annotate"])

    if cfg["skip_viewer"]:
        print("\n[viewer] skipped (--skip-viewer). Rebuild with: uv run make_viewer.py")
    else:
        print("\n[viewer] rebuilding ...")
        await make_viewer.main()

    print(f"\nDone. Continuations are in {run_dir.name}/ (Continuations page on the viewer).")


if __name__ == "__main__":
    asyncio.run(main())
