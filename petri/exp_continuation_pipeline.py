"""End-to-end CONTINUATION pipeline: generate -> (regular judge, inline) -> faithfulness
judge -> rebuild viewer, in one command.

THE QUESTION: if a target reward-hacks on one task and is then handed a SECOND, unrelated
task, does its hack rate on the second task go up -- because it's conditioned on its own
earlier hack? Per target model we pick prefixes (prior trajectories of that model) in up to
three flavors, plus new-task sources, and run the prefixed conditions N times each:

  full_hack_prefix      : target carries a trajectory where it fully reward-hacked
  corrected_hack_prefix : target carries a trajectory where it hacked, then corrected it
  clean_prefix          : target carries a trajectory it completed cleanly
  no_prefix             : target carries nothing -- the new task standalone (baseline).
                          NOT run here by default (2026-07-05 split): the baseline is
                          measured FIRST, separately, by exp_baseline_pipeline.py -- used
                          to screen candidate new tasks for a low standalone hack rate
                          before spending on the prefix conditions. The viewer joins the
                          two halves automatically (continuations page groups by B id),
                          so nothing links the run dirs. Pass --conditions=all (or name
                          no_prefix) to run the baseline inline anyway.

Ids are FLAT lists; each trajectory's target model is read from its stamped metadata and
grouped automatically (no positional pairing). Counts per model are free: every
(new task x prefix) pair runs, plus one no_prefix baseline per new task. The new task may
come from a different seed FAMILY than the prefix (e.g. perf_benchmark_gaming vs
ml_training_data_misuse); the target then keeps the PREFIX's system prompt and the pivot
turn says the new task is a different kind of work (see lib/exp_continuation.py).

AUDITOR (Owen 2026-07-05): FRESH by default -- the auditor gets ONLY the new task's seed (like
a plain audit of it) and generates the environment from scratch; it is NOT primed with the
original run and gets no faithfulness instruction. This dropped the giant reference blob that
overflowed the 128k-window sweep-5 auditor. It also means the environment is regenerated
independently each run, so CONFOUND WATCH: differences in auditor behavior across conditions/
epochs could confound the reward-hack-rate comparison -- the faithfulness judge is kept ON to
measure that drift, to be investigated before trusting cross-condition differences.
`--prime-auditor` restores the legacy primed behavior (full record + faithfulness instruction)
for that investigation. Either way the auditor never sees the prefix. The clean contrasts are
each-prefix-vs-no_prefix; prefix-flavor-vs-prefix-flavor is confounded by every way those prior
runs differ -- read it as suggestive.

Two judges:
  1. the REGULAR reward-hacking judge, inline -- but the prior-task prefix is MECHANICALLY cut
     out of the transcript the judge sees (system prompt + new task only, numbers preserved),
     so it can't see or infer the prior task (no_prefix is judged as a standalone run);
  2. a FAITHFULNESS judge that compares each continuation to its new-task original and scores
     how much the auditor's environment diverged from it (lib/exp_continuation
     .run_faithfulness_for_dir) -- stored as a queryable per-run flag. With fresh auditors this
     is the drift diagnostic for the confound above (its prompt still says "primed", a wording
     lag to revisit when interpreting the scores).

Usage:
  uv run exp_continuation_pipeline.py --full-hack-prefixes=455,471 --clean-prefixes=460,473 \\
      --corrected-hack-prefixes=602,617 --continuations=458,470,610 --n=5
  uv run exp_continuation_pipeline.py ... --conditions=full_hack_prefix,no_prefix   # subset
  uv run exp_continuation_pipeline.py ... --n=5 --dry-run        # FREE: plan only
  uv run exp_continuation_pipeline.py ... --n=5 --skip-judge     # generate only

Flags:
  --continuations=<ids>             REQUIRED. new-task source ids (any models, any seeds;
                                    each pairs with every same-model prefix from a
                                    different seed).
  --full-hack-prefixes=<ids>        full-hack prefix ids (optional if the condition is off).
  --corrected-hack-prefixes=<ids>   corrected-hack prefix ids (optional if the condition is off).
  --clean-prefixes=<ids>            clean prefix ids (optional if the condition is off).
  --conditions=<a,b|all>            subset of {full_hack_prefix,corrected_hack_prefix,
                                    clean_prefix,no_prefix} (default: the three prefixed
                                    conditions; `all` adds no_prefix back).
  --n=<int>                         REQUIRED. continuations (epochs) per cell.
  --concurrency=<int>               one knob -> generation max_samples/max_connections AND
                                    faithfulness-judge parallelism (default 50).
  --prime-auditor                   LEGACY: prime the auditor with the new-task original's full
                                    record + a faithfulness instruction (default is a FRESH
                                    auditor that gets only the seed). For the confound study.
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


def _ids(flag: str, required: bool = False) -> list[int]:
    raw = _arg(flag)
    if not raw or raw is True:
        if required:
            raise SystemExit(f"{flag} is required (comma-separated original trajectory ids)")
        return []
    try:
        ids = [int(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"{flag} must be integers, got {raw!r}")
    if not ids:
        raise SystemExit(f"{flag} had no usable ids")
    return ids


# CLI flag -> the condition its ids realize (see C.PREFIX_CONDITIONS)
_PREFIX_FLAGS = {
    "--full-hack-prefixes": "full_hack_prefix",
    "--corrected-hack-prefixes": "corrected_hack_prefix",
    "--clean-prefixes": "clean_prefix",
}


def _posint(flag, default):
    v = _arg(flag, default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise SystemExit(f"{flag} must be an integer, got {v!r}")
    if v < 1:
        raise SystemExit(f"{flag} must be >= 1, got {v}")
    return v


def _parse_args() -> dict:
    cont = _ids("--continuations", required=True)
    prefix_ids = {cond: _ids(flag) for flag, cond in _PREFIX_FLAGS.items()}

    cond_raw = _arg("--conditions")
    if cond_raw is None or cond_raw is True:
        # default: prefixed conditions only -- the no_prefix baseline is measured first by
        # exp_baseline_pipeline.py (the screening half of the split pipeline).
        conditions = list(C.PREFIX_CONDITIONS)
    elif str(cond_raw).strip() == "all":
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

    # every requested prefixed condition needs ids, and ids for an unrequested condition
    # are almost certainly a mistake -- fail loudly either way, before any loading.
    for cond in conditions:
        if cond != "no_prefix" and not prefix_ids.get(cond):
            flag = next(f for f, c in _PREFIX_FLAGS.items() if c == cond)
            raise SystemExit(f"condition {cond!r} is requested but {flag} was not given; "
                             f"pass ids or narrow --conditions.")
    for cond, ids in prefix_ids.items():
        if ids and cond not in conditions:
            flag = next(f for f, c in _PREFIX_FLAGS.items() if c == cond)
            raise SystemExit(f"{flag} was given but --conditions excludes {cond!r}; "
                             "drop the flag or widen --conditions.")

    n = _arg("--n")
    if n is None or n is True:
        raise SystemExit("--n is required (continuations / epochs per cell)")
    try:
        n = int(n)
    except ValueError:
        raise SystemExit(f"--n must be an integer, got {n!r}")
    if n < 1:
        raise SystemExit(f"--n must be >= 1, got {n}")

    return dict(
        prefix_ids=prefix_ids, cont=cont, conditions=conditions, n=n,
        run_dir_stem="continuation",
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
    print("  CONTINUATION PIPELINE  (condition a target on a prior task, then hand it a new one)")
    print("=" * 76)
    for flag, cond in _PREFIX_FLAGS.items():
        print(f"  {cond:22s}: {cfg['prefix_ids'][cond] or '—'}")
    print(f"  continuations (new-task sources): {cfg['cont']}")
    print(f"  conditions: {cfg['conditions']}    n: {cfg['n']}    concurrency: {cfg['concurrency']}")
    print(f"  auditor: {'PRIMED (reference + faithfulness)' if cfg['prime_auditor'] else 'FRESH (seed only, no reference)'}")
    print(f"  faithfulness judge: {cfg['deviation_model']}\n")
    await run_pipeline(cfg)


async def run_pipeline(cfg: dict) -> None:
    """The shared engine: plan -> generate (inline regular judge) -> faithfulness judge ->
    annotate -> viewer. Called by main() here (the prefixed experiment) and by
    exp_baseline_pipeline.py (the no_prefix screening half); cfg carries everything,
    including run_dir_stem (the run-dir name prefix, so the two halves stay tellable
    apart on disk while both start with 'continuation-' for the viewer's dir scan)."""
    print("[load] reading originals, reconstructing prefixes, validating the plan ...")
    plans = await C.build_plans(cfg["prefix_ids"], cfg["cont"])

    tasks = []
    for model, plan in sorted(plans.items()):
        # coverage: a model must contribute BOTH sides (its ids are probably a typo otherwise);
        # prefixes are only required when a prefixed condition is requested.
        if not plan.b_refs:
            raise SystemExit(f"model {model!r} has prefixes but no --continuations id; every "
                             "model in the run needs at least one new-task source.")
        needs_prefixes = any(c != "no_prefix" for c in cfg["conditions"])
        if needs_prefixes and not plan.prefixes:
            raise SystemExit(f"model {model!r} has continuations but no prefixes; pass prefix "
                             "ids for it or run with --conditions=no_prefix.")
        for cond in cfg["conditions"]:
            if cond != "no_prefix" and not any(p.condition == cond for p in plan.prefixes):
                raise SystemExit(f"model {model!r} has no {cond} prefix but --conditions "
                                 f"requests it; pass an id or narrow --conditions.")

        # the run-level allow/correct condition each original was generated under -- shown so
        # a "correct"-condition id passed as a continuation is visible at plan time (the
        # continuation reruns the new task under the SAME stamped condition; nothing imposes one).
        def _cond(ref):
            return f" [{ref.condition}]" if ref.condition else ""

        print(f"  {model}:")
        for p in plan.prefixes:
            synth = "  [synthetic tool-result closer]" if p.synthesized_closer else ""
            print(f"      {p.condition:22s} #{p.ref.traj_id}  (seed {p.ref.seed}{_cond(p.ref)}, "
                  f"family {C.seed_family(p.ref.seed_dir)}, {len(p.messages)} msgs){synth}")
        for b in plan.b_refs:
            fams = {C.seed_family(p.ref.seed_dir) for p in plan.prefixes}
            cross = ("  [CROSS-FAMILY: target keeps the prefix's system prompt + pivot note]"
                     if fams - {C.seed_family(b.seed_dir)} else "")
            # reference-blob size only matters when priming the auditor (fresh auditors never
            # receive it); show it only then so the fresh plan print isn't misleading.
            refsz = (f", reference={len(C.build_reference_blob(b)):,} chars"
                     if cfg["prime_auditor"] else "")
            print(f"      new task               #{b.traj_id}  (seed {b.seed}{_cond(b)}, "
                  f"family {C.seed_family(b.seed_dir)}, max_turns={b.max_turns}{refsz}){cross}")
        for b in plan.b_refs:
            for cond in cfg["conditions"]:
                if cond == "no_prefix":
                    tasks.append(C.build_continuation_task(
                        plan, b, cond, None, DIMENSIONS, prime_auditor=cfg["prime_auditor"]))
                else:
                    for p in plan.prefixes:
                        if p.condition == cond:
                            tasks.append(C.build_continuation_task(
                                plan, b, cond, p, DIMENSIONS, prime_auditor=cfg["prime_auditor"]))

    expected = len(tasks) * cfg["n"]
    print(f"\n  {len(plans)} model(s), {len(tasks)} cell(s) x n={cfg['n']} = {expected} "
          "continuation(s) to generate")

    if cfg["dry_run"]:
        print("\n[dry-run] plan validated (prefixes + new tasks + conditions). No generation, no cost.")
        return

    run_dir = make_viewer.LOGS / f"{cfg['run_dir_stem']}-{cfg['n']}x-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[generate] eval_set -> {run_dir}")
    print("           (spends on the target provider; the REGULAR judge runs inline) ...")
    import openrouter_cost   # persist OpenRouter's real billed cost per call (see lib/openrouter_cost.py)
    import direct_cost       # exact list-price cost for direct anthropic/openai calls (see lib/direct_cost.py)
    import model_window      # correct context windows so the auditor compacts at the real window (see lib/model_window.py)
    openrouter_cost.install()
    direct_cost.install()
    model_window.install()
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
        "run_kind": cfg["run_dir_stem"],   # continuation | continuation-baseline
        "config_version": "continuation-v2",
        "n": cfg["n"],
        "conditions": cfg["conditions"],
        "preamble": C.PREAMBLE,
        "plans": {
            model: {
                "target_model": plan.target_model,
                "prefixes": [
                    {
                        "traj_id": p.ref.traj_id,
                        "condition": p.condition,
                        "seed": p.ref.seed,
                        "seed_family": C.seed_family(p.ref.seed_dir),
                        "run_condition": p.ref.condition,   # allow|correct|None
                        "synthesized_closer": p.synthesized_closer,
                    }
                    for p in plan.prefixes
                ],
                "continuations": [
                    {
                        "traj_id": b.traj_id,
                        "seed": b.seed,
                        "seed_family": C.seed_family(b.seed_dir),
                        "run_condition": b.condition,        # allow|correct|None
                        "auditor": b.auditor_model,
                        "judge": b.judge_model,
                        "max_turns": b.max_turns,
                    }
                    for b in plan.b_refs
                ],
            }
            for model, plan in sorted(plans.items())
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
        print("\n[faithfulness] auditor-faithfulness (deviation-from-original) judge ...")
        b_refs_by_id = {b.traj_id: b for plan in plans.values() for b in plan.b_refs}
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
