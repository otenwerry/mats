"""End-to-end CONTINUATION pipeline: generate -> (regular judge, inline) -> [faithfulness
judge: OFF by default] -> rebuild viewer, in one command.

THE QUESTION: if a target reward-hacks on one task and is then handed a SECOND, unrelated
task, does its hack rate on the second task go up -- because it's conditioned on its own
earlier hack?

ONE INVOCATION = ONE TREATMENT (2026-07-08 rewrite). You choose the prefix set with a single
`--prefixes` flag and label the run with `--treatment`; run the file again for another
treatment. This replaced the old fixed three-flavor scheme (--full-hack-prefixes /
--corrected-hack-prefixes / --clean-prefixes + --conditions) so you can define arbitrarily many
treatments without touching the code -- the viewer renders whatever treatments appear in the
data, each in its own spot.

  uv run exp_continuation_pipeline.py --treatment=full-hack --prefixes=455,471 \\
      --continuations=458,470,610 --n=5
  uv run exp_continuation_pipeline.py --treatment=clean --prefixes=460,473 \\
      --continuations=458,470,610 --n=5
  uv run exp_continuation_pipeline.py --treatment=no-prefix --prefixes=none \\
      --continuations=458,470,610 --n=5          # the baseline (no prior context)

`--prefixes=none` is the baseline (the new task run standalone): the target carries nothing, so
there is nothing to compare a prefix against but itself. The viewer joins treatments by B id --
run the baseline once per new task and every prefixed treatment sits beside it in the same
(model, new task) box.

Ids are FLAT lists; each trajectory's target model is read from its stamped metadata and
grouped automatically (no positional pairing). Counts per model are free: every
(new task x prefix) pair runs. The new task may come from a different seed FAMILY than the
prefix (e.g. p_hacking vs ml_training_data_misuse). All new originals share
seeds/SYSTEM_PROMPT.txt, and planning aborts before paid work if the selected A/B source
trajectories carry different recorded prompt text. The pivot turn names the new kind of
work (see lib/exp_continuation.py).

AUDITOR (Owen 2026-07-08): ALWAYS FAITHFUL -- the auditor is primed with the new task's full
original run + a faithfulness instruction and reproduces that environment as closely as it can.
This keeps the environment FIXED across treatments so the reward-hack-rate comparison isn't
confounded by the auditor improvising differently each run; the faithfulness judge still scores
residual drift. (The earlier fresh-auditor default existed to dodge a DeepSeek context overflow
that turned out to be an Inspect window-detection glitch, since fixed in lib/model_window.py, so
priming is safe again.) The auditor never sees the prefix -- Petri keeps the auditor's and
target's message lists separate. The clean contrast is each-prefix-vs-baseline;
prefix-vs-prefix is confounded by every way those prior runs differ -- read it as suggestive.

Two judges:
  1. the REGULAR reward-hacking judge, inline -- but the prior-task prefix is MECHANICALLY cut
     out of the transcript the judge sees (system prompt + new task only, numbers preserved),
     so it can't see or infer the prior task (a baseline is judged as a standalone run);
  2. a FAITHFULNESS judge that compares each continuation to its new-task original and scores
     how much the auditor's environment diverged from it (lib/exp_continuation
     .run_faithfulness_for_dir) -- stored as a queryable per-run flag. OFF BY DEFAULT (Owen
     2026-07-09): the auditor is still primed to be faithful, we just don't score it for now.
     Pass --faithfulness-judge to run it, or run it after the fact via
     tools/exp_rejudge_continuation_faithfulness.py. The judge code is untouched, only the
     pipeline no longer calls it unless asked.

Usage:
  uv run exp_continuation_pipeline.py --treatment=full-hack --prefixes=455,471 \\
      --continuations=458,470,610 --n=5
  uv run exp_continuation_pipeline.py ... --n=5 --dry-run             # FREE: plan only
  uv run exp_continuation_pipeline.py ... --n=5 --faithfulness-judge  # also score faithfulness

Flags:
  --treatment=<slug>                REQUIRED. free-form label for this run (lowercase letters,
                                    digits, hyphens; e.g. full-hack, clean, no-prefix). Names
                                    the run's spot in the viewer; runs with the SAME treatment
                                    + same new task pool together (that's how you add epochs).
  --prefixes=<ids|none>             REQUIRED. prefix trajectory ids the target carries (any
                                    models, any seeds), or `none` for a baseline (no prefix).
  --continuations=<ids>             REQUIRED. new-task source ids (any models, any seeds; each
                                    pairs with every same-model prefix from a different seed).
  --n=<int>                         REQUIRED. continuations (epochs) per cell.
  --concurrency=<int>               one knob -> generation max_samples/max_connections AND
                                    faithfulness-judge parallelism (default 50).
  --deviation-model=<m>             Anthropic model for the faithfulness judge (default
                                    claude-opus-4-8).
  --annotate-model=<m>              Anthropic model for the hack-turn annotator (default
                                    claude-opus-4-8).
  --dry-run                         load + build + validate the plan and print it; no cost.
  --faithfulness-judge              run the auditor-faithfulness judge (OFF by default). The
                                    regular reward-hacking judge is inline regardless.
  --skip-annotate                   skip the hack-turn annotation of full-hack continuations.
  --skip-viewer                     don't rebuild the viewer at the end.
  --force-judge                     re-run (re-spend on) faithfulness judgments already stored;
                                    implies --faithfulness-judge.
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

import viewer
import exp_continuation as C
from judge_models import SECONDARY_JUDGE
from exp_annotate_hacks import DEFAULT_MODEL as ANNOTATE_DEFAULT_MODEL

DEFAULT_CONCURRENCY = 50
# Secondary judge role (Anthropic SDK only) -- see lib/judge_models.py
DEFAULT_DEVIATION_MODEL = SECONDARY_JUDGE
# The continuation annotate stage delegates to the AGENTIC annotator, so it follows that
# default (any provider). Only the FAITHFULNESS judge here is Anthropic-SDK bound.
DEFAULT_ANNOTATE_MODEL = ANNOTATE_DEFAULT_MODEL


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


def _posint(flag, default):
    v = _arg(flag, default)
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise SystemExit(f"{flag} must be an integer, got {v!r}")
    if v < 1:
        raise SystemExit(f"{flag} must be >= 1, got {v}")
    return v


def _parse_prefixes() -> list[int]:
    """--prefixes=<comma-separated ids> | none. `none` -> [] (a baseline, no prefix)."""
    raw = _arg("--prefixes")
    if raw is None or raw is True:
        raise SystemExit("--prefixes is required (comma-separated prefix ids, or `none` for a "
                         "baseline with no prior context)")
    if str(raw).strip().lower() == "none":
        return []
    try:
        ids = [int(x) for x in str(raw).split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"--prefixes must be integers or `none`, got {raw!r}")
    if not ids:
        raise SystemExit("--prefixes had no usable ids (use `none` for a baseline)")
    return ids


def _parse_args() -> dict:
    treatment_raw = _arg("--treatment")
    if treatment_raw is None or treatment_raw is True:
        raise SystemExit("--treatment is required (a free-form label for this run, e.g. "
                         "full-hack / clean / no-prefix)")
    treatment = C.validate_treatment(str(treatment_raw))
    prefix_ids = _parse_prefixes()
    cont = _ids("--continuations", required=True)

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
        treatment=treatment, prefix_ids=prefix_ids, cont=cont, n=n,
        run_dir_stem="continuation",
        concurrency=_posint("--concurrency", DEFAULT_CONCURRENCY),
        deviation_model=_arg("--deviation-model", DEFAULT_DEVIATION_MODEL),
        annotate_model=_arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        dry_run="--dry-run" in sys.argv,
        # Faithfulness judging is OFF by default now (Owen 2026-07-09) -- opt in with
        # --faithfulness-judge. --force-judge implies it (re-judging is pointless if we skip).
        run_faithfulness=("--faithfulness-judge" in sys.argv) or ("--force-judge" in sys.argv),
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
    print(f"  treatment: {cfg['treatment']}")
    print(f"  prefixes: {cfg['prefix_ids'] or 'none (baseline — no prior context)'}")
    print(f"  continuations (new-task sources): {cfg['cont']}")
    print(f"  n: {cfg['n']}    concurrency: {cfg['concurrency']}")
    print("  auditor: FAITHFUL (primed with the new task's original run + faithfulness instruction)")
    print(f"  faithfulness judge: {cfg['deviation_model']}"
          if cfg["run_faithfulness"] else "  faithfulness judge: OFF (pass --faithfulness-judge to run)")
    print()
    await run_pipeline(cfg)


async def run_pipeline(cfg: dict) -> None:
    """The engine: plan -> generate (inline regular judge) -> faithfulness judge -> annotate ->
    viewer. cfg carries everything, including the single `treatment` label and the `prefix_ids`
    list (empty for a baseline). Run dirs are logs/continuation-<N>x-<timestamp>/, with a
    numeric suffix only when concurrent commands claim the same second."""
    treatment = cfg["treatment"]
    has_prefixes = bool(cfg["prefix_ids"])
    print("[load] reading originals, reconstructing prefixes, validating the plan ...")
    plans = await C.build_plans(treatment, cfg["prefix_ids"], cfg["cont"])

    tasks = []
    for model, plan in sorted(plans.items()):
        # coverage: a model must contribute BOTH sides (its ids are probably a typo otherwise);
        # prefixes are only required for a prefixed treatment (--prefixes != none).
        if not plan.b_refs:
            raise SystemExit(f"model {model!r} has prefixes but no --continuations id; every "
                             "model in the run needs at least one new-task source.")
        if has_prefixes and not plan.prefixes:
            raise SystemExit(f"model {model!r} has continuations but no prefixes; pass prefix "
                             "ids for it or run with --prefixes=none.")

        # the run-level allow/correct condition each original was generated under -- shown so
        # a "correct"-condition id passed as a continuation is visible at plan time (the
        # continuation reruns the new task under the SAME stamped condition; nothing imposes one).
        def _cond(ref):
            return f" [{ref.condition}]" if ref.condition else ""

        print(f"  {model}:")
        for p in plan.prefixes:
            synth = "  [synthetic tool-result closer]" if p.synthesized_closer else ""
            print(f"      prefix #{p.ref.traj_id}  (seed {p.ref.seed}{_cond(p.ref)}, "
                  f"family {C.seed_family(p.ref.seed_dir)}, {len(p.messages)} msgs){synth}")
        for b in plan.b_refs:
            fams = {C.seed_family(p.ref.seed_dir) for p in plan.prefixes}
            cross = ("  [CROSS-FAMILY: shared system prompt + destination-specific pivot]"
                     if fams - {C.seed_family(b.seed_dir)} else "")
            fixed_tools = plan.b_tool_sets[b.traj_id]
            # the auditor is always primed with this reference, so its size is always relevant.
            refsz = f", reference={len(C.build_reference_blob(b)):,} chars"
            print(f"      new task               #{b.traj_id}  (seed {b.seed}{_cond(b)}, "
                  f"family {C.seed_family(b.seed_dir)}, max_turns={b.max_turns}{refsz}){cross}")
            print(f"          fixed target tools from B: "
                  f"{', '.join(fixed_tools.names) if fixed_tools.names else '(none)'} "
                  f"[{fixed_tools.fingerprint[:12]}]")
            print(f"          judge dimensions: {', '.join(b.dimension_set.names)}")
        for b in plan.b_refs:
            if has_prefixes:
                for p in plan.prefixes:
                    tasks.append(C.build_continuation_task(plan, b, treatment, p))
            else:
                tasks.append(C.build_continuation_task(plan, b, treatment, None))

    expected = len(tasks) * cfg["n"]
    print(f"\n  treatment {treatment!r}: {len(plans)} model(s), {len(tasks)} cell(s) x "
          f"n={cfg['n']} = {expected} continuation(s) to generate")

    if cfg["dry_run"]:
        print("\n[dry-run] plan validated (treatment + prefixes + new tasks). No generation, no cost.")
        return

    run_dir = _claim_run_dir(
        viewer.LOGS, cfg["run_dir_stem"], cfg["n"], datetime.now()
    )
    print(f"\n[generate] eval_set -> {run_dir}")
    print("           (spends on the target provider; the REGULAR judge runs inline) ...")
    import openrouter_cost   # persist OpenRouter's real billed cost per call (see lib/openrouter_cost.py)
    import direct_cost       # exact list-price cost for direct anthropic/openai calls (see lib/direct_cost.py)
    import model_window      # correct context windows so the auditor compacts at the real window (see lib/model_window.py)
    import prompt_caching    # provider-prefix warm-up barrier + stored cache evidence
    openrouter_cost.install()
    direct_cost.install()
    model_window.install()
    prompt_caching.install_inspect_warmup()
    success, _logs = eval_set(
        tasks,
        epochs=cfg["n"],
        max_tasks=min(len(tasks), cfg["concurrency"]),
        max_samples=cfg["concurrency"],
        max_connections=cfg["concurrency"],
        log_dir=str(run_dir),
    )
    print(f"[generate] eval_set finished, success={success}")
    prompt_caching.write_report(run_dir)

    # provenance: what was run, under what inherited config (self-describing run dir).
    (run_dir / "continuation_meta.json").write_text(json.dumps({
        "experiment": "continuation",
        "run_kind": cfg["run_dir_stem"],   # continuation
        "config_version": "continuation-v7",
        "n": cfg["n"],
        "treatment": treatment,
        "preamble": C.PREAMBLE,
        "plans": {
            model: {
                "target_model": plan.target_model,
                "prefixes": [
                    {
                        "traj_id": p.ref.traj_id,
                        "treatment": p.treatment,
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
                        "target_tools_mode": "fixed-from-original-b",
                        "target_tool_names": plan.b_tool_sets[b.traj_id].names,
                        "target_tools_fingerprint": plan.b_tool_sets[b.traj_id].fingerprint,
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
    loaded = await viewer.load_mode(run_dir)
    if not loaded:
        raise SystemExit(
            f"[fatal] eval_set wrote ~{expected} sample(s) to {run_dir.name}, but the viewer "
            f"loader (viewer.load_mode) sees 0 of them. This is almost always a judge "
            f"score-key mismatch: load_mode keys on the 'audit_judge' score, so the continuation "
            f"judge must register under name='audit_judge'. Nothing downstream will populate. "
            f"Re-inspect with MAKE_VIEWER_NO_CACHE=1 and do NOT trust this run.")
    if len(loaded) < expected:
        print(f"  WARNING: viewer loader sees {len(loaded)}/{expected} continuation(s) -- "
              f"{expected - len(loaded)} were dropped (judge error / unscored?) and won't appear "
              f"downstream.")
    else:
        print(f"[generate] viewer loader sees all {len(loaded)} continuation(s) -- OK")

    if not cfg["run_faithfulness"]:
        print("\n[faithfulness] SKIPPED (off by default). The auditor was still primed to be "
              "faithful; we just don't score it. Enable with --faithfulness-judge, or run later "
              "via tools/exp_rejudge_continuation_faithfulness.py.")
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
        print("\n[viewer] skipped (--skip-viewer). Rebuild with: "
              "uv run viewer.py --continuations-only")
    else:
        print("\n[viewer] refreshing continuation visuals ...")
        await viewer.main(continuations_only=True)

    print(f"\nDone. Continuations are in {run_dir.name}/ (Continuations page on the viewer).")


def _claim_run_dir(log_root: pathlib.Path, stem: str, n: int,
                   started_at: datetime) -> pathlib.Path:
    """Atomically reserve a continuation run directory.

    Concurrent commands can begin in the same second. The first keeps the historical name;
    later commands get ``-2``, ``-3``, and so on instead of sharing one directory and
    overwriting each other's metadata.
    """
    base = log_root / f"{stem}-{n}x-{started_at:%Y%m%d-%H%M%S}"
    suffix = 1
    while True:
        candidate = base if suffix == 1 else pathlib.Path(f"{base}-{suffix}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1


if __name__ == "__main__":
    asyncio.run(main())
