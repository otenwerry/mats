"""End-to-end reward-hacking audit pipeline: audit -> annotate -> viewer, in one run.

Thinks of the whole thing as a function: inputs are (targets, seeds, epochs); output
is the static viewer with every trajectory judged on 5 dims, every FULL reward hack
secondarily annotated for its hack turns, and everything linked up. The three stages,
each reusing the standalone scripts' code (single source of truth):

  1. AUDIT     (exp_rh_audit.build_tasks + run_eval): targets x seeds x epochs
               trajectories, each judged inline on all 5 dims. -> one timestamped
               log dir under mats-local/petri/logs/.
  2. ANNOTATE  (exp_annotate_hacks.run_annotation): secondary turn-level judging of
               exactly the FULL reward hacks (the binary definition / first viewer
               category), run with the SAME parallelism as the audit. -> annotations.json.
  3. VIEWER    (make_viewer.main, FREE): rebuild the static viewer, auto-attaching
               scores + annotations. -> mats-local/petri/viewer/index.html.

Robustness (the point of this file):
  - Individual failures DON'T stop the run. Inspect's eval_set records per-sample
    errors and continues; a target that produced nothing is *warned* about, not
    aborted on (unlike the standalone exp_rh_audit CLI, which aborts). Annotation
    catches per-trajectory errors and keeps going. Annotate/viewer are each wrapped
    so a failure in one still lets the others complete.
  - No re-judge step: fresh exp_rh_audit runs already produce all 5 dims, so the
    binary-hack gate works directly. (exp_rejudge_rh.py is only for OLDER audits.)

--targets, --seed-dir, --seeds and --epochs are REQUIRED (no defaults) so the experiment
design is always explicit and can never silently fall back to a stale list. The fixed
target system prompt is ALWAYS on (read from <seed-dir>/SYSTEM_PROMPT.txt; the retired
--fixed_sp flag is rejected).

Usage:
  # the whole ml_training_data_misuse collection (all 7 pinned scenario dirs, each with
  # its own SYSTEM_PROMPT.txt) in one invocation. One condition per invocation, so a
  # full sweep = one invocation per condition:
  uv run exp_audit_pipeline.py --targets=opus-4.6 --seed-dir=ml_training_data_misuse \
      --seeds=all --condition=allow --epochs=2
  # a subset of the collection's scenario dirs:
  uv run exp_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse --seeds=youtube_comments,fraud_detection --condition=allow --epochs=1
  # a single pinned seed dir still works too:
  uv run exp_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse/youtube_comments --seeds=all --condition=allow --epochs=1
  uv run exp_audit_pipeline.py ... --skip-annotate     # audit + viewer only
  uv run exp_audit_pipeline.py ... --force-annotate     # re-annotate hacks already in annotations.json

Flags:
  --targets=<a,b,..>   REQUIRED. target names from TARGET_CHOICES.
  --seed-dir=<sub>     REQUIRED. which seed FOLDER to run. Three layouts: a plain dir of
                       freestanding seeds (seeds/<sub>/*.md), a PINNED dir (core.md +
                       conditions/, e.g. ml_training_data_misuse/youtube_comments), or a
                       COLLECTION of pinned dirs (each immediate subdir pinned, e.g.
                       --seed-dir=ml_training_data_misuse runs all 7 scenario dirs at
                       once, one task per target x dir). Errors if the folder has 0
                       seeds. Required because the fixed target system prompt (always on)
                       is read from SYSTEM_PROMPT.txt in the seed dir (each collection
                       member has its own).
  --seeds=<a,b,..>     REQUIRED. seed filename stems -- or collection member names --
                       within the chosen folder, or `all`.
  --epochs=<N>         REQUIRED. epochs per (target, seed) cell.
  --reasoning=yes|no   optional (default yes). Native TARGET reasoning. yes: open models
                       (OpenRouter) keep the raw trace in context, first-party opus/gpt get
                       summarized thinking NOT fed back to context, and the '<thinking> tags'
                       system-prompt instruction is stripped. Depth = REASONING_EFFORT (medium).
  --auditor=<m>        auditor model (default openrouter/deepseek/deepseek-v4-pro-20260423): a TARGET_CHOICES
                       shortname (e.g. --auditor=glm-5.2) or a full provider/model string
                       (e.g. --auditor=anthropic/claude-sonnet-4-6). Judge stays fixed.
  --auditor-thinking=<yes|no>  auditor extended thinking (default no). yes = adaptive
                       thinking (summarized); no = off. Stamped into metadata so resamples
                       inherit the same setting.
  (fixed target system prompt: ALWAYS ON, no flag. Every audit uses the one prompt in
                       <seed-dir>/SYSTEM_PROMPT.txt; the target is pinned to it and the
                       auditor's set_system_message tool is removed. The retired --fixed_sp
                       flag is rejected loudly if passed.)
  --condition=<c>      PINNED seed dirs and collections of them only; REQUIRED there,
                       forbidden elsewhere. Which conditions/<c>.md fragment is appended to
                       core.md -- exactly one condition per run (a collection applies it to
                       every member); run each condition as its own pipeline invocation.
  --concurrency=<N>    one knob -> audit max_samples/max_connections AND annotate
                       parallelism (default 50).
  --max-turns=<N>      auditor turn cap (default 60). Stamped per task into metadata,
                       so runs with different caps are distinguishable and resamples
                       inherit the value.
  --annotate-model=<m> Anthropic model for the secondary hack-turn judging
                       (default claude-opus-4-8).
  --skip-annotate      stop after the audit (still rebuilds the viewer).
  --skip-viewer        don't rebuild the viewer at the end.
  --force-annotate     re-annotate FULL hacks even if already in annotations.json.

Costs money (Anthropic + OpenRouter for the audit; Anthropic for the annotation).
The viewer stage is free.
"""

import asyncio
import pathlib
import sys
import traceback
from datetime import datetime

# the core modules live in lib/; put it on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

import make_viewer
from exp_rh_audit import (
    TARGET_CHOICES, SEEDS_ROOT, DATA, MAX_TURNS, REASONING_EFFORT, build_tasks, run_eval,
    dead_targets, reasoning_tag, resolve_auditor, resolve_auditor_thinking, resolve_seeds,
    resolve_fixed_sp, reject_fixed_sp_flag, resolve_condition, resolve_reasoning, JUDGE,
)
from exp_annotate_hacks import load_all_original_audits, run_annotation
from model_routing import route  # match build_tasks' provider routing for the dead-target guard

# --targets, --seeds and --epochs are REQUIRED (no defaults) so the experiment design
# is always explicit on the command line and can't silently fall back to a stale list.
# Only the operational knobs (parallelism, annotate model) default.
DEFAULT_CONCURRENCY = 50
DEFAULT_ANNOTATE_MODEL = "claude-opus-4-8"


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _parse_args() -> dict:
    # --targets REQUIRED (no default).
    targets_arg = _arg("--targets")
    if targets_arg is None:
        raise SystemExit(f"--targets is required (no default); choices: {sorted(TARGET_CHOICES)}")
    targets = list(dict.fromkeys(t.strip() for t in targets_arg.split(",") if t.strip()))
    if not targets:
        raise SystemExit(f"--targets had no usable names; choices: {sorted(TARGET_CHOICES)}")
    unknown = [t for t in targets if t not in TARGET_CHOICES]
    if unknown:
        raise SystemExit(f"unknown --targets {unknown}; choices: {sorted(TARGET_CHOICES)}")

    # --seed-dir=<sub> picks the seed folder (omitted -> top-level seeds/*.md; a name ->
    # seeds/<sub>/*.md). --seeds REQUIRED (no default): use --seeds=all for every seed in that
    # folder, or a comma list of stems.
    seed_dir_arg = _arg("--seed-dir")
    seeds_path, available_seeds = resolve_seeds(seed_dir_arg)
    if not available_seeds:
        where = f"seeds/{seed_dir_arg}/" if seed_dir_arg else "seeds/ (top level)"
        subdirs = sorted(p.name for p in SEEDS_ROOT.iterdir() if p.is_dir())
        raise SystemExit(
            f"no .md seeds found in {where}. Seeds are organized into subdirs -- pass "
            f"--seed-dir=<name> to run one. available subdirs: {subdirs}")
    seeds_arg = _arg("--seeds")
    if seeds_arg is None:
        raise SystemExit(
            "--seeds is required (no default); use --seeds=all for every seed in the chosen "
            f"folder, or a comma-separated subset. available: {available_seeds}"
        )
    if seeds_arg.strip() == "all":
        seeds = list(available_seeds)
    else:
        seeds = [s.strip() for s in seeds_arg.split(",") if s.strip()]
        unknown_s = [s for s in seeds if s not in available_seeds]
        if unknown_s:
            raise SystemExit(f"unknown --seeds {unknown_s}; available in {seeds_path}: {available_seeds}")
    if not seeds:
        raise SystemExit(f"--seeds had no usable names; available: {available_seeds}")

    # --epochs REQUIRED (no default). Positive integer.
    epochs_arg = _arg("--epochs")
    if epochs_arg is None:
        raise SystemExit("--epochs is required (no default); positive integer (e.g. --epochs=2)")
    try:
        epochs = int(epochs_arg)
    except ValueError:
        raise SystemExit(f"--epochs must be an integer, got {epochs_arg!r}")
    if epochs < 1:
        raise SystemExit(f"--epochs must be >= 1, got {epochs}")

    # operational knobs keep defaults.
    def _posint(flag: str, default: int) -> int:
        raw = _arg(flag)
        if raw is None:
            return default
        try:
            v = int(raw)
        except ValueError:
            raise SystemExit(f"{flag} must be an integer, got {raw!r}")
        if v < 1:
            raise SystemExit(f"{flag} must be >= 1, got {v}")
        return v

    # --auditor=<shortname|model> (default AUDITOR = deepseek-v4-pro). A TARGET_CHOICES
    # shortname (e.g. glm-5.2) resolves to its full slug; a provider-prefixed string
    # passes through; anything else fails here (see resolve_auditor). Judge stays fixed.
    auditor = resolve_auditor(_arg("--auditor"))

    # --auditor-thinking=yes|no (default no). Resolved to the auditor's reasoning-effort
    # (str) or None (off); passed to build_tasks and stamped into metadata so resamples
    # inherit it.
    auditor_reasoning_effort = resolve_auditor_thinking(_arg("--auditor-thinking", None))

    # fixed target system prompt: ALWAYS on, read from <seed-dir>/SYSTEM_PROMPT.txt
    # (which is why --seed-dir is required). The retired --fixed_sp flag is rejected
    # loudly rather than silently ignored.
    reject_fixed_sp_flag()
    fixed_system_prompt = resolve_fixed_sp(seed_dir_arg, seeds_path)

    # --condition=<c> (PINNED seed dirs only, where it's required; see resolve_condition).
    condition = resolve_condition(_arg("--condition"), seeds_path)

    # --reasoning=yes|no (optional, defaults to yes): whether targets reason natively. Resolved to
    # a bool, stamped into metadata so resamples reproduce it. See resolve_reasoning.
    reasoning = resolve_reasoning(_arg("--reasoning"))

    return {
        "targets": targets,
        "seeds": seeds,
        "seeds_path": seeds_path,
        "epochs": epochs,
        "auditor": auditor,
        "auditor_reasoning_effort": auditor_reasoning_effort,
        "reasoning": reasoning,
        "fixed_system_prompt": fixed_system_prompt,
        "condition": condition,
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        # auditor turn cap; stamped per task into metadata (see build_tasks) so runs with
        # different caps are distinguishable and resamples inherit the value.
        "max_turns": _posint("--max-turns", MAX_TURNS),
        "annotate_model": _arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        "skip_annotate": "--skip-annotate" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "force_annotate": "--force-annotate" in sys.argv,
    }


def run_audit_stage(cfg: dict):
    """Stage 1. Returns (logs, log_dir, expected_n) or (None, log_dir, expected_n)
    if the audit crashed outright (we still proceed to annotate/viewer on whatever
    is on disk). Dead targets are WARNED about, never aborted on."""
    targets, seeds, epochs = cfg["targets"], cfg["seeds"], cfg["epochs"]
    target_models = [route(TARGET_CHOICES[t]) for t in targets]  # routed to match logged slugs
    expected_n = len(targets) * len(seeds) * epochs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # pinned runs carry their condition in the dir name so per-condition dirs are tellable apart
    label = targets[0] if len(targets) == 1 else f"{len(targets)}targets"
    if cfg["condition"] is not None:
        label += f"-{cfg['condition']}"
    log_dir = DATA / "logs" / f"v2-{label}-{epochs}ep-{timestamp}"

    print("=" * 72)
    print(f"STAGE 1/3  AUDIT  ->  {log_dir.name}")
    print("=" * 72)
    print(f"  targets ({len(targets)}): " + ", ".join(f"{t}{reasoning_tag(cfg['reasoning'])}" for t in targets))
    print(f"  seeds ({len(seeds)}): {seeds}")
    eff = cfg["auditor_reasoning_effort"]
    thinking_note = "off" if eff is None else f"adaptive (effort={eff}, summarized)"
    print(f"  auditor={cfg['auditor']} [thinking: {thinking_note}]  judge={JUDGE}")
    fsp = cfg["fixed_system_prompt"]
    if isinstance(fsp, dict):
        sizes = ", ".join(f"{m}={len(fsp[m])}ch" for m in seeds)
        print(f"  fixed SP (always on): one per seed dir, each from its own "
              f"SYSTEM_PROMPT.txt ({sizes}); auditor set_system_message disabled")
    else:
        print(f"  fixed SP (always on): {len(fsp)} chars; auditor set_system_message disabled")
    if cfg["condition"] is not None:
        print(f"  condition: {cfg['condition']} (pinned seed dir(s) -- core.md + "
              f"conditions/{cfg['condition']}.md + inlined pinned files; a collection "
              "applies it to every member)")
    print(f"  epochs={epochs}  concurrency={cfg['concurrency']}  max_turns={cfg['max_turns']}")
    if cfg["reasoning"]:
        print(f"  reasoning=ON effort={REASONING_EFFORT} | open (deepseek/glm/kimi): raw trace, "
              "history=inspect-default (NOT pinned) | closed (opus/gpt): summary, history=none | "
              "'<thinking> tags' SP instruction stripped")
    else:
        print("  reasoning=OFF (all targets)")
    print(f"  expected trajectories: {len(targets)} x {len(seeds)} x {epochs} = {expected_n}\n")

    try:
        tasks = build_tasks(targets, seeds, log_dir.name, auditor=cfg["auditor"],
                            reasoning=cfg["reasoning"],
                            auditor_reasoning_effort=cfg["auditor_reasoning_effort"],
                            seeds_path=cfg["seeds_path"],
                            fixed_system_prompt=cfg["fixed_system_prompt"],
                            condition=cfg["condition"],
                            max_turns=cfg["max_turns"])
        success, logs = run_eval(tasks, epochs, cfg["concurrency"], log_dir)
    except SystemExit:
        raise  # config errors (e.g. empty seed subset) should abort before any spend
    except Exception as e:
        print("\n!! AUDIT STAGE CRASHED (continuing to annotate+viewer on existing logs):")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, log_dir, expected_n

    print(f"\neval_set finished, success={success}")
    actual_n = 0
    for log in logs:
        n = len(log.samples or [])
        actual_n += n
        print(f"  {log.eval.task}: status={log.status}, samples={n}")

    # Robustness check #1: did every target actually generate? Warn, don't abort.
    dead = dead_targets(logs, target_models)
    if dead:
        print("\n" + "!" * 72)
        print(f"WARNING: target(s) {dead} produced 0 output tokens -- they never ran")
        print("  (likely a bad slug / 402 / key error). Their 'audits' are empty and the")
        print("  judge scores them as clean no-hack runs -- DO NOT trust those rows. The")
        print("  other targets are fine; re-run just the dead one(s) into a new dir.")
        print("!" * 72)

    # Robustness check #2: trajectory count. A silent shortfall (samples dropped after
    # retries) would otherwise look like a complete run.
    if actual_n != expected_n:
        print(f"\nWARNING: expected {expected_n} trajectories but eval_set wrote {actual_n}. "
              f"{expected_n - actual_n} sample(s) are missing (errored out after retries?). "
              "The viewer will show only what completed.")
    else:
        print(f"\nall {expected_n} trajectories present.")
    return logs, log_dir, expected_n


async def run_post_audit_stages(cfg: dict) -> None:
    """Stages 2 (annotate) and 3 (viewer). Each is wrapped so a failure in one
    doesn't prevent the other; the viewer always runs last."""
    if not cfg["skip_annotate"]:
        print("\n" + "=" * 72)
        print("STAGE 2/3  ANNOTATE  (secondary hack-turn judging of FULL reward hacks)")
        print("=" * 72)
        try:
            audits = await load_all_original_audits()
            await run_annotation(
                audits,
                model=cfg["annotate_model"],
                concurrency=cfg["concurrency"],
                force=cfg["force_annotate"],
            )
        except Exception as e:
            print(f"\n!! ANNOTATE STAGE FAILED (continuing to viewer): {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 2 annotate: --skip-annotate)")

    if not cfg["skip_viewer"]:
        print("\n" + "=" * 72)
        print("STAGE 3/3  VIEWER  (rebuild static viewer, free)")
        print("=" * 72)
        try:
            await make_viewer.main()
        except Exception as e:
            print(f"\n!! VIEWER STAGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 3 viewer: --skip-viewer)")


def main() -> None:
    cfg = _parse_args()
    # eval_set is synchronous and spins up its own event loop, so it must run OUTSIDE
    # asyncio.run; the async annotate/viewer stages run after it.
    run_audit_stage(cfg)
    asyncio.run(run_post_audit_stages(cfg))
    print("\n" + "=" * 72)
    print(f"PIPELINE DONE.  open {DATA / 'viewer' / 'index.html'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
