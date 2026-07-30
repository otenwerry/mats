"""End-to-end reward-hacking audit pipeline: audit -> annotate -> viewer, in one run.

Thinks of the whole thing as a function: inputs are (targets, seeds, epochs); output
is the static viewer with every trajectory judged on 5 dims, every FULL reward hack
secondarily annotated for its hack turns, and everything linked up. The three stages,
each reusing the standalone scripts' code (single source of truth):

  1. AUDIT     (exp_rh_audit.build_tasks + run_eval): targets x seeds x epochs
               trajectories, each judged inline on its global + seed-scoped dimensions.
               -> one timestamped
               log dir under mats-local/petri/logs/.
  2. ANNOTATE  (exp_annotate_hacks.run_annotation): secondary turn-level judging of
               exactly the FULL reward hacks (the binary definition / first viewer
               category), run with the SAME parallelism as the audit. -> annotations.json.
  3. VIEWER    (viewer.main, FREE): rebuild the static viewer, auto-attaching
               scores + annotations. -> mats-local/petri/viewer/index.html.

Robustness (the point of this file):
  - Unknown or malformed command-line flags fail before any API calls. A typo must never
    silently fall back to an operational default such as the 60-turn cap.
  - Individual failures DON'T stop sibling samples. Inspect's eval_set records
    per-sample errors and continues. The pipeline still builds the viewer so failed
    samples remain inspectable, then exits non-zero if any expected sample, target
    response, or primary judge result is unusable.
  - No re-judge step: fresh exp_rh_audit runs already produce their routed dimensions, so the
    binary-hack gate works directly. (exp_rejudge_rh.py is only for OLDER audits.)

--targets, --seed-dir, --seeds, --epochs and --max-turns are REQUIRED (no defaults) so
the experiment design is always explicit and can never silently fall back to a stale
list or turn cap. The fixed target system prompt is ALWAYS on (read from
seeds/SYSTEM_PROMPT.txt; the retired --fixed_sp flag is rejected).

Usage:
  # the whole ml_training_data_misuse collection (all pinned scenario dirs sharing the
  # global SYSTEM_PROMPT.txt) in one invocation. One condition per invocation, so a
  # full sweep = one invocation per condition:
  uv run exp_audit_pipeline.py --targets=opus-4.6 --seed-dir=ml_training_data_misuse \
      --seeds=all --condition=allow --epochs=2 --max-turns=200
  # a subset of the collection's scenario dirs:
  uv run exp_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse --seeds=youtube_comments,fraud_detection --condition=allow --epochs=1 --max-turns=200
  # a single pinned seed dir still works too:
  uv run exp_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse/youtube_comments --seeds=all --condition=allow --epochs=1 --max-turns=200
  uv run exp_audit_pipeline.py ... --skip-annotate     # audit + viewer only
  uv run exp_audit_pipeline.py ... --force-annotate     # re-annotate hacks already in annotations.json

Flags:
  --targets=<a,b,..>   REQUIRED. target names from TARGET_CHOICES.
  --seed-dir=<sub>     REQUIRED. which seed FOLDER to run. Three layouts: a plain dir of
                       freestanding seeds (seeds/<sub>/*.md), a PINNED dir (core.md +
                       conditions/, e.g. ml_training_data_misuse/youtube_comments), or a
                       COLLECTION of pinned dirs (each immediate subdir pinned, e.g.
                       --seed-dir=ml_training_data_misuse runs every active pinned member
                       at once, one task per target x dir). Errors if the folder has 0
                       seeds. Every seed uses the global fixed target system prompt from
                       seeds/SYSTEM_PROMPT.txt.
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
                       seeds/SYSTEM_PROMPT.txt; the target is pinned to it and the
                       auditor's set_system_message tool is removed. The retired --fixed_sp
                       flag is rejected loudly if passed.)
  --condition=<c>      PINNED seed dirs and collections of them only; optional, defaults
                       to allow, and forbidden elsewhere. Which conditions/<c>.md fragment
                       is appended to core.md -- exactly one condition per run (a collection
                       applies it to every member); run each condition as its own invocation.
  --concurrency=<N>    one knob -> audit max_samples/max_connections AND annotate
                       parallelism (default 50).
  --max-turns=<N>      REQUIRED. Auditor turn cap. Stamped per task into metadata, so
                       runs with different caps are distinguishable and resamples inherit
                       the value.
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

import viewer
from exp_rh_audit import (
    TARGET_CHOICES, SEEDS_ROOT, DATA, REASONING_EFFORT, build_tasks, run_eval,
    dead_targets, reasoning_tag, resolve_auditor, resolve_auditor_thinking, resolve_seeds,
    resolve_fixed_sp, reject_fixed_sp_flag, resolve_condition, resolve_reasoning, JUDGE,
    print_dimension_plan,
)
from exp_annotate_hacks import load_all_original_audits, run_annotation
from model_routing import route  # match build_tasks' provider routing for the dead-target guard
from viewer_load import (
    blocking_target_provider_events,
    context_calls_for_role,
    judge_score_status,
    target_output_tokens,
)

# --targets, --seeds, --epochs and --max-turns are REQUIRED (no defaults) so the experiment
# design is always explicit on the command line and can't silently fall back to a stale list
# or turn cap.
# Only the operational knobs (parallelism, annotate model) default.
DEFAULT_CONCURRENCY = 50
DEFAULT_ANNOTATE_MODEL = "claude-opus-4-8"

_VALUE_FLAGS = {
    "--targets", "--seed-dir", "--seeds", "--epochs", "--reasoning", "--auditor",
    "--auditor-thinking", "--condition", "--concurrency", "--max-turns",
    "--annotate-model",
}
_SWITCH_FLAGS = {"--skip-annotate", "--skip-viewer", "--force-annotate"}


def _validate_cli_args() -> None:
    """Reject typos and malformed flags before config resolution or paid work begins."""
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for arg in sys.argv[1:]:
        flag, separator, _ = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(f"{flag} requires a value in the form {flag}=<value>")
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {arg!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _parse_args() -> dict:
    # Keep the retired flag's specific migration error, then reject every other unknown or
    # malformed argument. This must happen before resolving seeds or constructing tasks.
    reject_fixed_sp_flag()
    _validate_cli_args()

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

    # fixed target system prompt: ALWAYS on, read from seeds/SYSTEM_PROMPT.txt.
    fixed_system_prompt = resolve_fixed_sp(seed_dir_arg, seeds_path)

    # --condition=<c> (PINNED seed dirs only; omitted -> allow; see resolve_condition).
    condition = resolve_condition(_arg("--condition"), seeds_path)

    # --reasoning=yes|no (optional, defaults to yes): whether targets reason natively. Resolved to
    # a bool, stamped into metadata so resamples reproduce it. See resolve_reasoning.
    reasoning = resolve_reasoning(_arg("--reasoning"))

    max_turns_arg = _arg("--max-turns")
    if max_turns_arg is None:
        raise SystemExit(
            "--max-turns is required (no default); pass the explicit auditor turn cap "
            "for this run (for example, --max-turns=200)"
        )

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
        # Required auditor turn cap; stamped per task into metadata (see build_tasks) so
        # different runs are distinguishable and resamples inherit the value.
        "max_turns": _posint("--max-turns", 1),
        "annotate_model": _arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        "skip_annotate": "--skip-annotate" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "force_annotate": "--force-annotate" in sys.argv,
    }


def run_audit_stage(cfg: dict):
    """Stage 1.

    Returns ``(logs, log_dir, expected_n, integrity_ok)``. An outright crash still
    proceeds to annotate/viewer on whatever reached disk, but the final process exits
    non-zero after those recovery stages.
    """
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
    print_dimension_plan(seeds, cfg["seeds_path"])
    fsp = cfg["fixed_system_prompt"]
    resolved_sp = next(iter(fsp.values())) if isinstance(fsp, dict) else fsp
    print(f"  fixed SP (always on): seeds/SYSTEM_PROMPT.txt "
          f"({len(resolved_sp)} chars, shared by every seed); "
          "auditor set_system_message disabled")
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
        return None, log_dir, expected_n, False

    print(f"\neval_set finished, success={success}")
    actual_n = 0
    for log in logs:
        n = len(log.samples or [])
        actual_n += n
        print(f"  {log.eval.task}: status={log.status}, samples={n}")

    # Robustness check #1: did every target actually generate? Warn, don't abort.
    dead = unrecovered_dead_targets(logs, target_models)
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

    integrity_failures = audit_integrity_failures(logs)
    if integrity_failures:
        print("\n" + "!" * 72)
        print(
            f"WARNING: {len(integrity_failures)} trajectory(ies) have data-integrity "
            "failures. They remain visible in the viewer but are excluded from analysis:"
        )
        for failure in integrity_failures:
            print(
                f"  {failure['task']}/{failure['sample']} epoch {failure['epoch']}: "
                f"{', '.join(failure['issues'])}"
            )
        print("!" * 72)

    integrity_ok = (
        bool(success)
        and actual_n == expected_n
        and not dead
        and not integrity_failures
    )
    return logs, log_dir, expected_n, integrity_ok


def audit_integrity_failures(logs: list) -> list[dict]:
    """Per-sample primary-judge and target-provider failures in fresh eval logs.

    This is the run-time counterpart of ``viewer_load.finalize_audit_integrity``.
    It deliberately runs before annotation so the terminal result cannot call a
    sample set complete merely because every ``EvalSample`` object exists.
    """
    failures: list[dict] = []
    for log in logs or []:
        roles = log.eval.model_roles or {}
        target_model = (
            getattr(roles.get("target"), "model", None)
            or str(roles.get("target", "?"))
        )
        declared = list((log.eval.metadata or {}).get("judge_dimensions") or [])
        for sample in log.samples or []:
            score = (sample.scores or {}).get("audit_judge")
            status, _issue = judge_score_status(score)
            values = (
                score.value
                if score is not None and isinstance(getattr(score, "value", None), dict)
                else {}
            )
            missing_dimensions = [
                dimension
                for dimension in declared
                if not isinstance(values.get(dimension), (int, float))
            ]
            target_usage = context_calls_for_role(sample, "target", target_model)
            issues: list[str] = []
            if status != "usable":
                issues.append(f"judge_score_{status}")
            if missing_dimensions:
                issues.append(
                    "judge_dimensions_missing:" + ",".join(missing_dimensions)
                )
            if (
                target_output_tokens(sample, target_model) == 0
                and not target_usage.get("visible_responses")
            ):
                issues.append("target_no_output")
            issues.extend(
                f"target_provider_{event['kind']}:attempt{event.get('attempt', '?')}"
                for event in blocking_target_provider_events(target_usage)
            )
            if issues:
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": list(dict.fromkeys(issues)),
                })
    return failures


def unrecovered_dead_targets(logs: list, target_models: list[str]) -> list[str]:
    """Token-based dead-target guard, repaired by direct visible-response evidence.

    Inspect's aggregate usage is normally the cheapest reliable guard against a bad
    slug/key/402. Some provider responses omit usage metadata, though; a target that
    visibly answered must not be declared dead solely because that counter stayed zero.
    """
    dead = set(dead_targets(logs, target_models))
    if not dead:
        return []
    for log in logs or []:
        roles = log.eval.model_roles or {}
        target_model = (
            getattr(roles.get("target"), "model", None)
            or str(roles.get("target", "?"))
        )
        target_name = target_model.split("/")[-1]
        if target_name not in dead:
            continue
        if any(
            context_calls_for_role(sample, "target", target_model).get("visible_responses")
            for sample in (log.samples or [])
        ):
            dead.remove(target_name)
    return sorted(dead)


async def run_post_audit_stages(cfg: dict) -> bool:
    """Stages 2 (annotate) and 3 (viewer). Each is wrapped so a failure in one
    doesn't prevent the other; the viewer always runs last."""
    ok = True
    if not cfg["skip_annotate"]:
        print("\n" + "=" * 72)
        print("STAGE 2/3  ANNOTATE  (secondary hack-turn judging of FULL reward hacks)")
        print("=" * 72)
        try:
            audits = await load_all_original_audits()
            annotation_stats = await run_annotation(
                audits,
                model=cfg["annotate_model"],
                concurrency=cfg["concurrency"],
                force=cfg["force_annotate"],
            )
            if annotation_stats.get("failed"):
                ok = False
        except Exception as e:
            ok = False
            print(f"\n!! ANNOTATE STAGE FAILED (continuing to viewer): {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 2 annotate: --skip-annotate)")

    if not cfg["skip_viewer"]:
        print("\n" + "=" * 72)
        print("STAGE 3/3  VIEWER  (rebuild static viewer, free)")
        print("=" * 72)
        try:
            await viewer.main()
        except Exception as e:
            ok = False
            print(f"\n!! VIEWER STAGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 3 viewer: --skip-viewer)")
    return ok


def main() -> None:
    cfg = _parse_args()
    # eval_set is synchronous and spins up its own event loop, so it must run OUTSIDE
    # asyncio.run; the async annotate/viewer stages run after it.
    _logs, _log_dir, _expected_n, audit_ok = run_audit_stage(cfg)
    post_ok = asyncio.run(run_post_audit_stages(cfg))
    print("\n" + "=" * 72)
    if audit_ok and post_ok:
        print(f"PIPELINE DONE.  open {DATA / 'viewer' / 'index.html'}")
    else:
        print(
            "PIPELINE FINISHED WITH INTEGRITY FAILURES. "
            f"Inspect {DATA / 'viewer' / 'index.html'} and the warnings above."
        )
    print("=" * 72)
    if not (audit_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
