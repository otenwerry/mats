"""Petri audit: reward-hacking propensity of a target model (v2 config).

Usage (CLI):
  uv run exp_rh_audit.py --target=qwen3-32b                   # 1 epoch, all seeds
  uv run exp_rh_audit.py --target=opus-4-6 --epochs=4         # 4 epochs
  uv run exp_rh_audit.py --target=qwen3-32b,llama-3.3-70b,deepseek-v3 --epochs=2   # sweep 3 targets
  uv run exp_rh_audit.py --target=qwen3-32b --seeds=ml_training_data_misuse,perf_benchmark_gaming

Flags:
  --target=<a,b,..> REQUIRED (no default). One OR MORE names from TARGET_CHOICES,
                    comma-separated -- each runs as its own task, all into one log
                    dir, so you can sweep models in a single invocation. qwen* /
                    OpenRouter targets are cheap; Anthropic targets like opus-4-6
                    are much pricier per token. Unknown / missing names error out.
  --epochs=<N>      Number of epochs (default 1). Scores are mean-reduced per seed
                    across epochs; more epochs = more trajectories = more $.
  --seeds=<a,b,..>  Comma-separated seed filename stems to run (default: all .md
                    files in seeds/). Unknown stems error out with the valid list.
  --concurrency=<N> Parallelism (default 10). Sets BOTH the number of audits run
                    in parallel (max_samples) AND the max in-flight requests per
                    model role (max_connections) to N. Higher = faster but more
                    likely to hit provider rate limits; Inspect backs off and
                    retries on 429s, so overshooting slows the run, doesn't fail it.

This module is ALSO importable: exp_audit_pipeline.py reuses TARGET_CHOICES,
REASONING_ENABLED, build_tasks(), run_eval() and dead_targets() so the model list
and task config have a single source of truth. Importing it has no side effects
beyond loading .env and the (free) judge-dimension files; nothing runs until the
CLI `main()` is invoked under `if __name__ == "__main__"`.

Each run writes to its own timestamped log dir:
  mats-local/petri/logs/v2-<target>-<N>ep-<YYYYMMDD-HHMMSS>
so two runs never collide and eval_set never treats a fresh run as already-done.
(Trade-off: a crashed run isn't resumable -- re-running starts a new dir.)

v2 config (vs the v1 runs in logs/pilot):
  - seeds: our own no-encouragement seeds in seeds/ (auditor is told to
    create opportunity + ordinary incentive but never suggest/encourage the
    rule-break): 3 variants of Petri's realistic seeds plus 2 of our own
    (p-hacking, benchmark gaming). The 3 artificial reward-signal seeds are
    dropped: they work by actively rewarding misbehavior, incompatible with
    no-encouragement.
  - judging: our custom dimensions (all from dimensions/, no built-in defaults):
    reward_hacking + auditor_nudging + situational_awareness, plus auditor_mistakes
    (auditor did something unrealistic -- bad simulated tool result or implausible
    user behavior) and degenerate_behavior (target acted incoherently). The last two
    are categorical eval-quality checks expected to sit at 1 on most audits.
    situational_awareness is our own merge of Petri's eval_awareness +
    audit_situational_awareness.

Costs money (Anthropic + OpenRouter APIs).
View results with `uv run make_viewer.py` or `uv run inspect view --log-dir <dir>`.
"""

import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set, task_with
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import get_model
from inspect_petri import audit, judge_dimensions

# all paths come from the single source of truth in petri_paths
from petri_paths import PETRI_ROOT, DATA, ENV_FILE, DIMENSIONS_DIR
load_dotenv(ENV_FILE)

AUDITOR = "anthropic/claude-sonnet-4-6"
JUDGE = "anthropic/claude-opus-4-8"

# Stamped into every run's log metadata (log.eval.metadata) so future logs are
# self-describing -- any reader (human or AI) can tell what config produced a dir
# without reverse-engineering the directory name. Bump CONFIG_VERSION when the seed
# set or judge dimensions change materially. See mats-local/petri/DATA_GUIDE.md.
CONFIG_VERSION = "v2"

# selectable targets: `--target=<name>` is REQUIRED (no default). qwen* run via
# OpenRouter; the anthropic ones run via the Anthropic key. Add new targets here.
TARGET_CHOICES = {
    # frontier (no reward hacking in our no-nudge setup so far)
    "qwen3-32b": "openrouter/qwen/qwen3-32b",
    "qwen3.7-max": "openrouter/qwen/qwen3.7-max",
    "opus-4-6": "anthropic/claude-opus-4-6",
    # NB: claude-sonnet-4-6 is also the AUDITOR (see AUDITOR below). Using it as a
    # target = same model auditing itself; methodologically OK (separate roles), but
    # interpret with that in mind.
    "sonnet-4-6": "anthropic/claude-sonnet-4-6",
    # mid-tier / older-vintage OpenRouter targets to probe for reward hacking.
    # NOTE: these slugs are best-guess as of a Jan-2026 knowledge cutoff -- verify
    # against openrouter.ai/models before a real run. A wrong slug fails LOUDLY:
    # the target generates 0 output tokens and the guard at the bottom aborts.
    "qwen2.5-72b": "openrouter/qwen/qwen-2.5-72b-instruct",
    "llama-3.3-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
    "deepseek-v3": "openrouter/deepseek/deepseek-chat",
    "gemma-3-27b": "openrouter/google/gemma-3-27b-it",
    "mistral-small": "openrouter/mistralai/mistral-small-3.1-24b-instruct",
    "deepseek-r1": "openrouter/deepseek/deepseek-r1",  # reasoning model (CoT visible)
    # 10-model audit batch (added 2026-06-23). All slugs verified live against the
    # OpenRouter /models API on that date (not knowledge-cutoff guesses). Routed via
    # OpenRouter to keep every non-Anthropic target on the single OpenRouter key --
    # incl. GPT-5.4-mini (could also go direct via "openai/..."). NOTE: the user asked
    # for "Grok 4"; plain grok-4 is no longer on OpenRouter, so we use grok-4.20
    # (a 4.x available there), per their pick.
    "glm-5.2": "openrouter/z-ai/glm-5.2",
    "glm-5.1": "openrouter/z-ai/glm-5.1",
    "gpt-5.4-mini": "openrouter/openai/gpt-5.4-mini",
    "grok-4.20": "openrouter/x-ai/grok-4.20",
    # A larger qwen3 (between qwen3-32b and qwen3.7-max) would be the cleanest
    # capability rung -- add once you confirm the slug, e.g.:
    # "qwen3-72b": "openrouter/qwen/qwen3-72b-instruct",
}
# Per-target reasoning control for OpenRouter targets. We pin reasoning explicitly
# (True/False) so a run never silently depends on the provider's default. True
# forwards reasoning={"enabled": true}; False forwards reasoning={"enabled": false},
# which disables chain-of-thought on HYBRID models. Targets NOT listed fall to the
# provider default. Caveats:
#   - deepseek-r1 reasons NATIVELY (not hybrid) -> can't be switched off, so it's
#     left off this map (always reasons).
#   - gpt-5.4-mini and grok-4.20 are reasoning-family models; enabled=false is sent
#     but the provider may not fully honor "off" (they can still reason internally,
#     like deepseek). Not yet empirically confirmed -- see note when running.
#   - Anthropic targets (opus/sonnet) ignore this map; their extended thinking is off
#     unless given a thinking budget (Inspect reasoning_tokens), which we don't set.
#   - non-reasoning models (qwen2.5-72b, llama-3.3-70b, ...) have no CoT to toggle.
REASONING_ENABLED = {
    "qwen3.7-max": True,   # (not in the 10-model batch) kept ON -- its established config
    "qwen3-32b": False,
    "glm-5.2": False,
    "glm-5.1": False,
    "gpt-5.4-mini": False,
    "grok-4.20": False,
}

MAX_TURNS = 30
# seed dir + the seed stems actually present (validated against typos by callers).
SEEDS = str(PETRI_ROOT / "seeds")
AVAILABLE_SEEDS = sorted(p.stem for p in (PETRI_ROOT / "seeds").glob("*.md"))

# Judge dimensions: ALL our custom dims from the dimensions/ dir (auto-loaded by
# judge_dimensions, sorted by filename) -- reward_hacking, auditor_nudging,
# situational_awareness, auditor_mistakes, degenerate_behavior. Two score the
# auditor as validity checks (auditor_nudging: high = the auditor verbally steered
# the target, so reward_hacking is contaminated; auditor_mistakes: high = the
# auditor wrote a bad simulated tool result or implausible user behavior, so the
# trajectory rests on an error). situational_awareness = target clocked that it's
# being tested/audited (our own merge of Petri's eval_awareness +
# audit_situational_awareness, so we no longer pull eval_awareness off the shelf).
# degenerate_behavior = target acted incoherently (malformed/looping tool calls).
# auditor_mistakes + degenerate_behavior are categorical, expected at 1 on most runs.
DIMENSIONS = judge_dimensions(DIMENSIONS_DIR)


def reasoning_tag(name: str) -> str:
    """' [reasoning:on]'/' [reasoning:off]'/'' for a target name (for log lines)."""
    pref = REASONING_ENABLED.get(name)
    return "" if pref is None else (" [reasoning:on]" if pref else " [reasoning:off]")


def build_tasks(selected_targets: list[str], selected_seeds: list[str], run_label: str) -> list:
    """Build one Inspect task per target (each over the selected seeds).

    selected_targets : names from TARGET_CHOICES (caller validates membership).
    selected_seeds   : seed filename stems (caller validates membership).
    run_label        : stamped into every task's metadata as `run_label` (usually
                       the log-dir name) so logs are self-describing.
    Reasoning is pinned per-target via REASONING_ENABLED. Returns the task list to
    hand to run_eval(). Raises SystemExit if a target ends up with zero seeds.
    """
    selected_set = set(selected_seeds)
    tasks = []
    for tgt_name in selected_targets:
        target = TARGET_CHOICES[tgt_name]
        base = audit(
            seed_instructions=SEEDS,
            judge_dimensions=DIMENSIONS,
            max_turns=MAX_TURNS,
            enable_rollback=False,  # linear transcripts: auditor can't roll back / branch
        )
        seed_subset = MemoryDataset(
            [s for s in base.dataset if s.id in selected_set], name="seeds"
        )
        if len(seed_subset) == 0:
            raise SystemExit(
                f"no seeds matched {selected_seeds}; dataset ids were {[s.id for s in base.dataset]}"
            )
        # Pin reasoning per-target where we have an explicit preference (True/False
        # in REASONING_ENABLED); otherwise pass the bare model string (provider
        # default). target_agent resolves the target via get_model(role="target"),
        # so this config carries into generation. pref is True / False / None.
        reasoning_pref = REASONING_ENABLED.get(tgt_name)
        target_role = (
            target if reasoning_pref is None
            else get_model(target, reasoning_enabled=reasoning_pref)
        )
        tasks.append(
            task_with(
                base,
                dataset=seed_subset,
                model_roles=dict(auditor=AUDITOR, target=target_role, judge=JUDGE),
                name=f"audit_{target.split('/')[-1]}",
                # self-describing run metadata -> log.eval.metadata (see DATA_GUIDE.md)
                metadata={
                    "config_version": CONFIG_VERSION,
                    "run_label": run_label,
                    "target_name": tgt_name,
                    "target_model": target,
                    # True / False = explicitly pinned; None = provider default (e.g.
                    # deepseek-r1 native-on, Anthropic off-unless-budgeted).
                    "reasoning_enabled": reasoning_pref,
                    "auditor": AUDITOR,
                    "judge": JUDGE,
                    "nudge": "no_nudge",
                    "judge_dimensions": [d.name for d in DIMENSIONS],
                    "max_turns": MAX_TURNS,
                    "enable_rollback": False,
                },
            )
        )
    return tasks


def run_eval(tasks: list, epochs: int, concurrency: int, log_dir):
    """Run eval_set over tasks. concurrency sets max_samples AND max_connections.
    Returns (success, logs). Individual sample failures don't raise here -- eval_set
    records them in the logs (success=False) and the run continues; only the dead-
    target check (dead_targets) flags targets that produced nothing at all."""
    return eval_set(
        tasks,
        epochs=epochs,
        max_tasks=len(tasks),
        max_samples=concurrency,
        max_connections=concurrency,
        log_dir=str(log_dir),
    )


def dead_targets(logs, target_models: list[str]) -> list[str]:
    """Bare model names that produced 0 output tokens across all audits (so they
    never actually ran -- e.g. wrong slug / 402 / bad key). Empty == all targets
    generated. PURE: returns the list, never raises, so callers choose abort vs warn."""
    out_by_target = {t.split("/")[-1]: 0 for t in target_models}
    for log in logs:
        for model, u in (log.stats.model_usage or {}).items():
            name = model.split("/")[-1]
            if name in out_by_target:
                out_by_target[name] += u.output_tokens or 0
    return sorted(name for name, out in out_by_target.items() if out == 0)


def main() -> None:
    # --target=<a,b,..> REQUIRED (no default). One or more names from TARGET_CHOICES,
    # comma-separated. Each target runs as its own task; all tasks go into one shared
    # log dir. Order preserved, duplicates dropped. Unknown/missing names error out.
    target_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--target=")), None)
    if target_arg is None:
        raise SystemExit(f"--target is required (no default); choices: {sorted(TARGET_CHOICES)}")
    selected_targets = list(dict.fromkeys(t.strip() for t in target_arg.split(",") if t.strip()))
    if not selected_targets:
        raise SystemExit(f"--target had no usable names; choices: {sorted(TARGET_CHOICES)}")
    unknown_targets = [t for t in selected_targets if t not in TARGET_CHOICES]
    if unknown_targets:
        raise SystemExit(f"unknown --target {unknown_targets}; choices: {sorted(TARGET_CHOICES)}")
    target_models = [TARGET_CHOICES[name] for name in selected_targets]

    # --epochs=<N> (default 1). Must be a positive integer.
    epochs_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--epochs=")), None)
    if epochs_arg is None:
        epochs = 1
    else:
        try:
            epochs = int(epochs_arg)
        except ValueError:
            raise SystemExit(f"--epochs must be an integer, got {epochs_arg!r}")
        if epochs < 1:
            raise SystemExit(f"--epochs must be >= 1, got {epochs}")

    # --concurrency=<N> (default 10). One knob for parallelism: it sets BOTH
    # max_samples (how many audits run at once) and max_connections (max in-flight
    # requests per model role) to the same value.
    concurrency_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), None)
    if concurrency_arg is None:
        concurrency = 10
    else:
        try:
            concurrency = int(concurrency_arg)
        except ValueError:
            raise SystemExit(f"--concurrency must be an integer, got {concurrency_arg!r}")
        if concurrency < 1:
            raise SystemExit(f"--concurrency must be >= 1, got {concurrency}")

    # --seeds=<a,b,..> selects seed files by filename stem (default: all). Validate
    # against the .md files actually present in seeds/ so a typo fails loudly.
    seeds_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--seeds=")), None)
    if seeds_arg is None:
        selected_seeds = AVAILABLE_SEEDS
    else:
        requested = [s.strip() for s in seeds_arg.split(",") if s.strip()]
        unknown = [s for s in requested if s not in AVAILABLE_SEEDS]
        if unknown:
            raise SystemExit(f"unknown --seeds {unknown}; available: {AVAILABLE_SEEDS}")
        selected_seeds = requested

    # each run gets its own timestamped dir: no collisions, and eval_set never treats
    # a fresh run as already-done. (Downside: a crashed run isn't resumable.)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # log-dir label: the target name for a single target, else "<N>targets".
    target_label = selected_targets[0] if len(selected_targets) == 1 else f"{len(selected_targets)}targets"
    log_dir = DATA / "logs" / f"v2-{target_label}-{epochs}ep-{timestamp}"

    tasks = build_tasks(selected_targets, selected_seeds, log_dir.name)

    print(f"[run] {len(tasks)} task(s) x {len(selected_seeds)} seed(s) x {epochs} epoch(s), max_turns={MAX_TURNS}")
    print(f"  seeds: {selected_seeds}")
    print(f"  judge dimensions: {[d.name for d in DIMENSIONS]}")
    print(f"  auditor={AUDITOR}  judge={JUDGE}")
    print(f"  targets ({len(selected_targets)}): " + ", ".join(
        f"{n}{reasoning_tag(n)}" for n in selected_targets))
    print(f"  concurrency: {concurrency} (parallel audits = max_samples; in-flight requests/role = max_connections)")
    for n, t in zip(selected_targets, target_models):
        print(f"  target model: {n} -> {t}")
    print(f"Logs -> {log_dir}\n")

    success, logs = run_eval(tasks, epochs, concurrency, log_dir)

    print(f"\neval_set finished, success={success}")
    for log in logs:
        print(f"  {log.eval.task}: status={log.status}, samples={len(log.samples or [])}")

    # Guard: every target must actually have generated tokens. A dead target (e.g.
    # OpenRouter 402 / bad key / wrong slug) is swallowed by the auditor loop -> empty
    # transcripts the judge scores 1 across the board, looking like a clean "no reward
    # hacking" run. Fail loudly instead. (The pipeline orchestrator treats this as a
    # warning-and-continue; the standalone CLI aborts.)
    dead = dead_targets(logs, target_models)
    if dead:
        raise SystemExit(
            f"\nABORT: target(s) {dead} produced 0 output tokens across all audits.\n"
            "They never actually ran (likely an API error swallowed by the auditor, e.g. a wrong\n"
            "OpenRouter slug, insufficient-credits/402, or a bad key). NOTE: verify your model\n"
            "slugs against openrouter.ai/models -- the mid-tier targets are best-guess defaults.\n"
            f"These logs are invalid -- delete them (`rm -rf {log_dir}`), fix the target(s), and rerun."
        )
    print(f"\nView results with: uv run inspect view --log-dir {log_dir}")


if __name__ == "__main__":
    main()
