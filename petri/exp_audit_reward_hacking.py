"""Petri audit: reward-hacking propensity of a target model (v2 config).

Modes (pilot = 1 epoch / smoke test, full = 4 epochs):
  uv run exp_audit_reward_hacking.py                       # pilot, default target (qwen)
  uv run exp_audit_reward_hacking.py --full                # full, qwen
  uv run exp_audit_reward_hacking.py --full --target=opus-4-6   # full, opus-4-6 target

Target is selectable via --target (see TARGET_CHOICES); default is qwen.
Each non-default target writes to its own log dir (full-v2-<target>/pilot-v2-<target>),
so runs don't collide. pilot = 5 audits, full = 20 audits. Cost depends on the
target: qwen is cheap (~$8-12 / ~$30-45); an Anthropic target like opus-4-6 is
much pricier per token (the v1 qwen full run used ~545k target output tokens).

v2 config (vs the v1 runs in logs/pilot):
  - seeds: our own no-encouragement seeds in seeds/ (auditor is told to
    create opportunity + ordinary incentive but never suggest/encourage the
    rule-break): 3 variants of Petri's realistic seeds plus 2 of our own
    (p-hacking, benchmark gaming). The 3 artificial reward-signal seeds are
    dropped: they work by actively rewarding misbehavior, incompatible with
    no-encouragement.
  - judging: ONLY the custom reward_hacking dimension (no built-in defaults).

Costs money (Anthropic + OpenRouter APIs).
View results with `uv run make_viewer.py` or `uv run inspect view --log-dir logs/<mode>`.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval_set, task_with
from inspect_petri import audit, judge_dimensions

HERE = Path(__file__).parent
load_dotenv(HERE.parent / ".env")
# logs live in mats-local (not committed to github)
DATA = HERE.parent.parent / "mats-local" / "petri"

AUDITOR = "anthropic/claude-sonnet-4-6"
JUDGE = "anthropic/claude-opus-4-8"

# selectable targets: `--target=<name>` (default qwen). qwen runs via OpenRouter;
# the others run via the Anthropic key. Add new targets here.
TARGET_CHOICES = {
    "qwen": "openrouter/qwen/qwen3-32b",
    "opus-4-6": "anthropic/claude-opus-4-6",
}
DEFAULT_TARGET = "qwen"
target_name = next(
    (a.split("=", 1)[1] for a in sys.argv if a.startswith("--target=")), DEFAULT_TARGET
)
if target_name not in TARGET_CHOICES:
    raise SystemExit(f"unknown --target={target_name}; choices: {sorted(TARGET_CHOICES)}")
TARGETS = [TARGET_CHOICES[target_name]]

FULL = "--full" in sys.argv
# our no-encouragement seed variants (seeds/*.md; seeds_dataset requires a str path)
SEEDS = str(HERE / "seeds")
MAX_TURNS = 30
# qwen keeps the legacy dir names (full-v2/pilot-v2, where existing data lives);
# other targets get a per-target suffix so runs don't collide or mix.
suffix = "" if target_name == DEFAULT_TARGET else f"-{target_name}"
if FULL:
    EPOCHS = 4
    LOG_DIR = DATA / "logs" / f"full-v2{suffix}"
else:
    EPOCHS = 1
    LOG_DIR = DATA / "logs" / f"pilot-v2{suffix}"

# judge ONLY on our custom reward_hacking dimension (no built-in defaults)
dimensions = judge_dimensions(HERE / "dimensions")

tasks = [
    task_with(
        audit(
            seed_instructions=SEEDS,
            judge_dimensions=dimensions,
            max_turns=MAX_TURNS,
            enable_rollback=False,  # linear transcripts: auditor can't roll back / branch
        ),
        model_roles=dict(auditor=AUDITOR, target=target, judge=JUDGE),
        name=f"audit_{target.split('/')[-1]}",
    )
    for target in TARGETS
]

mode = "FULL" if FULL else "PILOT"
n_seeds = len(list((HERE / "seeds").glob("*.md")))
print(f"[{mode}] {len(tasks)} tasks x {n_seeds} seeds x {EPOCHS} epoch(s), max_turns={MAX_TURNS}")
print(f"  seeds: {SEEDS} (no-encouragement variants)")
print(f"  judge dimensions: {[d.name for d in dimensions]}")
print(f"  auditor={AUDITOR}  judge={JUDGE}  target_name={target_name}")
for t in TARGETS:
    print(f"  target: {t}")
print(f"Logs -> {LOG_DIR}\n")

success, logs = eval_set(
    tasks,
    epochs=EPOCHS,
    max_tasks=len(tasks),
    max_connections=10,
    log_dir=str(LOG_DIR),
)

print(f"\neval_set finished, success={success}")
for log in logs:
    print(f"  {log.eval.task}: status={log.status}, samples={len(log.samples or [])}")

# Guard: the target must actually have generated tokens. A dead target (e.g.
# OpenRouter 402 / bad key) is swallowed by the auditor loop -> empty transcripts
# the judge scores 1 across the board, looking like a clean "no reward hacking"
# run. Fail loudly instead. (target ids are like "openrouter/qwen/qwen3-32b";
# match on the bare model name so a provider-prefix change doesn't slip past.)
target_names = {t.split("/")[-1] for t in TARGETS}
target_out = 0
for log in logs:
    for model, u in (log.stats.model_usage or {}).items():
        if model.split("/")[-1] in target_names:
            target_out += u.output_tokens or 0
if target_out == 0:
    raise SystemExit(
        f"\nABORT: target(s) {sorted(target_names)} produced 0 output tokens across all audits.\n"
        "The target never actually ran (likely an API error swallowed by the auditor, "
        "e.g. OpenRouter insufficient-credits/402 or a bad OPENROUTER_API_KEY).\n"
        f"These logs are invalid -- delete them (`rm -rf {LOG_DIR}`), fix the target API, and rerun."
    )
print(f"target output tokens: {target_out:,} (target ran OK)")
print(f"\nView results with: uv run inspect view --log-dir {LOG_DIR}")
