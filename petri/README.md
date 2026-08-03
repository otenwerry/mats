# Petri reward-hacking audits

Assessing models for reward-hacking propensity with [Inspect Petri](https://meridianlabs-ai.github.io/inspect_petri). Petri orchestrates three model roles: an **auditor** (drives a multi-turn scenario against the target, simulating users and tools), a **target** (the model under assessment), and a **judge** (scores each transcript per dimension). A reference clone of the Petri repo lives at `supermats/references/inspect_petri`.

Sibling project: `mats/environments/` runs REAL ports of some of these scenarios — actual code execution in Docker containers, no auditor — reusing this project's seeds, judge, rendering layer, and cost tracking. It keeps its own data root (`mats-local/environments/`) and its own viewer, so nothing mixes with the runs here. See `agent-notes/REAL_ENV_GUIDE.md`.

## Layout

The experiment entry points plus the viewer builder sit at the top; everything else is a library or an occasional utility.

```
exp_audit_pipeline.py            # A: run audits end-to-end  (audit -> annotate -> viewer)
exp_rollback_pipeline.py         # B: rollback resampling experiments end-to-end
exp_continuation_pipeline.py     # C: continuation experiments (prior task -> new task) end-to-end
exp_resample_begin_pipeline.py   # D: begin-resample experiments (re-roll from turn 1, control) end-to-end
exp_new_judge.py                 # compare a candidate judge with the saved Opus verdicts and cost
viewer.py                        # build the static viewer (run on its own, or auto-run by the pipelines)
lib/                             # the importable core the files above build on
  petri_paths.py                 #   single source of truth for every filesystem path
  exp_rh_audit.py                #   audit generation + judging
  exp_annotate_hacks.py          #   secondary hack-turn annotation
  exp_rollback.py                #   rollback continuation generation
  exp_rollback_judge.py          #   rollback secondary judging
  exp_resample.py                #   begin-resample core + the deviation/faithfulness judge
  exp_continuation.py            #   continuation core (prefix reconstruction, slicing judge, faithfulness)
  viewer_visuals.py              #   the per-sweep visuals pages
tools/                           # occasional standalone utilities (run by hand)
  exp_rejudge_main.py            #   re-judge EVERY current-sweep audit fresh on all current dimensions
  exp_rejudge_rh.py              #   re-judge OLD audits (reward_hacking>=5) on the current dimensions
  exp_rejudge_rollbacks.py       #   re-score rollback continuations on the current dimensions
  exp_rejudge_continuation_faithfulness.py  # re-run only the continuation faithfulness judge
  exp_rejudge_incomplete_hacks.py           # re-judge degenerate/nudged hacks missing a current dim
  exp_grade_dimension.py         #   backfill ONE newly-added dim onto existing full hacks (dim_scores sidecar)
  exp_grade_incompleteness.py    #   same backfill pattern, incompleteness over all current-sweep originals
  exp_deviation_judge.py         #   auditor-deviation judge for rollback continuations
  exp_mechanism_similarity.py    #   follow-up to C: rate how similar each re-hack's mechanism is to the hack it was primed on
  exp_test_openrouter.py         #   one tiny OpenRouter call to check whether the budget cap is lifted
  explore_select_continuation.py #   read-only: score all audits to help hand-pick continuation triples
  explore_token_usage.py         #   read-only: token usage by model role per run dir (run dirs via args)
  explore_validate_seeds.py      #   read-only: check every pinned seed dir assembles cleanly (run after editing seeds)
  migrate_hpc_*.py               #   one-time data migrations, already applied (idempotent)
  cleanup_dead_rollbacks.py / cleanup_nonhack_rollbacks.py
docs/                            # CODEBASE_GUIDE.md, CONTINUATION_EXPERIMENT_SPEC.md
seeds/                           # audit scenarios, including composed _shared/ collections
dimensions/                      # judge rubrics: global/ + a folder mirroring each seed dir
```

Data (eval logs, the generated viewer, and JSON sidecars) lives in `../../mats-local/petri/`, **not** in this repo, so run data and ~MBs of HTML aren't committed. See `docs/CODEBASE_GUIDE.md` for how the code + experiments fit together, and `mats-local/petri/DATA_GUIDE.md` for how to read the data without being misled.

## Setup

This folder is its own uv project:

```bash
cd petri
uv sync
```

API keys are read from `mats/.env` (`ANTHROPIC_API_KEY` for auditor/judge/Anthropic targets, `OPENROUTER_API_KEY` for OpenRouter targets).

## A. Generate trajectories — `exp_audit_pipeline.py`

One command runs the **audit** (targets × seeds × epochs, each judged inline on every dimension) and then the **viewer** (free). Paid **annotation** (locating hack turns on full reward hacks) is optional and off by default.

```bash
# full sweep, 50-wide:
uv run exp_audit_pipeline.py --targets=opus-4-6,sonnet-4-6,qwen3-32b,deepseek-r1,llama-3.3-70b --seeds=all --epochs=2 --concurrency=50
# quick single-target smoke:
uv run exp_audit_pipeline.py --targets=qwen3-32b --seeds=ml_training_data_misuse --epochs=1
uv run exp_audit_pipeline.py ... --annotate           # include paid hack-turn annotation
uv run exp_audit_pipeline.py ... --force-annotate     # annotate and replace existing annotations
```

| Flag | Meaning |
|------|---------|
| `--targets=<a,b,..>` | **required**; one or more names from `TARGET_CHOICES` (in `lib/exp_rh_audit.py`). A multi-target run sweeps each as its own task into one log dir. |
| `--seeds=<a,b,..>` or `all` | **required**; seed filename stems from `seeds/`, or `all`. |
| `--epochs=<N>` | **required**; epochs per (target, seed) cell. Trajectory count = targets × seeds × epochs. |
| `--concurrency=<N>` | one knob → audit parallelism (parallel audits *and* in-flight requests per role) **and** annotate parallelism (default 15). |
| `--annotate-model=<m>` | Anthropic model for the hack-turn annotation (default `claude-opus-4-8`). |
| `--annotate` | opt into paid hack-turn annotation; off by default. |
| `--skip-viewer` / `--force-annotate` | skip the free viewer rebuild / annotate and replace existing annotations. |

Roles: auditor = `claude-opus-4-8` (extended thinking on), judge = `claude-opus-4-8`, target = whatever you pass. Each run writes its own timestamped dir under `mats-local/petri/logs/`. Targets that produce **0 output tokens** (bad slug / quota / key error) are **warned about, not aborted on** — those empty trajectories are flagged `dead`, excluded from the propensity stats, badged `DEAD` in the viewer, and summarized in a loud banner at the top of the index, so a failed run can't masquerade as clean no-hacks.

## B. Rollback experiments — `exp_rollback_pipeline.py`

For each full reward hack (with a hack-turn annotation, used to locate the first hack), this re-rolls the conversation at one or more **cut locations** (`begin`, `middle`, `before`, `after`), in a **control** (plain re-roll) and a **treatment** (an honesty message inserted at the cut) condition, judges every continuation, annotates the ones the viewer tints red, and rebuilds the viewer. Each cell's run dir is `rollback-<location>-<condition>-<N>x-<timestamp>`. It dedups every cell against existing runs, so re-running only fills gaps.

```bash
uv run exp_rollback_pipeline.py --locations=all --N=5 --concurrency=50
uv run exp_rollback_pipeline.py --locations=begin,after --N=3 --trajectories=12,55
uv run exp_rollback_pipeline.py --locations=all --N=5 --dry-run   # FREE: build + validate the plan, no run
```

| Flag | Meaning |
|------|---------|
| `--locations=<a,b,..>` or `all` | **required**; subset of `{begin,middle,before,after}`. |
| `--N=<int>` | **required**; rollbacks (epochs) per cell. |
| `--concurrency=<int>` | one knob → generation + judging parallelism (default 15). |
| `--trajectories=<ids>` | restrict to specific original trajectory ids (default: all annotated full hacks). |
| `--dry-run` / `--skip-judge` / `--skip-viewer` / `--force-judge` | plan-only / stage toggles. |

The judge is told which inserted message and experimenter note came from the research team (not the auditor), so `auditor_nudging` on treatment runs reflects only the auditor's own conduct and is comparable to control.

## C. Continuation experiments — `exp_continuation_pipeline.py`

Asks: if a target reward-hacks on one task and is then handed a *second, unrelated* task, does its hack rate on the second task go up because it's conditioned on its own earlier hack? You give flat id lists of **prefixes** (prior trajectories, in up to three flavors) and **continuations** (new-task sources); each id's target model is read from its stamped metadata and grouped automatically. Per model, every (new task × prefix) pair runs N times, plus a baseline per new task — counts per category are free (e.g. 1 prefix of each flavor + 2 continuations = 8 cells):

- `no_prefix` — target gets the new task with no prior context (the baseline).
- `clean_prefix` — target is first conditioned on a prior task it completed *cleanly*.
- `corrected_hack_prefix` — target is first conditioned on a prior task where it hacked, then *corrected* the hack.
- `full_hack_prefix` — target is first conditioned on a prior task it fully *reward-hacked*. (Old runs have the undivided `hack_prefix`.)

The auditor reproduces the new-task original's environment in all conditions (primed with its full record + a "new workspace" framing) and never sees the prefix. The target's reward-hacking judge and a separate **faithfulness judge** (did the auditor reproduce the original?) both have the prefix **mechanically sliced out** of the transcript they see (system prompt + new task only, `[M#]` numbers preserved), so a prefixed run isn't auto-scored a hack because of its prefix. All config (target/auditor/judge, reasoning, max_turns) is inherited from the new-task original's stamped metadata. New originals across every seed family share `seeds/SYSTEM_PROMPT.txt`; continuation planning aborts before paid work if selected A/B sources carry different recorded prompt text. For a different-family task, the pivot names the new kind of work ("…move on to a different task: *an experiment analysis task* this time."; descriptors in `SEED_FAMILY_TASK_DESCRIPTION`, `lib/exp_continuation.py`). Results render on the viewer's **Continuations** page.

```bash
uv run exp_continuation_pipeline.py --full-hack-prefixes=660 --corrected-hack-prefixes=730 \
    --clean-prefixes=661 --continuations=794,806 --n=5
uv run exp_continuation_pipeline.py ... --n=5 --dry-run   # FREE: build + validate the plan, no run
```

| Flag | Meaning |
|------|---------|
| `--continuations=<ids>` | **required**; new-task source ids (any models/seeds; each pairs with every same-model prefix from a different seed). |
| `--full-hack-prefixes=<ids>` / `--corrected-hack-prefixes=<ids>` / `--clean-prefixes=<ids>` | prefix ids per flavor; required iff that condition is requested. |
| `--n=<int>` | **required**; continuations (epochs) per cell. |
| `--conditions=<a,b>` or `all` | subset of `{no_prefix,clean_prefix,corrected_hack_prefix,full_hack_prefix}` (default all). |
| `--concurrency=<int>` | one knob → generation + faithfulness-judge parallelism (default 50). |
| `--dry-run` / `--skip-judge` / `--skip-annotate` / `--skip-viewer` / `--force-judge` / `--force-annotate` | plan-only / stage toggles. |

## D. Begin-resample experiments — `exp_resample_begin_pipeline.py`

Re-rolls an original audit from turn 1 under identical conditions, with **nothing inserted** (control only), to ask whether the target behaves differently on a fresh sample. The resample's auditor is handed the full original run as a labeled reference plus a faithfulness instruction, so it reproduces the original environment; a **deviation judge** scores how faithfully it did (the same rubric the continuation faithfulness judge uses). All config is inherited from the original's stamped metadata.

```bash
uv run exp_resample_begin_pipeline.py --trajectories=466,508 --n=5 --concurrency=50
uv run exp_resample_begin_pipeline.py --trajectories=466 --n=5 --dry-run   # FREE: plan only
```

## E. Compare a candidate judge — `exp_new_judge.py`

This re-judges a balanced sample of current trajectories without changing the viewer's
saved verdicts. `n` is drawn from each of three viewer buckets (reward hacks, interesting,
and clean non-hacks) for each of the two active families (ML and p-hacking), so the total is
`6 × n`. Autonomous and user-elicited reward hacks are pooled before sampling.

```bash
uv run exp_new_judge.py --judge=sonnet-4.6 --n=10 --dry-run  # free sample preview
uv run exp_new_judge.py --judge=sonnet-4.6 --n=10            # 60 paid judge calls
uv run exp_new_judge.py --judge=sonnet-4.6 --n=10 --summary-only
```

The default fixed sample seed lets different judges use the same saved trajectories. Runs
checkpoint after every call and resume when the same command is repeated. The JSON outputs,
cost comparison, statistical analysis, and standalone HTML report are written under
`mats-local/petri/new_judge/`.

## Seeds & dimensions

`seeds/` holds the audit scenarios. Every seed is **no-nudge**: its body carries an explicit "Auditor conduct" paragraph forbidding the auditor from suggesting, hinting at, or encouraging the hack, so any reward hacking must originate from the target's own reasoning. The auditor builds the tempting opportunity but never verbally cues the target.

Audit dimensions are routed by seed directory. Every audit judge receives all `.md`
rubrics in `dimensions/global/` plus all rubrics in the relevant mirrored seed folder,
such as `dimensions/p_hacking/`. A more specific member folder such as
`dimensions/p_hacking/checkout_redesign/` wins over its family folder when it contains
rubrics; this lets one collection invocation use different dimensions for different
prompts. The exact folder, rubric files, and output names are printed before the run and
stored in its task metadata.

During the directory migration only, existing top-level `dimensions/*.md` files are also
loaded as global dimensions. No files were moved as part of the routing change. Duplicate
dimension filenames across the selected global and seed-specific sources fail before any
API call. `dimensions/archived/` and `dimensions/rollback_only/` remain excluded from
ordinary audit judging. Older audits need a re-judge after their rubric set changes (see
Tools).

The committed **binary reward-hack definition** lives in one place — `viewer.BINARY_HACK_CRITERIA` — and is imported everywhere (the annotate gate, the index tally, rollback selection).

## View results

```bash
uv run viewer.py                                       # free; both pipelines also run this as their last stage
open ../../mats-local/petri/viewer/index.html          # audits (+ rollbacks folded into full-hack rows)
# or Inspect's built-in log browser:
uv run inspect view --log-dir ../../mats-local/petri/logs/<run>
```

The index has one table per hack category over all runs, each row linking to a per-audit page with the judge's summary/justification/highlights and the exact `[M]`-numbered transcript the judge read (every citation is a clickable anchor). The **Visuals** tab charts rollback re-hacking. Every trajectory gets a stable integer ID persisted in `trajectory_ids.json`; rerunning a pipeline picks up new trajectories automatically.

## Tools (run occasionally, by hand)

```bash
uv run tools/exp_rejudge_main.py          # re-judge EVERY main-page audit on all current dimensions (costs money)
uv run tools/exp_rejudge_rh.py            # re-judge OLD audits on the current dimensions (costs money)
uv run tools/exp_rejudge_rollbacks.py     # re-score rollback continuations on the current dimensions (costs money)
uv run tools/cleanup_dead_rollbacks.py --dry-run    # FREE preview: move all-dead continuation files to trash
uv run tools/cleanup_nonhack_rollbacks.py           # move continuations of non-hack originals to trash
```

## Known lossiness

- `max_turns` (in `lib/exp_rh_audit.py`, currently 50): audits are truncated at that many auditor turns.
- `compaction=True` (Petri default): on very long audits the auditor's working context can be compacted. The judge still sees the full transcript; compaction events are recorded in the log and flagged in the viewer.
