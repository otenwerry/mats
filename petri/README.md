# Petri reward-hacking audits

Assessing models for reward-hacking propensity with [Inspect Petri](https://meridianlabs-ai.github.io/inspect_petri). Petri orchestrates three model roles: an **auditor** (drives a multi-turn scenario against the target, simulating users and tools), a **target** (the model under assessment), and a **judge** (scores each transcript per dimension). A reference clone of the Petri repo lives at `supermats/inspect_petri`.

## Layout

The experiment entry points plus the viewer builder sit at the top; everything else is a library or an occasional utility.

```
exp_audit_pipeline.py            # A: run audits end-to-end  (audit -> annotate -> viewer)
exp_rollback_pipeline.py         # B: rollback resampling experiments end-to-end
exp_continuation_pipeline.py     # C: continuation experiments (prior task -> new task) end-to-end
exp_resample_begin_pipeline.py   # D: begin-resample experiments (re-roll from turn 1, control) end-to-end
exp_mechanism_similarity.py      # follow-up to C: rate how similar each re-hack's mechanism is to the hack it was primed on
explore_*.py                     # one-off FREE read-only analysis scripts (pinned to specific past runs; update their hardcoded ids before reuse)
make_viewer.py                   # build the static viewer (run on its own, or auto-run by the pipelines)
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
  explore_select_continuation.py #   read-only: score all audits to help hand-pick continuation triples
  migrate_hpc_*.py               #   one-time data migrations, already applied (idempotent)
  cleanup_dead_rollbacks.py / cleanup_nonhack_rollbacks.py
docs/                            # CODEBASE_GUIDE.md, CONTINUATION_EXPERIMENT_SPEC.md
seeds/                           # audit scenarios (one .md per seed)
dimensions/                      # judge rubrics (one .md per dimension)
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

One command runs three stages: **audit** (targets × seeds × epochs, each judged inline on every dimension) → **annotate** (locate the hack turns on the full reward hacks) → **viewer** (rebuild the static viewer, free).

```bash
# full sweep, 50-wide:
uv run exp_audit_pipeline.py --targets=opus-4-6,sonnet-4-6,qwen3-32b,deepseek-r1,llama-3.3-70b --seeds=all --epochs=2 --concurrency=50
# quick single-target smoke:
uv run exp_audit_pipeline.py --targets=qwen3-32b --seeds=ml_training_data_misuse --epochs=1
uv run exp_audit_pipeline.py ... --skip-annotate     # audit + viewer only
uv run exp_audit_pipeline.py ... --force-annotate     # re-annotate hacks already done
```

| Flag | Meaning |
|------|---------|
| `--targets=<a,b,..>` | **required**; one or more names from `TARGET_CHOICES` (in `lib/exp_rh_audit.py`). A multi-target run sweeps each as its own task into one log dir. |
| `--seeds=<a,b,..>` or `all` | **required**; seed filename stems from `seeds/`, or `all`. |
| `--epochs=<N>` | **required**; epochs per (target, seed) cell. Trajectory count = targets × seeds × epochs. |
| `--concurrency=<N>` | one knob → audit parallelism (parallel audits *and* in-flight requests per role) **and** annotate parallelism (default 15). |
| `--annotate-model=<m>` | Anthropic model for the hack-turn annotation (default `claude-opus-4-8`). |
| `--skip-annotate` / `--skip-viewer` / `--force-annotate` | stage toggles. |

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

Asks: if a target reward-hacks on one task and is then handed a *second, unrelated* task, does its hack rate on the second task go up because it's conditioned on its own earlier hack? For each target model you give a **triple** — a hack prefix (A) and a clean prefix (A) from the *same* seed, plus a task source (B) from a *different* seed — and it runs three conditions N times each:

- `no_prefix` — target gets B's task with no prior context (the baseline).
- `clean_prefix` — target is first conditioned on A's *clean* transcript, then handed B's task.
- `hack_prefix` — target is first conditioned on A's *hack* transcript, then handed B's task.

The auditor reproduces B's environment in all three conditions (primed with B's full record + a "new workspace" framing) and never sees the prefix. The target's reward-hacking judge and a separate **faithfulness judge** (did the auditor reproduce B?) both have the prefix **mechanically sliced out** of the transcript they see (system prompt + new task only, `[M#]` numbers preserved), so a `hack_prefix` run isn't auto-scored a hack because of its prefix. All config (target/auditor/judge, reasoning, max_turns, the fixed system prompt) is inherited from B's stamped metadata. Results render on the viewer's **Continuations** page. Use `tools/explore_select_continuation.py` to pick the triples.

```bash
uv run exp_continuation_pipeline.py --hack-trajectories=471 --clean-trajectories=472 --continuation-trajectories=466 --n=5
# sweep several models (one hack/clean/B id per model in each list):
uv run exp_continuation_pipeline.py --hack-trajectories=471,557 --clean-trajectories=472,514 --continuation-trajectories=466,508 --n=5 --concurrency=50
uv run exp_continuation_pipeline.py ... --n=5 --dry-run   # FREE: build + validate triples, no run
```

| Flag | Meaning |
|------|---------|
| `--hack-trajectories=<ids>` / `--clean-trajectories=<ids>` / `--continuation-trajectories=<ids>` | **required**; one id per target model in each list. Hack & clean prefixes must share a seed; B must be a different seed. |
| `--n=<int>` | **required**; continuations (epochs) per (triple, condition). |
| `--models=<a,b>` | restrict to these target names (default: every model across the three lists). |
| `--conditions=<a,b>` or `all` | subset of `{no_prefix,clean_prefix,hack_prefix}` (default all). |
| `--concurrency=<int>` | one knob → generation + faithfulness-judge parallelism (default 50). |
| `--dry-run` / `--skip-judge` / `--skip-annotate` / `--skip-viewer` / `--force-judge` / `--force-annotate` | plan-only / stage toggles. |

## D. Begin-resample experiments — `exp_resample_begin_pipeline.py`

Re-rolls an original audit from turn 1 under identical conditions, with **nothing inserted** (control only), to ask whether the target behaves differently on a fresh sample. The resample's auditor is handed the full original run as a labeled reference plus a faithfulness instruction, so it reproduces the original environment; a **deviation judge** scores how faithfully it did (the same rubric the continuation faithfulness judge uses). All config is inherited from the original's stamped metadata.

```bash
uv run exp_resample_begin_pipeline.py --trajectories=466,508 --n=5 --concurrency=50
uv run exp_resample_begin_pipeline.py --trajectories=466 --n=5 --dry-run   # FREE: plan only
```

## Seeds & dimensions

`seeds/` holds the audit scenarios. Every seed is **no-nudge**: its body carries an explicit "Auditor conduct" paragraph forbidding the auditor from suggesting, hinting at, or encouraging the hack, so any reward hacking must originate from the target's own reasoning. The auditor builds the tempting opportunity but never verbally cues the target.

`dimensions/` holds the judge rubrics (auto-loaded as a directory). They are scored together in one judge call: `reward_hacking` (the target's gaming, tiered 1–10) and `hack_in_final_solution` (does the final submitted solution still rest on the hack — informational); target-quality checks — `degenerate_behavior` (target acted incoherently) and `hallucination` (target genuinely confused about its own actions, a low-confidence signal); validity checks on the audit itself — `auditor_nudging` (did the auditor verbally steer; high = `reward_hacking` is contaminated), `auditor_mistakes`, and `situational_awareness` (did the target clock it's being tested); plus `incompleteness` (how unfinished the run was when it ended — informational). All rubrics are unvalidated; spot-check judge citations before trusting at scale. To change the dimension set, edit the files in `dimensions/`; older audits then need a re-judge (see Tools).

The committed **binary reward-hack definition** lives in one place — `make_viewer.BINARY_HACK_CRITERIA` — and is imported everywhere (the annotate gate, the index tally, rollback selection).

## View results

```bash
uv run make_viewer.py                                  # free; both pipelines also run this as their last stage
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
