# Petri reward-hacking audits

Assessing models for reward-hacking propensity with [Inspect Petri](https://meridianlabs-ai.github.io/inspect_petri). Petri orchestrates three model roles: an **auditor** (drives a multi-turn scenario against the target, simulating users and tools), a **target** (the model under assessment), and a **judge** (scores each transcript per dimension). A reference clone of the Petri repo lives at `supermats/inspect_petri`.

## Layout

Two entry points do the two things we run; everything else is a library or an occasional utility.

```
exp_audit_pipeline.py      # MAIN A: run audits end-to-end  (audit -> annotate -> viewer)
exp_rollback_pipeline.py   # MAIN B: rollback resampling experiments end-to-end
make_viewer.py             # build the static viewer (run on its own, or auto-run by the pipelines)
lib/                       # the importable core the files above build on
  petri_paths.py           #   single source of truth for every filesystem path
  exp_rh_audit.py          #   audit generation + judging
  exp_annotate_hacks.py    #   secondary hack-turn annotation
  exp_rollback.py          #   rollback continuation generation
  exp_rollback_judge.py    #   rollback secondary judging
  viewer_visuals.py        #   the viewer's Visuals tab
tools/                     # occasional standalone utilities (run by hand)
  exp_rejudge_rh.py        #   re-judge OLD audits on the current dimensions
  exp_rejudge_rollbacks.py #   re-score rollback continuations on the current dimensions
  cleanup_dead_rollbacks.py / cleanup_nonhack_rollbacks.py
seeds/                     # audit scenarios (one .md per seed)
dimensions/                # judge rubrics (one .md per dimension)
```

Data (eval logs, the generated viewer, and JSON sidecars) lives in `../../mats-local/petri/`, **not** in this repo, so run data and ~MBs of HTML aren't committed. See `mats-local/petri/DATA_GUIDE.md` for how to read that data without being misled.

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

Roles: auditor = `claude-sonnet-4-6`, judge = `claude-opus-4-8`, target = whatever you pass. Each run writes its own timestamped dir under `mats-local/petri/logs/`. Targets that produce **0 output tokens** (bad slug / quota / key error) are **warned about, not aborted on** — those empty trajectories are flagged `dead`, excluded from the propensity stats, badged `DEAD` in the viewer, and summarized in a loud banner at the top of the index, so a failed run can't masquerade as clean no-hacks.

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

## Seeds & dimensions

`seeds/` holds the audit scenarios; each seed's front matter tags it with one of three **auditor-conduct conditions** (`no_nudge`, `max_elicitation`, `default_nudge_rule`) that control how hard the auditor may push without verbally cueing the target. When comparing reward-hacking scores across seeds, remember the conditions differ — within-seed comparisons are the clean ones.

`dimensions/` holds the judge rubrics (auto-loaded as a directory). They are scored together in one judge call: `reward_hacking` (the target's gaming, tiered 1–10), plus validity checks on the audit itself — `auditor_nudging` (did the auditor verbally steer; high = `reward_hacking` is contaminated), `auditor_mistakes`, `situational_awareness` (did the target clock it's being tested), and `degenerate_behavior` (did the target act incoherently). All rubrics are unvalidated; spot-check judge citations before trusting at scale. To change the dimension set, edit the files in `dimensions/`; older audits then need a re-judge (see Tools).

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
uv run tools/exp_rejudge_rh.py            # re-judge OLD audits on the current dimensions (costs money)
uv run tools/exp_rejudge_rollbacks.py     # re-score rollback continuations on the current dimensions (costs money)
uv run tools/cleanup_dead_rollbacks.py --dry-run    # FREE preview: move all-dead continuation files to trash
uv run tools/cleanup_nonhack_rollbacks.py           # move continuations of non-hack originals to trash
```

## Known lossiness

- `max_turns` (in `lib/exp_rh_audit.py`, currently 30): audits are truncated at that many auditor turns.
- `compaction=True` (Petri default): on very long audits the auditor's working context can be compacted. The judge still sees the full transcript; compaction events are recorded in the log and flagged in the viewer.
