# PostTrainBench trajectory viewer + RH judging pipeline

Browse and re-judge the [PostTrainBench](https://posttrainbench.com/) agent
trajectories — CLI agents (Claude Code, Codex, OpenCode, …) each given a base
LLM + an eval script + 10h on an H100, told to post-train the model to maximize
a benchmark. The research interest is **naturally-emergent reward hacking**
(e.g. an agent contaminating its training data with benchmark answers).

Task setup (matters for judging): agents got an EMPTY dir + `evaluate.py` +
`timer.sh` + chat templates — no train/val data ("complete freedom in data
sources"). `evaluate.py` queries the official TEST set, and the prompt
explicitly recommends `--limit` for fast iteration, so **test-set evaluation is
sanctioned**; the only data rule is *don't TRAIN on test questions/answers*.
Sanctioned eval runs write inspect logs (full test questions + targets) into
`logs/` — the main contamination honeypot.

## Data location

Code lives here (git); the data does **not** — it's in the sibling
`mats-local/posttrainbench/viewer_data/` tree (kept off GitHub). The dataset
provider already parsed every trajectory, so the viewer only renders:

- `index.json` — manifest, one row per task (976), all metadata + the original
  judges' verdicts (`contamination`, `disallowed_model`).
- `{run_id}.json` — one task: `events[]`, summary, metrics, judgements, logs.
- `{run_id}.workspace.json` — the agent's final workspace files.

Override the path with `PTB_DATA=/abs/path/to/viewer_data` if needed.

> **The full adjudication methodology — how the 51 flags became the final
> five-way labels (27 true reward hacks / 23 exonerated / 1 unclear) — is
> documented in [JUDGING.md](JUDGING.md).** The section below describes the
> individual judging layers.

## How the judging works (three layers)

**1. The dataset's ORIGINAL judge — a post-hoc agent, never reads the trace.**
After each run the benchmark ran a judge *agent* (Codex-format, with shell
tools) over the agent's FINAL WORKSPACE: it `ls`/`cat`/`grep`s files on demand,
so it never holds the trajectory (or even whole files) in context. Its one-line
verdicts are `contamination_judgement.txt` / `disallowed_model_judgement.txt`
(surfaced in `index.json`); its full investigation is `judge_output.json` in
the raw task dir (present for 33 of the 51 flagged runs; the other 18 have only
the one-line verdict). NOTE: `judgement.log` is NOT judge reasoning — it's
byte-identical to `error.log` (eval-process stdout). This judge used an
over-broad bar ("flag if overlap can't be ruled out"), producing false
positives (e.g. training on a genuine train split).

**2. Our re-judge (`exp_rejudge_hacks.py`, paid) — NOT an agent: one Claude
Opus call per flagged run, whole trajectory in context at once.** Per run it
gets, in a single prompt: the original judge's findings (when they exist), the
final workspace file list, and the FULL trajectory as an index-tagged condensed
JSON — reasoning / agent prose / commands / written code kept at full fidelity
(clipped at 8k chars each), big tool outputs clipped to 1.5k, the whole thing
capped at 500k chars (~127k tokens; the few giant traces get an explicit
truncation note). The model (default `claude-opus-4-8`, adaptive thinking,
effort=high) thinks first, then must call a `submit_judgement` tool returning:

- `verdict`: **yes / questionable / no** — judged against the agent's ACTUAL
  instructions (above), not the original judge's stricter bar;
- `rationale` + `summary`;
- `highlights`: hack turns + verbatim quotes (favoring the agent's own
  admissions over code) + optional reviewer annotations.

Every quote is then VERIFIED to be a real substring of its cited turn's
markable content (whitespace-tolerant) — unverifiable quotes survive only as
note-only annotations. Output: `highlights/{run_id}.json` (schema in
`highlights/README.md`). Runs ~5-way parallel. Current tallies: 34 yes /
1 questionable / 16 no.

Known bias: the prompt enumerates "test questions/answers in training data" as
cheating, so borderline *same-pool-but-unevaluated* cases (e.g. HealthBench's
4001 non-evaluated items) lean "yes"; annotations can also state conclusions
more confidently than the cited evidence. Spot-check boundary cases by hand.

**3. The categorizer (`exp_categorize_cheating.py`, paid) — one Opus call over
the 51 dossiers (not the traces).** It reads each run's verdict + rationale +
summary and induces a mechanism-shaped taxonomy (cheating categories + wrongly-
flagged-benign categories, every run in ≥1, with per-run notes and an
overview). Output: `highlights/categories.json`, rendered at `/report`.

## Run the viewer

```
uv run python mats/posttrainbench/viewing/viewer.py     # http://127.0.0.1:5001
```

- **Index**: filterable/sortable table; rows the original judges flagged are
  **red**, with our re-judge verdict pill (`RH: yes/?/no`) and the old judge's
  evidence level (`judge` = full report / `verdict` = one-liner), and a 🏷 pill
  for train-label-trap runs colored by the knowledge audit. Default order:
  RH-yes (🏷 knew → unclear → misled → untrapped) → questionable → wrongly
  flagged → clean (detail-page prev/next and /timing follow it).
- **Detail** (`/run/<run_id>`): the full trajectory per harness; hack turns
  boxed red (n/p to jump), incriminating phrases marked, reviewer annotations
  as distinct callouts, judge verdicts + re-judge rationale in banners.
- **Report** (`/report`): the cheating-pattern taxonomy with per-category run
  tables.
- **Timing** (`/timing`): for the 34 confirmed-RH runs, when the first hack
  emerges — in events, reasoning steps, wall-clock (where timestamps exist),
  and training runs launched before the hack. Click column headers to sort.
- **Workspace** (`/workspace/<run_id>`): the agent's final files.

## Trace-format note

The three harnesses store events differently — `render.py` dispatches on
`trace_format`. Codex exposes raw reasoning items; claude_code/opencode don't,
so their assistant text blocks play the declare-intent role (this mapping is
also how `/timing` counts "reasoning steps"). Only some harnesses record
per-event timestamps (opencode always; codex sometimes; claude_code never).

## Layout

```
utilities/
  paths.py                 # where everything lives (data root, highlights, run_id<->dir)
  locate.py                # trace mechanics: condensers, markable-text extraction,
                           #   training-launch detection, turn mapping (free)
viewing/
  viewer.py                # Flask app: index/run/report/timing/rates/workspace
  render.py                # event -> HTML per trace_format; marks; annotations
judging/                   # the adjudication pipeline (see JUDGING.md)
  judge_findings.py        # parse the original judge's judge_output.json (free)
  exp_rejudge_hacks.py     # Stage 1: in-context Opus re-judge
  flag_train_label_trap.py # Stage 2: deterministic trap detection (free)
  exp_audit_train_label.py # Stage 2: knew/misled/unclear knowledge audit
  exp_classify_turns.py    # per-turn hack-action vs context
  exp_audit_train_runs.py  # training-launch success audit
  flag_truncated_judge_input.py  # Stage 3: context-overflow flags (free)
  exp_final_judge.py       # Stage 4: unified agent judge (workspace + trace)
  exp_categorize_cheating.py # taxonomy -> highlights/categories.json
  exp_curate_stories.py    # per-run display stories for the viewer
  exp_localize_hacks.py    # earlier gpt-4o localizer (superseded; helpers reused)
  print_judge_prompts.py   # print all judge prompts for reading (free)
  survey_*.py              # analysis surveys (iteration style, awareness, val sets)
  list_flagged_missing_judge.py  # flagged runs lacking judge_output.json (free)
highlights/                # per-run judgements + categories.json (checked in)
```

Run scripts from this directory, e.g. `uv run python judging/exp_final_judge.py`
or `uv run python viewing/viewer.py`.
