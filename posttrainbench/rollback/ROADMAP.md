# Rollback experiments — build-out roadmap

Goal: turn the working-but-bespoke rollback pipeline into infrastructure where
Owen can run **any rollback experiment as a parameterized command**, across
**all 30 reward-hack trajectories**, without per-run hand-holding.

This is the forward-looking plan. The mechanics of the *current* pipeline are in
`README.md` (read that first).

## Overnight validation results (2026-06-15) — box-smoke GREEN

On a real H100 (Lambda west-filesystem), the fast smoke harness PASSED for both
core engines, validating the whole chain (reconstruct → resume → 1 agent step →
base-model materialize → vLLM scoring → archive → pull → viewerize/quarantine):
- **opencode** (bfcl_codexmax): resumed ses_424638 faithfully, coherent multi-step.
- **claude** (humaneval/gemma cut67): `claude --resume` loaded the reconstructed
  session (33,835 tok cached) and continued on-task — **resolves recon.py's open
  question: pure `--resume` works, no `--continue` fallback needed.**
This covers the resume mechanics for all 23 non-codex trajectories.
Also validated: **programmatic launch** (`launch_box.sh` / `ptb_lib.ptb_launch_box`)
— discovery (fs/region/key/instance-type) + launch API + terminate all confirmed.
Harness bugs found & fixed: HF cache dir uses `--` not `-` (base-model
materialize); shell→python bool interpolation in smoke_report.
Known minor: harmless `rsync mkstempsock` warning on pull (a socket file in the
workspace); add `--no-specials --no-devices` to suppress.

## Decisions locked (2026-06-14)

- **Compute: Lambda only.** RunPod parked (apptainer/privileged friction; revisit
  later if funding pressure demands — likely via Dockerizing `standard.def`).
- **Fidelity: each agent gets a single H100** (matches the originals). Multi-GPU
  is for **parallelism** — run N agents on an N×H100 box, one GPU each
  (`CUDA_VISIBLE_DEVICES` per cell) — NOT for giving one agent more GPUs.
- **Scope: all 30 trajectories** with `final.is_reward_hack` (in
  `highlights/*.json`). Scaffolds: opencode ~8, claude-API ~10, claude_non_api
  ~4, codex_non_api ~7, qwen3max 1.
- **Subscription auth = the faithful path** for `*_non_api` targets (not a cost
  hack). Claude headless works via `CLAUDE_CODE_OAUTH_TOKEN`; codex via
  `forced_login_method="chatgpt"` + `auth.json` (proven in PTB's own
  `agents/*_non_api/solve.sh`).
- **First deliverable: a fast end-to-end smoke harness** (validate every code
  path in minutes for pennies, before any multi-hour run).

## The experiment as a function (target CLI)

One `exp_*` entrypoint that wraps build → provision → launch → pull → viewerize:

```
exp_run.py
  --trajectory  {key | "all" | comma-list}     # iterate the 30
  --cut         {N | "auto"}                    # auto = cut-finder agent
  --prompt      {name}                          # reads prompts/<name>.txt
  --condition   {control | treatment | both}
  --seeds       N                               # resamples per cell
  --gpus        N                               # box size; 1 cell per GPU
  --smoke       {off | fast}                    # fast = 1 step + jump to end
```

Variables to externalize out of `config.py`:
- intervention prompt text → `prompts/<name>.txt` (today: hardcoded
  `INTERVENTION_REMINDER`).
- cut location → `--cut` / per-trajectory, with an `auto` cut-finder.

## Phased plan

- **P0 — cleanup + foundation**
  - Remove the dead `rollback_*.json` from `highlights/` (Owen handles git).
  - Extract intervention text → `prompts/*.txt`; control stem stays the inert
    "Please continue." stem.
- **P1 — fast smoke harness (FIRST)**
  - A mode that: provision/locate box → prep → let the agent take ONE real step
    → **force the end-condition** → score (or stub) → archive → pull →
    viewerize. Must exercise EVERY path that a real run hits, so a green smoke
    means a real run won't die at hour 2. Cheap (~pennies), fast (minutes).
- **P2 — unified experiment CLI** (the function above): trajectory iteration,
  prompt files, `condition both`, seeds.
- **P3 — provisioning automation** (launcher DONE; scheduler pending):
  `run/exp_launch_box.sh` + `run/ptb_lib.sh` programmatically launch a Lambda box
  (auto-discovers filesystem+region+SSH-key+H100 instance type w/ capacity,
  polls to active, optional `--bootstrap`); `--gpus N` for an N×H100 box.
  `ptb_lib.sh` is the SINGLE source of truth for SSH opts + Lambda API + fs
  detection — exp_run_experiment / exp_pull_result / exp_smoke now source it
  (no more copy-pasted SSH/curl). TODO: one-cell-per-GPU scheduler (N agents on
  N×H100, 1 GPU each) + exp_smoke auto-provisioning via ptb_launch_box.
- **P4 — engine coverage for all scaffolds** (toward one control run on all 30).
  Verified staging matrix (from adjudication data, 2026-06-14):

  | scaffold (engine) | count | trace_format | status |
  |---|---|---|---|
  | opencode | 8 | opencode | ✅ proven |
  | claude (API) | 10 | claude_code | engine scaffolded, UNPROVEN on box |
  | claude_non_api | 4 | claude_code | = claude engine + OAuth auth |
  | qwen3max | 1 | claude_code | likely = claude engine; defer |
  | codex_non_api | 7 | codex | NEW: recon_codex + engine + ptbio codex_item |

  **`claude_code` covers 15 of 30** → one engine path (recon.py + native_cli +
  solve_intervention.sh) unlocks half. **Only 7 are prep-free** (cheapest first
  controls): 4 opencode, 2 codex, 1 claude.

  - **P4a DONE — auto-enumeration backbone** (`build_trajectory_manifest.py` ->
    `trajectory_manifest.json`; `config.ALL_TRAJECTORIES`, `get_trajectory`).
    All 30 addressable by run_id; cuts = `first_hack_event`; `needs_prep` =
    `trainings_before_first_hack>0`; curated entries override derivable fields,
    manifest authoritative for needs_prep/auth/trace_format.
  - **P4b — claude path to parity** (DONE offline; box-validation pending):
    native_cli run_config now carries the box-runner fields (model_to_train,
    eval_model_dir, eval_limit, prep_commands, resume_prompt, scaffold, auth);
    `run_rollout_on_box.sh` is auth-aware (injects ANTHROPIC_API_KEY /
    CLAUDE_CODE_OAUTH_TOKEN, passes RESUME_MODE/AUTH); `solve_intervention.sh`
    handles oauth (claude_non_api); `viewerize.parse_continuation` is
    scaffold-aware (claude stream-json parser, verified on a real claude trace).
    Verified offline: a claude cell prepares (recon ok at cut 67), run_config
    complete, parser yields 763 events on the humaneval trace, engine routing ok.
    **Still needs a box to validate**: the actual `claude --resume` mechanics
    (recon.py's open question — pure-resume vs --continue) and auth injection.
    **qwen3max**: confirmed it ran ON Claude Code via Qwen's Anthropic-compatible
    DashScope endpoint (agents/qwen3max/solve.sh) — so the claude engine is
    faithful for it, but it needs a small DashScope auth variant
    (ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN=DASHSCOPE key). 1 trajectory; minor.
  - **P4c — codex path** (SET UP, not yet runnable — needs the subscription):
    built `engine/codex_cli.py`, `session/recon_codex.py`,
    `run/solve_intervention_codex.sh`, ptbio `codex_item` support (verified:
    89 tool calls, backward-rebuild ready), viewerize codex parser (verified:
    231 events), engine routing, per-scaffold container (`codex.sif`), and the
    ChatGPT-subscription `auth.json` deployment (fill-in path
    `~/.config/ptb/codex_auth.json`). Resume confirmed supported (`codex exec
    resume <thread_id> --json`, replays a rollout JSONL).
    **2 things left, gated on the MATS ChatGPT subscription:**
      1. capture a real `rollout-*.jsonl` and finalize the ONE validation seam
         `recon_codex.to_rollout_lines()` (build_session refuses until
         `PTB_CODEX_ROLLOUT_VALIDATED=1` so we don't waste quota);
      2. drop the subscription `auth.json` at `~/.config/ptb/codex_auth.json`.
    (No separate container: our standard.def bundles claude+codex+opencode CLIs,
    so codex runs in standard.sif like everything else.)
  - **P4d — prep auto-derivation** (candidate generator DONE; on-box validation
    next). `derive_prep.py` extracts candidate prep_commands from the trace:
    pre-cut python-script executions, exclude eval/probe/analysis, stage-classify
    (prep/train/merge), collapse version-iterated + re-run scripts (train → the
    single last run; prep/merge → last per distinct script), strip shell noise,
    order by event. **Reproduces both curated preps (kimi_humaneval,
    healthbench_kimi17) exactly.** Across the 23: **16 derive 'clean' (single
    canonical train, prep-before/merge-after, ≤4 steps — trustworthy pending
    validation), 7 need review** (agent interleaved prep/train, or huge pipelines
    like the codex reprompt cut2412 with 20+ build_* scripts).
    NOT yet wired into real runs (a wrong prep wastes a GPU run). **Next: the
    on-box prep-smoke** — run a derived prep, score the regenerated model, confirm
    it matches the recorded pre-cut baseline (Trajectory.precut_eval_file) within
    stderr; only validated preps get promoted/used. Needs a GPU box.
- **P5 — cut-finder agent (refinement, not a blocker)**: cuts already
  auto-derive from `first_hack_event`. This refines them to clean message
  boundaries (recon.validate_prefix) and can re-judge onset; build when the
  auto-cuts prove too coarse.
- **P6 — (later) RunPod** via Dockerized `standard.def`, if funding makes it
  worthwhile.

## Open questions / risks to revisit

- claude/codex CLI resume fidelity: does each support session reconstruction +
  resume as cleanly as opencode? (opencode is proven; the others are not.)
- codex traces lack timestamps (like claude) → timer falls back to
  sleep-calibration; verify that path.
- subscription rate limits / ToS when running many `*_non_api` resumes headless.
- Lambda capacity for 4×/8× boxes when we want parallelism.
