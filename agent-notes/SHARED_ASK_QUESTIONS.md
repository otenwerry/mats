# SHARED_ASK_QUESTIONS — the cross-env ask-questions endpoint

> **Summary:** `shared/exp_ask_questions.py` is the single endpoint for asking
> questions on resumed contexts in ANY environment (`--env=petri` /
> `posttrainbench`). Shared logic (question sets, transition policy, EM judge,
> fan-out, results layout, cost totals) lives in the driver; resume mechanics
> live in one adapter per env: `petri/lib/exp_ask_env.py` and
> `posttrainbench/lib/exp_ask_env.py` (interface documented in
> `shared/ask_common.py`). Built 2026-07-15.

## Rules when editing

- Anything that should behave the same in every environment belongs in the
  DRIVER (or a new shared module), never in an adapter. Adding a question set =
  edit `QUESTION_SETS` in the driver only.
- Anything about how a context is rebuilt/resumed, what a baseline means, or
  env-only flags belongs in that env's ADAPTER. Adding an environment = one new
  adapter module + one branch in `load_adapter()`.
- Output locations and results.json / fidelity.json schemas are CONTRACTS with
  the two viewers (petri/viewer.py `load_em_blocks`, posttrainbench/viewer.py
  EM + context-recon sections). The shared endpoint writes to the same per-env
  roots with the same schemas as the retired-in-place originals; don't change
  keys without updating the viewers.
- Adapters never judge and never write fidelity.json / probe_result.md — the
  driver does, after judging. Adapters write only env artifacts
  (response.json / probe_out.jsonl / resume_turns.jsonl / context.jsonl).

## Two-venv gotcha

petri is its OWN uv project (`petri/.venv`, has inspect_ai); posttrainbench
runs in the root `mats/.venv`. The driver re-execs itself under the right
interpreter per `--env` (`ensure_venv()`), so it can be launched from anywhere
with either venv. If an import of inspect_ai ever fails inside the shared
endpoint, suspect that re-exec first.

## Transitional state (2026-07-15)

The originals — `petri/exp_ask_questions.py` and
`posttrainbench/exp_ask_questions.py` — still exist UNCHANGED because Owen is
using them while the shared endpoint beds in. They share no code with the
shared endpoint (the adapters hold moved copies), so a fix applied to an
original does NOT propagate; apply fixes to the driver/adapters. Once Owen
switches, delete the originals and update DOCS.md (collapse the two per-env
"Ask questions" entries into the Shared section) and this note.

Verified 2026-07-15 by free `--dry-run` comparison against both originals on
petri id1414 and PTB run `claude_claude-opus-4-6_10h_run1__arenahardwriting_...`:
identical bundles, flags, counts, and cost estimates (incl. `--baseline=yes`,
`--only`, and the context set).
