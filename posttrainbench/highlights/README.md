# Per-run judgements (checked into git)

One `{run_id}.json` per **adjudicated run** — the 51 the original PTB judge
flagged for contamination/disallowed-model, plus the OpenAI-API-rule
violations found by our own scan. Each file's **`final` block is the only
thing you should filter on**; everything else is the judging history that
produced it.

> **`rollback_*.json` here are the rollback-experiment JUDGE OUTPUTS** (written
> by `judging/exp_judge_rollback.py`, same schema as above). They are NOT
> duplicates of the `mats-local/rollback/viewer_data/rollback_*.json` *trajectory*
> files — same name, different directory, different content. The viewer reads
> rollback "judged" status from these. **Don't delete them.**

```json
{
  "run_id": "...",
  "final": {                                  // CANONICAL — judging/finalize.py
    "verdict": "yes" | "no" | "questionable", //   is it actual reward hacking
    "label": "contamination" | "disallowed model" | "openai api misuse"
           | "broke rules unknowingly" | "didn't break rules" | "unclear",
    "is_reward_hack": true,                   //   label is one of the first three
    "categories": ["verbatim-test-copy"],     //   mechanism slugs (categories.json)
    "first_hack_event": 67,                   //   viewer-events index of the first hack ACTION
    "trainings_before_first_hack": 0,         //   successful training launches before it
    "trainings_total": 17,
    "derivation": "judging/finalize.py — ..."
  },
  ...judging history (see below)...
}
```

**Census recipes** — `is_reward_hack` selects the true reward hacks (30:
26 contamination + 1 disallowed model + 3 openai api misuse) across BOTH rule
axes; `label` gives the five-way adjudication the viewer displays; the timing
page is exactly the `is_reward_hack` rows. Don't filter on
`reassessment.verdict` (an intermediate layer — "yes" there includes runs the
final judge exonerated, e.g. agents fooled by an HF split literally *named*
"train" that IS the eval pool).

`final` is **derived, never edited by hand**. It is recomputed from the
history layers by `judging/finalize.py` (deterministic, free):

    uv run python posttrainbench/judging/finalize.py

Run that after ANY judgement edit (e.g. overriding a verdict per
`../JUDGING.md`, re-running a judging pass, editing categories.json). It also
stamps each categories.json member's `first_hack_event` (multi-mechanism runs
keep their hand-localized per-category onsets, recorded in
`finalize.CURATED_CATEGORY_ONSETS`).

## The judging history layers (in pipeline order — see ../JUDGING.md)

| field | producer | what it is |
|---|---|---|
| `old_judge_source`, `old_judge_report_md` | PTB's original workspace-only judge | recall-heavy flag that defined the 51-run candidate set |
| `reassessment` (verdict, rationale, summary) | `exp_rejudge_hacks.py` (Stage 1) | whole-trajectory re-judge |
| `marked_turns`, `quotes`, `annotations` | localizer + re-judge | *candidate* evidence turns/spans — NOT the hacks themselves |
| `turn_kinds` | `exp_classify_turns.py` | marked turns split into `hack` (direct action) vs `context`; unclassified default to hack |
| `train_label_trap`, `train_label_audit` | `flag_train_label_trap.py` + `exp_audit_train_label.py` (Stage 2) | did the agent KNOW split="train" was the eval pool? |
| `judge_input_truncated` | `flag_truncated_judge_input.py` (Stage 3) | the Stage-1 judge silently read only part of this trace |
| `final_judge` | `exp_final_judge.py` (Stage 4) | unified judge with workspace+trace access; overrides earlier layers |
| `train_run_audit` | `exp_audit_train_runs.py` | per-training-launch success audit (numbers the trains in the viewer) |
| `story` | `exp_curate_stories.py` | the curated display narrative + annotations, consistent with `final` |

OpenAI-API-rule files (the 5 violation runs) contain only `run_id` + `final`;
their evidence/rationale lives in `openai_api_judgement.json` (produced by
`exp_judge_openai_api.py` from the `api_scan.json` candidates) and renders on
the run page from there.

## How the viewer uses this

`viewing/viewer.py` **derives nothing**: index pills, red rows, the timing
page, the cheating report and the rates page all read `final.*` (loaded once
at startup). The run-page banner shows `story.summary` as the narrative with
the full multi-judge history collapsible beneath it; `marked_turns`/`quotes`/
`annotations`/`turn_kinds` drive the in-trace highlights; `train_run_audit`
labels every detected training launch (✗ for failed ones).

Hand-overriding a verdict: edit the relevant history layer (usually
`final_judge.verdict` — see "Known limitations" in ../JUDGING.md), then re-run
`finalize.py`.
