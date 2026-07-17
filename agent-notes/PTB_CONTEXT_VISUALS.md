# PostTrainBench context visuals

> **Covers:** the Trajectories / visuals page for reward-hack timing versus context usage and compaction.
> **Read when:** changing PTB token accounting, hack-turn annotations, context-window capacities, compaction parsing, or the main viewer visuals.
> **Last updated:** 2026-07-16

The main viewer's `/trajectory-visuals` page is built by
`posttrainbench/lib/context_visuals.py`. It emits one coverage record for every
confirmed reward-hack trajectory; unsupported records are not dropped.

Token definitions:

- Claude: `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`
  from one unique main-thread message id.
- OpenCode: `input + cache.read` from each `step_finish`. `total` is not used
  because it also includes output and reasoning.
- Codex: unavailable. Its `turn.completed` records are cumulative totals across
  many model requests, not per-call context sizes.
- Qwen: unavailable. Its assistant messages saved zero input tokens; session
  result totals cannot localize usage to a hack event.

Normal Claude runs used the 200k setup; only run-directory names containing
`_1m` use 1,000,000. OpenCode capacities match its v1.1.59 model limits:
Gemini 3.1 Pro 1,048,576; GPT-5.1 Codex Max 400,000; Kimi models 262,144.

Viewer event numbers remain the reference frame. An assistant annotation maps
to its containing call. A user tool-result annotation maps to the immediately
preceding call that issued the tool action. The canonical `first_hack_event` is
always included, along with every `turn_kinds` entry explicitly classified as
`hack`.

Compaction markers use saved Claude `system/compact_boundary` events only.
No compaction is inferred from a token drop. Calls since compaction count the
first model call after the boundary as 1. None of the eight OpenCode hack runs
reached the run-era automatic-compaction threshold.

The page renders all supported timelines and exact backing tables: there is no
sampling or top-N cap. Coverage rows expose `complete`, `partial`, or
`unavailable` in `data-context-coverage`, with a per-trajectory omission reason.
