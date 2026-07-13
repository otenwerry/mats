# Reconstruction issues tracker

One section per issue flag from the context-reconstruction work. The per-run tags
render in the **issues** column of the viewer's context-reconstruction window
(`/continuations/hacks`, legend at the bottom of that page); the universal caveats
apply to every probe and are listed after them.

**Maintenance rule: whenever an issue is fixed, investigated, or its status
changes, update this file AND the viewer tags/legend in the same change.**

Tag colors in the viewer (= status; own palette, distinct from the red/amber
contamination and fidelity flags — key shown at the top of the page): **purple**
permanent data loss, nothing will fix it · **blue** unresolved, still
investigating · **green** fixed in code — the tag survives only because that
run's existing continuation predates the fix, so re-running the continuation
clears it.

Status snapshot (2026-07-13): 30 reward-hack trajectories. 9 have completely
verbatim reconstructed contexts (Claude API runs whose visible window starts at a
compaction summary) — but no probe yet counts as a fully faithful end-to-end
resume (see Universal caveats).

---

## Per-run tags

### `nudge restarts ×N` (2 runs: both codex `*_reprompt`)

Restarts by the agents whose launch script contains the reprompt loop: when the
agent finished early, the script relaunched it with
"You still have Xh Ym remaining. Please continue improving your result and
maximize performance."

- **Known:** the wording, verbatim from `agents/codex_non_api_high_reprompt/solve.sh`
  (and siblings). Confirmed behaviorally: loop agents show 14–19 mid-run inits,
  non-loop codex agents show exactly one.
- **Known but not yet used:** the exact time values were never streamed, but these
  runs have line timestamps (`[2026-04-06T14:29:59Z]…`), so the true remaining
  time at each relaunch is recoverable arithmetically.
- **Moot for now:** codex can't be reconstructed anyway (see `edits lost`).
- **TODO (code, if codex ever becomes reconstructable):** wire timestamp-derived
  times into the reconstruction instead of estimates.

### `unexplained restarts ×N` (4 runs: claude bfcl run1 ×6, opus-1M ×1 and ×5, qwen ×22)

Mid-run agent restarts with **no known mechanism**. Evidence collected 2026-07-12/13:

- No relaunch loop exists for agents `claude` / `claude_non_api` / `qwen3max` in
  ANY branch of the harness repo (all 13 branches searched).
- The paper (arxiv 2603.08640) describes no relaunching — early stopping is
  listed as a limitation, and 20-hour runs were "discontinued" because agents
  stopped early. The reprompt agents are a post-paper addition.
- Post-restart replies discriminate: all 11 claude restart replies address
  **background-task completion notifications** ("task b92a6ea is stale, work is
  complete"); qwen's mostly hit **API errors immediately** (DashScope rate
  limit / input length), i.e. crash-retries. None reference time remaining.
- No trajectory anywhere echoes the nudge template.
- **Consequence:** `lib/recon_claude.py` currently inserts the nudge template (with
  an interpolated time estimate) at every mid-run init — for these 4 runs that is
  probably **wrong content**. Not yet changed; what to insert instead is an open
  design decision.
- **This is positive evidence of a different mechanism, not just caution:** the
  replies (12/12 address stale background tasks, 0 mention time), the scripts
  (loop agents restart 14–19×, non-loop codex agents exactly 0× — yet non-loop
  claude/qwen restarted), and the shape (restarts fire back-to-back, ~3 records
  apart, clustered after "done"; reprompt runs have real work between nudges).
- **Leading hypothesis (claude):** Claude Code itself continues the session to
  deliver "background task finished" notifications. Rival hypothesis: an
  external watchdog re-ran `claude --continue` (would unify with qwen's
  crash-retries — a crashed CLI can't re-enter itself).
- **Plausibly resolvable, not permanently uncertain:** if the notification
  format is a deterministic template, the injected content is reconstructable —
  the stale task ids are quoted in the replies and the original background jobs
  are in the stream.
- **TEST RESULT #1 (2026-07-13, CLI 2.1.207):** in `-p` mode the current CLI
  does NOT wait for background tasks — it killed the job ~5s after the final
  answer and exited with a single init. So CLI-internal re-entry can't produce
  the PTB pattern on today's CLI. Also: the run-era PTB streams contain NO
  task_* event subtypes (only init/status/compact_boundary), and the stale task
  ids the models cite (e.g. b92a6ea) enter their context via the ordinary
  launch tool_result — so the "notification" the models answered was injected
  at resume, not streamed.
- **TEST (script updated, re-run it):** `exp_restart_repro.py` now runs two
  phases — the background-job run, then `claude --continue` on the same session
  (exactly the external-watchdog move) — and dumps whatever the resume injects.
  Run once with the current CLI and once with
  `--cli "npx @anthropic-ai/claude-code@2.1.76"` (run-era). The injected format,
  if any, is what reconstruction should insert at PTB restart points.
- **Extra unfaithfulness in the existing probes:** the 3 restart runs' pre-fix
  probe contexts ENDED on the synthetic nudge text (their `--turn last` cut fell
  right after a relaunch). `--turn end` re-runs fix the ending; the mid-context
  inserts stay until the mechanism is known.
- **OWEN:** ask Maksym what relaunched the non-reprompt agents (and whether some
  infra-level retry wrapper existed outside the repo).

### `task prompt rebuilt` (grey; now tags only runs with a pre-fix continuation — currently the 2 opus-1M rows)

The initial task prompt was a CLI argument, never recorded in any stream. Runs
that never compacted keep it in every reconstructed context, so we regenerate it
with the harness's `get_prompt.py`.

- **Drift measured (2026-07-13):** `prompt.txt` changed exactly once in the
  window our runs span — commit `aebea2f` (2026-04-04), the same commit that
  added line timestamps. So: no timestamps in a trace ⇒ pre-April harness ⇒ the
  true prompt said "equiped" (typo) where the current template says "equipped".
  That one word is the **entire** difference (both template and generator diffed).
- **FIXED (2026-07-13):** `lib/prompts.py` now era-matches — runs without line
  timestamps get the original "equiped" wording restored, making regeneration
  byte-exact for the public-harness window. Reconstructions stamp a
  `prompt_era_matched` fidelity flag; the viewer tag survives only on runs whose
  existing continuation predates the fix (re-run clears it).
- Residual risk: the run-time harness might not match any public commit; one
  saved `prompt.txt` from Maksym would settle it completely.

### `edits lost` (all 7 codex runs)

Codex traces record only `{path, kind}` per file edit — the patch bodies were
never saved. This is the model's own *output*, so the context cannot be rebuilt.

- **Fundamental.** The rollout files (codex's resumable session state) lived in
  the container's temp home dir and were destroyed at job cleanup — verified in
  `run_task.sh`: only `task/` and named files are copied out. Not in the HF
  dataset; not a download problem.

### `reasoning lost` (7 codex + 3 opencode thinking-model runs)

Reasoning IS replayed within a tool-call chain (all providers) — and a 10-hour
agent run is essentially one giant chain — so mid-run cuts need it. Between user
turns it is dropped (Claude and OpenAI alike).

- **claude:** nothing lost — the stream holds the exact replayable objects
  (thinking summary + signature; that pair is all any client ever replays).
  Interpretability caveat only: the text is a summary, not raw CoT.
- **codex:** the replayable (encrypted) reasoning items lived in the destroyed
  rollout files. Fundamental, same as `edits lost`.
- **opencode:** the stream saves no reasoning at all. Tagged only for the three
  thinking-model runs (kimi-k2-thinking, gpt-5.1-codex-max, gemini-3.1-pro).
- **OPEN QUESTION:** did the opencode runtime replay reasoning within the chain
  during the original runs? If it never sent reasoning back, its resumes are
  accidentally faithful (the original model didn't see prior reasoning either).
  Check opencode's request-building code / docs for the kimi + OpenAI providers.
- **ASSUMPTION TO VERIFY (Owen or quick research):** kimi-k2.5 treated as
  NON-thinking (its 5 rows untagged). If it has a hidden reasoning channel,
  those rows need the tag.

### `effort not replicated` (4 claude_non_api runs)

Original subscription runs passed `--effort high`; the first probe campaign
resumed at default effort.

- **FIXED in code (2026-07-13):** `exp_probe_context.py` now passes the original
  agent's effort (`--effort high` for claude_non_api,
  `CLAUDE_CODE_EFFORT_LEVEL=max` for claude_non_api_max) and stamps an
  `effort_replicated` fidelity flag. The viewer tag clears automatically for
  post-fix probes.
- **TODO (paid, small):** re-run the 4 pre-fix probes to clear the tags.

---

## Universal caveats (every probe; not tagged per run)

### Resume-injected turns — MOSTLY FIXED for end-of-run probes (2026-07-13)

`claude --resume` on a context ending in a tool result injects a synthetic user
turn ("Continue from where you left off.") plus an assistant reply ("No response
requested."). Not our bug — it's the CLI's bridging behavior for resuming
mid-agentic-turn — but it fired on our probes because `--turn last` deliberately
cut *before* the final assistant message (a definition inherited from the
resample use-case).

- **FIXED for end probes:** new `--turn end` cuts *after* the final assistant
  message — the session ends on an assistant turn, the CLI appends the probe
  directly, no bridging, and the agent's real closing summary stays in context.
  The campaign runner now uses it.
- **Structural remainder:** 7 of the 15 claude/qwen reward-hack runs genuinely
  died mid-tool (final assistant message has an unanswered tool call), so their
  end cut snaps back and bridging is unavoidable — that's the trajectory's real
  shape, recorded in the cut notes. Bridging also returns at any mid-trajectory
  tool-result cut in future resample experiments: there it's a structural cost
  of native CLI resume that intervention design must account for.
- Injected turns remain recorded per probe (`resume_turns.jsonl`) and rendered
  on continuation pages.

### Local CLI attachments — FIXED in code (2026-07-13), pending one validation probe

The local Claude Code attaches its own system-reminder content to the probe turn
(tool/agent/skill listings from Owen's machine — visible in `resume_turns.jsonl`,
noted on continuation pages). The original agents never saw these.

- **Perfectly fixable — subtractive, not generative:** the contamination is
  *added* personal config, so removing it requires inventing nothing, and it
  only ever touched the new probe turn (original turns are verbatim regardless).
- **FIXED:** `exp_probe_context.py` now defaults to a fresh per-probe
  `CLAUDE_CONFIG_DIR` (auth via `ANTHROPIC_API_KEY`; `--user-config` restores
  the old behavior). Flags: `probe_config_clean` / `probe_config_personal`.
- **Residual, bounded, known (not guesses):** CLI built-in listings (killable by
  pinning the run's original CLI version — still on npm) and the regenerated
  system prompt's environment block (cwd/platform/date — the original cwd
  `/home/ben/task` is recorded in init events; fully closing this overlaps with
  the deferred workspace-reconstruction phase).
- **TODO:** validate on ONE probe via `resume_turns.jsonl` before the batch
  re-run (also confirms clean-config auth works and, if pinning, that the old
  CLI resumes cleanly).

### CLI version drift

Probes run CLI 2.1.207; originals used 2.1.9–2.1.76 → different system prompt,
tool schemas, current date. Deprioritized by Owen (recorded for exhaustiveness);
mostly eliminated for free by version pinning above.

### World/workspace mismatch

The conversation references a container workspace, files, GPU, running jobs; the
probe resumes in an empty scratch dir. Fine for no-tool probes; the main blocker
for real resample experiments. Workspace reconstruction is the deferred phase 2.

---

## Can't-reconstruct rows (greyed in the viewer)

| scaffold | verdict | what it would take |
|---|---|---|
| codex (7) | **fundamental** — patch bodies + rollouts destroyed | nothing recoverable; approximate-only, permanent flag |
| opencode (8) | logistics | install opencode CLI + provider auth for the original models; reasoning caveat above for 3 rows |
| qwen3max (1) | logistics | it's Claude Code + DashScope endpoint: needs a DashScope key + env overrides (**OWEN:** obtain key if we want this row) |

---

## Open questions / actions, collected

- [ ] **Owen → Maksym:** what relaunched the non-reprompt agents? (unexplained restarts)
- [ ] **Owen → Maksym:** one saved `prompt.txt` from any pre-April run (settles task-prompt drift completely)
- [x] ~~decide era-exact prompt regeneration~~ — DONE 2026-07-13 (`lib/prompts.py` era-matches; `prompt_era_matched` flag)
- [ ] **Owen:** decide what recon should insert at unexplained restarts instead of the nudge template
- [ ] **Owen:** DashScope key if we want the qwen3max row resumable; opencode CLI + provider auth if we want the 8 opencode rows
- [ ] **Test (exp approval, ~$0.05):** `exp_restart_repro.py` — background-task → re-entry hypothesis for claude restarts (also try `--cli` pinned to 2.1.76)
- [ ] **Test (paid, one probe):** validate clean-config + `--turn end` on one trajectory via `resume_turns.jsonl`, then batch `exp_run_context_recon.py --rerun` — clears all green tags and removes bridging/attachments for the 8 runs that end on an assistant message
- [x] ~~clean CLAUDE_CONFIG_DIR implementation~~ — DONE 2026-07-13 (default in exp_probe_context; `--user-config` opts out); CLI version pinning still optional/untested
- [ ] **Research:** does opencode replay reasoning mid-chain? is kimi-k2.5 a thinking model?
