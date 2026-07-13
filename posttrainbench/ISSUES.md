# Reconstruction issues tracker

One section per issue flag from the context-reconstruction work. The per-run tags
render in the **issues** column of the viewer's context-reconstruction window
(`/continuations/hacks`, legend at the bottom of that page); the universal caveats
apply to every probe and are listed after them. Kept current as findings land —
each section records what we know, what's still open, and who owes what.

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
- **Leading hypothesis (claude):** Claude Code itself continues the session to
  deliver "background task finished" notifications — inits cluster right after
  the agent declared itself done, and injected turns wouldn't be streamed.
- **TEST (cheap, needs exp approval):** run `claude -p` locally with a
  backgrounded sleep task; check whether the stream emits a second `init` after
  the task completes. Confirms/kills the hypothesis for ~$0.05.
- **OWEN:** ask Maksym what relaunched the non-reprompt agents (and whether some
  infra-level retry wrapper existed outside the repo).

### `task prompt rebuilt` (11 runs: never-compacted claude/qwen + all opencode)

The initial task prompt was a CLI argument, never recorded in any stream. Runs
that never compacted keep it in every reconstructed context, so we regenerate it
with the harness's `get_prompt.py`.

- **Drift is now measured, not hypothetical (2026-07-13):** `prompt.txt` changed
  exactly once in the window our runs span — commit `aebea2f` (2026-04-04), the
  same commit that added line timestamps. So: no timestamps in a trace ⇒
  pre-April harness ⇒ the true prompt said "equiped" (typo) where our
  regeneration says "equipped". That one word is the **entire** difference (both
  template and generator diffed).
- **DECISION (Owen):** whether to make regeneration era-exact (a two-line change
  to `lib/prompts.py`: use the old wording for runs without line timestamps).
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

### Resume-injected turns

`claude --resume` on a context ending in a tool result injects a synthetic user
turn ("Continue from where you left off.") plus an assistant reply ("No response
requested.") before the probe question; cuts ending on an assistant message get
just the assistant reply. Recorded per probe in `resume_turns.jsonl` and rendered
on continuation pages (2026-07-12). Must be accounted for in any
intervention/resample design.

### Local CLI attachments

The local Claude Code attaches its own system-reminder content to the probe turn
(tool/agent/skill listings from Owen's machine — visible in `resume_turns.jsonl`,
noted on continuation pages). The original agents never saw these.

- **Fix recipe for the next campaign (not yet implemented):**
  1. run probes with a fresh `CLAUDE_CONFIG_DIR` + `ANTHROPIC_API_KEY` auth →
     removes everything personal;
  2. optionally pin the run's original CLI version
     (`@anthropic-ai/claude-code@2.1.9` etc., still on npm) → removes the
     version-baked built-ins too, and kills most system-prompt drift;
  3. verify on ONE probe via `resume_turns.jsonl` before running the batch
     (also checks that the old CLI resumes cleanly).

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
- [ ] **Owen:** decide era-exact prompt regeneration ("equiped" typo) — worth it?
- [ ] **Owen:** decide what recon should insert at unexplained restarts instead of the nudge template
- [ ] **Owen:** DashScope key if we want the qwen3max row resumable; opencode CLI + provider auth if we want the 8 opencode rows
- [ ] **Test (exp approval, ~$0.05):** background-task → second init hypothesis for claude restarts
- [ ] **Test (paid, small):** re-run the 4 claude_non_api probes with effort replication
- [ ] **Next campaign prep:** clean CLAUDE_CONFIG_DIR (+ optional CLI version pinning), verified on one probe first
- [ ] **Research:** does opencode replay reasoning mid-chain? is kimi-k2.5 a thinking model?
