# Reconstruction issues tracker

> **Covers:** the PostTrainBench context-reconstruction issue tracker — per-run fidelity tags (nudge restarts, background restarts, edits lost, reasoning lost, effort, task-prompt rebuild, resume turns) and the universal caveats. Ties to `lib/recon_claude.py`, `exp_ask_questions.py`, `lib/prompts.py`, and the viewer's issues column on `/continuations/hacks`.
> **Read when:** doing PTB context-reconstruction / probe work, or reading a run's reconstruction-fidelity flags.
> **Last updated:** 2026-07-15
> **NB (2026-07-15):** `exp_probe_context.py` and `exp_run_context_recon.py` are RETIRED — the probe now runs as `exp_ask_questions.py --questions=context` (DEFAULT_PROBE + resolve_cli moved there verbatim). Mentions of the old files below are historical; the fixes they describe live on in the shared recon/ask code paths.
> **Original path:** posttrainbench/ISSUES.md
> **Note:** the file's own rule — when an issue changes, update this file AND the viewer tags/legend in the same change.

One section per issue flag from the context-reconstruction work. The per-run tags
render in the **issues** column of the viewer's context-reconstruction window
(`/continuations/hacks`; the tag legend sits at the top of that page, closed by
default); the universal caveats apply to every probe and are listed after them.

**Maintenance rule: whenever an issue is fixed, investigated, or its status
changes, update this file AND the viewer tags/legend in the same change.**

Tag colors in the viewer (= status; own palette, distinct from the red/amber
contamination and fidelity flags — key shown at the top of the page): **purple**
permanent data loss, nothing will fix it · **blue** unresolved, still
investigating · **teal** structural — an unavoidable artifact of resuming this
trajectory's shape, not data loss and not fixable, but known and recorded ·
**green** fixed in code — the tag survives only because that run's existing
continuation predates the fix, so re-running the continuation clears it.

Status snapshot (2026-07-13): 30 reward-hack trajectories. 11 have completely
verbatim reconstructed contexts (visible window starts at a compaction summary,
no mid-run restarts: 9 API runs + the 2 subscription-sonnet runs #22 #23) — but
no probe yet counts as a fully faithful end-to-end resume (see Universal
caveats). The pre-fix systematic probes were reclassified into the viewer's
**early tests** window on 2026-07-13; the current context-reconstruction campaign
is in `probes_context_recon_current/`. Its first batch attempted 13 Claude
trajectories: 11 completed with sensible context reports and are marked
**approximately reconstructed** in the viewer; 2 subscription-Sonnet probes
failed before inference because their run-era CLI rejects `--effort` (fixed in
code; retry pending). “Approximately” retains the universal local-environment /
missing-workspace caveat and, on 7 rows, the teal structural resume bridge.

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

### `background restarts ×N` (4 runs: claude bfcl run1 ×6, opus-1M ×1 and ×5, qwen ×22)

**MECHANISM CONFIRMED (2026-07-13, three-part repro + crosscheck):** the run-era
CLI (tested: 2.1.76) **waits for background tasks in `-p` mode and re-enters the
session in-process once per completed task**, emitting a new `system:init` under
the same session id and injecting a plain (non-meta, unstreamed) user turn:

    <task-notification>
    <task-id>b5z23std2</task-id>
    <tool-use-id>toolu_01WWJ7GEq8GHu68GeCPFswb1</tool-use-id>
    <output-file>/private/tmp/claude-.../tasks/b5z23std2.output</output-file>
    <status>completed</status>
    <summary>Background command "sleep 45 && echo BACKGROUND_DONE" completed (exit code 0)</summary>
    </task-notification>
    Read the output file to retrieve the result: <output-file>

Crosscheck against the PTB streams: background launches match re-entries in
every affected run — bfcl run1 **6/6**, opus-1M run3 **5 re-entries / 6
launches** (one presumably finished in-session), qwen **22/22**. Qwen's
API-error replies were the re-entry calls hitting DashScope rate limits — same
mechanism, no external watchdog anywhere. (The current CLI behaves differently:
it kills background tasks ~5s after the final answer, which is why the first
repro pass pointed the wrong way.)

**PARTIALLY FIXED (2026-07-13) — 12 of 34 inserts hand-reviewed and wired in
(11 fully + 1 with a permanent gap); 22 deferred:**

- **Reviewed + reconstructed (green tag):** bfcl run1 (6) and opus-1M run3 (5).
  Manual review matched each restart to its background job via the ids/content
  the replies quote (notification order matched launch order in every checkable
  case). Assignments live in `restart_assignments.json` (verbatim
  task-id/tool-use-id/command/output-file from the streams + inferred
  status/exit-code + per-entry evidence and confidence); `recon_claude` builds
  the `<task-notification>` turn from each entry (provenance
  `reconstructed_notification`, flag `restart_notification_reconstructed`).
  Known caveats, flagged: exit codes inferred (2 failures identified from
  replies, rest assumed 0); one low-confidence entry (bfcl's vLLM-server task,
  assigned by elimination). The failed-task summary wording was VERIFIED by
  the `--fail` repro (2026-07-13): `failed with exit code 1` — our earlier
  guess had the wrong format ("failed (exit code 1)"), fixed in recon_claude.
- **Reconstructed with a permanent gap (purple tag, shown in the viewer as
  `missing helper agent text` — renamed 2026-07-13 for readability; the row
  itself moved to the page's deferred section, Owen 2026-07-13):**
  opus-1M run1's (#15) single restart. The `--agent-task` repro (2026-07-13, CLI 2.1.76) pinned the
  agent-task notification wording (summary `Agent "<description>" completed` +
  `<result>` + `<usage>` + "Full transcript available at:" footer). All
  surviving fields are inserted verbatim (agentId, tool-use-id, output-file,
  description), but the `<result>` text — the subagent's actual report — and
  the `<usage>` numbers were **never recorded**: background-subagent
  transcripts are absent from claude streams (foreground sidechains ARE
  recorded; the transcript file died with the container). The reconstruction
  omits those lines and flags them (`restart_notification_reconstructed`,
  severity warn). **NEW data-loss class worth remembering:** any run using
  background subagents has this hole.
- **Deferred (blue tag), Owen's call still open:** qwen's 22 (replies are API
  errors — nothing to review; assignments would be launch-order guesses).
  Keeps the old time-nudge insert with a loud KNOWN-WRONG flag
  (`continuation_prompt_estimated`).
- The failed-task wording test RAN (`--fail` mode, Owen, 2026-07-13): the
  run-era CLI says `Background command "..." failed with exit code 1` (no
  parentheses — our guessed format was wrong; fixed in recon_claude). All
  three notification wordings are now repro-verified.

Original evidence trail (2026-07-12/13, kept for the record):

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
- **Consequence (since addressed):** `lib/recon_claude.py` used to insert the
  nudge template (with an interpolated time estimate) at every mid-run init —
  wrong content for these 4 runs. It now inserts reconstructed notifications
  where reviewed (see above); only deferred restarts still get the nudge, loudly
  flagged.
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
- **TEST RESULT #2 (2026-07-13) — intermediate conclusion, SUPERSEDED by the
  2.1.76 repro (see the section header: the run-era CLI re-enters by itself;
  no external relauncher).** Running
  `claude --continue` on a session with a pending background task reproduced
  the full PTB signature in one shot: multiple `system:init` events under one
  session id, an injected user turn that does NOT appear in the stream, and the
  model replying about the stale task. The injected turn's format (CLI 2.1.207):

      <task-notification>
      <task-id>bstr6vigs</task-id>
      <tool-use-id>toolu_...</tool-use-id>
      <status>stopped</status>
      <summary>No completion record was found for this background shell
      command from the previous session. ...</summary>

  So this pass concluded the PTB restarts = something external re-ran
  `claude --continue`, with the CLI injecting task-notifications at resume.
  (Both then-open unknowns were since resolved by the 2.1.76 repro: the run-era
  wording is captured verbatim, and there was no external relauncher — hence no
  relauncher prompt text to recover. Note for pinned-CLI runs: use
  `npx -y`, plain npx hangs on a hidden install prompt.)
- **Extra unfaithfulness in the existing probes:** the 3 restart runs' pre-fix
  probe contexts ENDED on the synthetic nudge text (their `--turn last` cut fell
  right after a relaunch). `--turn end` re-runs fix the ending; the mid-context
  inserts stay until the mechanism is known.
- ~~**OWEN:** ask Maksym what relaunched the non-reprompt agents~~ — resolved,
  no Maksym needed (the CLI relaunched itself).

### `extra resume turns` (teal on the 7 mid-action enders: #1 #3 #5 #7 #9 #10 #13; green on clean enders with a pre-fix probe)

Added 2026-07-13 (Owen: structural resume artifacts should be tagged per run,
in their own category/color). Resuming a context that ends on an unanswered
tool call makes the CLI perform a synthetic bridge of its own ("Continue from
where you left off." / "No response requested.") before our probe question.
The fresh run-era 2.1.9 probes persist only the assistant half in the session
file; the continuation viewer now anchors on the exact probe question and shows
that turn without misclassifying old records that the CLI re-appends/normalizes.

- **Teal (structural):** the 7 runs above genuinely died mid-action — their
  final assistant message left a tool call unanswered, so the end cut snaps to
  before it and the insert is unavoidable. This is the trajectory's real ending
  shape, not a reconstruction bug. Stored per reconstruction as the
  `resume_bridging_structural` fidelity flag (fires whenever a rebuilt context
  ends on a tool_result); the injected turns themselves are saved in each
  probe's `resume_turns.jsonl` and rendered on the continuation page.
- **Green:** a clean-ending run whose existing probe predates the `--turn end`
  fix also got the inserts (the old `--turn last` cut fell before the final
  message); re-running clears those.
- Mid-trajectory cuts at tool results (future resample experiments) hit the
  same inserts — intervention design must account for them.

### `task prompt rebuilt` (green; tags only runs with a pre-fix continuation — currently #15 and #20)

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

### `reasoning lost` (7 codex + ALL 8 opencode runs — purple)

Reasoning IS replayed within a tool-call chain (all providers) — and a 10-hour
agent run is essentially one giant chain — so mid-run cuts need it. Between user
turns it is dropped (Claude and OpenAI alike).

- **claude:** nothing lost — the stream holds the exact replayable objects
  (thinking summary + signature; that pair is all any client ever replays).
  Interpretability caveat only: the text is a summary, not raw CoT.
- **codex:** the replayable (encrypted) reasoning items lived in the destroyed
  rollout files. Fundamental, same as `edits lost`.
- **opencode — RESOLVED 2026-07-13 (source-read of v1.1.59 at its tag, plus its
  pinned AI-SDK packages; clone in references/opencode):**
  - opencode STORED reasoning parts locally and REPLAYED them into every
    subsequent request — kimi models as plaintext `reasoning_content`, gemini
    as thought parts + signatures, gpt as encrypted reasoning items. A missing
    reasoning part is silently tolerated at resume.
  - The runs' stdout (`--format json` WITHOUT `--thinking`) skipped reasoning
    parts by construction — so the reasoning existed in the containers'
    storage and was simply never captured. The containers are gone: PERMANENT.
    Resumes from our rebuilt storage send less context than the original
    runtime did. Tag upgraded blue → purple.
  - ~~ASSUMPTION: kimi-k2.5 non-thinking~~ — WRONG: kimi-k2.5 is a reasoning
    model, thinking ON by default (models.dev `reasoning: true` today and in
    the run-era catalog; Moonshot's model card). Its 5 rows now carry the tag
    too → all 8 opencode rows, not 3.
  - Possible partial mitigation, unexplored: the printer emitted full part
    JSON for text/tool parts incl. metadata, so gemini thought-signatures and
    openai reasoning item-ids attached to TEXT/tool parts may be recoverable
    from our streams (the reasoning TEXT itself is not).

### `effort not replicated` (pre-fix probes for 2 subscription-Opus runs)

Historical correction from the harness git history (2026-07-13): subscription
effort was not uniform. `--effort high` was added to
`agents/claude_non_api/solve.sh` on 2026-03-26. The Sonnet runs used CLI 2.1.34
and predate that change, so their original setting was the CLI default. The
later Opus-1M runs used CLI 2.1.76 and did pass `--effort high`.

- **FIXED in code (2026-07-13):** `exp_probe_context.py` era-matches both cases
  (2.1.34/default; 2.1.76/`--effort high`) and stamps an `effort_replicated`
  fidelity flag. An unknown subscription CLI version now fails closed instead
  of silently guessing.
- **FIRST CURRENT-BATCH RESULT:** the one non-deferred Opus-1M probe completed
  successfully at high effort. Both Sonnet probes failed before inference
  because the initial implementation incorrectly passed the unsupported
  `--effort high` flag to 2.1.34. The campaign runner now treats a fidelity file
  without a model answer as incomplete, so the same command retries those two.
- The other Opus-1M run is individually deferred for its permanently missing
  helper-agent report; its old probe remains in **early tests**.

---

## Universal caveats (every probe; not tagged per run)

### Resume-injected turns — no longer universal; now the per-run `extra resume turns` tag (2026-07-13)

`claude --resume` on a context ending in a tool result injects a synthetic user
turn ("Continue from where you left off.") plus an assistant reply ("No response
requested."). Not our bug — it's the CLI's bridging behavior for resuming
mid-agentic-turn — but it fired on every first-campaign probe because
`--turn last` deliberately cut *before* the final assistant message (a
definition inherited from the resample use-case).

- **FIXED for end probes:** new `--turn end` cuts *after* the final assistant
  message — the session ends on an assistant turn, the CLI appends the probe
  directly, no inserts, and the agent's real closing summary stays in context.
  The campaign runner now uses it.
- **Structural remainder — tagged per run since 2026-07-13:** see the
  `extra resume turns` section above (teal on the 7 mid-action enders,
  `resume_bridging_structural` fidelity flag; green where only a pre-fix probe
  is affected).
- Injected turns remain recorded per probe (`resume_turns.jsonl`) and rendered
  on continuation pages.

### Local CLI attachments — FIXED and batch-validated (2026-07-13)

The local Claude Code attaches its own system-reminder content to the probe turn
(tool/agent/skill listings from Owen's machine — visible in `resume_turns.jsonl`,
noted on continuation pages). The original agents never saw these.

- **Perfectly fixable — subtractive, not generative:** the contamination is
  *added* personal config, so removing it requires inventing nothing, and it
  only ever touched the new probe turn (original turns are verbatim regardless).
- **FIXED:** `exp_probe_context.py` now defaults to a fresh per-probe
  `CLAUDE_CONFIG_DIR` (auth via `ANTHROPIC_API_KEY`; `--user-config` restores
  the old behavior). Flags: `probe_config_clean` / `probe_config_personal`.
- **Residual, bounded, known (not guesses):** CLI built-in listings (now
  handled — probes pin the run's original CLI version by default, see "CLI
  version drift") and the regenerated system prompt's environment block
  (cwd/platform/date — the original cwd `/home/ben/task` is recorded in init
  events; fully closing this overlaps with the deferred
  workspace-reconstruction phase).
- **VALIDATED:** all 11 successful current-campaign probes used a clean config;
  none recorded a local attachment on the probe turn.

### CLI version drift — FIXED and batch-validated (2026-07-13)

Originals used CLI 2.1.9 / 2.1.34 / 2.1.76; the first probe campaign ran the
local 2.1.207 → different system-prompt instructions, tool schemas, CLI
behavior.

- **FIXED:** `exp_probe_context.py` now pins the run's own CLI version by
  default (`--cli original` → `npx -y @anthropic-ai/claude-code@<ver>`;
  `--cli local` restores the old behavior). Flags: `probe_cli_pinned` /
  `probe_cli_version_drift`. All three run-era versions invoke fine via npx
  (verified) and use the SAME session-dir naming as today (read from their
  cli.js bundles — an earlier note claiming older versions named dirs
  differently was wrong).
- **What pinning does NOT fix:** the system prompt's `<env>` block (working
  directory, platform, OS version, today's date) is generated at runtime by
  every CLI version — verified in the 2.1.34 bundle. A pinned probe still
  says Owen's machine + today's date, not the container + April. That
  residual belongs to the deferred workspace-reconstruction phase.
- **VALIDATED:** 11 successful current-campaign probes ran across the pinned
  run-era versions (2.1.9, 2.1.34, and 2.1.76). Two additional 2.1.34 Sonnet
  probes reached the CLI but rejected the initially wrong `--effort` option;
  their retry is tracked in the effort section above.

### World/workspace mismatch

The conversation references a container workspace, files, GPU, running jobs; the
probe resumes in an empty scratch dir. Fine for no-tool probes; the main blocker
for real resample experiments. Workspace reconstruction is the deferred phase 2.

---

## Can't-reconstruct rows (greyed in the viewer; codex + qwen + #15 live below the "deferred" divider on /continuations/hacks, boxes closed by default; opencode stays in the top section as planned work)

| scaffold | verdict | what it would take |
|---|---|---|
| codex (7) | **fundamental** — patch bodies + rollouts destroyed | nothing recoverable; approximate-only, permanent flag |
| opencode (8) | logistics | CLI DONE (pinned 1.1.59 in mats-local/tools/, 2026-07-13); remaining: an opencode zen key (`OPENCODE_API_KEY` in mats/.env — all 8 runs used the zen gateway). Permanent `reasoning lost` caveat on all 8 rows; the kimi-k2-thinking model is deprecated on zen (may refuse) |
| qwen3max (1) | logistics | it's Claude Code + DashScope endpoint: needs a DashScope key + env overrides (**OWEN:** obtain key if we want this row) |

---

## Open questions / actions, collected

- [x] ~~what relaunched the non-reprompt agents~~ — RESOLVED 2026-07-13: the run-era CLI itself (background-task re-entry; see `background restarts`). No Maksym needed.
- [ ] **Owen → Maksym:** one saved `prompt.txt` from any pre-April run (settles task-prompt drift completely)
- [x] ~~background-restarts reconstruction fix~~ — 12/34 reviewed + wired in 2026-07-13 (restart_assignments.json; run #15's agent-task wording pinned by the --agent-task repro, its subagent report permanently lost → purple); **still open:** qwen's 22, deferred by Owen — revisit when qwen becomes resumable
- [x] ~~decide era-exact prompt regeneration~~ — DONE 2026-07-13 (`lib/prompts.py` era-matches; `prompt_era_matched` flag)
- [ ] **Owen:** DashScope key if we want the qwen3max row (#54) resumable
- [x] ~~`exp_restart_repro.py` re-entry test~~ — DONE 2026-07-13 (three repro runs + `--agent-task`; mechanism confirmed, wordings captured)
- [x] ~~pin the FAILED-task notification wording~~ — DONE 2026-07-13 (Owen ran `--fail` on 2.1.76): `failed with exit code 1`, guessed format was wrong, recon_claude fixed
- [ ] **Test (paid, 2 retries):** rerun `exp_run_context_recon.py --scaffold claude`; the 11 completed probes are skipped and only the two failed subscription-Sonnet probes retry. The deferred #15 remains excluded by shared campaign policy.
- [x] ~~clean CLAUDE_CONFIG_DIR implementation~~ — DONE 2026-07-13 (default in exp_probe_context; `--user-config` opts out)
- [x] ~~CLI version pinning~~ — DONE + batch-validated 2026-07-13 (default `--cli original` via npx; 11 successful current-campaign probes used their run-era versions)
- [x] ~~does opencode replay reasoning mid-chain? is kimi-k2.5 a thinking model?~~ — BOTH RESOLVED 2026-07-13 (v1.1.59 source-read): yes it replays (three mechanisms), and yes kimi-k2.5 thinks by default → `reasoning lost` is purple on all 8 opencode rows
- [ ] **Owen:** create an opencode zen API key (sign in at https://opencode.ai/auth, add billing, copy the key) and add `OPENCODE_API_KEY=...` to mats/.env — unlocks the 8 opencode rows (with the permanent reasoning caveat; the kimi-k2-thinking row may refuse — that model was deprecated on zen 2026-03-06)
