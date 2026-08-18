# Environment batch and overnight wrappers (2026-08-15)

> **Summary:** Use this checklist before writing any custom script that launches,
> resumes, or retries several real-environment AWS campaigns. The 2026-08-15 incident
> showed that correct per-campaign commands are not enough: a wrapper must preserve
> the account-wide VM budget, share refreshable AWS credentials safely across controller
> processes, stream controller progress, retain exact recovery state, and distinguish a
> controller failure from a trajectory integrity failure. Read this together with
> `REAL_ENV_GUIDE.md`. **Last updated:** 2026-08-15

## Required shape

- A paid wrapper is named `exp_*.sh` or `exp_*.py`. It has a free `--plan`/`--dry-run`
  mode that validates every stored input and prints exact cell counts, per-controller VM
  caps, harnesses, and commands without AWS calls, model calls, or filesystem writes.
- Pin the exact matrix in arrays/data rather than discovering “latest” campaigns or
  prefixes at run time. Revalidate stored campaign identities, payload hashes, harnesses,
  terminal states, and expected retry counts before spend.
- Sum every simultaneously active `--vm-concurrency` cap and fail unless it is within
  the current account-wide allowance. Partition subscription concurrency separately by
  account/model constraints. A per-controller cap is not an account-wide cap.
- Run AWS setup once before starting child controllers. Use the ordinary `mats-run`
  profile path; never export access keys into the wrapper or snapshot temporary
  credentials into environment variables. `exp_aws_trajectory.aws_clients` now routes
  every controller through `tools/aws_credential_process.py`, whose file lock and
  mode-0600 cache make refresh single-flight across processes.
- Stagger child-controller starts to reduce EC2 launch and polling bursts. Use the
  controller's bounded EC2 launch-throttle retries; do not add an unbounded shell retry.
- Give every child `PYTHONUNBUFFERED=1`, prefix each live output line with its campaign
  label, tee it to a dedicated log, and print a periodic aggregate heartbeat while any
  child remains alive. The user should see VM launch, result import, and termination
  messages from the normal pipeline controller in real time.
- Pass `--skip-viewer` to children and run `uv run viewer.py` once after all children
  exit. Record each child exit status and the viewer status in one final JSON manifest.
- Use a run-specific log directory, an active-run lock, and—when repeating the exact
  slate would duplicate paid work—a durable started marker. Keep the wrapper awake on
  macOS because controllers must launch later VM waves, import results, and clean S3.
- A child failure must not cancel unrelated campaigns. It also must not be retried
  blindly. Subscription quota failures, provider errors, wall-clock outcomes, judge
  failures, and controller/auth failures require different recovery choices.

## Recovery semantics

- `--resume-campaign` monitors/imports an existing campaign and never relaunches work.
  Use it first after a controller crash or expired host login; workers continue without
  the controller.
- `--retry-failed` normally selects never-launched and infrastructure-failure cells.
  Add `--retry-pipeline-failures` only after inspecting the parent and intentionally
  deciding to rerun completed transport cells with nonzero pipeline exits.
- A pipeline exit of 1 is not synonymous with lost evidence. It can accompany a saved
  trajectory that failed benchmark integrity. Preserve/import first; classify second.
- Do not automatically retry subscription rate-limit/session-limit failures. Let that
  campaign exit, keep the others moving, and form an exact-cell retry after the account
  reset if the user wants one.
- After any recovery, verify terminal campaign counts, S3 cleanup state, local imported
  `.eval` counts, integrity-sidecar counts, and viewer indexing. Rebuild the viewer only
  after imports are stable.
- The 2026-08-15 continuation push has a pinned resume-only salvage entry point at
  `environments/recover_continuation_push_20260815.py`. It excludes the cancelled
  multi-agent ML campaigns and never launches or retries a cell. Run this import pass
  before constructing any exact retry for that incident.

## 2026-08-15 incident record

The first `exp_tonight_20260815.sh` launched 19 independent controllers at once. Its
problems were hidden child output, a simultaneous control-plane burst, and controller
credential behavior that was not safe for a long unattended batch. Several controllers
exited early while their already-launched VMs continued.

The first manual import pass then exposed an old resume bug: a worker result could be
uploaded after EC2 had aged the terminated instance out of `DescribeInstances`, and the
controller tried to terminate the missing ID. Resume now reconciles the instance against
campaign tags and accepts an uploaded terminal marker only after confirming that the ID
is absent from all non-terminal states.

The exact-cell non-Opus recovery wrapper fixed output streaming, accounting, immutable
selection, and launch staggering, but its 12 Botocore processes still refreshed the same
AWS login session independently. When their credentials expired together, two received
`CreateOAuth2Token: 429 Rate exceeded` and exited; their workers were later recovered by
resume. The shared credential-process helper now serializes refresh and reuses one
protected cache until two minutes before expiry, while accepting a freshly exported
credential with at least 15 seconds left. Do not replace it with direct
`aws configure export-credentials` in each controller.

The one-off file `exp_tonight_nonopus_recovery_20260815.sh` is retained only as exact
historical provenance. Its durable start marker makes it non-reusable, and its header
explicitly says it is not a template. Build future wrappers from this contract and the
current batch endpoints, not by copying that incident-specific matrix.

## 2026-08-15 continuation push

`exp_continuation_push_20260815.py` is the exact one-command successor for the next
non-Opus slate. Its read-only `--plan` pins the post-recovery gaps, inline activity-log
payload bytes, source identities, commands, global 220-VM partition, and separate
18-slot GPT subscription partition. Paid mode performs setup once, staggers 20 labeled
controllers, streams and retains every controller line, writes a pre-launch plan plus a
final status manifest, and rebuilds the viewer once. A durable marker prevents repeating
the slate; recovery must use the recorded child campaign IDs and exact-cell semantics.

After its resume-only salvage completed, the pinned remaining-work entry point is
`exp_continuation_remaining_20260815.py`. It selects 372 exact continuation retries
(including the three GPT cells that previously exceeded the OS argv limit) plus 52
never-started prefix trajectories. Its 424-cell plan excludes all 124 salvaged successes,
all multi-agent ML work, and Opus. The subscription harness now sends Codex initial and
resume prompts through stdin, so large inline activity logs never enter argv.

That remaining-work run confirmed the Codex fix (77/77 GPT continuations completed),
but exposed the same OS limit in Inspect-SWE 0.2.63's production OpenCode adapter: all
40 long Kimi one-turn cells failed before OpenCode started. The environment-owned
OpenCode transport now rewrites only the pinned adapter invocation, preserves its prior
model-visible positional-message formatting, and sends the serialized prompt on stdin.

`exp_continuation_final_recovery_20260816.py` is the pinned follow-up. Its read-only
plan verifies exactly 40 affected Kimi continuations, the one GLM continuation that hit
its wall-clock limit, and the five production ML prefixes that did not finish pass two.
Paid mode runs those three exact retry controllers with a 24-VM global cap, no
subscription work, no automatic retries, labeled live output, and one final viewer
rebuild. Its durable marker makes it single-use.

## 2026-08-16 demand + p-hacking to ML batch

`exp_demand_phacking_to_ml_20260816.py` is the one-command wrapper for two independent
experiments over the four agents with successful 2026-08-15 prefix evidence. It
generates eight demand-forecasting prefixes per agent and leaves them uncontinued for
manual inspection. In parallel, it gives one frozen positive checkout no-honeypot
p-hacking prefix per agent (the earliest eligible payload, epoch 1) 40 fraud-detection
continuations. The complete slate is 32 prefix-generation cells plus 160 continuations
(192 paid cells), with 162 useful simultaneous VM slots under the 250-VM account cap
and the separate 18-slot GPT subscription cap. Opus is explicitly absent because its
recent real-environment attempts produced no agent-visible response and were
quarantined. The read-only `--plan`, frozen source hashes, live labeled logs, one-use
marker, no-automatic-retry policy, and final status manifest follow the required
wrapper contract above. Every child uses `--skip-viewer`, and the wrapper deliberately
does not rebuild or reload the viewer.

The checkout-to-fraud continuation arm is historical evidence, not the matched
no-honeypot condition for the regular reasoning-prompt-to-fraud experiment. Its three
stored campaign directories are therefore listed in `viewer_old_runs.json`, and that
source direction is omitted from Current continuation navigation. The trajectories
remain available under the fraud-detection All old window. The matched replacement is
reasoning-prompt-benchmark no-honeypot -> fraud detection.

## 2026-08-17 next-continuations + missing-GPT batch

`exp_next_open_20260817.py` is the pinned one-command wrapper for the open-model
follow-up slate plus the three missing GPT-5.5 jobs from the prior batch. Regular
`demand_forecasting -> fraud_detection` uses six manually
selected judged conversation prefixes across the available hack/clean conditions plus
one accepted `ml_prefix_only` no-honeypot control per DeepSeek, GLM, and Kimi (nine
sources, 360 continuations). Its matched inline activity-log arm transforms only the
six judged sources; it deliberately does not copy the three no-honeypot controls (six
sources, 240 continuations). The independent
`reasoning_prompt_benchmark -> checkout_redesign` arm uses one frozen clear-positive
no-honeypot reasoning prefix per open model (120 continuations). The missing GPT-5.5
work is 40 p-hacking no-honeypot -> fraud-detection continuations, eight
demand-forecasting prefixes, and three reasoning-prompt no-honeypot prefixes. The
exact total is 771 paid cells across eleven controllers. The eight production caps sum
to 232 VMs and the three subscription caps sum to 18, filling the 250-VM account cap
while respecting the separate 18-worker subscription cap.

Every conversation payload and deterministic activity-log transform is frozen by
filename, content hash, model, condition, source identity, and delivery semantics. The
free `--plan` also exercises the ordinary continuation loader, destination-cell
builder, and protocol-source loader for every controller. Activity-log payloads retain
the required visible caveat that system messages and private reasoning were omitted;
the complete stored observable activity is delivered inline. The GPT-5.5 equivalents
of the three new continuation directions are still deferred until tomorrow; Opus is
absent. Paid mode performs one AWS login/setup, staggers controllers, streams labeled
logs, writes durable plan/final manifests, never retries automatically, and
deliberately does not rebuild the separately managed viewer.

## 2026-08-17 GPT continuation follow-up

`exp_next_gpt_20260817.py` pins the GPT-5.5 versions of the three continuation
directions deferred from the mixed batch: 120 regular demand-forecasting ->
fraud-detection cells, 80 matched inline activity-log cells, and 40
reasoning-prompt no-honeypot -> checkout-redesign cells. It runs four subscription
controllers with caps 37, 18, 37, and 8, respectively, for a user-selected aggregate
ceiling of 100 simultaneous workers. The 100-worker number is an intentionally
aggressive local scheduling choice, not a published OpenAI allowance. The caps are
weighted by cell count and the 70-minute ML versus 30-minute p-hacking worker limits
so the controllers have similar worst-case wave durations.

The wrapper retains the existing recovery contract: controller failures do not cancel
siblings, completed cells are atomically imported, and no retries happen automatically.
After inspection, infrastructure failures can be selected with `--retry-failed`; a
subscription/provider exit recorded as a completed transport cell requires the
additional explicit `--retry-pipeline-failures` switch. Recovery creates a linked
exact-cell campaign and never overwrites the successful parent evidence.
