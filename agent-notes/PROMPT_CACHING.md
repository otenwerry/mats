# Prompt caching across Petri and PTB

> **Summary:** Petri uses provider prompt-prefix caching, not Inspect response
> replay. `petri/lib/prompt_caching.py` warms repeated prefixes before parallel
> fan-out and writes per-run evidence. PTB propensity asks use a stricter
> warm-up/reuse check through the shared ask runner. Read before adding or
> changing any repeated-context paid pipeline.

## The distinction that matters

Inspect's `cache=` / `CachePolicy` is a generation-response cache. A hit returns
an old model output. Do not enable it for independent experimental samples.

Provider prompt caching stores only the prompt's KV prefix. Each request still
samples a fresh output. Inspect 0.3.239 enables provider prompt caching by
default for supported APIs; direct Anthropic SDK calls need `cache_control`.
Petri's target agents pass `cache=False` only to the response-cache layer. That
does not turn provider caching off.

## Petri

`petri/lib/prompt_caching.py` is the shared implementation:

- `install_inspect_warmup()` wraps Inspect's direct Anthropic, direct OpenAI,
  and OpenRouter providers. Requests with the same long stable history wait for
  one real response, then fan out normally. No response is shared.
- `cached_system()` / `cached_user_prefix()` add explicit five-minute
  Anthropic breakpoints for direct SDK calls. `run_direct_cached()` applies the
  same warm-first rule.
- `write_report(run_dir)` stores `prompt_cache_report.json`: input, output,
  cache-read, cache-write, cost, cache-read fraction, and an honest per-model
  status. Only a positive provider cache-read count is called verified.
- Repeated OpenAI continuations, resamples, rollbacks, and resumed-context asks
  provide a stable `prompt_cache_key` for cache-shard routing. OpenAI still
  combines the key with the exact prompt prefix; the key cannot cause content
  collisions or response reuse.
- Cache-sized OpenRouter calls receive a stable `session_id` derived from the
  model plus the first system and first non-system messages. It remains constant
  as the trajectory grows. This makes provider stickiness start after the first
  successful request; an explicit caller-supplied `session_id` always wins.

The audit, continuation, and rollback generation pipelines install the barrier
automatically. The shared Petri ask adapter does too and stores per-ask cache
usage in its normal result schema. Direct hack annotation, rollback annotation,
continuation faithfulness, and resample deviation calls use explicit Anthropic
breakpoints. Current specialized/legacy one-off tools were not generalized;
audit them explicitly if they become active experiments.

Historical evidence is strong. For example,
`continuation-10x-20260709-162948` recorded positive cache reads for all six
models. Approximate read fractions of prompt tokens were 97% for the Anthropic
target, 96% for OpenAI, 93% for the DeepSeek auditor, 91% for GLM, and 78% for
Kimi. The inline Anthropic judge also read cached tokens, though its changing
full transcripts made the aggregate fraction much lower (16%).

### 2026-07-23 continuation check and fixes

The smaller `continuation-3x-20260723-112753` check showed positive aggregate
cache reads, but its first target and auditor calls did not reliably reuse the
shared prefix across epochs. Two independent causes were found:

1. B's tools were being recreated by the auditor and could differ across
   epochs. `continuation-v5` fixes this part by loading B's exact recorded tools,
   pre-registering them for every treatment, and preventing tool changes.
2. The warm-up grouping key serializes Inspect message objects including
   internal message IDs. Provider-visible prompt text can therefore be
   identical while the local key treats epochs as different. The v2 key fixes
   this by removing only the top-level `ChatMessage.id`; provider-visible
   tool-call IDs and all prompt content remain part of the key.

Local replay of the saved three-epoch check now produces one first-auditor
warm-up group and, with fixed tools, one first-target group for each of the five
models. New reports identify the v2 key under
`warmup_barrier.key_version`. This verifies local grouping, not provider
behavior. Later calls within individual trajectories can still hit provider
caches, so an aggregate positive read count does not prove that the expensive
shared initial prefix was reused. Do not call continuation caching
provider-verified until a small paid run confirms first-call cache reads.

The paid repeat `continuation-3x-20260723-142551` then verified all 10 auditor
reuse calls and 9 of 10 target reuse calls. Claude, GPT, DeepSeek, and GLM were
perfect; one of Kimi's two expected first-target reuses missed despite all three
requests having the same local key. OpenRouter documents that, without an
explicit `session_id`, provider stickiness begins only after it observes a cache
hit—not after the first successful cache-writing request. The shared wrapper now
adds that stable session ID automatically. This last routing change has local
test coverage but has not yet had a paid verification run.

## PTB shared asks

**PAID PTB EXPERIMENTS ARE HARD BLOCKED AS OF 2026-07-23.** Free viewing,
reconstruction, and dry-runs still work. The shared ask adapter, legacy ask
runner, and restart reproduction script all call
`posttrainbench/lib/experiment_readiness.py` before paid work. Do not remove the
block until Owen has chosen the original-CLI fidelity versus newer-CLI verified
cache protocol and the documentation is updated at the same time.

`shared/exp_ask_questions.py --env=posttrainbench` plus
`posttrainbench/lib/exp_ask_env.py` implement `--prefix-cache=auto|required|off`.
`auto` means required for propensity and off for older EM/context runs.

Required mode gives every ask a fresh session and empty filesystem while keeping
one stable cwd string for Claude Code's cache key. Asks are serial within a
trajectory and trajectories remain parallel. The first ask warms; the second
must read at least 80% of at least 4,096 cacheable first-iteration tokens before
any ask three begins. It requests a one-hour TTL and aborts on missing usage,
wrong TTL, cache miss, CLI error, or missing result. Evidence is stored per ask
and summarized in `results.json`.

Claude Code added the generic one-hour cache flag in 2.1.108. PTB originals such
as 2.1.9/2.1.34/2.1.76 cannot honor it. Required mode therefore refuses those
CLIs before an API ask. Use `--cli=local` or a newer explicit command and accept
the stored scaffold-version fidelity caveat, or explicitly select
`--prefix-cache=off`. This is an experiment-design tradeoff, not something code
should choose silently.

Required caching is Claude-only. PTB OpenCode resumes are refused before API
spend in required mode; `--prefix-cache=off` is the explicit override. The
legacy per-environment ask runners are also not protected—use the shared runner.

Focused no-API tests:

- `petri/tests/test_prompt_caching.py`
- `posttrainbench/tests/test_prompt_cache.py`
