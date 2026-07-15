"""Register our curated context windows with Inspect so auditor compaction triggers
at each model's REAL window, not Inspect's 128k default.

Inspect's compaction (CompactionAuto, threshold 0.9) derives its trigger from the
model window via get_model_input_tokens(). Two failure modes for our models:
  - UNKNOWN to Inspect's bundled static DB (our post-cutoff 2026 slugs --
    deepseek-v4-pro, glm-5.1/5.2, grok-4.20, qwen3.7-max, mimo, minimax, ...): the
    lookup returns None and Inspect falls back to DEFAULT_CONTEXT_WINDOW = 128_000,
    so e.g. a 1M-window DeepSeek auditor compacts at ~115k (0.9 x 128k).
  - KNOWN but STALE: Inspect's context_length is materially below the real window
    (deepseek-chat / deepseek-r1 = 65,536; gemma-3-27b = 65,536; qwen3-32b = 40,960).
Either way the auditor compacts far too early -- the same 128k default that killed
the sweep-5 baseline screening with 'Compaction insufficient'.

Fix: feed lib/model_prices.CONTEXT_WINDOWS (our curated windows -- sourced from
openrouter.ai/api/v1/models 2026-07-06 + provider tiers, already what the viewer
treats as authoritative) into Inspect via its supported set_model_info() hook, but
ONLY where Inspect is missing the model or its window is SMALLER than ours. We
deliberately DON'T touch models whose Inspect context_length already matches ours
(anthropic, gpt-5.5, gpt-5.4-mini, llama, kimi): their lower input_tokens is an
intentional output reserve (context_length - output_tokens) that is exactly the
right compaction budget, and overriding it would remove that headroom.

For a KNOWN-but-stale model we MERGE (copy the existing ModelInfo and bump only
context_length / input_tokens) so it keeps its family / reasoning / organization
metadata; for an unknown model we register a fresh ModelInfo(context_length=win).
Compaction counts INPUT tokens, and the 0.9 threshold leaves the output headroom
(>=10% of the window), so setting input_tokens = full window is safe for every
window we register (all >= 128k, most >= 1M).

Keys: registered under the bare canonical slug (matches Model.canonical_name() for
OpenRouter-routed models) AND the 'openrouter/'-prefixed form (matches str(model));
first-party anthropic/openai slugs already carry their provider prefix in
CONTEXT_WINDOWS, so str(model) matches them directly.

Call install() once before launching an eval (beside direct_cost.install()).
Idempotent; fails soft (logs a warning, no-ops) if Inspect's internals move.
"""
import logging

import model_prices

logger = logging.getLogger(__name__)
_installed = False


def install() -> bool:
    """Correct Inspect's per-model context window from model_prices.CONTEXT_WINDOWS so
    the auditor's compaction threshold matches the real window. Returns True if it ran
    (even partially), False if the Inspect hooks were unavailable. Per-model failures
    are logged and skipped -- a runner never crashes over window bookkeeping."""
    global _installed
    if _installed:
        return True
    try:
        from inspect_ai.model import get_model_info, set_model_info
        from inspect_ai.model._model_data.model_data import ModelInfo
    except Exception as e:  # Inspect internals moved -- don't take the runner down
        logger.warning("model_window: could not import Inspect model-info hooks (%s); "
                       "auditor compaction may fire early on unknown/stale windows", e)
        return False

    fixed, left_as_is = [], []
    for slug, win in model_prices.CONTEXT_WINDOWS.items():
        existing = get_model_info(slug) or get_model_info(f"openrouter/{slug}")
        ctx = getattr(existing, "context_length", None) if existing else None
        # Never shrink a correct window: skip when Inspect already knows a window
        # at least as large as ours (its lower input_tokens is a deliberate reserve).
        if ctx is not None and ctx >= win:
            left_as_is.append(slug)
            continue
        if existing is not None:  # known-but-stale: keep metadata, bump only the window
            info = existing.model_copy(update={"context_length": win})
            info._input_tokens = win
        else:                     # unknown to Inspect: minimal entry with just the window
            info = ModelInfo(context_length=win)
        for key in (slug, f"openrouter/{slug}"):
            try:
                set_model_info(key, info)
            except Exception as e:
                logger.warning("model_window: failed to register %s (%s)", key, e)
        was = f"was {ctx:,}" if ctx is not None else "was 128k default (unknown to Inspect)"
        fixed.append(f"{slug}={win:,} ({was})")

    _installed = True
    logger.info("model_window: corrected context windows for %d model(s): %s",
                len(fixed), "; ".join(fixed) or "none")
    if left_as_is:
        logger.info("model_window: left %d model(s) unchanged (Inspect window already >= ours): %s",
                    len(left_as_is), ", ".join(left_as_is))
    return True
