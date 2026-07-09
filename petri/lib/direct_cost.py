"""Feed first-party LIST prices into Inspect's native cost path for DIRECT calls.

Inspect already computes `usage.total_cost` from a per-model price table
(`compute_model_cost`), but ships no prices for our (post-cutoff) models, so it
never fires. This registers the LIST prices in lib/model_prices (the "-list"
entries, via `model_prices.list_priced()`) using Inspect's supported
`set_model_cost()` hook. Once registered, every DIRECT Anthropic/OpenAI call
gets an exact `total_cost` set at generate-time — the SAME ModelUsage.total_cost
field that lib/openrouter_cost.py sets for OpenRouter calls, and that the viewer
(lib/model_prices.sample_cost) already reads as the EXACT, no-`~` per-trajectory
cost. So this is the direct-API analogue of openrouter_cost: both funnel into
total_cost, and between them every model in a trajectory can be exact.

Cost is priced from token buckets (input / output / cache-read / cache-write),
with cache-write at the 5-min-TTL rate (1.25x input); targets run cache=False so
cache tokens are ~0 and this is moot in practice.

Call `install()` once before launching an eval (beside openrouter_cost.install()).
Idempotent; fails soft (logs a warning, no-ops) if Inspect's internals move or a
slug isn't in Inspect's model database — a runner never crashes over cost bookkeeping.
"""
import logging

import model_prices

logger = logging.getLogger(__name__)
_installed = False


def install() -> bool:
    """Register model_prices' LIST prices with Inspect so direct calls get an exact
    total_cost. Returns True if registration ran (even partially), False if the hook
    was unavailable. Per-model failures are logged and skipped."""
    global _installed
    if _installed:
        return True
    try:
        from inspect_ai.model import set_model_cost
        from inspect_ai.model._model_data.model_data import ModelCost
    except Exception as e:  # Inspect internals moved — don't take the runner down over cost
        logger.warning("direct_cost: could not import set_model_cost/ModelCost (%s); "
                       "direct Anthropic/OpenAI costs will be ~estimates", e)
        return False

    registered, skipped = [], []
    for slug, p in model_prices.list_priced().items():
        try:
            set_model_cost(slug, ModelCost(
                input=p["input"],
                output=p["output"],
                input_cache_write=p["cache_write"],
                input_cache_read=p["cache_read"],
            ))
            registered.append(slug)
        except Exception as e:  # e.g. slug not in Inspect's model DB — skip, keep going
            skipped.append(f"{slug} ({type(e).__name__}: {e})")

    _installed = True
    logger.info("direct_cost: registered LIST prices for %d model(s): %s",
                len(registered), ", ".join(registered) or "none")
    if skipped:
        logger.warning("direct_cost: skipped %d model(s) (fall back to ~estimate): %s",
                       len(skipped), "; ".join(skipped))
    return True
