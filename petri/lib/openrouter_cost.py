"""Persist OpenRouter's REAL billed per-call cost into the eval log, per trajectory.

OpenRouter already returns the actual dollar cost of every call (`response.usage.cost`),
but (1) inspect's OpenRouter provider never reads it, and (2) inspect condenses the raw
response out of the log for all but the first few calls — so the cost would be lost. This
monkeypatches `OpenRouterAPI.generate` to copy `response.usage.cost` into
`ModelUsage.total_cost` while the fresh response is still in hand. `total_cost` is a
first-class ModelUsage field that survives condensing and is summed by ModelUsage.__add__,
so it accumulates into each sample's `model_usage[model].total_cost` — which the viewer then
reads as the EXACT (no-`~`) per-trajectory cost (see lib/model_prices.py).

Only takes effect for models routed through `openrouter/…`. Anthropic-direct calls are
untouched (they carry no cost field) and keep falling back to the price×token estimate.
Because our model slugs aren't in inspect's built-in price table, inspect's own cost path
leaves `total_cost` alone, so the value set here is not overwritten.

Call `install()` once before launching an eval; it's idempotent and fails soft (logs a
warning and no-ops) if inspect's internals ever move, so a runner never crashes over cost
bookkeeping.
"""
import logging

logger = logging.getLogger(__name__)
_installed = False


def _apply_real_cost(output, call) -> None:
    """Set output.usage.total_cost from OpenRouter's response.usage.cost, if present."""
    usage = getattr(output, "usage", None)
    if usage is None or call is None:
        return
    raw = getattr(call, "response", None)
    ru = raw.get("usage") if isinstance(raw, dict) else None
    cost = ru.get("cost") if isinstance(ru, dict) else None
    if isinstance(cost, (int, float)):
        usage.total_cost = float(cost)


def install() -> bool:
    """Monkeypatch OpenRouterAPI.generate to capture real billed cost. Returns True if the
    patch is in place (or already was), False if it couldn't be applied."""
    global _installed
    if _installed:
        return True
    try:
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
    except Exception as e:  # inspect internals moved — don't take the runner down over cost
        logger.warning("openrouter_cost: could not import OpenRouterAPI (%s); "
                       "real-cost capture DISABLED, costs will be ~estimates", e)
        return False

    _orig_generate = OpenRouterAPI.generate

    async def generate(self, *args, **kwargs):
        result = await _orig_generate(self, *args, **kwargs)
        try:
            if isinstance(result, tuple) and len(result) == 2:
                output, call = result
                # output can be an Exception on a failed call — guard via getattr in helper
                _apply_real_cost(output, call)
        except Exception:
            logger.debug("openrouter_cost: failed to set total_cost", exc_info=True)
        return result

    OpenRouterAPI.generate = generate
    _installed = True
    logger.info("openrouter_cost: real-cost capture INSTALLED (response.usage.cost -> total_cost)")
    return True
