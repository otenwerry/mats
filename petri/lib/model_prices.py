"""Per-model token prices + per-trajectory cost, for the viewer's cost line.

Two cost regimes, unified by `sample_cost`:
  - APPROXIMATE (retroactive, part a): price x token-count. Prices below. All OpenRouter
    prices were DERIVED from the `cost_details` OpenRouter already stamped on the
    un-condensed early calls in our own logs (tools/explore_derive_prices.py) — i.e. the
    real per-token rates we were charged — so they're accurate, but price x tokens is still
    only an estimate because (1) OpenRouter routes a slug to different upstream providers at
    slightly different rates (e.g. deepseek-v4-pro input 0.748 vs 1.13) and (2) the Anthropic
    prices are ASSUMED from public Opus/Sonnet tiers (we never billed them via OpenRouter, so
    they can't be derived). Hence these render with a leading `~`.
  - EXACT (going forward, part b): OpenRouter's real billed `usage.cost`, captured into
    `ModelUsage.total_cost` at generate-time (see lib/openrouter_cost.py). When a trajectory's
    every model call carries a real total_cost, `sample_cost` uses it and drops the `~`.

Prices are $ per MILLION tokens. `cache_read`/`cache_write` price the separately-counted
cache tokens (inspect's input_tokens already EXCLUDES them). cache_write defaults to the
input rate where a provider doesn't bill writes separately (OpenRouter reported
cache_write_tokens=0 for all our models, so it's moot for the derived set).

Keys are canonical slugs with any leading "openrouter/" stripped, so both
"openrouter/anthropic/claude-opus-4-8" (post-migration) and "anthropic/claude-opus-4-8"
(today) resolve to the same entry.
"""
from __future__ import annotations


def _p(inp, out, cache_read=None, cache_write=None, source="openrouter-derived"):
    return {"input": inp, "output": out,
            "cache_read": inp if cache_read is None else cache_read,
            "cache_write": inp if cache_write is None else cache_write,
            "source": source}


# OpenRouter models — DERIVED from our logs' cost_details (median of the per-token rates
# OpenRouter actually charged). See tools/explore_derive_prices.py.
PRICES: dict[str, dict] = {
    "deepseek/deepseek-chat":              _p(0.20, 0.80, 0.20),
    "deepseek/deepseek-r1":                _p(0.70, 2.50, 0.70),
    "deepseek/deepseek-v4-pro-20260423":   _p(0.748, 1.496, 0.0624),
    "minimax/minimax-m2.7":                _p(0.18, 1.00, 0.053),
    "moonshotai/kimi-k2.6-20260420":       _p(0.95, 3.60, 0.125),
    "openai/gpt-5.5-20260423":             _p(5.00, 30.00, 0.50),
    "qwen/qwen-2.5-72b-instruct":          _p(0.38, 0.40, 0.38),
    "qwen/qwen3-32b":                      _p(0.08, 0.30, 0.10),
    "qwen/qwen3.7-max":                    _p(1.25, 3.75, 0.25),
    "xiaomi/mimo-v2.5-pro":                _p(0.60, 3.00, 0.26),
    "z-ai/glm-5.1":                        _p(0.966, 3.036, 0.179),
    "z-ai/glm-5.2":                        _p(1.1522, 3.6212, 0.214),
    # Owen-provided (2026-07-06). couldn't derive from logs (no cost-stamped calls); cache_read
    # for grok = 0.25x input (xAI cached-input ratio), llama has no cache discount on OpenRouter.
    "meta-llama/llama-3.3-70b-instruct":   _p(0.10, 0.32, 0.10, source="openrouter-assumed"),
    "x-ai/grok-4.20":                      _p(1.25, 2.50, 0.3125, source="openrouter-assumed"),
    # --- First-party LIST prices (source ends "-list"). Registered into Inspect's cost
    # path via direct_cost.install(), so DIRECT calls to these models get an exact
    # total_cost the same way OpenRouter calls do (see lib/direct_cost.py). These entries
    # then serve only as a fallback; the exactness comes from the Inspect-set total_cost.
    # Anthropic: real list prices (Opus $5/$25, Sonnet $3/$15; cache_read 0.1x input,
    # cache_write 1.25x input = the standard 5-min-TTL ratio).
    "anthropic/claude-opus-4-8":           _p(5.0, 25.0, 0.50, 6.25, "anthropic-list"),
    "anthropic/claude-opus-4-6":           _p(5.0, 25.0, 0.50, 6.25, "anthropic-list"),
    "anthropic/claude-sonnet-4-6":         _p(3.0, 15.0, 0.30, 3.75, "anthropic-list"),
    # OpenAI DIRECT slugs (dated snapshots, keyed to match model_routing.route()). Real
    # OpenAI list prices (Owen-provided 2026-07-06): cache_read = OpenAI's cached-input rate
    # (0.1x input); cache_write defaults to input (OpenAI has no cache-write surcharge, and
    # it's moot anyway since targets run cache=False). "-list" -> registered with Inspect via
    # direct_cost.install(), so direct GPT calls get an exact total_cost like Anthropic's.
    "openai/gpt-5.5-2026-04-23":           _p(5.00, 30.00, 0.50, source="openai-list"),
    "openai/gpt-5.4-mini-2026-03-17":      _p(0.75, 4.50, 0.075, source="openai-list"),
    # Historical: gpt slugs as billed via OpenRouter (old logs still reference these; keep
    # so past trajectories stay priced). New runs route to the openai/… slugs above.
    "openai/gpt-5.5-20260423":             _p(5.00, 30.00, 0.50),
}


def list_priced() -> dict[str, dict]:
    """First-party LIST-price entries (source ends '-list'), keyed by their DIRECT slug.
    Consumed by direct_cost.install() to feed Inspect's cost path so direct Anthropic/
    OpenAI calls get an exact total_cost. Excludes OpenRouter-derived/assumed estimates."""
    return {slug: p for slug, p in PRICES.items() if str(p.get("source", "")).endswith("-list")}


# Context window (max tokens per request) per model, $ per model, for the viewer's
# per-role "context used" line. Keyed by CANONICAL slug (canon() strips a leading
# 'openrouter/'). Anthropic/OpenAI from Inspect's static model DB; the OpenRouter
# entries from openrouter.ai/api/v1/models (fetched 2026-07-06; deepseek-v4-pro and
# kimi-k2.6 read off their undated variant, which shares the window). context_window()
# falls back to Inspect's DB for any model not listed, so a new target still resolves.
CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-4-8":                 1_000_000,
    "anthropic/claude-opus-4-6":                 1_000_000,
    "anthropic/claude-sonnet-4-6":               1_000_000,
    "openai/gpt-5.5-2026-04-23":                 1_050_000,
    "openai/gpt-5.5-20260423":                   1_050_000,   # historical OpenRouter slug
    "openai/gpt-5.4-mini-2026-03-17":              400_000,
    "deepseek/deepseek-chat":                      131_072,
    "deepseek/deepseek-r1":                        163_840,
    "deepseek/deepseek-v4-pro-20260423":         1_048_576,
    "google/gemma-3-27b-it":                       131_072,
    "meta-llama/llama-3.3-70b-instruct":           131_072,
    "minimax/minimax-m2.7":                        204_800,
    "mistralai/mistral-small-3.1-24b-instruct":    128_000,
    "moonshotai/kimi-k2.6-20260420":               262_144,
    "qwen/qwen-2.5-72b-instruct":                  131_072,
    "qwen/qwen3-32b":                              131_072,
    "qwen/qwen3.7-max":                          1_000_000,
    "x-ai/grok-4.20":                            2_000_000,
    "xiaomi/mimo-v2.5-pro":                      1_048_576,
    "z-ai/glm-5.1":                                202_752,
    "z-ai/glm-5.2":                              1_048_576,
}


def context_window(slug: str) -> int | None:
    """Max tokens per request for `slug`. CONTEXT_WINDOWS first (canonical key), then
    Inspect's static model DB (`get_model_info(...).context_length`, free/offline) for
    models we didn't hardcode. None when unknown -> the viewer surfaces it as '?'."""
    if not slug:
        return None
    win = CONTEXT_WINDOWS.get(canon(slug))
    if win is not None:
        return win
    try:
        from inspect_ai.model import get_model_info
        info = get_model_info(slug)
        return getattr(info, "context_length", None) if info else None
    except Exception:
        return None


def canon(slug: str) -> str:
    """Canonical price key: drop a leading 'openrouter/' router prefix (so a model routed
    through OpenRouter and the same model called directly share one price entry)."""
    return slug[len("openrouter/"):] if slug.startswith("openrouter/") else slug


def price_for(slug: str) -> dict | None:
    return PRICES.get(canon(slug))


def _usage_cost(u: dict, p: dict) -> float:
    """price x tokens for one model's usage dict, mirroring inspect's compute_model_cost
    (`input` already excludes the separately-counted cache tokens). `u` keys:
    input / output / cache_read / cache_write (see viewer_load.usage_to_dict)."""
    cost = (u.get("input") or 0) / 1e6 * p["input"]
    cost += (u.get("output") or 0) / 1e6 * p["output"]
    cost += (u.get("cache_read") or 0) / 1e6 * p["cache_read"]
    cost += (u.get("cache_write") or 0) / 1e6 * p["cache_write"]
    return cost


def sample_cost(model_usage: dict) -> dict:
    """Per-trajectory cost from a trajectory's model_usage ({slug: usage-dict}, the plain
    dicts stored by viewer_load.usage_to_dict — each has input/output/cache_read/cache_write
    and total_cost).

    For each model: use its real billed `total_cost` when present (EXACT); else fall back to
    price x tokens from PRICES (approximate); else — no price known — record it as unknown and
    contribute nothing. Returns:
      {total: float, by_model: {slug: {cost, exact, source}}, exact: bool, unknown: [slug,...]}
    `exact` is True only when every contributing model reported a real total_cost AND no model
    was unpriced — that's the flag the viewer uses to drop the `~`.
    """
    by_model: dict[str, dict] = {}
    unknown: list[str] = []
    total = 0.0
    any_est = False
    for slug, u in (model_usage or {}).items():
        real = u.get("total_cost")
        if isinstance(real, (int, float)):
            by_model[slug] = {"cost": float(real), "exact": True, "source": "billed"}
            total += float(real)
            continue
        p = price_for(slug)
        if p is None:
            unknown.append(slug)
            any_est = True
            continue
        c = _usage_cost(u, p)
        by_model[slug] = {"cost": c, "exact": False, "source": p["source"]}
        total += c
        any_est = True
    exact = (not any_est) and bool(by_model)
    return {"total": total, "by_model": by_model, "exact": exact, "unknown": unknown}


def cost_by_role(role_usage: dict, role_slugs: dict) -> dict:
    """Per-role cost for one trajectory. `role_usage` = {role: usage-dict} (viewer_load's
    role_usage, keyed auditor/target/judge); `role_slugs` = {role: model-slug} (that
    trajectory's target/auditor/judge). Prices each role by ITS slug — so a slug serving two
    roles is still split correctly. Returns {role: {cost, exact, unpriced}}. Real billed
    total_cost wins over the price×token estimate, same as sample_cost."""
    out: dict[str, dict] = {}
    for role, u in (role_usage or {}).items():
        real = u.get("total_cost")
        if isinstance(real, (int, float)):
            out[role] = {"cost": float(real), "exact": True, "unpriced": False}
            continue
        p = price_for(role_slugs.get(role, ""))
        out[role] = {"cost": (_usage_cost(u, p) if p else 0.0), "exact": False,
                     "unpriced": p is None}
    return out
