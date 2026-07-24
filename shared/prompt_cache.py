"""Provider-neutral bookkeeping for prompt-cache usage and reuse checks.

This module is deliberately passive: it never calls an API and it does not decide
how requests are scheduled.  Experiment adapters feed it the usage object returned
by a provider, store the normalized record, and use :func:`assess_reuse` when a
stable prefix is expected to have been warmed by an earlier independent request.

The normalized field names follow Anthropic's response schema because those are the
only cache counters currently exposed by the PTB scaffold.  Callers for other
providers can construct the same four-count dict before using the assessment and
summary helpers.
"""
from __future__ import annotations

from typing import Any, Iterable


CACHE_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)

TTL_USAGE_KEYS = (
    "ephemeral_5m_input_tokens",
    "ephemeral_1h_input_tokens",
)


def _nonnegative_int(value: Any) -> int:
    """Return a safe token count; malformed/missing provider fields become zero."""
    if isinstance(value, bool):
        return 0
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def normalize_usage(usage: dict | None, *, first_iteration: bool = False) -> dict:
    """Normalize one provider usage object into stable, queryable token fields.

    Claude CLI result events include both aggregate counts and an ``iterations``
    list.  Cache reuse between independent asks must be checked on the FIRST model
    request: later iterations may read a cache created by tool use inside the same
    ask and would produce a false positive.  ``first_iteration=True`` selects that
    first request when it exists, falling back loudly via ``source`` when it does
    not.
    """
    aggregate = usage if isinstance(usage, dict) else {}
    selected = aggregate
    source = "aggregate"
    if first_iteration:
        iterations = aggregate.get("iterations")
        if isinstance(iterations, list) and iterations and isinstance(iterations[0], dict):
            selected = iterations[0]
            source = "first_iteration"
        else:
            source = "aggregate_fallback_no_iterations"
    out = {key: _nonnegative_int(selected.get(key)) for key in CACHE_USAGE_KEYS}
    creation = selected.get("cache_creation")
    creation = creation if isinstance(creation, dict) else {}
    for key in TTL_USAGE_KEYS:
        out[f"cache_creation_{key}"] = _nonnegative_int(creation.get(key))
    out["cache_creation_ttl_detail_available"] = bool(creation)
    out["cacheable_input_tokens"] = (
        out["cache_creation_input_tokens"] + out["cache_read_input_tokens"]
    )
    out["source"] = source
    return out


def assess_reuse(reference: dict, current: dict, *, min_reuse_fraction: float = 0.8,
                 min_cacheable_tokens: int = 4096) -> dict:
    """Check whether ``current`` read the stable prefix observed in ``reference``.

    The reference may itself have been a cache hit (for example, a prior process
    warmed the same prefix), so the denominator is all cacheable input it observed:
    cache reads plus cache writes.  The threshold is intentionally fractional to
    tolerate a small varying suffix and run-era tokenizer/tool-schema differences.
    """
    expected = _nonnegative_int(reference.get("cacheable_input_tokens"))
    read = _nonnegative_int(current.get("cache_read_input_tokens"))
    fraction = (read / expected) if expected else None
    if expected < min_cacheable_tokens:
        verified = False
        reason = (f"warm-up exposed only {expected:,} cacheable input tokens; "
                  f"need at least {min_cacheable_tokens:,} to verify a substantial prefix")
    elif fraction is None or fraction < min_reuse_fraction:
        verified = False
        shown = "unknown" if fraction is None else f"{fraction:.1%}"
        reason = (f"cache read covered {shown} of the warm-up prefix "
                  f"({read:,}/{expected:,} tokens), below the required "
                  f"{min_reuse_fraction:.0%}")
    else:
        verified = True
        reason = (f"cache read covered {fraction:.1%} of the warm-up prefix "
                  f"({read:,}/{expected:,} tokens)")
    return {
        "verified": verified,
        "reason": reason,
        "reference_cacheable_input_tokens": expected,
        "cache_read_input_tokens": read,
        "reuse_fraction": fraction,
        "min_reuse_fraction": min_reuse_fraction,
        "min_cacheable_tokens": min_cacheable_tokens,
    }


def summarize(records: Iterable[dict]) -> dict | None:
    """Aggregate stored ``prompt_cache`` records without hiding missing usage."""
    items = [r.get("prompt_cache") for r in records if isinstance(r, dict)]
    items = [x for x in items if isinstance(x, dict)]
    if not items:
        return None
    usages = [x.get("usage") for x in items if isinstance(x.get("usage"), dict)]
    checks = [x.get("reuse_check") for x in items
              if isinstance(x.get("reuse_check"), dict)]
    return {
        "mode": items[0].get("mode"),
        "n_records": len(items),
        "n_with_usage": len(usages),
        "cache_creation_input_tokens": sum(
            _nonnegative_int(x.get("cache_creation_input_tokens")) for x in usages),
        "cache_read_input_tokens": sum(
            _nonnegative_int(x.get("cache_read_input_tokens")) for x in usages),
        "cache_creation_5m_input_tokens": sum(
            _nonnegative_int(x.get("cache_creation_ephemeral_5m_input_tokens"))
            for x in usages),
        "cache_creation_1h_input_tokens": sum(
            _nonnegative_int(x.get("cache_creation_ephemeral_1h_input_tokens"))
            for x in usages),
        "n_reuse_checks": len(checks),
        "n_reuse_verified": sum(x.get("verified") is True for x in checks),
        "n_reuse_failed": sum(x.get("verified") is False for x in checks),
        "all_reuse_checks_passed": bool(checks) and all(
            x.get("verified") is True for x in checks),
    }
