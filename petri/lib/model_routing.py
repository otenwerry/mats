"""Single source of truth for WHICH API serves each model.

Policy: models from vendors we hold first-party API keys for (Anthropic, OpenAI)
are called via that vendor's own API; everything else goes through OpenRouter
(the only way we can reach deepseek / qwen / glm / grok / kimi / minimax / …).

`route()` normalizes any model slug to obey this policy, and every place that
turns a target/auditor/judge slug into a model calls it — so the decision lives
here and propagates everywhere. It's idempotent (routing an already-routed slug
is a no-op), so it's safe to apply at multiple layers.

Why this isn't a pure prefix-strip: OpenRouter and OpenAI date their model
snapshots differently (OpenRouter `gpt-5.5-20260423` vs OpenAI `gpt-5.5-2026-04-23`),
so the OpenAI ids need an explicit remap. Anthropic ids are identical on both, so
they only need the prefix dropped (and today's slugs are already `anthropic/…`).
"""
from __future__ import annotations

# Vendors we call via their first-party API. Everything else stays on OpenRouter.
# This set IS the toggle: to move OpenAI back onto OpenRouter, drop "openai".
DIRECT_VENDORS = {"anthropic", "openai"}

# OpenRouter slug -> first-party slug, for vendors whose model id differs between
# the two (OpenAI's dated snapshots). Anthropic needs no entry (ids match; the
# generic prefix-strip below handles it). Verify a new OpenAI id against the live
# OpenAI model list before trusting the exact snapshot string — a wrong id fails
# LOUDLY at generation (0 output tokens -> the pipeline's dead-target guard aborts).
_OPENROUTER_TO_DIRECT = {
    "openrouter/openai/gpt-5.5-20260423": "openai/gpt-5.5-2026-04-23",
    "openrouter/openai/gpt-5.4-mini":     "openai/gpt-5.4-mini-2026-03-17",
}


def route(slug: str) -> str:
    """Return the slug to actually call, per the direct-vs-OpenRouter policy.

    - Known OpenAI OpenRouter slugs -> their first-party dated id (explicit map).
    - Any other `openrouter/<vendor>/…` where <vendor> is a DIRECT_VENDORS member
      -> the vendor-direct slug (drop the `openrouter/` prefix).
    - Everything else (already-direct slugs, non-direct OpenRouter vendors) -> unchanged.
    """
    if not slug or "/" not in slug:
        return slug
    if slug in _OPENROUTER_TO_DIRECT:
        return _OPENROUTER_TO_DIRECT[slug]
    if slug.startswith("openrouter/"):
        rest = slug[len("openrouter/"):]
        vendor = rest.split("/", 1)[0]
        if vendor in DIRECT_VENDORS:
            return rest
    return slug


def openai_compatible(slug: str) -> bool:
    """True when Inspect sends OpenAI-style tool schemas for this slug, so the
    target needs `strict_tools=False` (else an optional param in an auditor-created
    tool 400s the request; see the petri-auditor-empty-tool-params note). Covers
    direct OpenAI (`openai/…`) AND every OpenRouter route (non-OpenAI OpenRouter
    providers ignore the flag, so passing it there is a harmless no-op — matching
    the pre-existing behavior). Anthropic-direct is False (its provider rejects the
    kwarg). Routes the slug first so a post-route `openai/…` is caught.
    """
    s = route(slug)
    return s.startswith("openrouter/") or s.startswith("openai/")
