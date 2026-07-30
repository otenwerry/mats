"""One place that decides which model judges a trajectory.

The ALIGNMENT judge (the model that scores a transcript on every dimension) is chosen
here and nowhere else, so switching it is a one-line change and every experiment --
petri audits, re-judges, dimension backfills, rollbacks, and the real-environment runs in
mats/environments/ -- moves together.

Three ways to choose, in order of precedence:
  1. an explicit argument / CLI flag (``--judge=<shortname|provider/model>``)
  2. the ``PETRI_JUDGE`` environment variable (handy for a one-off sweep)
  3. DEFAULT_JUDGE below

DEFAULT_JUDGE is gpt-5.6-luna since 2026-07-30 (Owen: cheaper and, on his own judge
comparison, as good as the incumbent Opus 4.8). Cost per million tokens, first-party
OpenAI list: $0.20 in / $1.20 out, against Opus 4.8's $5.00 / $25.00 -- roughly 20x
cheaper per judged trajectory.

NOTE: the judge model is stamped into every run's metadata (``judge``), so old runs stay
self-describing and a mixed-judge comparison is always visible in the viewer rather than
silent. Runs judged by different models are NOT automatically comparable; that is the one
thing to keep in mind when varying it.

This module is deliberately dependency-light (only model_routing) so anything can import
it without pulling in an experiment module.
"""

from __future__ import annotations

import os

from model_routing import route

# Judge shortnames. Add a row to make a model selectable as `--judge=<name>`; the value is
# a full provider/model string, routed at resolve time (model_routing sends anthropic and
# openai first-party, everything else through OpenRouter).
#
# BEFORE adding a row: register the model in model_prices.PRICES + CONTEXT_WINDOWS and in
# viewer._PRETTY_MODELS, or its judge cost is reported as unpriced and the viewer shows the
# raw slug. gpt-5.6-terra is deliberately absent for exactly that reason -- it has no price
# entry yet, and a judge whose cost we cannot report is worse than one fewer option. (A
# full slug still works via resolve_judge, so a one-off experiment is never blocked.)
JUDGE_CHOICES: dict[str, str] = {
    # cheap tier
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "deepseek-v4-pro": "openrouter/deepseek/deepseek-v4-pro-20260423",
    # frontier tier
    "gpt-5.6-sol": "openai/gpt-5.6-sol",
    "opus-4.8": "anthropic/claude-opus-4-8",
    "sonnet-4.6": "anthropic/claude-sonnet-4-6",
}

DEFAULT_JUDGE_NAME = "gpt-5.6-luna"
JUDGE_ENV_VAR = "PETRI_JUDGE"

# ---- secondary judge roles ------------------------------------------------- #
# The secondary judges (auditor deviation, auditor faithfulness, the rollback follow-up,
# mechanism similarity) call the Anthropic SDK DIRECTLY -- client.messages.parse() with a
# pydantic output_format -- rather than going through inspect's model layer. That client
# can only reach Anthropic models, so pointing these at the default judge
# (openai/gpt-5.6-luna) would fail at request time, not at startup.
#
# So they get their own centralized default: the cheapest priced Anthropic model. Owen
# asked for Luna here too (2026-07-30); using it literally requires porting those five
# files off the Anthropic SDK first, which is real work on code he does not expect to run
# again -- so this is the cheap-and-safe version, and resolve_secondary_judge() REFUSES a
# non-Anthropic model so the mistake cannot be made quietly later.
SECONDARY_JUDGE_NAME = "sonnet-4.6"
SECONDARY_JUDGE_SLUG = JUDGE_CHOICES[SECONDARY_JUDGE_NAME]      # anthropic/claude-sonnet-4-6
# The Anthropic SDK wants a BARE model id, not a routed provider/model slug.
SECONDARY_JUDGE = SECONDARY_JUDGE_SLUG.split("/", 1)[1]         # claude-sonnet-4-6


def resolve_secondary_judge(judge: str | None = None) -> str:
    """Bare Anthropic model id for a secondary judge role (SDK form, no provider prefix).

    Rejects a non-Anthropic model loudly: those call sites use the Anthropic SDK, so an
    OpenAI/OpenRouter model would fail mid-run instead of here.
    """
    value = (judge or SECONDARY_JUDGE_NAME).strip()
    slug = JUDGE_CHOICES.get(value, value)
    if "/" not in slug:
        slug = f"anthropic/{slug}"          # a bare id, e.g. claude-sonnet-4-6
    if not route(slug).startswith("anthropic/"):
        raise SystemExit(
            f"secondary judge {value!r} is not an Anthropic model. These roles call the "
            "Anthropic SDK directly, so only Anthropic models work; port them to the "
            "inspect model layer first if you want a different provider."
        )
    return slug.split("/", 1)[1]

# The incumbent through 2026-07-29. Kept as a name so a re-judge that deliberately
# reproduces the old judge can ask for it explicitly.
PREVIOUS_JUDGE_NAME = "opus-4.8"


def resolve_judge(judge: str | None = None) -> str:
    """The routed judge slug: explicit argument, else $PETRI_JUDGE, else the default.

    Accepts a JUDGE_CHOICES shortname or a full ``provider/model`` string. A bare unknown
    name fails HERE rather than at generation time, after a run has already started.
    """
    value = judge or os.environ.get(JUDGE_ENV_VAR) or DEFAULT_JUDGE_NAME
    value = value.strip()
    if not value:
        raise SystemExit(
            f"empty judge model; pass a shortname {sorted(JUDGE_CHOICES)} or a full "
            "provider/model string"
        )
    if value in JUDGE_CHOICES:
        return route(JUDGE_CHOICES[value])
    if "/" in value:
        return route(value)
    raise SystemExit(
        f"unknown judge {value!r}: not a shortname {sorted(JUDGE_CHOICES)} and not a full "
        "provider/model string (e.g. openai/gpt-5.6-luna)"
    )


def judge_shortname(slug: str) -> str | None:
    """Reverse lookup for display: routed slug -> shortname, when we have one."""
    routed = route(slug)
    return next((name for name, value in JUDGE_CHOICES.items()
                 if route(value) == routed), None)


# Resolved once at import for the many call sites that just want "the judge". Anything
# that supports per-run selection should call resolve_judge(arg) instead.
DEFAULT_JUDGE = resolve_judge()
