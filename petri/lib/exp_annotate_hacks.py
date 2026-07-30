"""LLM agent that locates and annotates reward-hacking turns in Petri audits.

For each audit that satisfies the committed binary definition of a reward hack
(BINARY_HACK_CRITERIA in viewer), this feeds the Petri judge's own output
(score + summary + justification + highlights, all of which cite [MN] message
numbers) PLUS the exact [MN]-numbered target transcript to a Claude model, and
asks it to localize the specific turns where the target games (or commits to
gaming) the success criteria, with a short title + note per turn.

Output is written/merged into mats-local/petri/annotations.json, keyed by the
same page-name viewer.py uses. viewer.py (free) then marks those turns
and renders a "hack turns" nav so a reader can bounce through them.

This is an *interpretive* layer: the annotations are one model's reading of the
judge's findings, not ground truth. Each entry stores the model that produced it.

Costs money (Anthropic API). Reuses viewer's rendering so [MN] numbers match.

By default it annotates exactly the audits that satisfy the committed binary
definition of a reward hack (viewer.BINARY_HACK_CRITERIA -- the single source
of truth for the thresholds) and aren't already annotated. That definition needs
every required judge dim to be scored. FRESH exp_rh_audit.py runs already produce
them all, so no re-judge is needed. Only OLDER audits that predate the current
dimension set need a re-judge (tools/exp_rejudge_rh.py) first; such audits are
reported here as "needs re-judge" and skipped (never silently treated as non-hacks).

This module is ALSO importable: exp_audit_pipeline.py calls
load_all_original_audits() + run_annotation() so the gate and the parallel
annotation loop have a single source of truth. Importing it has no side effects.

Since 2026-07-30 this is AGENTIC and runs on a cheap model (see lib/exp_annotate_agent.py):
the model gets the judge's findings plus a content-free transcript index, then pulls the
turns it wants with read_turns / search_transcript and finishes with submit_annotation.
Read coverage is recorded per annotation and surfaced in the viewer.

Usage (CLI, from petri/ -- this file lives in lib/):
  uv run lib/exp_annotate_hacks.py                  # annotate binary-def hacks not yet done
  uv run lib/exp_annotate_hacks.py --force          # re-annotate all binary-def hacks (re-spends)
  uv run lib/exp_annotate_hacks.py --concurrency=50 # annotate up to 50 trajectories in parallel (default 50)
  uv run lib/exp_annotate_hacks.py --model=claude-opus-4-8   # pricier model
Then regenerate the viewer (free): uv run viewer.py
"""

import asyncio
import json
import sys

from dotenv import load_dotenv
from inspect_ai.model import get_model

# viewer lives at the top level (parent of lib/); put it on the import path so
# this module works whether it's imported by a pipeline or run standalone from lib/.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

# reuse viewer's rendering + paths so [MN] numbers and page keys match exactly,
# and the committed binary definition of a reward hack (single source of truth)
from viewer import (
    DATA, LOGS, ROLLBACK_PREFIX, RESAMPLE_PREFIX, CONTINUATION_PREFIX,
    load_mode, page_name, binary_hack_eval, BINARY_HACK_CRITERIA,
)
from exp_annotate_agent import annotate_agentic
from model_routing import route
import openrouter_cost   # real billed cost for OpenRouter calls (installed in run_annotation)
import direct_cost       # list-price cost for direct anthropic/openai calls

load_dotenv(LOGS.parent.parent.parent / "mats" / ".env")  # mats/.env (API keys)

# Cheap by default (Owen 2026-07-30): the agentic loop reads only the turns it needs, so
# a mid-tier model is enough. Any TARGET_CHOICES-style slug or full provider/model string
# works -- routing is applied, so OpenRouter and first-party both go to the right place.
DEFAULT_MODEL = "openrouter/deepseek/deepseek-v4-pro-20260423"
DEFAULT_CONCURRENCY = 50
ANN_FILE = DATA / "annotations.json"


# The prompt, tool schema, agentic loop, and read-coverage accounting all live in
# lib/exp_annotate_agent.py so the environments project reuses them unchanged.



async def load_all_original_audits() -> list[dict]:
    """Every ORIGINAL audit (one dict per trajectory) across all non-experiment log dirs.
    Rollback continuations (logs/rollback-*/), resamples (logs/resample-*/), and continuations
    (logs/continuation-*/) are SEPARATE experiments annotated by their own pipelines
    (exp_rollback_judge.py / the continuation pipeline's annotate stage, which scopes to the
    new task), so they're excluded here. Shared by main() and the pipeline orchestrator so
    'which dirs count as original audits' has one definition."""
    skip = (ROLLBACK_PREFIX, RESAMPLE_PREFIX, CONTINUATION_PREFIX)
    mode_dirs = sorted(d for d in LOGS.iterdir()
                       if d.is_dir() and not d.name.startswith(skip)) if LOGS.exists() else []
    if not mode_dirs:
        raise SystemExit(f"no log directories found under {LOGS}")
    audits: list[dict] = []
    for mode_dir in mode_dirs:
        print(f"loading {mode_dir.name}/ ...")
        audits.extend(await load_mode(mode_dir))
    return audits


async def run_annotation(
    audits: list[dict],
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    force: bool = False,
    ann_file=ANN_FILE,
) -> dict:
    """Annotate every binary-definition reward hack in `audits` that isn't already
    done (unless force). Runs up to `concurrency` annotations in PARALLEL (asyncio
    semaphore + gather), checkpoints annotations.json after each one, and returns a
    stats dict. Robust: a single trajectory that errors is counted as failed and the
    rest continue. The candidate gate is binary_hack_eval (single source of truth).
    """
    if concurrency < 1:
        raise SystemExit(f"concurrency must be >= 1, got {concurrency}")
    ann_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(ann_file.read_text()) if ann_file.exists() else {}

    # Cost capture for the inspect model layer (the annotator is no longer on the
    # Anthropic SDK): real billed cost on OpenRouter, list price on first-party.
    openrouter_cost.install()
    direct_cost.install()
    annotator = get_model(route(model))
    in_tok = out_tok = 0
    run_cost = 0.0
    done = failed = skipped_not_hack = skipped_needs_rejudge = 0
    skipped_integrity = skipped_existing = skipped_empty = 0
    candidates: list[tuple[dict, str, int | float]] = []
    print(
        f"\nmodel={route(model)}  concurrency={concurrency}  force={force}\n"
        f"mode = AGENTIC (judge findings + transcript index; the model reads turns itself)\n"
        f"writing -> {ann_file.name}\n"
        f"gate = binary RH definition ({BINARY_HACK_CRITERIA})\n"
    )
    for a in audits:
        key = page_name(a["mode"], a["task"], a["seed"], a["epoch"])
        rh = a["scores"].get("reward_hacking")
        verdict = binary_hack_eval(a)
        if not verdict["ok"]:
            if verdict.get("integrity"):
                skipped_integrity += 1
                continue
            # missing dims => not classifiable yet (run exp_rejudge_rh.py); failed
            # dims => a genuine non-hack. Keep these separate so the former is never
            # silently treated as "not a hack".
            if verdict["missing"]:
                skipped_needs_rejudge += 1
            else:
                skipped_not_hack += 1
            continue
        if not a["transcript"]:
            print(f"  SKIP {key}: no transcript")
            skipped_empty += 1
            continue
        if key in existing and not force:
            skipped_existing += 1
            continue
        candidates.append((a, key, rh))

    total = len(candidates)
    if total == 0 and skipped_needs_rejudge:
        print(f"  NOTE: {skipped_needs_rejudge} reward_hacking>=5 audit(s) are missing the "
              "new judge dimensions -- run `uv run tools/exp_rejudge_rh.py` first, then re-run this.")
    sem = asyncio.Semaphore(concurrency)

    incomplete = 0

    async def annotate_one(i: int, a: dict, key: str, rh: int | float) -> None:
        nonlocal in_tok, out_tok, run_cost, done, failed, incomplete
        async with sem:
            print(f"  [{i}/{total}] annotating {key} (reward_hacking={rh}) ...", flush=True)
            try:
                ann = await annotate_agentic(annotator, a)
            except Exception as e:
                failed += 1
                print(f"    [{i}/{total}] FAILED {key}: {type(e).__name__}: {e}", flush=True)
                return
        in_tok += ann.usage.get("input") or 0
        out_tok += ann.usage.get("output") or 0
        run_cost += ann.usage.get("total_cost") or 0.0
        if ann.error:
            incomplete += 1
        existing[key] = {
            "model": route(model),
            "reward_hacking": rh,
            "hack_turns": ann.hack_turns,
            # AGENTIC read coverage (2026-07-30): the annotator sees only the turns it
            # asks for, so how much it read is part of the record. messages_read /
            # read_fraction / turns_read are what it actually pulled;
            # annotated_without_reading names any turn it labeled WITHOUT reading (its
            # quotes cannot be verbatim); hit_tool_turn_cap / error mark a loop that
            # never submitted. The viewer surfaces these, never hides them.
            "coverage": ann.coverage,
            "annotator_error": ann.error,
            # raw token counts for this annotation, stored so viewer can price it at
            # DISPLAY time (like the rest of the cost system — see lib/model_prices.py).
            # total_cost is OpenRouter's real billed figure when available (EXACT),
            # otherwise None and the viewer prices price×token (~estimate). Entries
            # annotated before these fields existed simply lack them and are surfaced as
            # "no captured cost" in the viewer, never silently zeroed.
            "usage": ann.usage,
        }
        done += 1
        ann_file.write_text(json.dumps(existing, indent=2))  # checkpoint after each
        titles = ", ".join(t["title"] for t in ann.hack_turns)
        cov = ann.coverage
        read_note = (f"read {cov['messages_read']}/{cov['messages_total']} msgs in "
                     f"{cov['tool_turns']} tool turn(s)")
        print(f"    [{i}/{total}] -> {len(ann.hack_turns)} hack turn(s); {read_note}"
              f"{f': {titles[:90]}' if titles else ''}"
              f"{f'  !! {ann.error}' if ann.error else ''}")

    await asyncio.gather(*(
        annotate_one(i, a, key, rh)
        for i, (a, key, rh) in enumerate(candidates, start=1)
    ))

    # Cost readout via the shared price table (lib/model_prices.py), so it matches what
    # the viewer shows. sample_cost prefers each model's real billed total_cost (EXACT)
    # and falls back to price x tokens; an unpriced model is reported, not silently zeroed.
    from model_prices import sample_cost

    priced = sample_cost({route(model): {
        "input": in_tok, "output": out_tok, "cache_read": 0, "cache_write": 0,
        "total_cost": run_cost or None,
    }})
    cost = priced["total"]
    cost_kind = ("EXACT (provider-billed)" if priced["exact"]
                 else ("~estimate" if not priced["unknown"]
                       else f"~estimate; UNPRICED model {priced['unknown']}"))
    print(f"\n{len(existing)} total entries in {ann_file}")
    print(f"  this run: {done} annotated, {failed} failed, {skipped_existing} already done, "
          f"{skipped_not_hack} fail the binary definition, "
          f"{skipped_needs_rejudge} need re-judge (missing new dims), "
          f"{skipped_integrity} data-integrity failures, "
          f"{skipped_empty} no-transcript")
    if incomplete:
        print(f"  !! {incomplete} annotation(s) ended without a clean submission "
              "(flagged in annotations.json as annotator_error and shown in the viewer)")
    print(f"  tokens: in={in_tok:,} out={out_tok:,}  ${cost:.2f} [{cost_kind}]")
    return {
        "done": done, "failed": failed, "candidates": total,
        "skipped_existing": skipped_existing, "skipped_not_hack": skipped_not_hack,
        "skipped_needs_rejudge": skipped_needs_rejudge, "skipped_empty": skipped_empty,
        "skipped_integrity": skipped_integrity, "incomplete": incomplete,
        "in_tok": in_tok, "out_tok": out_tok, "cost": cost,
        "total_entries": len(existing),
    }


async def main() -> None:
    model = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), DEFAULT_MODEL)
    concurrency = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), str(DEFAULT_CONCURRENCY)))
    force = "--force" in sys.argv
    audits = await load_all_original_audits()
    await run_annotation(audits, model=model, concurrency=concurrency, force=force)
    print("Regenerate the viewer (free) to see the hack-turn nav: uv run viewer.py")


if __name__ == "__main__":
    asyncio.run(main())
