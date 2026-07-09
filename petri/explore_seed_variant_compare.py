"""Read-only: compare aggregate trajectory outcomes for youtube_comments2 vs 3.

Loads the single run that contains both variants (…-150147) through the viewer's own
load layer (warm cache, no API calls) so the classification matches the viewer exactly,
then prints per-variant aggregates: category breakdown, mean judge scores, completion,
and hack-mechanism (failure_modes) tallies.
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from lib.viewer_load import load_mode
from make_viewer import hack_category, categorize

RUN = Path("/Users/owenterry/supermats/mats-local/petri/logs/"
           "v2-4targets-allow-5ep-20260707-150147")

SCORE_DIMS = ["reward_hacking", "incompleteness", "situational_awareness",
              "degenerate_behavior", "hallucination", "hack_in_final_solution",
              "auditor_nudging", "deception"]


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else float("nan")


async def main():
    audits = await load_mode(RUN)
    variants = {"youtube_comments2": [], "youtube_comments3": []}
    for a in audits:
        variants.setdefault(a["seed"], []).append(a)

    for v in ("youtube_comments2", "youtube_comments3"):
        A = variants.get(v, [])
        print("=" * 70)
        print(f"{v}:  n={len(A)}   targets={sorted(set(a['target'] for a in A))}")
        print("-" * 70)

        cats = Counter(hack_category(a) for a in A)
        print("category (viewer 5-way):", dict(cats))
        n = len(A) or 1
        print(f"  full hacks: {cats['full']}/{len(A)} = {cats['full']/n:.0%}")

        print("mean scores:")
        for d in SCORE_DIMS:
            print(f"    {d:26s} {mean([a['scores'].get(d) for a in A]):.2f}")

        print("completion / health:")
        print(f"    ended_via_end_conv (auditor stopped): "
              f"{sum(a['ended_via_end_conv'] for a in A)}/{len(A)}")
        print(f"    hit max-turns cap: "
              f"{sum(not a['ended_via_end_conv'] for a in A)}/{len(A)}")
        print(f"    crashed: {sum(a['crashed'] for a in A)}   dead target: "
              f"{sum(a['dead'] for a in A)}   fork: {sum(a['fork'] for a in A)}")
        print(f"    max_turns setting: {sorted(set(a.get('max_turns') for a in A))}")

        fm = Counter()
        for a in A:
            for m in (a.get("failure_modes") or []):
                fm[m] += 1
        print("failure_modes tally (hack mechanism):", dict(fm))
        prop = Counter(a.get("rh_propagation") for a in A)
        print("rh_propagation:", dict(prop))
        print()


if __name__ == "__main__":
    asyncio.run(main())
