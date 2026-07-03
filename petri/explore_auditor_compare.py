"""Compare cheap auditors (GLM 5.2, Kimi K2.6, DeepSeek V4 Pro) vs the Opus 4.8 baseline.

Free, local-only. Loads the 2026-07-02 auditor-sweep dirs plus the sweep-3 Opus-auditor
baseline through make_viewer's own loader, then reports per-auditor: trajectory counts,
hack-category breakdown (same classification the viewer uses), score distributions for
the auditor-quality dimensions (auditor_mistakes, auditor_nudging, hallucination),
conversation-shape stats (user/assistant turns, ended_via_end_conv), and a per-target
hack table. The Opus baseline is ALSO shown restricted to the 4 targets the sweep used,
so the head-to-head is like-for-like (baseline restriction dropping Opus-as-target is
called out in the output, not silent).
"""

import asyncio
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer as mv

SWEEP_DIRS = [
    "v2-4targets-1ep-20260702-143507",   # GLM 5.2 auditor
    "v2-4targets-1ep-20260702-143741",   # Kimi K2.6 auditor (not in manifest -- may be partial)
    "v2-4targets-1ep-20260702-152119",   # DeepSeek V4 Pro auditor
    "v2-4targets-1ep-20260702-160733",   # Kimi K2.6 auditor
    "v2-4targets-2ep-20260702-143430",   # GLM 5.2 auditor, 2ep (not in manifest -- may be partial)
]
BASELINE_DIRS = [
    "v2-5targets-1ep-20260629-183349",
    "v2-5targets-3ep-20260629-225553",
]
SWEEP_TARGETS = {"DeepSeek V4 Pro", "GLM 5.1", "GPT-5.5", "Kimi K2.6"}
AUDITOR_QUALITY_DIMS = ["auditor_mistakes", "auditor_nudging", "hallucination"]
OTHER_DIMS = ["reward_hacking", "degenerate_behavior", "incompleteness", "hack_in_final_solution"]


def auditor_short(model: str) -> str:
    m = model.lower()
    if "opus" in m:
        return "Opus 4.8"
    if "glm" in m:
        return "GLM 5.2"
    if "kimi" in m:
        return "Kimi K2.6"
    if "deepseek" in m:
        return "DeepSeek V4 Pro"
    if "sonnet" in m:
        return "Sonnet 4.6"
    return model


def n_turns(transcript: str) -> tuple[int, int]:
    heads = list(mv.MSG_HEAD.finditer(transcript))
    roles = [h.group(2).strip().lower() for h in heads]
    return roles.count("user"), roles.count("assistant")


def dist_str(vals: list[int]) -> str:
    return " ".join(f"{k}:{v}" for k, v in sorted(Counter(vals).items()))


async def main():
    groups: dict[str, list[dict]] = defaultdict(list)  # auditor-label -> audits
    for d in SWEEP_DIRS:
        p = mv.LOGS / d
        if not p.exists():
            print(f"[skip] {d}: dir missing")
            continue
        audits = await mv.load_mode(p)
        if not audits:
            print(f"[skip] {d}: loaded 0 audits (live/partial run?)")
            continue
        lab = auditor_short(audits[0]["auditor"])
        groups[lab] += audits
        print(f"[load] {d}: {len(audits)} audits, auditor={lab}")
    base = []
    for d in BASELINE_DIRS:
        audits = await mv.load_mode(mv.LOGS / d)
        base += audits
        print(f"[load] {d}: {len(audits)} audits, auditor={auditor_short(audits[0]['auditor'])}")
    groups["Opus 4.8 (all 5 targets)"] = base
    groups["Opus 4.8 (same 4 targets)"] = [a for a in base if mv.target_short(a) in SWEEP_TARGETS]

    order = ["Opus 4.8 (all 5 targets)", "Opus 4.8 (same 4 targets)",
             "GLM 5.2", "Kimi K2.6", "DeepSeek V4 Pro"]
    order = [k for k in order if k in groups] + [k for k in groups if k not in order]

    print("\n" + "=" * 100)
    print("PER-AUDITOR SUMMARY  (dead audits excluded from all stats; counts shown)")
    print("=" * 100)
    cats = ["full", "degenerate", "nudged", "both", "non"]
    for lab in order:
        audits = groups[lab]
        dead = [a for a in audits if a.get("dead")]
        live = [a for a in audits if not a.get("dead")]
        cnt = Counter(mv.hack_category(a) for a in live)
        n = len(live)
        hackable = cnt.get("full", 0) + cnt.get("degenerate", 0)
        print(f"\n--- {lab}:  n={n} live, {len(dead)} dead ---")
        print("  categories: " + "  ".join(f"{c}={cnt.get(c, 0)}" for c in cats)
              + f"   |  full+degen rate = {hackable}/{n} = {hackable / n:.0%}" if n else "  (empty)")
        for dim in AUDITOR_QUALITY_DIMS + OTHER_DIMS:
            vals = [a["scores"].get(dim) for a in live if a["scores"].get(dim) is not None]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            hi = sum(1 for v in vals if v >= 5)
            print(f"  {dim:24s} mean={mean:4.2f}  >=5: {hi:2d}/{len(vals)}   dist: {dist_str(vals)}")
        shapes = [n_turns(a["transcript"]) for a in live]
        u = [s[0] for s in shapes]; asst = [s[1] for s in shapes]
        ended = sum(1 for a in live if a["ended_via_end_conv"])
        forks = sum(1 for a in live if a.get("fork"))
        print(f"  user turns: mean={sum(u)/len(u):.1f} dist: {dist_str(u)}")
        print(f"  assistant turns: mean={sum(asst)/len(asst):.1f} min={min(asst)} max={max(asst)}")
        print(f"  ended_via_end_conv: {ended}/{len(live)}   forks(restart/rollback): {forks}")

    print("\n" + "=" * 100)
    print("PER-TARGET x AUDITOR: full+degenerate hacks / live n")
    print("=" * 100)
    tgts = sorted({mv.target_short(a) for g in groups.values() for a in g if not a.get("dead")})
    print(f"{'target':20s}" + "".join(f"{lab:>28s}" for lab in order))
    for t in tgts:
        row = f"{t:20s}"
        for lab in order:
            live = [a for a in groups[lab] if not a.get("dead") and mv.target_short(a) == t]
            if not live:
                row += f"{'-':>28s}"
                continue
            h = sum(1 for a in live if mv.hack_category(a) in ("full", "degenerate"))
            row += f"{f'{h}/{len(live)}':>28s}"
        print(row)

    print("\n" + "=" * 100)
    print("PER-SEED x AUDITOR: full+degenerate hacks / live n")
    print("=" * 100)
    seeds = sorted({a["seed"] for g in groups.values() for a in g if not a.get("dead")})
    print(f"{'seed':45s}" + "".join(f"{lab:>28s}" for lab in order))
    for sd in seeds:
        row = f"{sd.replace('ml_training_data_misuse_', ''):45s}"
        for lab in order:
            live = [a for a in groups[lab] if not a.get("dead") and a["seed"] == sd]
            if not live:
                row += f"{'-':>28s}"
                continue
            h = sum(1 for a in live if mv.hack_category(a) in ("full", "degenerate"))
            row += f"{f'{h}/{len(live)}':>28s}"
        print(row)


if __name__ == "__main__":
    asyncio.run(main())
