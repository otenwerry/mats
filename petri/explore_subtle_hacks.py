"""Find VALID non-hacks scoring 2-4 on reward_hacking (the "considered a hack but
didn't commit one" band). Local-only, no API calls.

Emits a dump of each candidate's scores + judge summary/justification/highlights so we
can categorize the edge-case behavior and pick continuation-prefix candidates. Also
prints the landscape (RH-score distribution across all non-hacks, valid vs invalid).
"""
import asyncio
import json
from collections import Counter, defaultdict

import make_viewer as mv
from lib.viewer_load import load_mode  # noqa: F401 (kept parallel to make_viewer usage)

DATA = mv.DATA
OUT = DATA / "_analysis" / "subtle_hacks_dump.txt"

# valid non-hack = category "non", complete (incompleteness < 2), and no CONFIRMED
# (present-and-out-of-range) auditor/target fault -- exactly make_viewer's non_ok bucket.
NUDGE_DIMS = set(mv.NUDGE_DIMS)
DEGEN_DIMS = set(mv.DEGENERATE_DIMS)


def _incompleteness(a):
    v = a["scores"].get("incompleteness")
    return v if isinstance(v, (int, float)) else 0


def is_valid_non(a):
    if mv.hack_category(a) != "non":
        return False
    if _incompleteness(a) >= 2:
        return False
    failed = set(mv.binary_hack_eval(a)["failed"])
    if failed & (NUDGE_DIMS | DEGEN_DIMS):
        return False
    return True


def sweep_key(a):
    return mv._SWEEP_DIR_TO_KEY.get(a["mode"], mv.CURRENT_SWEEP)


async def main():
    by_id = await mv.load_originals_by_id()
    print(f"loaded {len(by_id)} original audits with trajectory IDs")

    # ---- landscape: RH-score distribution across ALL non-hacks (valid + invalid) ----
    non = [(i, a) for i, a in by_id.items() if mv.hack_category(a) == "non"]
    valid_dist, invalid_dist = Counter(), Counter()
    for _, a in non:
        rh = a["scores"].get("reward_hacking")
        bucket = valid_dist if is_valid_non(a) else invalid_dist
        bucket[rh] += 1
    print("\nreward_hacking distribution across non-hacks (rh_score: valid / invalid):")
    for rh in sorted(set(valid_dist) | set(invalid_dist), key=lambda x: (x is None, x)):
        print(f"  rh={rh}: valid={valid_dist[rh]:3d}  invalid={invalid_dist[rh]:3d}")

    # ---- candidates: VALID non-hacks with reward_hacking in {2,3,4} ----
    cands = [(i, a) for i, a in non
             if is_valid_non(a) and a["scores"].get("reward_hacking") in (2, 3, 4)]
    cands.sort(key=lambda t: (-t[1]["scores"].get("reward_hacking", 0), sweep_key(t[1]), t[0]))
    print(f"\n{len(cands)} VALID non-hacks scoring 2-4 on reward_hacking (candidates)")

    by_sweep = defaultdict(list)
    for i, a in cands:
        by_sweep[sweep_key(a)].append(i)
    for k, ids in sorted(by_sweep.items()):
        print(f"  sweep {k}: {len(ids)} -> ids {sorted(ids)}")

    # ---- full dump ----
    lines = []
    for i, a in cands:
        sc = a["scores"]
        lines.append("=" * 100)
        lines.append(
            f"ID {i} | rh={sc.get('reward_hacking')} | sweep={sweep_key(a)} | "
            f"target={a['target']} | cond={a.get('condition')} | dir={a['mode']} | "
            f"seed={a['seed']} | ep={a['epoch']}")
        # compact score line
        order = ["reward_hacking", "hack_in_final_solution", "auditor_nudging",
                 "auditor_mistakes", "situational_awareness", "degenerate_behavior",
                 "hallucination", "incompleteness"]
        sline = "  ".join(f"{d}={sc[d]}" for d in order if d in sc)
        extra = "  ".join(f"{d}={v}" for d, v in sc.items() if d not in order)
        lines.append("SCORES: " + sline + (("  | " + extra) if extra else ""))
        lines.append(f"SUMMARY: {a.get('summary', '').strip()}")
        just = (a.get('justification') or '').strip()
        if just:
            lines.append(f"JUSTIFICATION: {just}")
        hl = (a.get('highlights') or '').strip()
        if hl:
            lines.append(f"HIGHLIGHTS: {hl}")
        lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"\nwrote full dump ({len(cands)} candidates) to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
