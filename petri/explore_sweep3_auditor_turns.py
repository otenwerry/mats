"""Sweep-3 auditor-variability analysis (free, local-only).

Question (Owen, 2026-07-02): in the sweep-3 runs (old sub-tab 3, the fixed-SP ML-training
sweep), how much of hack vs. non-hack is explained by what the AUDITOR chose to do —
in particular the "let the target finish cleanly, then invite it to improve its score"
move — rather than by the target's own propensity?

This script loads the two sweep-3 dirs through make_viewer's own loader (same data the
viewer shows), then:
  1. prints a per-trajectory summary table (category, scores, #user turns, #assistant
     turns, first-hack position);
  2. dumps every USER turn of every trajectory (auditor-written, per the transcript
     role convention) to mats-local/petri/_analysis/sweep3_user_turns_dump.txt for
     manual reading/labeling. Bodies are TRUNCATED to TRUNC chars per turn in the dump
     (marked with "...[+N chars]"); the summary table has no truncation.
"""

import asyncio
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer as mv

DIRS = ["v2-5targets-1ep-20260629-183349", "v2-5targets-3ep-20260629-225553"]
OUT_DIR = mv.DATA / "_analysis"
TRUNC = 700  # chars of each user-turn body kept in the dump


def parse_messages(transcript: str) -> list[tuple[int, str, str]]:
    """(m, role, body) for every [M#] message-head in a rendered transcript."""
    heads = list(mv.MSG_HEAD.finditer(transcript))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(transcript)
        body = transcript[h.end():end].strip()
        out.append((int(h.group(1)), h.group(2).strip().lower(), body))
    return out


async def main():
    audits = []
    for d in DIRS:
        audits += await mv.load_mode(mv.LOGS / d)
    audits = [a for a in audits if not a.get("dead")]
    ann_file = mv.DATA / "annotations.json"
    annotations = json.loads(ann_file.read_text()) if ann_file.exists() else {}

    rows = []
    dump_sections = defaultdict(list)  # category -> [text]
    for a in audits:
        msgs = parse_messages(a["transcript"])
        users = [(m, body) for m, role, body in msgs if role == "user"]
        n_assist = sum(1 for _, role, _ in msgs if role == "assistant")
        cat = mv.hack_category(a)
        page = mv.page_name(a["mode"], a["task"], a["seed"], a["epoch"])
        ann = annotations.get(page)
        fh_m = mv.first_hack_m(ann)
        fh_a = mv.assistant_turn_index(a["transcript"], fh_m)
        users_before_hack = (
            sum(1 for m, _ in users if m < fh_m) if isinstance(fh_m, int) else None
        )
        rows.append(dict(
            target=mv.target_short(a), seed=a["seed"].replace("ml_training_data_misuse_", ""),
            epoch=a["epoch"], mode=("1ep" if "1ep" in a["mode"] else "3ep"),
            cat=cat, rh=mv.rh_score(a), nudge=a["scores"].get("auditor_nudging"),
            n_user=len(users), n_assist=n_assist,
            first_hack_a=fh_a, users_before_hack=users_before_hack,
            ended_via_end_conv=a["ended_via_end_conv"],
        ))
        # dump user turns for manual reading
        head = (f"### {cat.upper()} | {mv.target_short(a)} | "
                f"{rows[-1]['seed']} e{a['epoch']} ({rows[-1]['mode']}) | "
                f"rh={rows[-1]['rh']} nudge={rows[-1]['nudge']} | "
                f"{len(users)} user turns, {n_assist} assistant turns | "
                f"first hack: {'A' + str(fh_a) if fh_a else '-'} (M{fh_m})")
        lines = [head]
        for m, body in users:
            marker = " <-- LAST USER TURN BEFORE FIRST HACK" if (
                isinstance(fh_m, int) and m < fh_m
                and all(not (m < m2 < fh_m) for m2, _ in users)) else ""
            body_txt = re.sub(r"\s+", " ", body)
            if len(body_txt) > TRUNC:
                body_txt = body_txt[:TRUNC] + f" ...[+{len(body_txt) - TRUNC} chars]"
            lines.append(f"  [M{m}]{marker} {body_txt}")
        dump_sections[cat].append("\n".join(lines))

    # ---- summary table ----
    rows.sort(key=lambda r: (r["cat"], r["target"], r["seed"], r["epoch"]))
    cols = ["cat", "target", "seed", "epoch", "mode", "rh", "nudge", "n_user",
            "n_assist", "first_hack_a", "users_before_hack", "ended_via_end_conv"]
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))

    # ---- aggregates ----
    print(f"\ntotal trajectories: {len(rows)}")
    print("\ncategory counts:", dict(Counter(r["cat"] for r in rows)))
    print("\nuser-turn count by category (mean / min / max / distribution):")
    for cat in ["full", "degenerate", "nudged", "both", "non"]:
        vals = [r["n_user"] for r in rows if r["cat"] == cat]
        if not vals:
            continue
        dist = dict(sorted(Counter(vals).items()))
        print(f"  {cat:11s} n={len(vals):3d}  mean={sum(vals)/len(vals):.1f}  "
              f"min={min(vals)}  max={max(vals)}  dist={dist}")
    print("\nuser turns BEFORE first hack (full+degenerate hacks with annotation):")
    vals = [r["users_before_hack"] for r in rows
            if r["users_before_hack"] is not None]
    print(f"  n={len(vals)}  dist={dict(sorted(Counter(vals).items()))}")
    print("\nfirst-hack assistant-turn index distribution:")
    fh = [r["first_hack_a"] for r in rows if r["first_hack_a"] is not None]
    print(f"  n={len(fh)}  dist={dict(sorted(Counter(fh).items()))}")

    # ---- per target x category ----
    print("\ntarget x category:")
    tgts = sorted({r["target"] for r in rows})
    cats = ["full", "degenerate", "nudged", "both", "non"]
    print("  " + "\t".join(["target"] + cats))
    for t in tgts:
        cnt = Counter(r["cat"] for r in rows if r["target"] == t)
        print("  " + "\t".join([t] + [str(cnt.get(c, 0)) for c in cats]))

    # ---- dump ----
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "sweep3_user_turns_dump.txt"
    parts = []
    for cat in ["full", "degenerate", "nudged", "both", "non"]:
        if dump_sections[cat]:
            parts.append(f"\n{'=' * 100}\nCATEGORY: {cat} ({len(dump_sections[cat])} trajectories)\n{'=' * 100}\n")
            parts.append("\n\n".join(dump_sections[cat]))
    out.write_text("\n".join(parts))
    print(f"\nwrote user-turn dump (bodies truncated to {TRUNC} chars): {out}")


if __name__ == "__main__":
    asyncio.run(main())
