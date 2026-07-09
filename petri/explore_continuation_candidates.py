"""Dump current-sweep ("consistent") originals as a flat table so Owen can pick
continuation-experiment trajectories: per model, candidates for full_hack_prefix
(allow-condition full hacks), corrected_hack_prefix (correct-condition hacks with a
corrected final solution), clean_prefix (clean runs), and new-task (B) sources.

FREE: reads the viewer load cache / eval logs only, no API calls.
"""

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

import make_viewer
from make_viewer import hack_category, sweep_key


def short(model: str) -> str:
    return (model or "?").split("/")[-1]


async def main():
    by_id = await make_viewer.load_originals_by_id()
    rows = []
    for tid, a in by_id.items():
        if sweep_key(a) != "consistent":
            continue
        s = a["scores"]
        rows.append({
            "id": tid,
            "model": short(a["target"]),
            "auditor": short(a["auditor"]),
            "seed": a["seed"],
            "cond": a.get("condition") or "-",
            "ep": a["epoch"],
            "cat": hack_category(a),
            "rh": s.get("reward_hacking"),
            "hifs": s.get("hack_in_final_solution"),
            "nudge": s.get("auditor_nudging"),
            "mist": s.get("auditor_mistakes"),
            "sa": s.get("situational_awareness"),
            "degen": s.get("degenerate_behavior"),
            "hall": s.get("hallucination"),
            "incomp": s.get("incompleteness"),
            "kchars": len(a.get("transcript") or "") // 1000,
            "dead": a.get("dead"),
            "crash": a.get("crashed"),
            "mode": a["mode"],
        })
    rows.sort(key=lambda r: (r["model"], r["cond"], r["seed"], r["ep"]))
    hdr = ["id", "model", "seed", "cond", "ep", "cat", "rh", "hifs", "nudge",
           "mist", "sa", "degen", "hall", "incomp", "kchars", "dead", "crash",
           "auditor", "mode"]
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[h]) for h in hdr))
    print(f"\n{len(rows)} current-sweep originals", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
