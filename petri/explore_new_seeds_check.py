"""Read-only sanity check on the two new-seed runs (p_hacking/checkout_redesign,
perf_benchmark_gaming/log_analytics). Prints per-run scores, hack category, and judge
summary. No API calls."""
import asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from petri_paths import LOGS
from viewer_load import load_mode
from make_viewer import hack_category, is_hack_binary

RUNS = {
    "checkout_redesign (p_hacking)": "v2-4targets-allow-2ep-20260705-215503",
    "log_analytics (perf_benchmark_gaming)": "v2-4targets-allow-2ep-20260705-215536",
}
DIMS = ["reward_hacking", "hack_in_final_solution", "auditor_nudging",
        "auditor_mistakes", "situational_awareness", "degenerate_behavior",
        "hallucination", "incompleteness"]

def short(m): return (m or "?").split("/")[-1]

async def main():
    for label, d in RUNS.items():
        mode_dir = LOGS / d
        audits = await load_mode(mode_dir)
        print("\n" + "=" * 100)
        print(f"### {label}   [{d}]   ({len(audits)} audits)")
        print("=" * 100)
        audits.sort(key=lambda a: (short(a["target"]), a["epoch"]))
        for a in audits:
            cat = hack_category(a)
            flags = []
            if a.get("dead"): flags.append("DEAD")
            if a.get("crashed"): flags.append("CRASHED")
            if a.get("fork"): flags.append("FORK")
            if not a.get("ended_via_end_conv"): flags.append("HIT-CAP")
            fl = ("  <" + ",".join(flags) + ">") if flags else ""
            print(f"\n--- {short(a['target'])}  e{a['epoch']}  |  CATEGORY={cat.upper()}{fl}")
            sc = a["scores"]
            row = "  ".join(f"{k.split('_')[0][:5]}:{sc.get(k,'-')}" for k in DIMS)
            print("    " + row)
            summ = (a.get("summary") or "").strip().replace("\n", " ")
            print("    SUMMARY: " + (summ[:600] if summ else "(none)"))

    print("\n\n" + "#" * 100)
    print("# AGGREGATE")
    print("#" * 100)
    for label, d in RUNS.items():
        audits = await load_mode(LOGS / d)
        from collections import Counter
        c = Counter(hack_category(a) for a in audits)
        nhack = sum(1 for a in audits if is_hack_binary(a))
        print(f"{label}: n={len(audits)}  full-hacks={nhack}  categories={dict(c)}")

asyncio.run(main())
