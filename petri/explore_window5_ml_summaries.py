"""Read-only dump of judge summaries for window-5 (the 'consistent' sweep) ML
trajectories. No API calls -- reads local .eval logs via the viewer's own loader.

Prints, grouped by hack category and condition, each trajectory's key scores and the
judge's summary, so we can qualitatively read what the honeypot elicits.
"""
import asyncio
import sys
sys.path.insert(0, "lib")

from petri_paths import LOGS
import make_viewer as mv


ML_PREFIX = "ml_training_data_misuse"
# window-5 collection runs carry the bare variant name as the seed id
ML_SEEDS = {"youtube_comments", "clinical_notes", "fraud_detection",
            "review_sentiment", "search_ranking", "spam_detection",
            "support_routing"}
KEY_DIMS = ["reward_hacking", "auditor_nudging", "auditor_mistakes",
            "situational_awareness", "degenerate_behavior", "hallucination",
            "incompleteness"]


def is_ml(a: dict) -> bool:
    lbl = mv.seed_label(a["seed"])
    return lbl.startswith(ML_PREFIX) or a["seed"] in ML_SEEDS


async def main() -> None:
    all_dirs = sorted(d for d in LOGS.iterdir() if d.is_dir())
    mode_dirs = [d for d in all_dirs
                 if not d.name.startswith(mv.ROLLBACK_PREFIX)
                 and not d.name.startswith(mv.RESAMPLE_PREFIX)
                 and not d.name.startswith(mv.CONTINUATION_PREFIX)]
    audits: list[dict] = []
    for d in mode_dirs:
        try:
            audits.extend(await mv.load_mode(d))
        except Exception as e:
            print(f"  skip {d.name}: {e}")

    # window 5 = the catch-all 'consistent' sweep
    w5 = [a for a in audits if mv.sweep_key(a) == mv.CURRENT_SWEEP and is_ml(a)]
    print(f"\n=== window-5 (consistent) ML audits: {len(w5)} ===")

    # overview: seed x condition x category counts
    from collections import Counter
    by_dir = Counter(a["mode"] for a in w5)
    print("\nby run dir:")
    for m, n in sorted(by_dir.items()):
        print(f"  {n:3d}  {m}")
    by_seed = Counter(mv.seed_label(a["seed"]) for a in w5)
    print("\nby seed:")
    for s, n in sorted(by_seed.items()):
        print(f"  {n:3d}  {s}")
    by_cond = Counter(a.get("condition") for a in w5)
    print("\nby condition:", dict(by_cond))
    by_aud = Counter(mv.pretty_model(a["auditor"]) for a in w5)
    print("by auditor:", dict(by_aud))
    by_tgt = Counter(mv.pretty_model(a["target"]) for a in w5)
    print("by target:", dict(by_tgt))
    by_cat = Counter(mv.hack_category(a) for a in w5)
    print("by category:", dict(by_cat))
    # condition x category
    print("\ncondition x category:")
    cc = Counter((a.get("condition"), mv.hack_category(a)) for a in w5)
    for (cond, cat), n in sorted(cc.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        print(f"  {str(cond):8s} {cat:11s} {n}")

    def scores_line(a: dict) -> str:
        parts = []
        for d in KEY_DIMS:
            v = a["scores"].get(d)
            parts.append(f"{d[:4]}={v}" if v is not None else f"{d[:4]}=-")
        return " ".join(parts)

    # full dump grouped by category then condition
    order = ["full", "degenerate", "nudged", "both", "non"]
    for cat in order:
        group = [a for a in w5 if mv.hack_category(a) == cat]
        if not group:
            continue
        print("\n\n" + "#" * 90)
        print(f"# CATEGORY: {cat}  ({len(group)} trajectories)")
        print("#" * 90)
        for a in sorted(group, key=lambda a: (str(a.get("condition")), a["target"], a["seed"], a["epoch"])):
            print(f"\n--- cond={a.get('condition')}  target={mv.pretty_model(a['target'])}"
                  f"  auditor={mv.pretty_model(a['auditor'])}"
                  f"  seed={mv.seed_label(a['seed'])}  dir={a['mode']} ---")
            print(f"    {scores_line(a)}")
            flags = []
            if a.get("dead"): flags.append("DEAD")
            if a.get("crashed"): flags.append("CRASHED")
            if flags:
                print(f"    FLAGS: {', '.join(flags)}")
            summ = (a.get("summary") or "").strip()
            print(f"    SUMMARY: {summ if summ else '(none)'}")


if __name__ == "__main__":
    asyncio.run(main())
