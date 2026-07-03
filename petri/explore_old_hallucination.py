"""Read-only exploration (no API calls): understand hallucination coverage on the OLD
trajectories for the "dumb" models (every target NOT among the five main-trajectory models).
Prints per-model base hallucination + the aggregate reward-hack hallucination picture so I can
pick a good operationalization before wiring it into the Visuals page."""
import asyncio
from collections import Counter, defaultdict

import make_viewer as mv


async def main() -> None:
    all_dirs = sorted(d for d in mv.LOGS.iterdir() if d.is_dir())
    mode_dirs = [d for d in all_dirs if not d.name.startswith(
        (mv.ROLLBACK_PREFIX, mv.RESAMPLE_PREFIX, mv.CONTINUATION_PREFIX))]
    audits: list[dict] = []
    for d in mode_dirs:
        audits.extend(await mv.load_mode(d))
    mv.assign_ids(audits)

    main = [a for a in audits if not mv.is_old_trajectory(a)]
    old = [a for a in audits if mv.is_old_trajectory(a)]
    main_models = {mv.pretty_model(a["target"]) for a in main}
    print(f"\nmain audits={len(main)}  old audits={len(old)}")
    print(f"MAIN models ({len(main_models)}): {sorted(main_models)}")

    dumb = [a for a in old if mv.pretty_model(a["target"]) not in main_models]
    print(f"\nDUMB old audits (target not in main): {len(dumb)}")

    def hal(a):
        v = a["scores"].get("hallucination")
        return v if isinstance(v, (int, float)) else None

    by = defaultdict(list)
    for a in dumb:
        by[mv.pretty_model(a["target"])].append(a)
    print(f"\n{'model':22s} {'n':>4} {'halNN':>6} {'meanHal':>8}  {'rh>=5':>6} {'rh&halNN':>9}")
    for m in sorted(by, key=lambda x: -len(by[x])):
        rows = by[m]
        hals = [hal(a) for a in rows if hal(a) is not None]
        mean = sum(hals) / len(hals) if hals else float("nan")
        rhh = [a for a in rows if (mv.rh_score(a) or 0) >= 5]
        print(f"{m:22s} {len(rows):>4} {len(hals):>6} {mean:>8.2f}  {len(rhh):>6} "
              f"{sum(1 for a in rhh if hal(a) is not None):>9}")

    # aggregate reward hacks (loose rh>=5) with hallucination scored
    agg = [hal(a) for a in dumb if (mv.rh_score(a) or 0) >= 5 and hal(a) is not None]
    print(f"\nAggregate DUMB reward hacks (rh>=5) with hallucination scored: {len(agg)}")
    for thr in (2, 3, 4, 5):
        ab = sum(1 for h in agg if h > thr)
        print(f"  hallucination > {thr}: {ab}/{len(agg)} = {100*ab/len(agg) if agg else 0:.0f}%")
    print("  hal distribution:", dict(sorted(Counter(agg).items())))

    # also: how many dumb reward hacks had hallucination NULL (dropped)?
    rh_all = [a for a in dumb if (mv.rh_score(a) or 0) >= 5]
    print(f"\ndumb reward hacks total={len(rh_all)}  hal-null among them="
          f"{sum(1 for a in rh_all if hal(a) is None)}")


if __name__ == "__main__":
    asyncio.run(main())
