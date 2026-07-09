"""Pick candidate new-task sources for exp_baseline_pipeline.py (FREE: reads the log cache).

For the split continuation experiment we want, per target model, a continuation (new-task
source, "B") whose task has a LOW standalone reward-hack rate. This script estimates that
prior from the existing sweep-5 runs: for every (model, seed) cell it pools all runs across
BOTH the allow and correct conditions (Owen 2026-07-05: pooling is fine for a first
estimate; the screening run gives the trustworthy number) and counts hack categories.

It then suggests candidates that are:
  - from a seed DIFFERENT from every prefix seed of that model (Owen's planned prefixes,
    hardcoded below from the 2026-07-05 draft command);
  - from the ALLOW condition (a correct-condition non-hack could be a corrected hack);
  - themselves clean, complete non-hacks (category non, incompleteness <= 1, not dead);
  - from the cell with the lowest estimated hack rate (fewest full + degenerate).
"""

import asyncio
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

import make_viewer as mv

# Owen's planned prefixes (draft command, 2026-07-05): their seeds are EXCLUDED per model.
PREFIXES = {
    "full_hack": [1240, 1136, 1159, 1210, 1218],
    "corrected": [973, 996, 1018, 733, 1077],
    "clean": [1114, 998, 1158, 1208, 1219],
}
# his draft continuation list, resolved for reference only
DRAFT_CONTINUATIONS = [1117, 760, 1131, 1150, 716, 1171, 1200, 803, 1230, 1232]


def _inc(a: dict) -> float:
    v = a["scores"].get("incompleteness")
    return v if isinstance(v, (int, float)) else 0


async def main() -> None:
    by_id = await mv.load_originals_by_id()
    sweep5 = {i: a for i, a in by_id.items() if mv.sweep_key(a) == "consistent"}
    print(f"loaded {len(by_id)} originals, {len(sweep5)} on sweep 5 (consistent)\n")

    def describe(tid: int) -> str:
        a = by_id.get(tid)
        if a is None:
            return f"  #{tid}: NOT FOUND"
        return (f"  #{tid}: {mv.pretty_model(a['target']):24s} seed={mv.seed_label(a['seed']):20s} "
                f"cond={a.get('condition') or '—':8s} cat={mv.hack_category(a):10s} "
                f"rh={mv.rh_score(a)} inc={_inc(a)}"
                f"{'  DEAD' if a.get('dead') else ''}{'  CRASHED' if a.get('crashed') else ''}")

    print("=== planned prefixes ===")
    for kind, ids in PREFIXES.items():
        print(f"{kind}:")
        for tid in ids:
            print(describe(tid))
    print("\n=== draft continuation list (reference) ===")
    for tid in DRAFT_CONTINUATIONS:
        print(describe(tid))

    # per-model excluded seeds = every seed used by any of that model's prefixes
    excluded: dict[str, set] = {}
    for ids in PREFIXES.values():
        for tid in ids:
            a = by_id.get(tid)
            if a is not None:
                excluded.setdefault(mv.pretty_model(a["target"]), set()).add(a["seed"])

    # (model, seed) cells over sweep 5, both conditions pooled
    cells: dict[tuple, list] = {}
    for tid, a in sweep5.items():
        cells.setdefault((mv.pretty_model(a["target"]), a["seed"]), []).append((tid, a))

    print("\n=== per-(model, seed) prior estimate (sweep 5, allow+correct pooled) ===")
    for model in sorted(excluded):
        print(f"\n{model}   (prefix seeds excluded: "
              f"{', '.join(sorted(mv.seed_label(s) for s in excluded[model]))})")
        rows = []
        for (m, seed), grp in cells.items():
            if m != model:
                continue
            cats = [mv.hack_category(a) for _, a in grp if not a.get("dead")]
            n = len(cats)
            n_hackish = sum(1 for c in cats if c in ("full", "degenerate"))
            n_full = cats.count("full")
            # allow-condition clean complete non-hacks = usable candidate B originals
            cand = [tid for tid, a in grp
                    if a.get("condition") == "allow" and mv.hack_category(a) == "non"
                    and _inc(a) <= 1 and not a.get("dead") and not a.get("crashed")]
            rows.append((n_hackish, n_full, n, seed, cand))
        for n_hackish, n_full, n, seed, cand in sorted(rows):
            mark = "  [PREFIX SEED]" if seed in excluded[model] else ""
            print(f"  {mv.seed_label(seed):22s} full {n_full}/{n}, full+degen {n_hackish}/{n}, "
                  f"allow clean non-hack ids: {cand or '—'}{mark}")


if __name__ == "__main__":
    asyncio.run(main())
