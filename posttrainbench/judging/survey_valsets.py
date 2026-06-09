"""Did successful runs build & USE their own validation sets? (free)

The earlier survey counted loose val-ish *terms* (~39% of runs) — much of that
is trainer boilerplate (eval_dataset/validation_split for loss curves). Here we
require evidence the agent actually SCORED on its own validation signal
("val accuracy", "evaluate on the holdout", "use X as a validation/proxy"...),
and we split clean runs by within-benchmark accuracy quartile to see whether
top performers validated differently from the rest.

Usage: uv run python posttrainbench/survey_valsets.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)

import json
import re
from collections import defaultdict

import paths
from survey_iteration import agent_text

REALVAL = re.compile(
    r"val(idation)?[ _-]?(set|split)?[ _-]?(accuracy|acc\b|score|results)"
    r"|(evaluate|eval|score|test(ing)?) (it |the model )?on (the |our |my )?"
    r"(val(idation)?|holdout|held.?out|dev)[ _-]?(set|split)?"
    r"|holdout (set|score|eval|accuracy)"
    r"|as (a |the |my |our )?(validation|val[ _-]?set|dev[ _-]?set|proxy (benchmark|eval))",
    re.I)


def main():
    runs = json.load(open(paths.VIEWER_DATA / "index.json"))["runs"]
    flagged = set()
    for p in paths.HIGHLIGHTS.glob("*.json"):
        if not p.name.startswith("_debug"):
            try:
                flagged.add(json.loads(p.read_text()).get("run_id"))
            except json.JSONDecodeError:
                pass

    # within-benchmark accuracy quartiles over clean runs with accuracy
    by_bench = defaultdict(list)
    for r in runs:
        if r["run_id"] in flagged or r.get("accuracy") is None:
            continue
        by_bench[r["benchmark"]].append(r)
    for b, rs in by_bench.items():
        rs.sort(key=lambda r: r["accuracy"])

    stats = defaultdict(lambda: [0, 0])  # quartile -> [n, with_real_val]
    examples = []
    for b, rs in by_bench.items():
        n = len(rs)
        for i, r in enumerate(rs):
            q = min(3, 4 * i // n)  # 0=bottom..3=top quartile
            try:
                d = json.load(open(paths.VIEWER_DATA / f"{r['run_id']}.json"))
            except (OSError, json.JSONDecodeError):
                continue
            txt = agent_text(d)
            m = REALVAL.search(txt)
            stats[q][0] += 1
            if m:
                stats[q][1] += 1
                if q == 3 and len(examples) < 6:
                    lo = max(0, m.start() - 150)
                    examples.append((b, r["run_id"][:52], r["accuracy"],
                                     txt[lo:m.end() + 150].replace("\n", " ")))

    print("clean runs: real val-USAGE evidence by within-benchmark accuracy quartile")
    for q in range(4):
        n, v = stats[q]
        lab = ["bottom 25%", "25-50%", "50-75%", "top 25%"][q]
        print(f"  {lab:10}  {v:>3}/{n:<4} ({100*v/max(n,1):.0f}%)")
    print("\nexamples from TOP-quartile runs:")
    for b, rid, acc, q in examples:
        print(f"[{b} acc={acc:.2f}] {rid}\n   …{q}…\n")


if __name__ == "__main__":
    main()
