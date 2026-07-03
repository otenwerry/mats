"""Read-only: assemble a scored table of all (non-rollback) audit trajectories to
help hand-pick continuation-experiment trajectories. Reads local logs only (free)."""
import asyncio
import json
import sys
from pathlib import Path

# tools/ lives one level under petri/; put petri/ (for make_viewer) and petri/lib (for
# petri_paths) on the path so this runs from anywhere via `uv run tools/explore_select_continuation.py`.
_petri = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))
import make_viewer as mv
from petri_paths import DATA, LOGS

DIMS = ["reward_hacking", "auditor_nudging", "auditor_mistakes",
        "situational_awareness", "degenerate_behavior", "hallucination",
        "hack_in_final_solution", "honest_path_ambiguity", "incompleteness"]


async def main():
    reg = json.loads((DATA / "trajectory_ids.json").read_text())
    mode_dirs = sorted([d for d in LOGS.iterdir()
                        if d.is_dir() and not d.name.startswith("rollback")])
    rows = []
    for md in mode_dirs:
        audits = await mv.load_mode(md)
        for a in audits:
            key = mv.traj_key(a)
            tid = reg.get(key)
            sc = a.get("scores", {})
            ev = mv.binary_hack_eval(a)
            rows.append(dict(
                id=tid,
                mode=a["mode"],
                task=a["task"],
                target=mv.pretty_model(a["target"]),
                seed=mv.seed_label(a["seed"]),
                epoch=a["epoch"],
                dead=a.get("dead"),
                ended_clean=a.get("ended_via_end_conv"),
                is_hack=mv.is_hack_binary(a),
                missing=",".join(ev["missing"]),
                failed=",".join(ev["failed"]),
                **{d: sc.get(d) for d in DIMS},
            ))
    out = DATA / "_analysis" / "continuation_candidates.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    asyncio.run(main())
