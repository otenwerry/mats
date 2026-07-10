"""Inspect the cut region of a trajectory by EYE (P5, manual approach).

With only 30 trajectories, we curate cuts by hand rather than guessing with a
heuristic auto-snapper. This shows, around a trajectory's `first_hack_event`
(the adjudication's hack onset), the event window + which indices are VALID cut
boundaries (a cut at index c is valid iff events[:c] has no dangling tool_use —
recon rejects otherwise). Read it, pick the right pre-hack cut on a clean
boundary, and record it as a curated override (config CUT_OVERRIDES / a curated
Trajectory.default_cut).

    uv run python -m rollback.inspect_cut <nick|run_id> [--before N] [--after N]
    uv run python -m rollback.inspect_cut --audit          # validity of every default cut
"""
from __future__ import annotations

import argparse
import sys

from . import config, ptbio
from .session import recon


def _summ(e: dict) -> str:
    t = e.get("type")
    if t == "codex_item":
        it = e.get("item") or {}
        body = it.get("command") or it.get("text") or ""
        return f"[{it.get('type')}] {body[:80]}"
    out = []
    for b in e.get("blocks") or []:
        bt = b.get("type")
        if bt == "text":
            out.append((b.get("text") or "").strip()[:80])
        elif bt == "tool_use":
            inp = b.get("input") or {}
            arg = inp.get("command") or inp.get("file_path") or inp.get("filePath") or inp.get("path") or ""
            out.append(f"[tool_use {b.get('name')}] {str(arg)[:60]}")
        elif bt == "tool_result":
            out.append("[tool_result]")
    return " | ".join(out)


def show(traj: config.Trajectory, before: int, after: int) -> None:
    events = ptbio.load_events(traj)
    cut = traj.default_cut
    lo, hi = max(0, cut - before), min(len(events), cut + after)
    print(f"{traj.run_id}\n  scaffold={traj.scaffold} default_cut(first_hack_event)={cut} "
          f"valid={recon.validate_prefix(events, cut)['ok']}")
    # nearest valid cut <= default
    near = next((c for c in range(cut, max(0, cut - 30), -1)
                 if recon.validate_prefix(events, c)["ok"]), None)
    if near != cut:
        print(f"  nearest valid boundary <= default: {near}  (default is mid-tool-call)")
    print(f"  events [{lo},{hi}) — 'cut@i' valid = cutting BEFORE event i is clean:")
    for i in range(lo, hi):
        ok = "ok " if recon.validate_prefix(events, i)["ok"] else "BAD"
        mark = "  <== first_hack_event" if i == cut else ""
        print(f"    cut@{i} {ok}  {events[i].get('type'):9} {_summ(events[i])[:96]}{mark}")


def audit() -> None:
    print("validity of each trajectory's default cut (first_hack_event):")
    bad = 0
    for rid, t in config.ALL_TRAJECTORIES.items():
        events = ptbio.load_events(t)
        ok = recon.validate_prefix(events, t.default_cut)["ok"]
        if not ok:
            bad += 1
            near = next((c for c in range(t.default_cut, max(0, t.default_cut - 30), -1)
                         if recon.validate_prefix(events, c)["ok"]), None)
            print(f"  BAD  cut={t.default_cut:>4} (nearest valid {near}) {t.scaffold:8} {rid[:48]}")
        else:
            print(f"  ok   cut={t.default_cut:>4} {' ':>17} {t.scaffold:8} {rid[:48]}")
    print(f"\n{bad}/{len(config.ALL_TRAJECTORIES)} default cuts land mid-tool-call (need a curated boundary).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?")
    ap.add_argument("--before", type=int, default=8)
    ap.add_argument("--after", type=int, default=3)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    if args.audit or not args.target:
        audit()
        return
    show(config.get_trajectory(args.target), args.before, args.after)


if __name__ == "__main__":
    main()
