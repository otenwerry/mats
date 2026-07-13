"""Reconstruct agent context at a chosen turn of a PTB reward-hack trajectory.

Free (no API calls). The paid probe lives in exp_probe_context.py.

Usage:
  python reconstruct.py --list
  python reconstruct.py --report                 # parse+align+cut-check all 30
  python reconstruct.py --trajectory <run_id> [--turn <viewer event idx|last|first_hack>]

--turn is a VIEWER EVENT INDEX (same numbering as the viewer and as
first_hack_event/marked_turns in highlights), or one of two keywords: 'last'
(the default — the latest cuttable point, i.e. the context going into the
agent's final turn) and 'first_hack'. The cut goes immediately before the
resolved event; invalid positions are refused with nearby valid alternatives,
and mid-message positions snap down to their message boundary (recorded in
fidelity.json).

Output bundle: mats-local/posttrainbench2/reconstructions/<run_id>__ev<N>/
  fidelity.json     flags + stats + provenance summary (ALWAYS read this)
  provenance.json   per-message source (verbatim vs synthesized), raw line map
  context.jsonl     claude/qwen3max: the message list (API format)
  session_preview.jsonl  claude: native session file rendered with cwd=/home/ben/task
  storage/          opencode: native storage tree
  items_context.json     codex: verbatim kept items (report-only scaffold)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import recon_claude, recon_codex, recon_opencode, runs, stream  # noqa: E402

PROBE_SUPPORT = {
    "claude": "claude --resume (supported)",
    "qwen3max": "unsupported in v1 (claude-format data, but needs the qwen CLI + key)",
    "opencode": "opencode run --session (needs opencode CLI + provider auth)",
    "codex": "unsupported (data is lossy: no rollout, no patch bodies)",
}


def build_bundle(traj: runs.Trajectory, turn_arg: str):
    """Parse, align, resolve the cut, and build the scaffold bundle.
    Returns (parsed, plan, bundle, fidelity_dict). Raises stream.CutError."""
    parsed = stream.parse(traj)
    events = traj.viewer_events()
    alignment = stream.align(traj, parsed, events)
    if not alignment.ok:
        raise RuntimeError(f"raw stream does not align with viewer events: {alignment.detail}")

    if turn_arg == "end":
        # cut AFTER the final message: the completed trajectory, ending on the
        # agent's closing assistant message (no resume bridging turns)
        plan = stream.end_cut(traj, parsed, alignment, events)
        turn = plan.turn
    else:
        if turn_arg == "first_hack":
            if traj.first_hack_event is None:
                raise stream.CutError("no first_hack_event recorded for this trajectory")
            turn = traj.first_hack_event
        elif turn_arg == "last":
            turn = stream.last_valid_turn(traj, parsed, alignment, events)
        else:
            turn = int(turn_arg)
        plan = stream.resolve_cut(traj, parsed, alignment, events, turn)

    if traj.scaffold in ("claude", "qwen3max"):
        bundle = recon_claude.build_context(traj, parsed, plan)
        flags, stats = bundle.flags, bundle.stats
    elif traj.scaffold == "opencode":
        bundle = recon_opencode.build(traj, parsed, plan)
        flags, stats = bundle.flags, bundle.stats
    else:
        bundle = recon_codex.build(traj, parsed, plan)
        flags, stats = bundle.flags, bundle.stats

    fidelity = {
        "run_id": traj.run_id, "scaffold": traj.scaffold,
        "requested_turn": turn, "cut_event": plan.cut_event,
        "raw_cut_record": plan.raw_cut, "snapped": plan.snapped,
        "cut_notes": plan.notes,
        "alignment_ok": True, "skipped_noise_lines": parsed.skipped_lines,
        "probe_support": PROBE_SUPPORT[traj.scaffold],
        "flags": flags, "stats": stats,
    }
    return parsed, plan, bundle, fidelity


def write_bundle(traj: runs.Trajectory, plan, bundle, fidelity) -> Path:
    out = runs.OUT_ROOT / "reconstructions" / f"{traj.run_id}__ev{plan.cut_event}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "fidelity.json").write_text(json.dumps(fidelity, indent=2))

    if traj.scaffold in ("claude", "qwen3max"):
        prov = [{"i": i, "role": m.role, "provenance": m.provenance,
                 "raw_records": m.raw_records} for i, m in enumerate(bundle.messages)]
        (out / "provenance.json").write_text(json.dumps(prov, indent=2))
        with open(out / "context.jsonl", "w") as f:
            for m in bundle.messages:
                f.write(json.dumps({"role": m.role, "message": m.message,
                                    "provenance": m.provenance}) + "\n")
        sid, lines = recon_claude.emit_session(bundle, cwd="/home/ben/task",
                                               version=fidelity.get("original_cli_version") or "unknown")
        with open(out / "session_preview.jsonl", "w") as f:
            for ln in lines:
                f.write(json.dumps(ln) + "\n")
    elif traj.scaffold == "opencode":
        for rel, content in bundle.files.items():
            p = out / "storage" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
        (out / "provenance.json").write_text(json.dumps(
            {"session_id": bundle.session_id, "provider_model": bundle.provider_model,
             "synthesized": ["initial user prompt part", "message info records"],
             "verbatim": "all assistant parts"}, indent=2))
    else:
        (out / "items_context.json").write_text(json.dumps(bundle.items, indent=2))
    return out


def cmd_one(args):
    traj = runs.load(args.trajectory)
    print(f"[{traj.run_id}] scaffold={traj.scaffold} agent={traj.agent}")
    try:
        parsed, plan, bundle, fidelity = build_bundle(traj, args.turn)
    except stream.CutError as e:
        print(f"CUT REFUSED: {e}")
        if e.candidates:
            print(f"  try --turn {' or '.join(str(c) for c in e.candidates)}")
        sys.exit(2)
    if traj.scaffold in ("claude", "qwen3max"):
        fidelity["original_cli_version"] = recon_claude.original_cli_version(parsed)
        fidelity["original_model"] = recon_claude.original_model(parsed)
    out = write_bundle(traj, plan, bundle, fidelity)
    print(f"cut: requested ev{fidelity['requested_turn']} -> effective ev{plan.cut_event}"
          + (" (snapped)" if plan.snapped else ""))
    for n in plan.notes:
        print(f"  note: {n}")
    print(f"stats: {json.dumps(fidelity['stats'])}")
    print("flags:")
    for fl in fidelity["flags"]:
        print(f"  [{fl['severity']:5}] {fl['code']}: {fl['detail'][:110]}")
    print(f"bundle: {out}")


def cmd_list(_args):
    print(f"{'scaffold':9} {'events':>6} {'first_hack':>10}  run_id")
    for traj in runs.reward_hacks():
        n = len(traj.viewer_events())
        print(f"{traj.scaffold:9} {n:>6} {str(traj.first_hack_event):>10}  {traj.run_id}")
    print()
    for k, v in PROBE_SUPPORT.items():
        print(f"  {k}: {v}")


def cmd_report(_args):
    ok = refused = failed = 0
    for traj in runs.reward_hacks():
        try:
            parsed, plan, bundle, fidelity = build_bundle(traj, "first_hack")
            worst = max((f["severity"] for f in fidelity["flags"]),
                        key=["info", "warn", "error"].index, default="info")
            snap = f" snap->ev{plan.cut_event}" if plan.snapped else ""
            s = fidelity["stats"]
            kept = s.get("n_messages") or s.get("messages_kept") or s.get("items_kept")
            print(f"OK      [{worst:5}] {traj.scaffold:9} ev{fidelity['requested_turn']}{snap}"
                  f" kept={kept}  {traj.run_id}")
            ok += 1
        except stream.CutError as e:
            print(f"REFUSED {traj.scaffold:9} {traj.run_id}: {e}")
            refused += 1
        except Exception as e:
            print(f"FAILED  {traj.scaffold:9} {traj.run_id}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{ok} ok, {refused} refused (need a different --turn), {failed} failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trajectory")
    ap.add_argument("--turn", default="last",
                    help="viewer event index, 'last' (default), or 'first_hack'")
    args = ap.parse_args()
    if args.list:
        cmd_list(args)
    elif args.report:
        cmd_report(args)
    elif args.trajectory:
        cmd_one(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
