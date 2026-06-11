"""Turn a rollback continuation (opencode --format json solve_out) into a
viewer_data trajectory, STITCHED onto the original prefill so the whole thing
renders as one run in the viewer with a clear ROLLBACK-CUT divider.

Layout produced (events[]):
    original events[0 : cut)          the prefill the agent had at the cut
    <CUT marker event>                a synthetic divider (rendered distinctly)
    <resume-prompt user event>        the control/treatment turn we injected
    converted continuation events     our agent's actions after the rollback

The continuation's opencode "tool" parts (which bundle input+output) are split
into an assistant tool_use event + a user tool_result event, matching how the
dataset provider parsed the originals. Output goes to mats-local
(config.ROLLBACK_VIEWER_DATA / <run_id>.json) — our own writable experiment
data, kept off github — and the viewer loads these additively (see
viewing/viewer.py).

Usage:
    python -m rollback.viewerize <solve_out.txt> --label control_completed \
        [--cut 419] [--resume "Please continue."]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config, ptbio

OUT_DIR = config.ROLLBACK_VIEWER_DATA


def _ms_to_iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_continuation(solve_out: Path) -> list[dict]:
    """opencode stream -> viewer events[] (assistant tool_use + user tool_result
    pairs, assistant text). step_start/step_finish are dropped (cosmetic)."""
    events: list[dict] = []
    for line in solve_out.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = d.get("type")
        p = d.get("part") or {}
        ts = _ms_to_iso(d["timestamp"]) if d.get("timestamp") else None
        if t == "tool_use":
            st = p.get("state") or {}
            inp = st.get("input") if isinstance(st.get("input"), dict) else {}
            cid = p.get("callID") or p.get("id")
            events.append({"type": "assistant", "ts": ts, "session_idx": 1,
                           "parent_tool_use_id": None,
                           "blocks": [{"type": "tool_use", "id": cid,
                                       "name": (p.get("tool") or "").capitalize(),
                                       "input": inp}]})
            events.append({"type": "user", "ts": ts, "session_idx": 1,
                           "parent_tool_use_id": None,
                           "blocks": [{"type": "tool_result", "tool_use_id": cid,
                                       "content": str(st.get("output", "")),
                                       "is_error": st.get("status") == "error"}]})
        elif t == "text" and (p.get("text") or "").strip():
            events.append({"type": "assistant", "ts": ts, "session_idx": 1,
                           "parent_tool_use_id": None,
                           "blocks": [{"type": "text", "text": p["text"]}]})
    return events


def _marker(text: str) -> dict:
    return {"type": "assistant", "ts": None, "session_idx": 1,
            "parent_tool_use_id": None, "rollback_marker": True,
            "blocks": [{"type": "text", "text": text}]}


def build(solve_out: Path, label: str, cut: int, resume: str,
          run_index: int = 1, traj: config.Trajectory = None,
          session_id: str = None, result_dir: str = None,
          condition: str = None) -> Path:
    traj = traj or config.TRAJECTORY
    orig = json.loads(traj.viewer_json.read_text())
    orig_exp = (orig.get("index_row") or {}).get("experiment", traj.experiment)
    prefill = orig["events"][:cut]
    cont = parse_continuation(solve_out)

    cut_div = _marker(
        f"════════ ROLLBACK CUT ════════\n"
        f"Original trajectory truncated before event {cut}. The agent was "
        f"resumed below from this point with the injected turn:")
    resume_evt = {"type": "user", "ts": None, "session_idx": 1,
                  "parent_tool_use_id": None, "rollback_resume": True,
                  "blocks": [{"type": "text", "text": resume}]}

    events = prefill + [cut_div, resume_evt] + cont

    run_id = f"rollback_{label}__{traj.run_name}"
    # experiment mirrors the label so the column is self-describing & unique:
    #   <original experiment>__rollback__<label>   (e.g. ...__rollback__treatment_run1)
    experiment = f"{orig_exp}__rollback__{label}"
    row = dict(orig.get("index_row") or {})
    row.update({
        "run_id": run_id,
        "experiment": experiment,
        "run_name": f"rollback_{label}",
        "agent_model": traj.policy_model_continuation or traj.policy_model_recorded,
        "trace_format": "opencode",
        "num_turns": len(cont),
        "accuracy": None, "stderr": None,
        "is_rollback": True,          # drives the 'rollback' tag in the viewer
    })

    out = {
        "index_row": row,
        "meta": {"rollback": True, "label": label, "cut_before_event": cut,
                 "source_trajectory": traj.run_id,
                 "policy_model": traj.policy_model_continuation,
                 "resume_prompt": resume, "n_prefill_events": len(prefill),
                 "n_continuation_events": len(cont), "cut_event_index": cut,
                 # dedup keys for sync_viewer (so a run is never re-added):
                 "condition": condition, "session_id": session_id,
                 "result_dir": result_dir},
        "summary": f"Rollback {label}: {traj.benchmark_id} cut before ev{cut}, "
                   f"resumed with {traj.policy_model_continuation}.",
        "sessions": [], "events": events, "metrics": {}, "judgements": {},
        "prompt": orig.get("prompt", ""),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{run_id}.json"
    path.write_text(json.dumps(out) + "\n")
    print(f"wrote {path}")
    print(f"  run_id={run_id}  prefill={len(prefill)} ev + marker/resume + "
          f"continuation={len(cont)} ev = {len(events)} total")
    print(f"  cut divider at index {len(prefill)}, resume at {len(prefill)+1}, "
          f"continuation starts {len(prefill)+2}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solve_out")
    ap.add_argument("--label", required=True, help="e.g. control_completed")
    ap.add_argument("--cut", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="the injected turn; default = the trajectory's control stem")
    ap.add_argument("--run-index", type=int, default=1,
                    help="i in experiment '<orig>__rollback__run{i}'")
    args = ap.parse_args()
    cut = args.cut if args.cut is not None else config.CUT_BEFORE_EVENT
    resume = args.resume if args.resume is not None else config.CONTROL_STEM
    build(Path(args.solve_out), args.label, cut, resume, run_index=args.run_index)


if __name__ == "__main__":
    main()
