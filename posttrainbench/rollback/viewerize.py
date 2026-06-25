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


def parse_continuation(solve_out: Path, trace_format: str = "opencode") -> list[dict]:
    """Continuation solve_out -> viewer events[], dispatched by the trajectory's
    trace_format (so claude/opencode/codex runs each parse with their own
    stream shape). All produce the same block format the viewer renders."""
    if trace_format == "claude_code":
        return _parse_continuation_claude(solve_out)
    if trace_format == "codex":
        return _parse_continuation_codex(solve_out)
    return _parse_continuation_opencode(solve_out)


def _parse_continuation_codex(solve_out: Path) -> list[dict]:
    """Codex `exec --json` stream -> viewer events[]. Each `item.completed`
    becomes a `codex_item` event carrying the raw item (reasoning / agent_message
    / command_execution / file_change), matching how the dataset provider parsed
    the original codex runs (viewer renders event['item'])."""
    events: list[dict] = []
    for line in solve_out.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "item.completed":
            continue
        item = d.get("item") or {}
        events.append({"type": "codex_item", "ts": None, "session_idx": 1,
                       "parent_tool_use_id": None, "subtype": item.get("type"),
                       "item": item})
    return events


def _stringify(content) -> str:
    """tool_result content may be a string or a list of content blocks."""
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content)
    return content if isinstance(content, str) else str(content)


def _parse_continuation_claude(solve_out: Path) -> list[dict]:
    """Claude Code stream-json -> viewer events[]. Each assistant/user line's
    message.content blocks (text / tool_use / tool_result) map straight through;
    system/result lines are dropped."""
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
        if t not in ("assistant", "user"):
            continue
        content = (d.get("message") or {}).get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        blocks = []
        for b in content or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text" and (b.get("text") or "").strip():
                blocks.append({"type": "text", "text": b["text"]})
            elif bt == "tool_use":
                blocks.append({"type": "tool_use", "id": b.get("id"),
                               "name": b.get("name") or "",
                               "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
            elif bt == "tool_result":
                blocks.append({"type": "tool_result",
                               "tool_use_id": b.get("tool_use_id") or b.get("id"),
                               "content": _stringify(b.get("content", "")),
                               "is_error": bool(b.get("is_error"))})
        if blocks:
            events.append({"type": t, "ts": None, "session_idx": 1,
                           "parent_tool_use_id": None, "blocks": blocks})
    return events


def _parse_continuation_opencode(solve_out: Path) -> list[dict]:
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


def _primary_score(score: dict | None) -> tuple:
    """(accuracy, stderr) from a score.json dict. Prefers 'accuracy'; else the
    first numeric metric that isn't an error bar."""
    if not score:
        return None, None
    acc = score.get("accuracy")
    if acc is None:
        for k, v in score.items():
            if isinstance(v, (int, float)) and k not in ("stderr", "std", "sem"):
                acc = v
                break
    return acc, score.get("stderr")


def build(solve_out: Path, label: str, cut: int, resume: str,
          run_index: int = 1, traj: config.Trajectory = None,
          session_id: str = None, result_dir: str = None,
          condition: str = None, score: dict = None,
          run_config: dict | None = None,
          prep_fidelity: dict | None = None,
          score_status: str | None = None) -> Path:
    traj = traj or config.TRAJECTORY
    orig = json.loads(traj.viewer_json.read_text())
    orig_exp = (orig.get("index_row") or {}).get("experiment", traj.experiment)
    prefill = orig["events"][:cut]
    trace_format = traj.trace_format or "opencode"
    cont = parse_continuation(solve_out, trace_format)
    # display name for the continuation policy (curated route, else recorded id)
    policy_display = traj.policy_model_continuation or traj.policy_model_recorded

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
    acc, stderr = _primary_score(score)
    row = dict(orig.get("index_row") or {})
    row.update({
        "run_id": run_id,
        "experiment": experiment,
        "run_name": f"rollback_{label}",
        "agent_model": policy_display,
        "trace_format": trace_format,
        "num_turns": len(cont),
        # benchmark score of THIS rollback's model (from the box scoring stage);
        # None until scored. The viewer shows it as "score (original)".
        "accuracy": acc, "stderr": stderr,
        "is_rollback": True,          # drives the 'rollback' tag in the viewer
    })

    # Reconstruction-fidelity + scoring caveats (lossy-processing surfacing): the
    # prep-fidelity gate is non-fatal (2026-06-17), so a run may resume on a model
    # that wasn't (or couldn't be) verified against the pre-cut baseline, and the
    # flaky vLLM scorer may leave a run unscored. Both are stored as queryable
    # flags AND a human-readable caveat list so no downstream consumer treats an
    # unverified/diverged/unscored run as clean.
    prep_fidelity_status = (prep_fidelity or {}).get("status")
    if prep_fidelity_status is None and prep_fidelity is not None:
        # Older prep_fidelity.json (pre-2026-06-17) has no 'status' field — derive
        # it from the faithful verdict so legacy runs flag correctly on re-sync.
        faithful = prep_fidelity.get("faithful")
        prep_fidelity_status = {True: "verified", False: "diverged",
                                None: "unverified_no_baseline"}.get(faithful)
    caveats = []
    if prep_fidelity_status == "diverged":
        caveats.append(
            f"prep fidelity DIVERGED: reconstructed pre-cut model scored "
            f"{(prep_fidelity or {}).get('retrained')} vs baseline "
            f"{(prep_fidelity or {}).get('baseline')} — agent resumed on a model "
            f"that may not match the original pre-cut state.")
    elif prep_fidelity_status == "unverified_no_score":
        caveats.append("prep fidelity UNVERIFIED: the fidelity scorer failed to "
                       "produce a number; reconstruction was not checked against baseline.")
    elif prep_fidelity_status == "unverified_no_baseline":
        caveats.append("prep fidelity UNVERIFIED: no recorded pre-cut baseline to "
                       "check the reconstructed model against.")
    if score_status == "failed":
        caveats.append("benchmark scoring FAILED (vLLM scorer crashed); this run "
                       "has no achieved score — rescore from the archived model.")
    elif score_status == "not_attempted":
        caveats.append("benchmark scoring was not attempted (no servable model / "
                       "no evaluate.py); this run has no achieved score.")

    out = {
        "index_row": row,
        "meta": {"rollback": True, "label": label, "cut_before_event": cut,
                 "source_trajectory": traj.run_id,
                 "policy_model": policy_display,
                 "resume_prompt": resume, "n_prefill_events": len(prefill),
                 "n_continuation_events": len(cont), "cut_event_index": cut,
                 # The agent resumed but emitted ZERO events after the injected
                 # user-turn -> a vacuous "control run" (no continuation to judge
                 # or score). Stored as a queryable flag so the orchestrator can
                 # fail the run and the index page can mark the trajectory as
                 # NOT-yet-successfully-controlled. Distinct from the prep-gate
                 # failure (which produces no solve_out / viewer_data at all).
                 "empty_continuation": len(cont) == 0,
                 # dedup keys for sync_viewer (so a run is never re-added):
                 "condition": condition, "session_id": session_id,
                 "result_dir": result_dir,
                 "score": score,   # raw metrics dict from the scoring stage
                 "prompt_contract": (run_config or {}).get("prompt_contract"),
                 "timeout_contract": (run_config or {}).get("timeout_contract"),
                 "workspace_audit": (run_config or {}).get("workspace_audit"),
                 "session_caveats": (run_config or {}).get("session_caveats") or [],
                 "prep_source": (run_config or {}).get("prep_source"),
                 "require_prep_fidelity": (run_config or {}).get("require_prep_fidelity"),
                 # Non-fatal-gate provenance (2026-06-17): queryable flags + caveats.
                 "prep_fidelity": prep_fidelity,
                 "prep_fidelity_status": prep_fidelity_status,
                 "score_status": score_status,
                 "caveats": caveats},
        "summary": f"Rollback {label}: {traj.benchmark_id} cut before ev{cut}, "
                   f"resumed with {policy_display}.",
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
