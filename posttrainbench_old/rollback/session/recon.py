"""Rebuild a Claude Code session JSONL truncated at the cut, for native resume.

Claude Code persists each session as a JSONL under
    ~/.claude/projects/<cwd-with-slashes-as-dashes>/<sessionId>.jsonl
one object per turn, each carrying: uuid, parentUuid (linking to the previous
turn), sessionId, cwd, version, type ("user"|"assistant"), timestamp, and the
Anthropic-format `message`. The dataset discarded these files (only task/ was
archived), so we reconstruct one from the viewer_data events.

We emit, in order:
  1. a leading user turn = the task PROMPT (Claude Code's first session entry;
     it isn't a stream event because it arrives as a CLI arg).
  2. events [0, cut) mapped to user/assistant turns (system/result skipped).
The rollback experiment's new user turn is NOT embedded here. It is appended by
the runner at resume time (`RESUME_MODE=continue_prompt`) so control and
treatment use the same experiment contract across scaffolds.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .. import config, ptbio, scaffold


CWD = "/home/ben/task"
DEFAULT_VERSION = "2.1.34"  # observed in the dataset's Claude traces


def _project_dir_name(cwd: str = CWD) -> str:
    """Claude Code's project folder name: cwd with '/' and '.' -> '-'."""
    return cwd.replace("/", "-").replace(".", "-")


def _uuid() -> str:
    # NOTE: real uuids; reproducibility of the *experiment* doesn't require
    # stable uuids (they only thread the session graph). Vary across rebuilds.
    return str(uuid.uuid4())


def _assistant_message(blocks: list[dict]) -> dict:
    content = []
    for b in blocks:
        t = b.get("type")
        if t == "text":
            content.append({"type": "text", "text": b.get("text", "")})
        elif t == "tool_use":
            content.append({"type": "tool_use", "id": b.get("id"),
                            "name": b.get("name"),
                            "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
    return {"role": "assistant", "content": content}


def _user_message(blocks: list[dict]) -> dict:
    content = []
    for b in blocks:
        t = b.get("type")
        if t == "tool_result":
            content.append({"type": "tool_result",
                            "tool_use_id": b.get("tool_use_id") or b.get("id"),
                            "content": b.get("content", "")})
        elif t == "text":
            content.append({"type": "text", "text": b.get("text", "")})
    return {"role": "user", "content": content}


def validate_prefix(events: list[dict], cut: int) -> dict:
    """Ensure the truncated prefix has no dangling tool_use and ends cleanly."""
    tu, tr = set(), set()
    for e in events[:cut]:
        for b in e.get("blocks") or []:
            if b.get("type") == "tool_use":
                tu.add(b.get("id"))
            elif b.get("type") == "tool_result":
                tr.add(b.get("tool_use_id") or b.get("id"))
    last = next((events[i] for i in range(cut - 1, -1, -1)
                 if events[i].get("type") in ("user", "assistant")), None)
    return {
        "unmatched_tool_use": sorted(x for x in (tu - tr) if x),
        "last_turn_type": last.get("type") if last else None,
        "ok": not (tu - tr),
    }


def build_session(spec: config.ExperimentSpec, out_dir: Path,
                  version: str = DEFAULT_VERSION) -> dict:
    """Write the reconstructed session JSONL. Returns metadata incl. the path
    Claude Code expects and the sessionId to resume."""
    traj = spec.trajectory
    cut = spec.cut_before_event
    events = ptbio.load_events(traj)

    val = validate_prefix(events, cut)
    if not val["ok"]:
        raise ValueError(f"cut {cut} leaves dangling tool_use: {val['unmatched_tool_use']}")

    session_id = str(uuid.uuid4())
    proj_dir = out_dir / ".claude" / "projects" / _project_dir_name()
    proj_dir.mkdir(parents=True, exist_ok=True)
    session_path = proj_dir / f"{session_id}.jsonl"

    lines: list[dict] = []
    parent = None
    base_ts = "2026-02-06T15:47:00.000Z"  # cosmetic; timer.sh governs real time

    def emit(turn_type: str, message: dict):
        nonlocal parent
        u = _uuid()
        lines.append({
            "parentUuid": parent, "isSidechain": False, "userType": "external",
            "cwd": CWD, "sessionId": session_id, "version": version,
            "gitBranch": "", "type": turn_type, "message": message,
            "uuid": u, "timestamp": base_ts,
        })
        parent = u

    # 1. leading prompt turn
    prompt = scaffold.generate_prompt(traj)
    emit("user", {"role": "user", "content": prompt})

    # 2. events [0, cut)
    last_user_line_idx = None
    for e in events[:cut]:
        typ = e.get("type")
        if typ == "assistant":
            emit("assistant", _assistant_message(e.get("blocks") or []))
        elif typ == "user":
            emit("user", _user_message(e.get("blocks") or []))
            last_user_line_idx = len(lines) - 1
        # system/result: not part of the on-disk message log

    with open(session_path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")

    return {
        "session_id": session_id,
        "session_path": str(session_path),
        "project_dir": str(proj_dir),
        "cwd": CWD,
        "version": version,
        "n_turns": len(lines),
        "cut_before_event": cut,
        "condition": spec.condition,
        "intervention_injected": False,
        "prompt_injected_by": "runner_resume_prompt",
        "prefix_validation": val,
    }
