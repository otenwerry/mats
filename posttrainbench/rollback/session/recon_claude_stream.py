"""FAITHFUL Claude Code session reconstruction from solve_out.txt's stream-json.

Why this exists (vs the sibling recon.py): recon.py rebuilds the session from the
dataset's *viewer events*, which are a normalized view that keeps only
text/tool_use/tool_result blocks — it drops the message id, model, usage,
stop_reason, context_management, and any thinking blocks. solve_out.txt, however,
is Claude Code's raw `--output-format stream-json`: every assistant/user turn
carries the COMPLETE Anthropic `message` object plus the real per-turn `uuid`.
Rebuilding the on-disk transcript from THIS source is byte-faithful for the
message bodies (same contract OpenCode enjoys), so claude (and qwen3max, which
runs through the same Claude Code harness) become faithfully reconstructable.

Claude Code persists each session as
    ~/.claude/projects/<cwd '/'and'.'->'-'>/<sessionId>.jsonl
one object per turn: {parentUuid, sessionId, cwd, version, type, message, uuid,
timestamp, ...}. We emit that, truncated at the cut.

SCOPE (this module): single-session runs (one `system:init` in the stream — true
for the shorter runs, e.g. healthbench). Multi-session 10h runs (stitched across
compaction boundaries) raise NotImplementedError with the segment info — handled
next by reconstructing only the cut's session segment.

The rollback experiment's new user turn is appended by the runner at resume time
(`RESUME_MODE=continue_prompt`), not embedded in the reconstructed transcript.
"""
from __future__ import annotations

import json
import uuid as _uuidlib
from pathlib import Path

from .. import config, ptbio, scaffold

CWD = "/home/ben/task"
DEFAULT_VERSION = "2.1.34"
_STREAM_TYPES = {"system", "assistant", "user", "result"}


def load_stream(traj: config.Trajectory) -> list[dict]:
    """Claude Code stream-json events from solve_out.txt (skips harness noise)."""
    out = []
    with open(traj.raw_dir / "solve_out.txt") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") in _STREAM_TYPES:
                out.append(d)
    return out


def _project_dir_name(cwd: str = CWD) -> str:
    return cwd.replace("/", "-").replace(".", "-")


def _cut_stream_index(stream: list[dict], events: list[dict], cut: int) -> int:
    """Index into `stream` of the first turn to DROP, anchored by the tool_use id
    of the first tool call at/after the cut (stable across viewer events and the
    raw stream). Falls back to ordinal turn count when the cut turn has no tool
    call. Raises if the prefix would split a tool_use/tool_result pair."""
    anchor = next((tc for tc in ptbio.iter_tool_calls(events) if tc.idx >= cut), None)
    if anchor is not None:
        for i, e in enumerate(stream):
            if e.get("type") != "assistant":
                continue
            for b in e.get("message", {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") == anchor.tool_id:
                    return i
        raise ValueError(f"anchor tool_use id {anchor.tool_id} not found in stream")
    # no tool call at/after the cut: map by ordinal user/assistant turn count
    n_turns = sum(1 for e in events[:cut] if e.get("type") in ("user", "assistant"))
    seen = 0
    for i, e in enumerate(stream):
        if e.get("type") in ("user", "assistant"):
            if seen == n_turns:
                return i
            seen += 1
    return len(stream)


def _session_segments(stream: list[dict]) -> list[tuple[int, int]]:
    """[(start, end)) per Claude Code session, split on system:init events."""
    starts = [i for i, e in enumerate(stream)
              if e.get("type") == "system" and e.get("subtype") == "init"]
    if not starts:
        starts = [0]
    bounds = starts + [len(stream)]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def build_session(spec: config.ExperimentSpec, out_dir: Path,
                  version: str = DEFAULT_VERSION) -> dict:
    """Write a faithful reconstructed Claude Code session JSONL truncated at the
    cut. Returns metadata incl. the path + sessionId to resume."""
    traj = spec.trajectory
    cut = spec.cut_before_event
    events = ptbio.load_events(traj)
    stream = load_stream(traj)
    if not stream:
        raise RuntimeError(f"empty stream in {traj.raw_dir / 'solve_out.txt'}")

    segments = _session_segments(stream)
    if len(segments) > 1:
        raise NotImplementedError(
            f"{traj.run_name}: {len(segments)} Claude Code sessions (compaction "
            f"boundaries at {[s for s, _ in segments]}). Multi-session resume = "
            f"reconstruct only the cut's segment; not yet implemented.")

    cut_idx = _cut_stream_index(stream, events, cut)
    kept = stream[:cut_idx]

    # dangling-tool_use guard: every tool_use in kept assistant turns must have a
    # matching tool_result in a kept user turn (else the resume is malformed).
    tu, tr = set(), set()
    for e in kept:
        for b in e.get("message", {}).get("content") or []:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                tu.add(b.get("id"))
            elif b.get("type") == "tool_result":
                tr.add(b.get("tool_use_id"))
    dangling = sorted(x for x in (tu - tr) if x)
    if dangling:
        raise ValueError(f"cut {cut} leaves dangling tool_use: {dangling}")

    session_id = str(_uuidlib.uuid4())
    proj_dir = out_dir / ".claude" / "projects" / _project_dir_name()
    proj_dir.mkdir(parents=True, exist_ok=True)
    session_path = proj_dir / f"{session_id}.jsonl"
    base_ts = "2026-02-06T15:47:00.000Z"  # cosmetic; timer.sh governs real budget

    lines: list[dict] = []
    parent = None

    def emit(turn_type: str, message: dict, src_uuid: str | None = None):
        nonlocal parent
        u = src_uuid or str(_uuidlib.uuid4())
        lines.append({
            "parentUuid": parent, "isSidechain": False, "userType": "external",
            "cwd": CWD, "sessionId": session_id, "version": version,
            "gitBranch": "", "type": turn_type, "message": message,
            "uuid": u, "timestamp": base_ts,
        })
        parent = u

    # 1. leading prompt turn (Claude Code's first entry — a CLI arg, not streamed)
    emit("user", {"role": "user", "content": scaffold.generate_prompt(traj)})

    # 2. kept stream turns — message bodies copied VERBATIM (the fidelity win),
    #    real per-turn uuid preserved to thread the graph.
    last_user_idx = None
    for e in kept:
        t = e.get("type")
        if t not in ("assistant", "user"):
            continue  # system/result are not transcript message records
        emit(t, e.get("message", {}), e.get("uuid"))
        if t == "user":
            last_user_idx = len(lines) - 1

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
        "cut_stream_index": cut_idx,
        "condition": spec.condition,
        "intervention_injected": False,
        "prompt_injected_by": "runner_resume_prompt",
        "source": "solve_out.txt stream-json (faithful)",
    }
