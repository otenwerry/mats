"""Loading + parsing of PostTrainBench trajectory data for rebuild/replay.

We treat the dataset-provider's pre-parsed `viewer_data/{run_id}.json` events[]
as the canonical event stream (clean, Anthropic-block format). The raw
`trace.txt`/`solve_out.txt` is only consulted for signals the viewer dropped
(e.g. timing). Event indices here are viewer_data indices, matching the
hack-localization / highlights tooling and config.CUT_BEFORE_EVENT.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_events(traj: config.Trajectory) -> list[dict]:
    with open(traj.viewer_json) as f:
        return json.load(f)["events"]


def load_record(traj: config.Trajectory) -> dict:
    with open(traj.viewer_json) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Session boundaries. A 10h run is stitched from several Claude Code sessions;
# each resumed session begins with a compaction-summary user turn whose text
# starts with "This session is being continued from a previous conversation".
# A cut point must stay within ONE session.
# --------------------------------------------------------------------------- #
_CONT_MARKER = "This session is being continued from a previous conversation"


def session_boundaries(events: list[dict]) -> list[int]:
    """Event indices that START a continued session (after a compaction)."""
    out = []
    for i, ev in enumerate(events):
        if ev.get("type") != "user":
            continue
        for b in ev.get("blocks") or []:
            if b.get("type") == "text" and _CONT_MARKER in (b.get("text") or ""):
                out.append(i)
                break
    return out


def session_of(events: list[dict], idx: int) -> tuple[int, int, int]:
    """(session_number, session_start_idx, session_end_idx_exclusive) for idx."""
    bounds = [0] + session_boundaries(events) + [len(events)]
    for s in range(len(bounds) - 1):
        if bounds[s] <= idx < bounds[s + 1]:
            return s, bounds[s], bounds[s + 1]
    return 0, 0, len(events)


# --------------------------------------------------------------------------- #
# Per-event content extraction (for forward replay and session reconstruction).
# Claude/OpenCode use blocks[]; Codex uses codex_item. We handle blocks here
# (the chosen run is Claude) and raise for codex until that path is needed.
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    idx: int                 # event index
    block_idx: int           # block position within the event
    tool_id: str             # tool_use id (for matching results)
    name: str                # Bash | Write | Edit | Read | ...
    input: dict


@dataclass
class FileOp:
    """A persistent filesystem mutation implied by a tool call."""
    idx: int
    kind: str                # "write" | "edit" | "bash"
    path: str | None         # target file (write/edit) if known
    payload: dict            # full tool input


_WRITE_TOOLS = {"write", "create"}
_EDIT_TOOLS = {"edit", "multiedit", "str_replace_editor", "notebook_edit"}


def _op_path(inp: dict) -> str | None:
    """Target file path across scaffold conventions (Claude: file_path,
    OpenCode: filePath, codex: path)."""
    return inp.get("file_path") or inp.get("filePath") or inp.get("path")


def iter_tool_calls(events: list[dict], upto: int | None = None) -> list[ToolCall]:
    """All tool_use calls in event order, optionally for events [0, upto).

    Handles both block-format scaffolds (Claude/OpenCode: assistant events with
    tool_use blocks) and Codex (codex_item events: command_execution -> a Bash
    call; file_change -> one write/edit call per changed path). Codex file_change
    carries only path+kind (no content), which is enough for the BACKWARD roller
    (first_write_index) but NOT for forward replay — see file_ops."""
    end = len(events) if upto is None else upto
    calls: list[ToolCall] = []
    for i in range(end):
        ev = events[i]
        if ev.get("type") == "codex_item":
            it = ev.get("item") or {}
            itype = it.get("type")
            if itype == "command_execution":
                calls.append(ToolCall(idx=i, block_idx=0, tool_id=it.get("id", ""),
                                      name="bash", input={"command": it.get("command", "")}))
            elif itype == "file_change":
                for k, ch in enumerate(it.get("changes") or []):
                    calls.append(ToolCall(
                        idx=i, block_idx=k, tool_id=it.get("id", ""),
                        name="write" if ch.get("kind") == "add" else "edit",
                        input={"path": ch.get("path")}))
            continue
        if ev.get("type") != "assistant":
            continue
        for bi, b in enumerate(ev.get("blocks") or []):
            if b.get("type") == "tool_use":
                calls.append(ToolCall(
                    idx=i, block_idx=bi,
                    tool_id=b.get("id") or b.get("tool_use_id") or "",
                    name=(b.get("name") or ""),
                    input=b.get("input") if isinstance(b.get("input"), dict) else {},
                ))
    return calls


def file_ops(events: list[dict], upto: int | None = None) -> list[FileOp]:
    """Ordered persistent-mutation ops (writes/edits/bash) for events [0, upto).

    Reads/Globs/Greps/WebSearch are non-mutating and excluded. Bash is included
    (it may write files); the forward executor runs it, the backward roller
    inspects the cited path. This is the spine of forward replay."""
    ops: list[FileOp] = []
    for tc in iter_tool_calls(events, upto):
        name = tc.name.lower()
        if name in _WRITE_TOOLS:
            ops.append(FileOp(tc.idx, "write", _op_path(tc.input), tc.input))
        elif name in _EDIT_TOOLS:
            ops.append(FileOp(tc.idx, "edit", _op_path(tc.input), tc.input))
        elif name == "bash":
            ops.append(FileOp(tc.idx, "bash", None, tc.input))
    return ops


def first_write_index(events: list[dict]) -> dict[str, int]:
    """basename/path -> earliest event index that Write/created it. Used by the
    backward roller to decide which final-workspace files are agent-created
    after the cut (delete) vs. before (keep)."""
    seen: dict[str, int] = {}
    for op in file_ops(events):
        if op.kind in ("write",) and op.path:
            for key in (op.path, Path(op.path).name):
                seen.setdefault(key, op.idx)
    return seen


# --------------------------------------------------------------------------- #
# The task prompt (first user turn of session 1). Claude Code passes it via CLI
# arg, so it may not appear as an event; regenerate deterministically from the
# repo when absent. Callers that need byte-fidelity should prefer get_prompt.py.
# --------------------------------------------------------------------------- #
def recorded_prompt(events: list[dict]) -> str | None:
    """The first user text that is NOT a continuation summary, if present."""
    for ev in events:
        if ev.get("type") != "user":
            continue
        for b in ev.get("blocks") or []:
            if b.get("type") == "text":
                t = b.get("text") or ""
                if t and _CONT_MARKER not in t:
                    return t
    return None
