"""Parse raw PTB trajectory streams and align them to viewer event indices.

The viewer (viewer_data/<run_id>.json "events") is the shared reference frame:
Owen navigates trajectories by viewer event index, and the highlights labels
(first_hack_event, marked_turns) are viewer indices. All --turn arguments are
therefore viewer event indices. This module parses the RAW stream itself
(solve_out.txt / trace.txt) and proves, per trajectory, that our parse lines
up with the viewer's events — reconstruction payloads always come from the
raw stream, never from the derived viewer JSON.

Alignment rules (verified empirically, re-verified per trajectory at runtime):
  claude/qwen3max: every parsed JSON line -> exactly one viewer event, in
      order (uuid-bearing lines are uuid-matched).
  opencode: step_start -> 1 system event; tool_use -> assistant(tool_use) +
      user(tool_result); text -> 1 assistant event; step_finish -> 1 system.
  codex: thread.started -> 1 system event; item.completed -> 1 codex_item
      event; item.started / turn.* are skipped by the viewer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .runs import Trajectory

_TS_PREFIX = re.compile(r"^\[[0-9TZ:.\-]+\] ")


@dataclass
class Parsed:
    records: list[dict]           # parsed JSON objects, stream order
    linenos: list[int]            # 1-based line number of each record
    skipped_lines: int            # non-JSON noise (harness banners etc.)


@dataclass
class Alignment:
    ok: bool
    ev_to_raw: list[tuple[int, str]]   # viewer idx -> (record idx, sub-part tag)
    detail: str = ""


class CutError(ValueError):
    def __init__(self, msg: str, candidates: list[int] | None = None):
        super().__init__(msg)
        self.candidates = candidates or []


@dataclass
class CutPlan:
    turn: int                     # requested viewer event index
    cut_event: int                # effective viewer event index (after snap)
    raw_cut: int                  # cut falls immediately BEFORE this record idx
    snapped: bool = False
    notes: list[str] = field(default_factory=list)


def parse(traj: Trajectory) -> Parsed:
    records, linenos, skipped = [], [], 0
    with open(traj.traj_file) as f:
        for i, line in enumerate(f, 1):
            line = _TS_PREFIX.sub("", line.strip())
            if not line.startswith("{"):
                skipped += 1
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                continue
            linenos.append(i)
    return Parsed(records, linenos, skipped)


def _align_claude(parsed: Parsed, events: list[dict]) -> Alignment:
    ev_to_raw = [(i, "") for i in range(len(parsed.records))]
    if len(events) != len(parsed.records):
        return Alignment(False, [], f"count mismatch: {len(parsed.records)} raw vs {len(events)} viewer")
    for i, (r, e) in enumerate(zip(parsed.records, events)):
        ru, eu = r.get("uuid"), e.get("uuid")
        if eu is not None and ru != eu:
            return Alignment(False, [], f"uuid mismatch at event {i}")
        if eu is None and r.get("type") != e.get("type"):
            return Alignment(False, [], f"type mismatch at event {i}")
    return Alignment(True, ev_to_raw)


_OPENCODE_EXPANSION = {"step_start": ["system"], "step_finish": ["system"],
                       "text": ["assistant"], "reasoning": ["assistant"],
                       "tool_use": ["assistant", "user"]}


def _align_opencode(parsed: Parsed, events: list[dict]) -> Alignment:
    ev_to_raw: list[tuple[int, str]] = []
    expected_types: list[str] = []
    for i, r in enumerate(parsed.records):
        kinds = _OPENCODE_EXPANSION.get(r.get("type"))
        if kinds is None:
            return Alignment(False, [], f"unknown opencode stream type {r.get('type')} at record {i}")
        for k in kinds:
            ev_to_raw.append((i, k))
            expected_types.append(k)
    if len(events) != len(ev_to_raw):
        return Alignment(False, [], f"count mismatch: expanded {len(ev_to_raw)} vs viewer {len(events)}")
    for j, e in enumerate(events):
        if e.get("type") != expected_types[j]:
            return Alignment(False, [], f"type mismatch at event {j}: viewer {e.get('type')} vs {expected_types[j]}")
    return Alignment(True, ev_to_raw)


def _align_codex(parsed: Parsed, events: list[dict]) -> Alignment:
    ev_to_raw: list[tuple[int, str]] = []
    expected: list[tuple[str, str | None]] = []
    for i, r in enumerate(parsed.records):
        t = r.get("type")
        if t == "thread.started":
            ev_to_raw.append((i, ""))
            expected.append(("system", "init"))
        elif t == "item.completed":
            ev_to_raw.append((i, ""))
            expected.append(("codex_item", (r.get("item") or {}).get("type")))
        elif t == "item.updated":
            ev_to_raw.append((i, ""))
            expected.append(("item.updated", None))
        elif t == "error":
            ev_to_raw.append((i, ""))
            expected.append(("error", None))
    if len(events) != len(ev_to_raw):
        return Alignment(False, [], f"count mismatch: expected {len(ev_to_raw)} vs viewer {len(events)}")
    for j, e in enumerate(events):
        if (e.get("type"), e.get("subtype")) != expected[j]:
            return Alignment(False, [], f"type mismatch at event {j}: viewer {(e.get('type'), e.get('subtype'))} vs {expected[j]}")
    return Alignment(True, ev_to_raw)


def align(traj: Trajectory, parsed: Parsed, events: list[dict]) -> Alignment:
    if traj.scaffold in ("claude", "qwen3max"):
        return _align_claude(parsed, events)
    if traj.scaffold == "opencode":
        return _align_opencode(parsed, events)
    if traj.scaffold == "codex":
        return _align_codex(parsed, events)
    raise ValueError(traj.scaffold)


# ---------------------------------------------------------------- cut logic

def _claude_msg_id(rec: dict) -> str | None:
    return (rec.get("message") or {}).get("id")


def _is_main(rec: dict) -> bool:
    return not rec.get("parent_tool_use_id")


def claude_valid_cuts(parsed: Parsed) -> list[int]:
    """Record indices that start a new main-thread assistant API message."""
    out, prev_id = [], None
    for i, r in enumerate(parsed.records):
        if not _is_main(r):
            continue
        if r.get("type") == "assistant":
            mid = _claude_msg_id(r)
            if mid != prev_id:
                out.append(i)
            prev_id = mid
        elif r.get("type") in ("user", "system"):
            prev_id = None
    return out


def resolve_cut_claude(parsed: Parsed, alignment: Alignment,
                       events: list[dict], turn: int) -> CutPlan:
    if not 0 <= turn < len(events):
        raise CutError(f"turn {turn} out of range (0..{len(events) - 1})")
    rec_idx, _ = alignment.ev_to_raw[turn]
    rec = parsed.records[rec_idx]
    valid = claude_valid_cuts(parsed)
    if not _is_main(rec):
        # inside a subagent sidechain: main-thread resume cannot represent it
        before = [v for v in valid if v < rec_idx]
        after = [v for v in valid if v > rec_idx]
        cands = ([before[-1]] if before else []) + ([after[0]] if after else [])
        raise CutError(
            f"event {turn} is inside a subagent sidechain "
            f"(parent_tool_use_id={rec.get('parent_tool_use_id')}); v1 only cuts "
            f"the main thread. Nearest main-thread cuts: {cands}", cands)
    if rec.get("type") != "assistant":
        before = [v for v in valid if v < rec_idx]
        after = [v for v in valid if v > rec_idx]
        cands = ([before[-1]] if before else []) + ([after[0]] if after else [])
        raise CutError(
            f"event {turn} is a {rec.get('type')} event; cuts go immediately "
            f"before a main-thread assistant message. Nearest: {cands}", cands)
    plan = CutPlan(turn=turn, cut_event=turn, raw_cut=rec_idx)
    if rec_idx not in valid:
        snapped = max(v for v in valid if v <= rec_idx)
        plan.cut_event, plan.raw_cut, plan.snapped = snapped, snapped, True
        plan.notes.append(
            f"event {turn} is mid-message (continuation block of the same API "
            f"message); snapped down to the message start at event {snapped}")
    return plan


def resolve_cut_opencode(parsed: Parsed, alignment: Alignment,
                         events: list[dict], turn: int) -> CutPlan:
    if not 0 <= turn < len(events):
        raise CutError(f"turn {turn} out of range (0..{len(events) - 1})")
    rec_idx, kind = alignment.ev_to_raw[turn]
    mid = parsed.records[rec_idx]["part"].get("messageID")
    first_rec = next(i for i, r in enumerate(parsed.records)
                     if r["part"].get("messageID") == mid)
    plan = CutPlan(turn=turn, cut_event=turn, raw_cut=first_rec)
    if first_rec != rec_idx or kind != "system" or parsed.records[rec_idx].get("type") != "step_start":
        # snap to the start of the assistant message containing this event
        ev_first = next(j for j, (ri, _) in enumerate(alignment.ev_to_raw) if ri == first_rec)
        plan.cut_event, plan.snapped = ev_first, True
        plan.notes.append(
            f"event {turn} is inside assistant message {mid}; snapped to the "
            f"message boundary at event {ev_first} (opencode messages are "
            f"resumable only at message boundaries)")
    return plan


def resolve_cut_codex(parsed: Parsed, alignment: Alignment,
                      events: list[dict], turn: int) -> CutPlan:
    if not 0 <= turn < len(events):
        raise CutError(f"turn {turn} out of range (0..{len(events) - 1})")
    rec_idx, _ = alignment.ev_to_raw[turn]
    return CutPlan(turn=turn, cut_event=turn, raw_cut=rec_idx)


def resolve_cut(traj: Trajectory, parsed: Parsed, alignment: Alignment,
                events: list[dict], turn: int) -> CutPlan:
    if traj.scaffold in ("claude", "qwen3max"):
        return resolve_cut_claude(parsed, alignment, events, turn)
    if traj.scaffold == "opencode":
        return resolve_cut_opencode(parsed, alignment, events, turn)
    return resolve_cut_codex(parsed, alignment, events, turn)
