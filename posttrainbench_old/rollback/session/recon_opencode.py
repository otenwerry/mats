"""Rebuild OpenCode session storage truncated at the cut, for native resume.

OpenCode (v1.1.59, pinned by the PTB container) persists sessions under
    $XDG_DATA_HOME/opencode/storage/   (default ~/.local/share/opencode/storage)
        migration                       schema migration counter ("2")
        project/global.json             project record (task dir is not a git
                                        repo -> projectID "global")
        session/global/<sessionID>.json session info
        message/<sessionID>/<msgID>.json one record per message (user|assistant)
        part/<msgID>/<partID>.json      one record per part

Unlike the Claude path (which reconstructs from viewer events), the OpenCode
solve_out.txt stream IS the part log: `opencode run --format json` prints every
part object verbatim (real ids, real timestamps, real token usage), so kept
parts are written back byte-faithfully. Only the *message info* records and the
initial user prompt (passed as a CLI arg, never streamed) are synthesized:
  - user message: id derived to sort before the first assistant id (OpenCode
    orders messages lexicographically by id), prompt text as its single part.
  - assistant messages: role/parentID/agent/path are fixed; time from the
    stream; cost/tokens/finish lifted from that message's step-finish part.
    modelID/providerID record the ORIGINAL policy model (drift, if any, applies
    only to newly sampled turns).

The cut: viewer event indices anchor to the stream via tool callIDs. We find
the first tool_use at/after the cut event, locate its messageID in the stream,
and keep every message strictly before it. A kept event sharing that messageID
(i.e. a cut through the middle of a message) is an error.

Resume mechanics (`opencode run --session <id> --model ... "<text>"`) ALWAYS
append a new user text part — there is no prompt-less resume — so the control/
treatment difference lives in that resume prompt, not in storage. Storage is
identical for both conditions.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config, ptbio, scaffold

CWD = "/home/ben/task"
OPENCODE_VERSION = "1.1.59"  # standard.def pins opencode-ai@1.1.59
_STREAM_TYPES = {"step_start", "tool_use", "text", "step_finish", "reasoning"}


def load_stream(traj: config.Trajectory) -> list[dict]:
    """Part-stream lines of solve_out.txt (skips the harness's non-JSON noise)."""
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
            if d.get("type") in _STREAM_TYPES and isinstance(d.get("part"), dict):
                out.append(d)
    return out


def group_messages(stream: list[dict]) -> list[tuple[str, list[dict]]]:
    """(messageID, [stream lines]) in order of first appearance."""
    order: list[str] = []
    by_id: dict[str, list[dict]] = {}
    for d in stream:
        mid = d["part"].get("messageID")
        if mid not in by_id:
            by_id[mid] = []
            order.append(mid)
        by_id[mid].append(d)
    return [(mid, by_id[mid]) for mid in order]


def cut_message_id(events: list[dict], cut: int, stream: list[dict]) -> str:
    """messageID of the first message dropped by the cut, anchored by the
    ORDINAL of the first tool_use at/after the cut (its position among all
    tool_uses), NOT by callID. Some runs reset/reuse callIDs across messages
    (e.g. functions.bash:0 recurs), which breaks ID-based matching; the viewer's
    tool_use order and the stream's tool_use order agree, so position is robust."""
    kept = sum(1 for tc in ptbio.iter_tool_calls(events) if tc.idx < cut)
    tool_uses = [d for d in stream if d["type"] == "tool_use"]
    if kept >= len(tool_uses):
        raise ValueError(f"no tool_use at/after cut {cut}; cannot anchor")
    return tool_uses[kept]["part"]["messageID"]


def validate_cut(events: list[dict], cut: int, stream: list[dict],
                 kept_ids: set[str], cut_mid: str) -> dict:
    """The kept viewer events' tool calls must map exactly onto kept messages —
    checked by ORDINAL (callID-agnostic). Requires the events' tool_use count to
    match the stream's (else the ordinal mapping is unsafe -> not ok)."""
    tool_uses = [d for d in stream if d["type"] == "tool_use"]
    n_events_tool = len(ptbio.iter_tool_calls(events))
    kept = sum(1 for tc in ptbio.iter_tool_calls(events) if tc.idx < cut)
    misplaced = [d["part"].get("callID") for d in tool_uses[:kept]
                 if d["part"]["messageID"] not in kept_ids]
    dropped_in_kept = [d["part"].get("callID") for d in tool_uses[kept:]
                       if d["part"]["messageID"] in kept_ids]
    return {
        "kept_tool_calls": kept,
        "stream_tool_uses": len(tool_uses),
        "events_tool_calls": n_events_tool,
        "count_match": n_events_tool == len(tool_uses),
        "misplaced_kept_calls": misplaced,
        "dropped_calls_in_kept_messages": dropped_in_kept,
        "ok": (n_events_tool == len(tool_uses)) and not misplaced and not dropped_in_kept,
    }


def _id_before(first_id: str, prefix: str) -> str:
    """An id with the right prefix that sorts strictly before first_id."""
    body = first_id.split("_", 1)[1]
    candidate = f"{prefix}_{body[:4]}{'0' * (len(body) - 4)}"
    if candidate >= first_id:  # body started with zeros; go shorter, '_'+0* < any longer body? no — use all-zero shorter body
        candidate = f"{prefix}_0"
    assert candidate < first_id
    return candidate


def _provider_model(recorded: str) -> tuple[str, str]:
    """'opencode/minimax-m2.5-free' -> ('opencode', 'minimax-m2.5-free')."""
    if "/" in recorded:
        p, m = recorded.split("/", 1)
        return p, m
    return "opencode", recorded


def build_session(spec: config.ExperimentSpec, job_home: Path) -> dict:
    """Write reconstructed OpenCode storage under job_home/.local/share/opencode.
    Returns metadata incl. the sessionID to resume. Storage does not depend on
    spec.condition (the reminder travels in the resume prompt)."""
    traj = spec.trajectory
    cut = spec.cut_before_event
    events = ptbio.load_events(traj)
    stream = load_stream(traj)
    if not stream:
        raise RuntimeError(f"empty part stream in {traj.raw_dir / 'solve_out.txt'}")

    session_id = stream[0]["sessionID"]
    messages = group_messages(stream)
    cut_mid = cut_message_id(events, cut, stream)
    mids = [m for m, _ in messages]
    if cut_mid not in mids:
        raise ValueError(f"cut messageID {cut_mid} not in stream order")
    kept = messages[: mids.index(cut_mid)]
    kept_ids = {m for m, _ in kept}

    val = validate_cut(events, cut, stream, kept_ids, cut_mid)
    if not val["ok"]:
        raise ValueError(f"cut {cut} does not fall on a message boundary: {val}")

    provider_id, model_id = _provider_model(traj.policy_model_recorded)
    storage = job_home / ".local" / "share" / "opencode" / "storage"
    t0_ms = stream[0]["timestamp"]
    t_last_ms = max(d["timestamp"] for _, lines in kept for d in lines) if kept else t0_ms

    def write(rel: str, obj) -> None:
        p = storage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2))

    write("migration", "2")
    write("project/global.json", {
        "id": "global", "worktree": "/", "sandboxes": [],
        "time": {"created": t0_ms - 10_000, "updated": t0_ms - 10_000},
    })
    write(f"session/global/{session_id}.json", {
        "id": session_id, "slug": "rollback-reconstruction",
        "version": OPENCODE_VERSION, "projectID": "global", "directory": CWD,
        "title": f"rollback {spec.cell_id} of {traj.run_id}",
        "time": {"created": t0_ms - 5_000, "updated": t_last_ms},
    })

    # user prompt message (CLI arg in the original run; never streamed)
    user_mid = _id_before(mids[0], "msg")
    prompt = scaffold.generate_prompt(traj)
    write(f"message/{session_id}/{user_mid}.json", {
        "id": user_mid, "sessionID": session_id, "role": "user",
        "time": {"created": t0_ms - 5_000},
        "agent": "build",
        "model": {"providerID": provider_id, "modelID": model_id},
    })
    first_part_id = stream[0]["part"]["id"]
    write(f"part/{user_mid}/{_id_before(first_part_id, 'prt')}.json", {
        "id": _id_before(first_part_id, "prt"), "sessionID": session_id,
        "messageID": user_mid, "type": "text", "text": prompt,
    })

    # kept assistant messages: synthesized info + verbatim parts
    n_parts = 0
    for mid, lines in kept:
        finish = next((d["part"] for d in reversed(lines)
                       if d["part"]["type"] == "step-finish"), None)
        times = [d["timestamp"] for d in lines]
        write(f"message/{session_id}/{mid}.json", {
            "id": mid, "sessionID": session_id, "role": "assistant",
            "time": {"created": min(times), "completed": max(times)},
            "parentID": user_mid,
            "modelID": model_id, "providerID": provider_id,
            "mode": "build", "agent": "build",
            "path": {"cwd": CWD, "root": CWD},
            "cost": (finish or {}).get("cost", 0),
            "tokens": (finish or {}).get("tokens") or {
                "input": 0, "output": 0, "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
            "finish": (finish or {}).get("reason"),
        })
        for d in lines:
            part = d["part"]
            write(f"part/{mid}/{part['id']}.json", part)
            n_parts += 1

    return {
        "session_id": session_id,
        "storage_dir": str(storage),
        "opencode_version": OPENCODE_VERSION,
        "cwd": CWD,
        "cut_before_event": cut,
        "cut_message_id": cut_mid,
        "messages_kept": len(kept),
        "messages_dropped": len(messages) - len(kept),
        "parts_written": n_parts,
        "user_message_id": user_mid,
        "prompt_chars": len(prompt),
        "condition": spec.condition,
        # storage is condition-independent; the reminder is the resume prompt
        "intervention_injected": False,
        "cut_validation": val,
    }
