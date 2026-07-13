"""Reconstruct a Claude Code context (and native session file) at a cut.

Model-visible context at a main-thread cut:
  - if no compaction happened before the cut: [task prompt] + all main-thread
    messages up to the cut;
  - if compaction(s) happened: the stream records each compact boundary
    followed by ONE synthetic user message holding the summary Claude Code
    actually put in front of the model. Context = that summary + everything
    after it (the pre-compaction history was no longer visible to the model).
  - each mid-run `system:init` is the run-era CLI re-entering the session to
    deliver a background-task completion notification (mechanism confirmed
    2026-07-13, see ISSUES.md): an unstreamed plain user turn in the
    <task-notification> format. Where the reviewed assignment file
    (restart_assignments.json) covers the restart we insert a reconstructed
    notification (flagged); otherwise we still insert the OLD time-nudge
    template, which is known-wrong content, with a loud flag (deferred runs).
  - subagent (sidechain) records never entered the main context and are
    excluded; only the Task tool_result the main thread received remains.

Assistant stream lines sharing one message.id are chunks of one API message
and get merged. All assistant/user content blocks (including thinking blocks
with their signatures) are copied verbatim from the stream.

Session file format (validated against a real `claude --resume` on the GPU
box in the first pass): one JSON object per turn with envelope fields
parentUuid/isSidechain/userType/cwd/sessionId/version/gitBranch/type/message/
uuid/timestamp under ~/.claude/projects/<cwd with '/' and '.' -> '-'>/.
Envelope fields are synthesized (flagged); message payloads are not.
"""
from __future__ import annotations

import json
import uuid as uuidlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from . import prompts
from .runs import Trajectory
from .stream import CutPlan, Parsed, _claude_msg_id, _is_main


@dataclass
class Msg:
    role: str                     # "user" | "assistant"
    message: dict                 # Anthropic-format message object
    provenance: str               # verbatim | verbatim_compact_summary |
                                  # regenerated_prompt | estimated_continuation
    raw_records: list[int] = field(default_factory=list)


@dataclass
class ContextBundle:
    traj: Trajectory
    plan: CutPlan
    messages: list[Msg]
    flags: list[dict]
    stats: dict


def _flag(code: str, detail: str, severity: str = "warn") -> dict:
    return {"code": code, "severity": severity, "detail": detail}


_ASSIGNMENTS_PATH = Path(__file__).resolve().parents[1] / "restart_assignments.json"
_assignments_cache: dict | None = None


def restart_assignments() -> dict:
    """The reviewed background-task-notification assignments (see ISSUES.md,
    'background restarts'). {'assignments': {run_id: {init_record: {...}}},
    'deferred': {run_id: {reason}}}."""
    global _assignments_cache
    if _assignments_cache is None:
        _assignments_cache = (json.loads(_ASSIGNMENTS_PATH.read_text())
                              if _ASSIGNMENTS_PATH.exists()
                              else {"assignments": {}, "deferred": {}})
    return _assignments_cache


def _task_notification_text(a: dict) -> str:
    """The run-era CLI's injected turn, rebuilt from a reviewed assignment.
    Both formats captured verbatim from the 2.1.76 repros (2026-07-13):
    shell tasks end with 'Read the output file...', agent (subagent) tasks
    carry <result>/<usage> and end with 'Full transcript available at:'.
    Failed-task wording is an assumption; agent <result>/<usage> lines are
    omitted when the data was never recorded (see the assignment's 'lost')."""
    head = ("<task-notification>\n"
            f"<task-id>{a['task_id']}</task-id>\n"
            f"<tool-use-id>{a['tool_use_id']}</tool-use-id>\n"
            f"<output-file>{a['output_file']}</output-file>\n"
            f"<status>{a['status']}</status>\n")
    if a.get("task_type") == "agent":
        body = f"<summary>Agent \"{a['description']}\" completed</summary>\n"
        if a.get("result") is not None:
            body += f"<result>{a['result']}</result>\n"
        u = a.get("usage")
        if u:
            body += (f"<usage><total_tokens>{u['total_tokens']}</total_tokens>"
                     f"<tool_uses>{u['tool_uses']}</tool_uses>"
                     f"<duration_ms>{u['duration_ms']}</duration_ms></usage>\n")
        footer = f"Full transcript available at: {a['output_file']}"
    else:
        verb = "completed" if a["status"] == "completed" else "failed"
        body = (f"<summary>Background command \"{a['command']}\" {verb} "
                f"(exit code {a['exit_code']})</summary>\n")
        footer = f"Read the output file to retrieve the result: {a['output_file']}"
    return head + body + "</task-notification>\n" + footer


def build_context(traj: Trajectory, parsed: Parsed, plan: CutPlan) -> ContextBundle:
    recs = parsed.records
    cut = plan.raw_cut
    flags: list[dict] = [_flag("envelope_synthesized",
                               "session envelope (uuids/timestamps/cwd/gitBranch) is synthetic; "
                               "message payloads are verbatim from the stream", "info")]

    init_idxs = [i for i, r in enumerate(recs)
                 if r.get("type") == "system" and r.get("subtype") == "init"]
    compact_idxs = [i for i, r in enumerate(recs)
                    if r.get("type") == "system" and r.get("subtype") == "compact_boundary"
                    and i < cut and _is_main(r)]

    messages: list[Msg] = []
    if compact_idxs:
        start = compact_idxs[-1]
        flags.append(_flag(
            "compaction_slice",
            f"{len(compact_idxs)} compaction(s) before the cut; context starts at the "
            f"synthetic summary after the boundary at record {start}. This matches what "
            f"the model saw (pre-compaction history was already gone)."))
    else:
        start = -1
        messages.append(Msg("user", {"role": "user", "content": prompts.task_prompt(traj)},
                            "regenerated_prompt"))
        flags.append(_flag("prompt_regenerated", prompts.prompt_provenance(traj), "info"))
        flags.append(_flag(
            "prompt_era_matched",
            "regeneration is era-exact since 2026-07-13 (this flag marks post-fix "
            "reconstructions; probes without it used the wrong prompt era)", "info"))

    n_side = 0
    cur_asst: Msg | None = None
    for i in range(start + 1, cut):
        r = recs[i]
        if not _is_main(r):
            n_side += 1
            cur_asst = None
            continue
        t = r.get("type")
        if t == "system":
            if r.get("subtype") == "status":
                continue  # e.g. "compacting" notice; must not break message grouping
            if r.get("subtype") == "init" and init_idxs and i != init_idxs[0]:
                a = restart_assignments()["assignments"].get(traj.run_id, {}).get(str(i))
                if a is not None:
                    messages.append(Msg(
                        "user", {"role": "user", "content": _task_notification_text(a)},
                        "reconstructed_notification", [i]))
                    lost = f" LOST CONTENT: {a['lost']}." if a.get("lost") else ""
                    flags.append(_flag(
                        "restart_notification_reconstructed",
                        f"restart at record {i}: background-task notification for "
                        f"{a['task_id']} reconstructed from the reviewed assignment file "
                        f"(confidence: {a['confidence']}; exit code inferred, not "
                        f"recorded).{lost} Evidence: {a['evidence']}",
                        "warn" if a.get("lost") else "info"))
                else:
                    text, basis = prompts.continuation_prompt(i / max(1, len(recs)),
                                                              traj.num_hours)
                    messages.append(Msg("user", {"role": "user", "content": text},
                                        "estimated_continuation", [i]))
                    reason = (restart_assignments()["deferred"]
                              .get(traj.run_id, {}).get("reason", "not yet reviewed"))
                    flags.append(_flag(
                        "continuation_prompt_estimated",
                        f"restart at record {i}: KNOWN-WRONG content — the real injected "
                        f"turn was a background-task notification (mechanism confirmed "
                        f"2026-07-13), but this restart has no reviewed assignment "
                        f"({reason}); the old time-nudge template is inserted instead. "
                        f"{basis}"))
            cur_asst = None
            continue
        if t == "result":
            cur_asst = None
            continue
        msg = r.get("message") or {}
        if t == "assistant":
            mid = _claude_msg_id(r)
            if cur_asst is not None and cur_asst.message.get("id") == mid:
                cur_asst.message["content"].extend(msg.get("content") or [])
                cur_asst.raw_records.append(i)
            else:
                m = {"id": mid, "role": "assistant", "model": msg.get("model"),
                     "content": list(msg.get("content") or [])}
                cur_asst = Msg("assistant", m, "verbatim", [i])
                messages.append(cur_asst)
        elif t == "user":
            prov = "verbatim_compact_summary" if r.get("isSynthetic") else "verbatim"
            messages.append(Msg("user", {"role": "user", "content": msg.get("content")},
                                prov, [i]))
            cur_asst = None

    # validation: no dangling tool_use, roles roughly alternate
    tu, tr = set(), set()
    for m in messages:
        blocks = m.message.get("content")
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if b.get("type") == "tool_use":
                tu.add(b.get("id"))
            elif b.get("type") == "tool_result":
                tr.add(b.get("tool_use_id"))
    dangling = sorted(x for x in (tu - tr) if x)
    if dangling:
        raise ValueError(f"cut at record {cut} leaves dangling tool_use ids: {dangling}")

    n_think = sum(1 for m in messages if isinstance(m.message.get("content"), list)
                  for b in m.message["content"] if b.get("type") == "thinking")
    if n_think:
        flags.append(_flag(
            "thinking_blocks_included",
            f"{n_think} thinking block(s) carried verbatim incl. signatures (note: for "
            f"Claude 4-family models the API returns summarized thinking, so the text is "
            f"the summary the client saw, which is also exactly what a resume replays)",
            "info"))
    if n_side:
        flags.append(_flag(
            "sidechain_excluded",
            f"{n_side} subagent (sidechain) stream records before the cut excluded from "
            f"the main context, matching what the main agent saw", "info"))

    stats = {"n_messages": len(messages),
             "n_verbatim": sum(1 for m in messages if m.provenance.startswith("verbatim")),
             "n_synthesized": sum(1 for m in messages if not m.provenance.startswith("verbatim")),
             "compactions_before_cut": len(compact_idxs),
             "relaunches_before_cut": sum(1 for j in init_idxs[1:] if start < j < cut),
             "sidechain_records_excluded": n_side,
             "thinking_blocks": n_think}
    return ContextBundle(traj, plan, messages, flags, stats)


def project_dir_name(cwd: str) -> str:
    """Claude Code >=2.1.181 maps every non-alphanumeric char to '-'
    (older versions kept '_'; use the convention of the CLI doing the resume)."""
    import re
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def emit_session(bundle: ContextBundle, cwd: str, version: str,
                 session_id: str | None = None) -> tuple[str, list[dict]]:
    """Render the context as native session-JSONL lines. Returns (session_id, lines)."""
    session_id = session_id or str(uuidlib.uuid4())
    lines, parent = [], None
    t0 = datetime(2026, 2, 6, 15, 0, 0)
    for k, m in enumerate(bundle.messages):
        u = str(uuidlib.uuid4())
        lines.append({
            "parentUuid": parent, "isSidechain": False, "userType": "external",
            "cwd": cwd, "sessionId": session_id, "version": version,
            "gitBranch": "", "type": m.role, "message": m.message, "uuid": u,
            "timestamp": (t0 + timedelta(seconds=k)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        })
        parent = u
    return session_id, lines


def original_cli_version(parsed: Parsed) -> str | None:
    for r in parsed.records:
        if r.get("type") == "system" and r.get("subtype") == "init":
            return r.get("claude_code_version") or r.get("version")
    return None


def original_model(parsed: Parsed) -> str | None:
    for r in parsed.records:
        if r.get("type") == "system" and r.get("subtype") == "init":
            return r.get("model")
    return None
