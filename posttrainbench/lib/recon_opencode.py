"""Reconstruct OpenCode session storage at a cut, for native resume.

The `opencode run --format json` stream IS the part log: every part object is
printed verbatim (real ids, timestamps, token usage), so kept parts are
written back byte-faithfully. Synthesized (and flagged): the message-info
records (role/parentID/agent metadata around each part group) and the initial
user prompt (a CLI arg, never streamed).

Storage layout (validated against opencode-ai@1.1.59 in the first pass):
    storage/migration                        "2"
    storage/project/global.json
    storage/session/global/<sessionID>.json
    storage/message/<sessionID>/<msgID>.json
    storage/part/<msgID>/<partID>.json
Resume: `opencode run --session <id> --model <provider/model> "<prompt>"`
always appends a new user text part.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import prompts
from .runs import Trajectory
from .stream import CutPlan, Parsed

OPENCODE_VERSION = "1.1.59"   # pinned by the PTB container
CWD = "/home/ben/task"


@dataclass
class OpencodeBundle:
    traj: Trajectory
    plan: CutPlan
    session_id: str
    files: dict[str, str]         # storage-relative path -> file content
    flags: list[dict]
    stats: dict
    provider_model: str           # "provider/model" for --model at resume


def _flag(code: str, detail: str, severity: str = "warn") -> dict:
    return {"code": code, "severity": severity, "detail": detail}


def _id_before(first_id: str, prefix: str) -> str:
    body = first_id.split("_", 1)[1]
    candidate = f"{prefix}_{body[:4]}" + "0" * (len(body) - 4)
    if candidate >= first_id:
        candidate = f"{prefix}_0"
    assert candidate < first_id
    return candidate


def _recorded_model(traj: Trajectory) -> str:
    """provider/model as the run dir encodes it, e.g. opencode_opencode_kimi-k2.5
    -> opencode/kimi-k2.5 ; opencode_zai_glm-5 -> zai/glm-5."""
    rest = traj.run_dir.name[len(traj.agent) + 1:]
    rest = rest.rsplit("_10h", 1)[0]
    provider, model = rest.split("_", 1)
    return f"{provider}/{model}"


def build(traj: Trajectory, parsed: Parsed, plan: CutPlan) -> OpencodeBundle:
    recs = parsed.records
    session_id = recs[0]["sessionID"]
    flags = [_flag("message_info_synthesized",
                   "per-message metadata records synthesized (role/parentID/agent); "
                   "parts are byte-verbatim from the stream", "info"),
             _flag("prompt_regenerated", prompts.prompt_provenance(traj), "info"),
             _flag("prompt_era_matched",
                   "regeneration is era-exact since 2026-07-13 (this flag marks "
                   "post-fix reconstructions)", "info")]

    # group records by messageID in order of first appearance; check contiguity
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(recs):
        mid = r["part"].get("messageID")
        if mid not in groups:
            groups[mid] = []
            order.append(mid)
        groups[mid].append(i)
    for mid, idxs in groups.items():
        if idxs != list(range(idxs[0], idxs[-1] + 1)):
            raise ValueError(f"message {mid} is not contiguous in the stream; "
                             f"cut anchoring would be unsafe")

    if plan.raw_cut >= len(recs):
        kept_mids = list(order)          # end cut: keep every message
    else:
        cut_mid = recs[plan.raw_cut]["part"].get("messageID")
        kept_mids = order[: order.index(cut_mid)]

    # reasoning-visibility check: billed reasoning tokens without reasoning parts
    reasoning_tokens = sum(
        (r["part"].get("tokens") or {}).get("reasoning", 0)
        for r in recs if r.get("type") == "step_finish")
    has_reasoning_parts = any(r.get("type") == "reasoning" for r in recs)
    if reasoning_tokens > 0 and not has_reasoning_parts:
        flags.append(_flag(
            "opencode_reasoning_not_streamed",
            f"model billed {reasoning_tokens} reasoning tokens but the stream has no "
            f"reasoning parts — the run's stdout printer skipped them (--format json "
            f"without --thinking). opencode v1.1.59 stored reasoning and REPLAYED it "
            f"into every subsequent request (verified in source, 2026-07-13), so the "
            f"original model DID see this reasoning in-context; the rebuilt storage "
            f"lacks it and a resume sends less context than the original runtime "
            f"did. Permanent: the container's storage is gone."))

    files: dict[str, str] = {}
    t0 = recs[0]["timestamp"]
    kept_idx = [i for mid in kept_mids for i in groups[mid]]
    t_last = max((recs[i]["timestamp"] for i in kept_idx), default=t0)

    files["migration"] = "2"
    files["project/global.json"] = json.dumps({
        "id": "global", "worktree": "/", "sandboxes": [],
        "time": {"created": t0 - 10_000, "updated": t0 - 10_000}}, indent=2)
    files[f"session/global/{session_id}.json"] = json.dumps({
        "id": session_id, "slug": "ptb2-reconstruction",
        "version": OPENCODE_VERSION, "projectID": "global", "directory": CWD,
        "title": f"ptb2 cut ev{plan.cut_event} of {traj.run_id}",
        "time": {"created": t0 - 5_000, "updated": t_last}}, indent=2)

    provider_model = _recorded_model(traj)
    provider_id, model_id = provider_model.split("/", 1)

    user_mid = _id_before(order[0], "msg")
    files[f"message/{session_id}/{user_mid}.json"] = json.dumps({
        "id": user_mid, "sessionID": session_id, "role": "user",
        "time": {"created": t0 - 5_000}, "agent": "build",
        "model": {"providerID": provider_id, "modelID": model_id}}, indent=2)
    first_part_id = recs[0]["part"]["id"]
    user_pid = _id_before(first_part_id, "prt")
    files[f"part/{user_mid}/{user_pid}.json"] = json.dumps({
        "id": user_pid, "sessionID": session_id, "messageID": user_mid,
        "type": "text", "text": prompts.task_prompt(traj)}, indent=2)

    n_parts = 0
    for mid in kept_mids:
        idxs = groups[mid]
        finish = next((recs[i]["part"] for i in reversed(idxs)
                       if recs[i]["part"]["type"] == "step-finish"), None)
        times = [recs[i]["timestamp"] for i in idxs]
        files[f"message/{session_id}/{mid}.json"] = json.dumps({
            "id": mid, "sessionID": session_id, "role": "assistant",
            "time": {"created": min(times), "completed": max(times)},
            "parentID": user_mid, "modelID": model_id, "providerID": provider_id,
            "mode": "build", "agent": "build",
            "path": {"cwd": CWD, "root": CWD},
            "cost": (finish or {}).get("cost", 0),
            "tokens": (finish or {}).get("tokens") or {
                "input": 0, "output": 0, "reasoning": 0,
                "cache": {"read": 0, "write": 0}},
            "finish": (finish or {}).get("reason")}, indent=2)
        for i in idxs:
            part = recs[i]["part"]
            files[f"part/{mid}/{part['id']}.json"] = json.dumps(part, indent=2)
            n_parts += 1

    stats = {"messages_kept": len(kept_mids),
             "messages_dropped": len(order) - len(kept_mids),
             "parts_written": n_parts,
             "n_verbatim": n_parts,
             "n_synthesized": 1 + len(kept_mids)}  # prompt part + message infos
    return OpencodeBundle(traj, plan, session_id, files, flags, stats, provider_model)
