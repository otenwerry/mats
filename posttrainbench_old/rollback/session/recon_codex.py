"""Rebuild a Codex CLI session (rollout JSONL) truncated at the cut, for native
`codex exec resume`. Mirrors recon.py (Claude) / recon_opencode.py.

HOW CODEX PERSISTS A SESSION
  Codex writes one rollout file per session at
      $CODEX_HOME/sessions/<YYYY>/<MM>/<DD>/rollout-<ts>-<thread_id>.jsonl
  JSONL: the FIRST line is session metadata; each subsequent line is a
  RolloutItem `{"timestamp": ..., "type": <kind>, "payload": {...}}`. The
  conversation lives in `type:"response_item"` lines whose payloads mirror the
  OpenAI Responses API items (message / reasoning / function_call /
  function_call_output). On `codex exec resume <thread_id>` (or `--last`), codex
  locates the rollout file and replays it to restore state, then APPENDS new
  items (so the thread_id is stable across resumes).
  Refs: developers.openai.com/codex/cli/reference ;
        deepwiki.com/openai/codex 3.5.2 (rollout persistence), 4.4 (resume).

WHAT WE HAVE
  The dataset kept only the codex EVENT stream (solve_out.txt: item.completed
  with reasoning / agent_message / command_execution / file_change items) and
  the viewer's `codex_item` events — NOT the rollout file. So we reconstruct the
  rollout from the events. `normalize_events()` (below) turns events[:cut] into
  an ordered, scaffold-neutral conversation; that part is verifiable offline.

  !!! VALIDATION SEAM — `to_rollout_lines()` !!!
  The EXACT RolloutItem line schema (field names of response_item payloads:
  message vs reasoning vs function_call/_output, role, content part types,
  call_id, name="shell"/"apply_patch", arguments JSON) must be confirmed against
  a REAL captured rollout before a paid run. To capture one (needs the ChatGPT
  subscription auth.json — see run/solve_intervention_codex.sh):
      codex exec --json "say hi and run `ls`"   # any tiny task
      cat $CODEX_HOME/sessions/**/rollout-*.jsonl   # <- copy the real schema here
  Until then build_session() refuses unless PTB_CODEX_ROLLOUT_VALIDATED=1, so we
  never burn subscription quota resuming a rollout codex would silently reject.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from .. import config, ptbio, scaffold

CWD = "/home/ben/task"
# MUST match the codex pinned in run/standard.def — the rollout schema is
# version-specific (0.79.0 used a `shell` tool; 0.139.x/0.140.x use `exec_command`).
# standard.def pins 0.140.0; the exec_command/custom_tool_call payload shapes are
# unchanged from the 0.139.0 rollout this was validated against, but the on-box
# codex CANARY is what confirms 0.140.0 actually resumes the reconstructed rollout.
CLI_VERSION = "0.140.0"
# Rollout line timestamps are cosmetic for resume (codex replays items, not
# clocks); a fixed base keeps reconstruction deterministic. timer.sh governs
# the real elapsed time on the box.
BASE_TS = "2026-02-06T15:47:00.000Z"


def _line(type_: str, payload: dict) -> dict:
    """A rollout JSONL line: every real line is {timestamp, type, payload}."""
    return {"timestamp": BASE_TS, "type": type_, "payload": payload}


def normalize_events(events: list[dict], cut: int) -> list[dict]:
    """events[:cut] -> ordered conversation turns (scaffold-neutral). VERIFIABLE
    offline. Each turn: {role, kind, text|command|output|paths, call_id}."""
    turns: list[dict] = []
    for e in events[:cut]:
        if e.get("type") != "codex_item":
            continue
        it = e.get("item") or {}
        kind = it.get("type")
        if kind == "reasoning":
            turns.append({"role": "assistant", "kind": "reasoning", "text": it.get("text", "")})
        elif kind == "agent_message":
            turns.append({"role": "assistant", "kind": "message", "text": it.get("text", "")})
        elif kind == "command_execution":
            turns.append({"role": "assistant", "kind": "function_call",
                          "call_id": it.get("id", ""), "name": "shell",
                          "command": it.get("command", ""),
                          "output": it.get("aggregated_output", ""),
                          "exit_code": it.get("exit_code")})
        elif kind == "file_change":
            turns.append({"role": "assistant", "kind": "apply_patch",
                          "call_id": it.get("id", ""),
                          "changes": [{"path": c.get("path"), "kind": c.get("kind", "update")}
                                      for c in (it.get("changes") or [])]})
    return turns


# apply_patch reconstruction is LOSSY: the dataset's file_change events recorded
# only the changed paths + kind, NOT the diff bodies. We therefore synthesize a
# header-only patch — valid as REPLAYED history (codex replays the call as
# context; it does not re-apply it), but the exact edited content is gone. This
# is surfaced in build_session()'s return meta (apply_patch_lossy) so any codex
# result inherits a visible caveat.
_PATCH_KIND = {"add": "Add File", "update": "Update File", "delete": "Delete File"}


def _synth_patch(changes: list[dict]) -> str:
    body = ["*** Begin Patch"]
    for ch in changes:
        body.append(f"*** {_PATCH_KIND.get(ch.get('kind', 'update'), 'Update File')}: {ch.get('path')}")
    body.append("*** End Patch")
    return "\n".join(body) + "\n"


def to_rollout_lines(turns: list[dict], thread_id: str, prompt: str,
                     intervention: str | None, cwd: str = CWD) -> list[dict]:
    """turns -> rollout JSONL lines matching the REAL codex 0.139.0 schema
    (validated by resuming a reconstructed rollout — see module docstring):
      - every line is {timestamp, type, payload};
      - session_meta carries id/cwd/originator/cli_version/source/model_provider;
      - a shell command is a `function_call` named "exec_command" whose arguments
        JSON is {"cmd": <command>, "workdir": <cwd>} (NOT the old 0.79.0 "shell"
        + {"command":[...]} shape), paired with a function_call_output."""
    lines: list[dict] = [
        _line("session_meta", {
            "id": thread_id, "timestamp": BASE_TS, "cwd": cwd,
            "originator": "codex_exec", "cli_version": CLI_VERSION,
            "source": "exec", "thread_source": "user", "model_provider": "openai"}),
        _line("response_item", {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": prompt}]}),
    ]
    for t in turns:
        kind = t["kind"]
        if kind == "reasoning":
            lines.append(_line("response_item", {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": t["text"]}]}))
        elif kind == "message":
            lines.append(_line("response_item", {
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": t["text"]}]}))
        elif kind == "function_call":  # shell command -> exec_command
            args = json.dumps({"cmd": t.get("command", ""), "workdir": cwd})
            lines.append(_line("response_item", {
                "type": "function_call", "name": "exec_command",
                "call_id": t["call_id"], "arguments": args}))
            lines.append(_line("response_item", {
                "type": "function_call_output", "call_id": t["call_id"],
                "output": str(t.get("output", ""))}))
        elif kind == "apply_patch":  # file edit -> custom_tool_call(apply_patch)
            lines.append(_line("response_item", {
                "type": "custom_tool_call", "status": "completed",
                "call_id": t["call_id"], "name": "apply_patch",
                "input": _synth_patch(t.get("changes", []))}))
            paths = ", ".join(ch.get("path", "") for ch in t.get("changes", []))
            lines.append(_line("response_item", {
                "type": "custom_tool_call_output", "call_id": t["call_id"],
                "output": f"Success. Updated the following files:\n{paths}\n"}))
    # Do not append the rollback experiment prompt here. The runner appends the
    # exact control/treatment user turn at resume time, matching other scaffolds.
    return lines


def build_session(spec: config.ExperimentSpec, out_dir: Path) -> dict:
    cut = spec.cut_before_event
    events = ptbio.load_events(spec.trajectory)
    turns = normalize_events(events, cut)
    thread_id = str(uuid.uuid4())
    prompt = scaffold.generate_prompt(spec.trajectory)

    # Gate lifted: the 0.139.0 rollout schema in to_rollout_lines() (exec_command
    # for shell, custom_tool_call for apply_patch) was validated by building a
    # rollout from a real trajectory and confirming codex 0.139.0 RESUMES it
    # (loads full history, no parse error). The only coupling is CLI_VERSION,
    # which MUST track the codex pinned in run/standard.def.
    lines = to_rollout_lines(turns, thread_id, prompt, None)
    # codex locates a session by scanning $CODEX_HOME/sessions/**/rollout-*.jsonl
    # and matching the id, so mirror the real on-disk layout exactly:
    #   sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<thread_id>.jsonl
    sess_dir = out_dir / ".codex" / "sessions" / "2026" / "02" / "06"
    sess_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = sess_dir / f"rollout-2026-02-06T15-47-00-{thread_id}.jsonl"
    with open(rollout_path, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    n_patch = sum(1 for t in turns if t["kind"] == "apply_patch")
    return {"session_id": thread_id, "rollout_path": str(rollout_path),
            "n_turns": len(turns), "cut_before_event": cut,
            "condition": spec.condition,
            "intervention_injected": False,
            "prompt_injected_by": "runner_resume_prompt",
            # LOSSY: the prefix contained file edits whose diff bodies the dataset
            # did not preserve; the reconstructed apply_patch calls are header-only
            # (see _synth_patch). Propagate so downstream results carry the caveat.
            "apply_patch_lossy": n_patch > 0, "n_apply_patch": n_patch}
