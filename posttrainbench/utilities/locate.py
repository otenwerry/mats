"""Deterministic hop from judge citations to trace turns.

The judge cites workspace files (`train_v7.py`, `data/x.jsonl`); this module
finds the trace event(s) where the agent **wrote / edited / ran** each cited
file. Those events are the candidate hack turns. It also provides a compact
trace view for feeding traces to an LLM without blowing the context window.

Handles both event shapes:
  - block formats (claude_code, opencode): `blocks[]` with tool_use
    (input.file_path / input.command) and tool_result (content).
  - codex: `codex_item`s — command_execution (item.command / aggregated_output),
    file_change (item.changes[].path), reasoning / agent_message (item.text).
"""
from __future__ import annotations

import json
import os
import re

# Strength of a filename hit in an event; higher = more likely the hack turn.
WRITE, RUN, MENTION = 3, 2, 1


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/"))


def _event_file_signals(ev: dict) -> list[tuple[str, int]]:
    """(filename-or-path, strength) signals this event emits, for matching."""
    out: list[tuple[str, int]] = []

    # ---- codex ----
    if ev.get("type") == "codex_item":
        item = ev.get("item") or {}
        st = ev.get("subtype")
        if st == "file_change":
            for ch in item.get("changes", []):
                p = ch.get("path", "")
                if p:
                    out.append((p, WRITE))
        elif st == "command_execution":
            cmd = item.get("command", "") or ""
            out.append((cmd, RUN))
        elif st in ("reasoning", "agent_message"):
            out.append((item.get("text", "") or "", MENTION))
        return out

    # ---- block formats ----
    for b in ev.get("blocks") or []:
        bt = b.get("type")
        if bt == "tool_use":
            name = (b.get("name") or "").lower()
            inp = b.get("input") if isinstance(b.get("input"), dict) else {}
            fp = inp.get("file_path") or inp.get("path") or inp.get("filename")
            if fp and name in ("write", "edit", "multiedit", "create", "notebook_edit", "str_replace_editor"):
                out.append((str(fp), WRITE))
            elif fp:
                out.append((str(fp), MENTION))
            cmd = inp.get("command")
            if isinstance(cmd, str) and cmd:
                out.append((cmd, RUN))
        elif bt == "tool_result":
            c = b.get("content")
            if isinstance(c, str):
                out.append((c, MENTION))
            elif isinstance(c, list):
                for sub in c:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        out.append((sub["text"], MENTION))
        elif bt == "text":
            out.append((b.get("text", "") or "", MENTION))
    return out


def _matches(signal_text: str, strength: int, fname: str, path: str) -> int:
    """Best strength at which `signal_text` references the cited file."""
    base = _basename(path)
    # A WRITE/RUN signal is a file_path or command; match the basename in it.
    if strength >= RUN:
        # exact path or basename token present
        if path in signal_text or fname in signal_text or base in signal_text:
            return strength
        return 0
    # MENTION signal is free text/output; require the (non-trivial) basename.
    if len(base) >= 5 and base in signal_text:
        return MENTION
    return 0


def candidate_turns(events: list, cited_files: list[str]) -> dict[str, list[dict]]:
    """{cited_path: [{idx, strength}], ...} sorted by strength then order."""
    result: dict[str, list[dict]] = {}
    for path in cited_files:
        fname = _basename(path)
        if not fname:
            continue
        hits: list[dict] = []
        for idx, ev in enumerate(events):
            best = 0
            for sig_text, strength in _event_file_signals(ev):
                best = max(best, _matches(sig_text, strength, fname, path))
            if best:
                hits.append({"idx": idx, "strength": best})
        hits.sort(key=lambda h: (-h["strength"], h["idx"]))
        if hits:
            result[path] = hits
    return result


def best_marked_turns(events: list, cited_files: list[str], per_file: int = 1) -> list[int]:
    """Flatten candidate_turns to a deduped, ordered list of event indices.

    Keeps the top `per_file` strongest events per cited file (preferring
    write/run over mere mentions)."""
    chosen: set[int] = set()
    for path, hits in candidate_turns(events, cited_files).items():
        strong = [h for h in hits if h["strength"] >= RUN] or hits
        for h in strong[:per_file]:
            chosen.add(h["idx"])
    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Training-run launch detection (shared by the viewer's badges, /timing and    #
# the train-run success audit). A launch = a command running a train-ish       #
# script; data-prep/eval tooling excluded. Misses training inside shell        #
# scripts. Detection says nothing about whether the run SUCCEEDED — see        #
# exp_audit_train_runs.py.                                                     #
# --------------------------------------------------------------------------- #
_LAUNCH_RE = re.compile(
    r"(?:python[\d.]*|accelerate\s+launch|torchrun|deepspeed)\s+(?:-\S+\s+)*(\S+\.py)", re.I)
_TRAINY = re.compile(r"train|finetune|fine[-_]tune|sft|dpo|grpo|rlhf", re.I)
_NOT_TRAINY = re.compile(
    r"prepare|prep_|preproc|data|gen|generate|eval|test|check|verify|monitor|"
    r"plot|analy|merge|convert|download|tokeniz|inspect", re.I)


def training_launches(ev: dict) -> int:
    """Number of training-run launches in this event's command(s)."""
    cmds = []
    if ev.get("type") == "codex_item":
        if ev.get("subtype") == "command_execution":
            cmds.append((ev.get("item") or {}).get("command", "") or "")
    else:
        for b in ev.get("blocks") or []:
            if b.get("type") == "tool_use" and isinstance(b.get("input"), dict):
                c = b["input"].get("command")
                if isinstance(c, str):
                    cmds.append(c)
    n = 0
    for cmd in cmds:
        for m in _LAUNCH_RE.finditer(cmd):
            base = m.group(1).rsplit("/", 1)[-1]
            if _TRAINY.search(base) and not _NOT_TRAINY.search(base):
                n += 1
    return n


# --------------------------------------------------------------------------- #
# Markable content — the RAW strings the viewer can <mark> for an event.       #
# Quotes must be substrings of one of these to highlight, so localization      #
# quotes are validated against these (NOT the scaffolded condensed view).      #
# --------------------------------------------------------------------------- #
def markable_pieces(ev: dict) -> list[str]:
    pieces: list[str] = []
    if ev.get("type") == "codex_item":
        item = ev.get("item") or {}
        for k in ("text", "command", "aggregated_output"):
            if isinstance(item.get(k), str) and item[k]:
                pieces.append(item[k])
        for ch in item.get("changes", []) or []:
            if ch.get("path"):
                pieces.append(ch["path"])
        return pieces
    for b in ev.get("blocks") or []:
        bt = b.get("type")
        if bt == "text" and b.get("text"):
            pieces.append(b["text"])
        elif bt == "tool_use":
            inp = b.get("input") if isinstance(b.get("input"), dict) else {}
            for v in inp.values():
                if isinstance(v, str) and v:
                    pieces.append(v)
            pieces.append(json.dumps(inp, indent=2))  # viewer renders args as JSON
        elif bt == "tool_result":
            c = b.get("content")
            if isinstance(c, str):
                pieces.append(c)
            elif isinstance(c, list):
                for sub in c:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        pieces.append(sub["text"])
    return pieces


def full_event(ev: dict, idx: int, clip: int = 3000) -> dict:
    """Raw markable content of an event, for the LLM to quote from verbatim."""
    return {"idx": idx, "kind": (ev.get("subtype") or ev.get("type")),
            "content": _clip("\n".join(markable_pieces(ev)), clip)}


# --------------------------------------------------------------------------- #
# Compact trace view for LLM context                                          #
# --------------------------------------------------------------------------- #
def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"… [+{len(s) - n} chars]"


def condense_event(ev: dict, idx: int, out_clip: int = 600, text_clip: int | None = None) -> dict | None:
    """One compact entry per event, keyed by event index for citation back.

    `out_clip` clips noisy tool OUTPUT; `text_clip` (defaults to out_clip) clips
    text-bearing content — reasoning, agent/assistant prose, and written
    code/commands. For the judge view we pass a large text_clip so reasoning and
    code are seen in full (where the agent might 'plead guilty'), while still
    truncating multi-MB logs."""
    tc = out_clip if text_clip is None else text_clip
    if ev.get("type") == "codex_item":
        item = ev.get("item") or {}
        st = ev.get("subtype")
        if st == "reasoning":
            return {"idx": idx, "kind": "reasoning", "text": _clip(item.get("text", ""), tc)}
        if st == "agent_message":
            return {"idx": idx, "kind": "assistant", "text": _clip(item.get("text", ""), tc)}
        if st == "command_execution":
            return {"idx": idx, "kind": "command", "text": _clip(item.get("command", ""), tc),
                    "output": _clip(item.get("aggregated_output", ""), out_clip)}
        if st == "file_change":
            return {"idx": idx, "kind": "file_change",
                    "text": ", ".join(c.get("path", "") for c in item.get("changes", []))}
        return None

    typ = ev.get("type")
    if typ not in ("assistant", "user"):
        return None
    parts: list[str] = []
    for b in ev.get("blocks") or []:
        bt = b.get("type")
        if bt == "text":
            parts.append(_clip(b.get("text", ""), tc))
        elif bt == "tool_use":
            inp = b.get("input") if isinstance(b.get("input"), dict) else {}
            detail = inp.get("command") or inp.get("content") or inp.get("file_text") \
                or inp.get("new_string") or inp.get("file_path") or json.dumps(inp)[:200]
            parts.append(f"[{b.get('name')}] {_clip(str(detail), tc)}")
        elif bt == "tool_result":
            c = b.get("content")
            txt = c if isinstance(c, str) else json.dumps(c)
            parts.append(f"[result] {_clip(txt, out_clip)}")
    if not parts:
        return None
    return {"idx": idx, "kind": "assistant" if typ == "assistant" else "tool",
            "text": "\n".join(parts)}


def condense_trace(events: list, max_chars: int = 120_000, out_clip: int = 600,
                   text_clip: int | None = None) -> list[dict]:
    """Compact, index-tagged view of the whole trace, capped at max_chars."""
    rows, total = [], 0
    for idx, ev in enumerate(events):
        row = condense_event(ev, idx, out_clip, text_clip)
        if row is None:
            continue
        size = len(json.dumps(row))
        if total + size > max_chars:
            rows.append({"idx": idx, "kind": "note", "text": f"… trace truncated at event {idx} of {len(events)} …"})
            break
        rows.append(row)
        total += size
    return rows


def condense_trace_tail(events: list, max_chars: int = 500_000, out_clip: int = 1500,
                        text_clip: int | None = 8000) -> tuple[list[dict], int]:
    """Like condense_trace but keeps the TAIL of the trace: walk backwards from
    the end until the budget is spent. Returns (rows in chronological order,
    first_included_event_idx). Used to re-check truncated runs whose prefix the
    prior judge already read."""
    rows, total = [], 0
    start_idx = 0
    for idx in range(len(events) - 1, -1, -1):
        row = condense_event(events[idx], idx, out_clip, text_clip)
        if row is None:
            continue
        size = len(json.dumps(row))
        if total + size > max_chars:
            start_idx = idx + 1
            rows.append({"idx": idx, "kind": "note",
                         "text": f"… events 0–{idx} elided (the prior judge read the beginning) …"})
            break
        rows.append(row)
        total += size
    rows.reverse()
    return rows, start_idx


if __name__ == "__main__":  # manual check
    import sys
    import judge_findings
    run_id = sys.argv[1]
    rec = json.load(open(__import__("paths").VIEWER_DATA / f"{run_id}.json"))
    f = judge_findings.parse(run_id)
    print("cited_files:", f.cited_files)
    ct = candidate_turns(rec["events"], f.cited_files)
    for path, hits in ct.items():
        print(f"  {path:42} -> {hits[:4]}")
    print("best_marked_turns:", best_marked_turns(rec["events"], f.cited_files))
