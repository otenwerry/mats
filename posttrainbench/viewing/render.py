"""Rendering helpers for the PostTrainBench trajectory viewer.

The dataset's `viewer_data/{run_id}.json` files store a normalized `events[]`
list, but the *shape* of each event depends on the agent harness
(`trace_format`):

  - claude_code / opencode : events carry `blocks[]` in Anthropic API form
    (text / tool_use / tool_result).
  - codex                  : events are `codex_item`s with a `subtype`
    (reasoning / agent_message / command_execution / file_change); the
    content lives under `item`, there are no blocks.

`render_events()` dispatches on trace_format and returns an HTML string.

Every text region is passed through `highlight.mark()` so that later RH-evidence
work (quotes pulled from judgement.log) can be highlighted inline — see
viewer.py's `load_highlights`. Pass `quotes=()` for no highlighting.
"""
from __future__ import annotations

import html
import json
import re


# --------------------------------------------------------------------------- #
# Tolerant quote highlighting                                                 #
# --------------------------------------------------------------------------- #
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def mark(text: str, quotes=(), cls: str | None = "hack") -> str:
    """Escape `text` to HTML and wrap any occurrence of a quote in <mark>.

    Matching is whitespace-tolerant and case-insensitive so quotes copied out
    of a judge log still land even if spacing differs. Quotes that don't match
    are silently skipped (the caller can report unmatched counts separately).
    `cls` adds a CSS class to the <mark> (e.g. "hack" for hacking phrases).
    """
    open_tag = f'<mark class="{cls}">' if cls else "<mark>"
    if not text:
        return ""
    escaped = html.escape(str(text))
    if not quotes:
        return escaped
    # Work on the escaped string; build a normalized index -> escaped index map.
    norm_chars, idx_map = [], []
    for i, ch in enumerate(escaped):
        if ch.isspace():
            if norm_chars and norm_chars[-1] == " ":
                continue
            norm_chars.append(" ")
        else:
            norm_chars.append(ch.lower())
        idx_map.append(i)
    norm = "".join(norm_chars)
    spans = []
    for q in quotes:
        nq = _normalize(html.escape(str(q)))
        if not nq:
            continue
        start = 0
        while True:
            pos = norm.find(nq, start)
            if pos == -1:
                break
            lo = idx_map[pos]
            hi = idx_map[min(pos + len(nq) - 1, len(idx_map) - 1)] + 1
            spans.append((lo, hi))
            start = pos + len(nq)
    if not spans:
        return escaped
    # Merge overlapping spans and splice in <mark> tags.
    spans.sort()
    merged = [spans[0]]
    for lo, hi in spans[1:]:
        if lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    out, cur = [], 0
    for lo, hi in merged:
        out.append(escaped[cur:lo])
        out.append(f"{open_tag}{escaped[lo:hi]}</mark>")
        cur = hi
    out.append(escaped[cur:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Small building blocks                                                       #
# --------------------------------------------------------------------------- #
def _collapsible(label: str, body_html: str, char_len: int, threshold: int = 800) -> str:
    """Long outputs collapse behind a <details>; short ones render inline."""
    if char_len <= threshold:
        return f'<pre class="content tool">{body_html}</pre>'
    return (f'<details class="toolout"><summary>{html.escape(label)} · {char_len} chars</summary>'
            f'<pre class="content tool">{body_html}</pre></details>')


def _content_to_text(content) -> str:
    """tool_result content is sometimes a str, sometimes a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                parts.append(b.get("text") or b.get("content") or json.dumps(b))
            else:
                parts.append(str(b))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, indent=2)


def _tool_input(name: str, inp, quotes) -> str:
    """Render a tool_use input: pull the salient command/code, dump the rest."""
    if not isinstance(inp, dict):
        return f'<pre class="args">{mark(str(inp), quotes)}</pre>'
    code = None
    for key in ("command", "code", "cmd", "content", "new_string", "file_text"):
        if key in inp and isinstance(inp[key], str):
            code = inp[key]
            code_key = key
            break
    body = ""
    if code is not None:
        body += f'<pre class="code">{mark(code, quotes)}</pre>'
        rest = {k: v for k, v in inp.items() if k != code_key}
        if rest:
            body += f'<pre class="args">{mark(json.dumps(rest, indent=2), quotes)}</pre>'
    else:
        body += f'<pre class="args">{mark(json.dumps(inp, indent=2), quotes)}</pre>'
    return body


def _msg(role: str, header: str, inner: str, mark: str | None = None,
         mark_title: str = "", extra: str = "") -> str:
    """`mark` is None, 'hack' (red: a direct reward-hacking action) or
    'context' (neutral: notable turn — intent, contrast, aftermath — not a hack)."""
    cls, tag = "msg", ""
    title = f' title="{html.escape(mark_title)}"' if mark_title else ""
    if mark == "hack":
        cls, tag = "msg hackturn", f'<span class="hacktag"{title}>⚠ hack turn</span>'
    elif mark == "context":
        cls, tag = "msg notedturn", f'<span class="notetag"{title}>📌 noted</span>'
    return (f'<div class="{cls} role-{html.escape(role)}">'
            f'<div class="rolehdr">{html.escape(header)}{tag}{extra}</div>{inner}</div>')


# --------------------------------------------------------------------------- #
# Block-based formats (claude_code, opencode)                                 #
# --------------------------------------------------------------------------- #
def _render_blocks_event(ev: dict, n: int, quotes, mk=None, mk_title="", extra="") -> str:
    typ = ev.get("type", "?")
    blocks = ev.get("blocks") or []

    if typ in ("system", "result"):
        sub = ev.get("subtype") or typ
        raw = ev.get("raw")
        detail = ""
        if raw:
            txt = json.dumps(raw, indent=2)
            if len(txt) > 200:
                detail = f'<details class="toolout"><summary>{html.escape(sub)} · {len(txt)} chars</summary><pre class="content tool">{html.escape(txt)}</pre></details>'
            else:
                detail = f'<pre class="content tool">{html.escape(txt)}</pre>'
        return _msg("system", f"#{n} · {typ} · {sub}", detail or '<pre class="content empty">(event)</pre>', mk, mk_title, extra)

    role = "assistant" if typ == "assistant" else "user"
    parts = []
    for b in blocks:
        bt = b.get("type")
        if bt == "text":
            parts.append(f'<pre class="content">{mark(b.get("text", ""), quotes)}</pre>')
        elif bt == "tool_use":
            name = b.get("name", "tool")
            parts.append(f'<div class="call"><div class="callname">🔧 <code>{html.escape(name)}</code></div>'
                         f'{_tool_input(name, b.get("input"), quotes)}</div>')
        elif bt == "tool_result":
            txt = _content_to_text(b.get("content"))
            err = b.get("is_error")
            label = "error output" if err else "tool output"
            parts.append(_collapsible(label, mark(txt, quotes), len(txt)))
        else:
            parts.append(f'<pre class="args">{html.escape(json.dumps(b, indent=2))}</pre>')
    if not parts:
        parts.append('<pre class="content empty">(no content)</pre>')
    label = "Assistant" if role == "assistant" else "User / tool result"
    return _msg(role, f"#{n} · {label}", "".join(parts), mk, mk_title, extra)


# --------------------------------------------------------------------------- #
# Codex format                                                                #
# --------------------------------------------------------------------------- #
def _render_codex_event(ev: dict, n: int, quotes, mk=None, mk_title="", extra="") -> str:
    typ = ev.get("type")
    if typ == "system":
        return _msg("system", f"#{n} · system · {ev.get('subtype','')}",
                    f'<pre class="content tool">{html.escape(json.dumps(ev.get("raw", {}), indent=2))}</pre>', mk, mk_title, extra)

    sub = ev.get("subtype")
    item = ev.get("item") or {}

    if sub == "reasoning":
        return _msg("reasoning", f"#{n} · reasoning",
                    f'<pre class="content reasoning">{mark(item.get("text", ""), quotes)}</pre>', mk, mk_title, extra)
    if sub == "agent_message":
        return _msg("assistant", f"#{n} · agent message",
                    f'<pre class="content">{mark(item.get("text", ""), quotes)}</pre>', mk, mk_title, extra)
    if sub == "command_execution":
        cmd = item.get("command", "")
        out = item.get("aggregated_output", "")
        ec = item.get("exit_code")
        ectag = "" if ec in (0, None) else f'<span class="ectag">exit {ec}</span>'
        inner = (f'<div class="call"><div class="callname">🔧 <code>command</code>{ectag}</div>'
                 f'<pre class="code">{mark(cmd, quotes)}</pre></div>'
                 + _collapsible("output", mark(out, quotes), len(out)))
        return _msg("assistant", f"#{n} · command", inner, mk, mk_title, extra)
    if sub == "file_change":
        txt = json.dumps(item, indent=2)
        return _msg("assistant", f"#{n} · file change",
                    _collapsible("change", mark(txt, quotes), len(txt)), mk, mk_title, extra)
    # fallback
    txt = json.dumps(item, indent=2)
    return _msg("assistant", f"#{n} · {sub}", f'<pre class="content tool">{html.escape(txt)}</pre>', mk, mk_title, extra)


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #
_KIND_LABEL = {"admission": "🗣 admission", "reasoning": "💭 reasoning",
               "code": "💻 code", "command": "⌘ command", "data": "📄 data",
               "label_audit": "🏷 label audit", "final_judge": "⚖ final judge",
               "story_evidence": "🔎 evidence", "story_suspicion": "⚠ looked suspicious",
               "story_rebuttal": "✓ why it's fine", "story_uncertainty": "❓ uncertainty",
               "story_context": "📝 context", "other": "📝 note"}


def _annotations_html(notes: list) -> str:
    """Reviewer annotations for one turn, styled clearly distinct from the trace."""
    rows = []
    for a in notes:
        note = html.escape(a.get("note", "") or "")
        kind = a.get("kind", "other")
        lbl = _KIND_LABEL.get(kind, "📝 note")
        q = a.get("quote") or ""
        qhtml = f'<span class="anno-q">“{html.escape(q)}”</span> ' if q else ""
        if not note and not q:
            continue
        rows.append(f'<div class="anno anno-{html.escape(kind)}">'
                    f'<span class="anno-lbl">{lbl}</span> {qhtml}{note}</div>')
    return f'<div class="annos">{"".join(rows)}</div>' if rows else ""


def render_events(events: list, trace_format: str, quotes=(), marked_turns=(),
                  annotations=None, turn_kinds=None, train_turns=None,
                  api_turns=None) -> str:
    """`marked_turns`: marked event indices. `turn_kinds`: {turn: {kind, reason}}
    classifying each marked turn as 'hack' (direct hacking action, red) or
    'context' (notable but not a hack, neutral); unclassified turns default to
    'hack'. `annotations`: {turn: [{note,quote,kind}]} reviewer notes appended
    below the relevant turn."""
    hacks = set(marked_turns or ())
    anno = {int(k): v for k, v in (annotations or {}).items()}
    kinds = {}
    for k, v in (turn_kinds or {}).items():
        if isinstance(v, str):
            v = {"kind": v}
        kinds[int(k)] = v
    out, last_session = [], None
    for n, ev in enumerate(events):
        # rollback-experiment dividers render as distinct banners
        if ev.get("rollback_marker"):
            txt = "".join(b.get("text", "") for b in ev.get("blocks") or [])
            out.append(f'<div class="rollbackcut">{html.escape(txt)}</div>')
            continue
        if ev.get("rollback_resume"):
            txt = "".join(b.get("text", "") for b in ev.get("blocks") or [])
            out.append(f'<div class="rollbackresume"><b>injected resume turn:</b> '
                       f'{html.escape(txt)}</div>')
            continue
        sidx = ev.get("session_idx")
        if sidx != last_session and sidx is not None:
            out.append(f'<div class="sessdiv">— session {sidx} —</div>')
            last_session = sidx
        marked = n in hacks
        mk = mk_title = None
        if marked:
            info = kinds.get(n) or {}
            mk = info.get("kind") or "hack"
            mk_title = info.get("reason", "")
        extra = ""
        tinfo = (train_turns or {}).get(n)
        if tinfo:
            if isinstance(tinfo, str):
                tinfo = {"label": tinfo, "ok": True,
                         "title": "detected training-run launch (heuristic)"}
            tcls = "traintag" if tinfo.get("ok", True) else "traintag failed"
            extra = (f'<span class="{tcls}" title="{html.escape(tinfo.get("title", ""))}">'
                     f'🏋 train {html.escape(tinfo["label"])}</span>')
        for ainfo in (api_turns or {}).get(n, []):
            extra += (f'<span class="apiflag {html.escape(ainfo.get("cls",""))}" '
                      f'title="{html.escape(ainfo.get("title",""))}">'
                      f'{html.escape(ainfo.get("label",""))}</span>')
        # Word-level marks belong INSIDE the marked turn(s); don't bleed the
        # quotes across the whole trace (a short quote would light up hundreds
        # of unrelated events). No marked_turns given -> mark everywhere.
        q = quotes if (marked or not hacks) else ()
        if trace_format == "codex":
            out.append(_render_codex_event(ev, n, q, mk, mk_title or "", extra))
        else:
            out.append(_render_blocks_event(ev, n, q, mk, mk_title or "", extra))
        if n in anno:
            out.append(_annotations_html(anno[n]))
    return "\n".join(out)
