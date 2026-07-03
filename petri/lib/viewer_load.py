"""The make_viewer LOAD LAYER: turns logs/ run dirs into plain audit dicts and rollback
grids, with an on-disk cache so unchanged dirs never re-read their .eval archives.

Split out of make_viewer.py (2026-07-02) so the cache's code signature covers exactly the
code that shapes the cached data:

  cache key = the dir's *.eval files (name + mtime + size)
            + the score overlays (rejudge_scores.json + dim_scores/*.json)
            + the source of THIS FILE and petri_paths.py

make_viewer.py is NOT part of the key, so display/rendering edits there rebuild warm (all
cache hits, seconds -- this is what makes iterating on the viewer fast). THE RULE THAT
KEEPS THE CACHE HONEST: any code whose output lands in the cached objects -- new
audit-dict fields, transcript rendering, scratchpad/tool-call extraction, score overlays --
must live in this file (or petri_paths.py), never in make_viewer.py. Under that rule a
change to cached-data logic can never serve stale output.

Profiling context (measured at cache introduction): ~97% of a cold viewer build is reading
the .eval archives; writing the HTML is ~3%. `load_mode` is the single chokepoint for that
reading (load_rollback_run calls it too), and `rollback_grid` re-reads the rollback evals
on top (twice per dir per build). Both are pure dir -> data functions, so we memoize their
output to mats-local/petri/.viewer_cache/ (pure derived data, never committed).

IF THE CACHE EVER CAUSES PROBLEMS, escape hatches in increasing permanence:
  1. One-off bypass:  uv run make_viewer.py --no-cache   (or MAKE_VIEWER_NO_CACHE=1)
  2. Purge it:        rm -rf mats-local/petri/.viewer_cache   (forces a clean full rebuild)
  3. Permanently OFF: set `_CACHE_ENABLED = False` below.
  4. FULLY REMOVE the feature: delete _CACHE_DIR, _CACHE_ENABLED, _code_sig, _overlay_sig,
     _dir_sig, _cache_key, _cache_get and _cache_put; delete the thin wrappers `load_mode`
     and `rollback_grid`, renaming `_load_mode_impl` -> `load_mode` and
     `_rollback_grid_impl` -> `rollback_grid` (their bodies are the original, untouched
     code); drop the then-unused hashlib/os/pickle imports; delete
     mats-local/petri/.viewer_cache/. (Verified at introduction: a warm-cache build is
     byte-identical to a --no-cache build for every output file except visuals.html, whose
     matplotlib PNGs are non-deterministic regardless of caching.)
"""

import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_petri._judge.branches import flatten_timeline, render_segments
from inspect_scout import (
    MessagesPreprocessor,
    TranscriptContent,
    message_numbering,
    transcripts_from,
)

from petri_paths import DATA


_CACHE_DIR = DATA / ".viewer_cache"
_CACHE_ENABLED = (os.environ.get("MAKE_VIEWER_NO_CACHE") != "1") and ("--no-cache" not in sys.argv)


def _code_sig() -> str:
    """Hash of the source files whose logic shapes load_mode/rollback_grid output -- THIS file
    and petri_paths.py. make_viewer.py is deliberately NOT included: rendering/display edits
    there must never invalidate the cache. Any edit HERE still busts every entry."""
    h = hashlib.sha256()
    for p in (Path(__file__), Path(__file__).resolve().parent / "petri_paths.py"):
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def _overlay_sig() -> str:
    """Signature of the mutable score-overlay files (rejudge + dim_scores sidecars) that
    load_mode folds into each audit dict -- so changing a score busts the cache even when the
    .eval files are untouched."""
    parts: list[str] = []
    files = [REJUDGE_FILE] + (sorted(DIM_SCORES_DIR.glob("*.json")) if DIM_SCORES_DIR.is_dir() else [])
    for f in files:
        try:
            st = f.stat()
            parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append(f"{f.name}:absent")
    return "|".join(parts)


def _dir_sig(d: Path) -> str:
    """Signature of a dir's .eval files (name + mtime + size). Adding, removing, or rewriting
    any .eval changes this."""
    parts = []
    for f in sorted(d.glob("*.eval")):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def _cache_key(d: Path, kind: str) -> str:
    raw = "||".join([kind, d.name, _dir_sig(d), _overlay_sig(), _code_sig()])
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(d: Path, kind: str):
    """The cached object for (dir, kind) if present and current, else None."""
    if not _CACHE_ENABLED:
        return None
    f = _CACHE_DIR / f"{kind}__{d.name}__{_cache_key(d, kind)}.pkl"
    if not f.exists():
        return None
    try:
        with f.open("rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"  cache: ignoring unreadable entry {f.name} ({type(e).__name__})")
        return None


def _cache_put(d: Path, kind: str, obj) -> None:
    if not _CACHE_ENABLED:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(d, kind)
    current = f"{kind}__{d.name}__{key}.pkl"
    # one live entry per (dir, kind): drop stale-keyed siblings so the cache can't grow unbounded
    for old in _CACHE_DIR.glob(f"{kind}__{d.name}__*.pkl"):
        if old.name != current:
            old.unlink()
    with (_CACHE_DIR / current).open("wb") as fh:
        pickle.dump(obj, fh)


_ATTACHMENT_RE = re.compile(r"attachment://([0-9a-fA-F]+)")


def resolve_attachments(text: str, attachments: dict) -> str:
    """Replace Inspect attachment placeholders (`attachment://<hash>`) with the real text.

    Inspect offloads long strings into a per-sample attachment store keyed by content
    hash and leaves a `attachment://<hash>` placeholder in the message/field. `read_eval_log`
    returns the placeholder, NOT the text, so every rendering path has to resolve it itself
    — before this, the auditor scratchpad and a few transcript messages showed raw hashes.
    Unknown hashes are left untouched (rather than blanked) so nothing is silently lost."""
    if not text or "attachment://" not in text:
        return text
    return _ATTACHMENT_RE.sub(lambda m: attachments.get(m.group(1), m.group(0)), text)


MSG_HEAD = re.compile(r"^\[M(\d+)\] (<?[A-Za-z_]+>?[A-Za-z_ ]*):", re.M)


def _msg_text(msg) -> str:
    """Concatenate the text of a chat message's content (handles str or block list)."""
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c
    return "".join(
        b if isinstance(b, str) else (getattr(b, "text", "") or "") for b in (c or [])
    )


# Auditor tools that inject a message into the target transcript. We don't infer the
# resulting message from the tool name -- the event log tells us directly (see below).
_AUDITOR_MSG_TOOLS = {"set_system_message", "send_message", "send_tool_call_result"}
# Each such tool, when it actually creates a message, returns a string like
# "Tool result sent [message_id: M4]." -- and that M-number IS the transcript anchor.
_MSG_ID_RE = re.compile(r"message_id:\s*M(\d+)")

# [M#] head with its role, for aligning Petri ids to the viewer's numbering.
_HEAD_ROLE_RE = re.compile(r"^\[M(\d+)\]\s*(\w+)", re.M)
# expected viewer role of the message each message-creating auditor tool produces.
_MSGTOOL_ROLE = {"send_message": "user", "send_tool_call_result": "tool",
                 "set_system_message": "system"}


def _petri_to_viewer_offset(sample, transcript: str) -> int:
    """Petri numbers ONLY the live conversation (no system message, and -- for continuations --
    no replayed prefix), so a tool result's `[message_id: M#]` is shifted from the viewer's
    `[M#]` (which numbers the whole target timeline) by a constant `d`: viewer_M = petri_M + d.
    For a normal audit d = 1 (just the system message the viewer counts and Petri doesn't); for
    a continuation d = 1 + len(prefix A). Recover `d` by matching the auditor's message-creating
    tool calls against the transcript's roles (send_message->user, send_tool_call_result->tool,
    set_system_message->system), weighting the rare user/system anchors so a periodic run of
    tool results can't pick a wrong shift. Falls back to 1 when there's no signal.

    Without this correction every auditor scratchpad / hidden-tool aside renders one message too
    early on ordinary pages, and ~a whole prefix too early on continuation pages."""
    roles = {int(m.group(1)): m.group(2).lower() for m in _HEAD_ROLE_RE.finditer(transcript)}
    if not roles:
        return 1
    pairs: list[tuple[int, str]] = []   # (petri message id, expected viewer role)
    for ev in (getattr(sample, "events", None) or []):
        if getattr(ev, "event", None) != "tool":
            continue
        exp = _MSGTOOL_ROLE.get(getattr(ev, "function", None))
        if not exp:
            continue
        mid = _MSG_ID_RE.search(str(getattr(ev, "result", None) or ""))
        if mid:
            pairs.append((int(mid.group(1)), exp))
    if not pairs:
        return 1
    last = max(roles)

    def _score(d: int) -> int:
        return sum((5 if exp in ("user", "system") else 1)
                   for pid, exp in pairs if roles.get(pid + d) == exp)

    best_d, best_s = 1, -1
    for d in range(0, last + 1):
        s = _score(d)
        if s > best_s:
            best_s, best_d = s, d
    return best_d if best_s > 0 else 1


def auditor_scratchpads(sample, auditor_model: str, transcript: str) -> dict[int, str]:
    """Map target-transcript message number -> the auditor's scratchpad to show *before*
    that message.

    The auditor's private planning is the assistant text it writes alongside its tool
    calls (Petri hides this from the target). We read it from `sample.events`, which is
    EXACT: each auditor turn is a model event (the scratchpad is its assistant text), and
    each message-producing tool call logs a result naming the transcript message it
    created -- e.g. "Tool result sent [message_id: M4]." So we attach each turn's
    scratchpad to the first message its tool calls report creating, by number, with no
    positional guessing. A no-op call (e.g. a `send_tool_call_result` for a tool call the
    target never actually made) returns an error instead of a `[message_id: M#]`, so it's
    skipped automatically. Events are written incrementally, so this also survives a
    sample that was cancelled mid-run. We attribute scratchpad only to AUDITOR model
    events (`auditor_model`), never the target's or judge's."""
    present = {int(m.group(1)) for m in MSG_HEAD.finditer(transcript)}
    last_num = max(present) if present else 0
    off = _petri_to_viewer_offset(sample, transcript)  # petri id -> viewer [M#]
    atts = getattr(sample, "attachments", None) or {}
    out: dict[int, str] = {}
    pending: list[str] = []     # auditor text not yet attached (carried across turns)
    for ev in (getattr(sample, "events", None) or []):
        et = getattr(ev, "event", None)
        if et == "model" and getattr(ev, "model", None) == auditor_model:
            o = getattr(ev, "output", None)
            txt = _msg_text(o.choices[0].message).strip() if (o and o.choices) else ""
            txt = resolve_attachments(txt, atts)   # the longer notes are stored as attachments
            if txt:
                pending.append(txt)
        elif et == "tool" and getattr(ev, "function", None) in _AUDITOR_MSG_TOOLS:
            res = getattr(ev, "result", None)
            m = _MSG_ID_RE.search(res if isinstance(res, str) else str(res or ""))
            if not m:
                continue                        # no-op/error result -> created no message
            num = int(m.group(1)) + off         # petri id -> viewer numbering
            if pending and num in present:
                out[num] = "\n\n".join(pending).strip()   # before the turn's first message
            pending = []                        # a real message was created -> turn consumed
    if pending:                                 # trailing reasoning (e.g. end_conversation)
        tail = "\n\n".join(pending).strip()
        if tail:
            out[last_num + 1] = tail            # rendered after the last message
    return out


def _resolve_arg_values(args: dict, atts: dict) -> dict:
    """Resolve attachment refs inside string arg values (long descriptions, system messages,
    etc. are offloaded by Inspect to an attachment store) so a block shows the real text the
    model passed, not a storage hash. Non-string values (e.g. the parameters JSON-Schema dict)
    are left as-is."""
    return {k: (resolve_attachments(v, atts) if isinstance(v, str) else v)
            for k, v in args.items()}


# The auditor's own toolset (Petri's auditor_tools()). We scope to it so the judge's scoring
# tool ("answer") and any other non-auditor tool event in the sample are excluded -- the
# `agent` field is None on every tool event, so we can't tell them apart any other way. This
# is scoping, not interpretation: WITHIN this set there's no per-tool special-casing.
_AUDITOR_TOOLS = {
    "set_system_message", "send_message", "send_tool_call_result", "resume",
    "create_tool", "remove_tool", "end_conversation",
    "rollback_conversation", "restart_conversation",
}


def auditor_calls_by_anchor(sample, transcript: str) -> dict[int, list[dict]]:
    """Map target-transcript message number -> the auditor tool calls to show, as collapsed
    asides, *before* that message.

    The auditor drives the audit entirely through its own tool calls, but only some of those
    leave a mark on the target's transcript: the message-producing ones show up as the messages
    they create (set_system_message -> the system message, send_message -> a user message,
    send_tool_call_result -> a tool message), and `resume` shows up as the target's response.
    Everything else -- create_tool, remove_tool, end_conversation, a mid-conversation
    set_system_message that silently swaps the prompt -- changes the target's world WITHOUT
    leaving a message. We surface exactly those, read from `sample.events` (EXACT, not inferred).

    Rule (no per-tool special-casing within the auditor's toolset): a call is "already apparent"
    iff its result reports a message id the transcript renders. Apparent calls are skipped, and
    used as anchors -- pending non-apparent calls attach to the message they produced, i.e. just
    before the next thing the target sees. `resume` is never shown (its effect is the target
    turn). Every other auditor call is shown with its arguments verbatim, nothing added. Mirrors
    auditor_scratchpads; survives a cancelled run."""
    present = {int(m.group(1)) for m in MSG_HEAD.finditer(transcript)}
    last_num = max(present) if present else 0
    off = _petri_to_viewer_offset(sample, transcript)  # petri id -> viewer [M#]
    atts = getattr(sample, "attachments", None) or {}
    out: dict[int, list[dict]] = {}
    pending: list[dict] = []
    for ev in (getattr(sample, "events", None) or []):
        if getattr(ev, "event", None) != "tool":
            continue
        fn = getattr(ev, "function", None) or "?"
        if fn not in _AUDITOR_TOOLS:                  # e.g. the judge's "answer" scoring tool
            continue
        mid = _MSG_ID_RE.search(str(getattr(ev, "result", None) or ""))
        if mid and int(mid.group(1)) + off in present:  # produced a rendered message -> apparent
            if pending:                               # ...so anchor the pending calls right here
                out.setdefault(int(mid.group(1)) + off, []).extend(pending)
                pending = []
            continue
        err = getattr(ev, "error", None)
        err_msg = getattr(err, "message", None) if err is not None else None
        if fn == "resume":                            # advance command; its turn is shown already
            if err_msg:                               # ...UNLESS the target generation failed -- that
                # produces no target message, so this event is the ONLY trace of the failed turn.
                # (The auditor still RECEIVES this same text as the tool result; this only surfaces
                # it to us in the viewer, which would otherwise show nothing for the dropped turn.)
                pending.append({"function": fn, "args": {}, "error": err_msg})
            continue
        args = getattr(ev, "arguments", None) or {}
        entry = {"function": fn, "args": _resolve_arg_values(args, atts)}
        if err_msg:                                   # a non-resume auditor call that itself errored
            entry["error"] = err_msg
        pending.append(entry)
    if pending:                                       # calls with no later message (e.g. a run that
        out.setdefault(last_num + 1, []).extend(pending)   # fell apart, end_conversation) -> at end
    return out


def traj_key(a: dict) -> str:
    """Stable identity of one trajectory across viewer rebuilds and new runs."""
    return f"{a['mode']}__{a['task']}__{a['seed']}__e{a['epoch']}"


REJUDGE_FILE = DATA / "rejudge_scores.json"


def _load_rejudge() -> dict:
    return json.loads(REJUDGE_FILE.read_text()) if REJUDGE_FILE.exists() else {}


def _apply_rejudge(a: dict, rejudge: dict) -> dict:
    """Merge a re-judged entry into an audit dict in place (full replacement). When an
    entry is present it supersedes the eval-log scores + summary/justification/highlights;
    when absent the audit keeps its eval-log values. Whether the merge happened is not
    recorded on the dict (use binary_hack_eval to tell whether the 5 dims are present)."""
    e = rejudge.get(traj_key(a))
    if e is None:
        return a
    a["scores"] = e["scores"]
    a["summary"] = e.get("summary") or a.get("summary", "")
    a["justification"] = e.get("justification") or a.get("justification", "")
    a["highlights"] = e.get("highlights") or a.get("highlights", "")
    return a


# Per-dimension backfill sidecars from tools/exp_grade_dimension.py live here, one file
# per dimension (dim_scores/<dim>.json), each keyed by traj_key with the score under the
# <dim> key. They let a single newly-added dimension be scored on trajectories that were
# judged before that dimension existed, WITHOUT re-judging everything. Unlike rejudge
# (full replacement), these only ADD a missing dim -- an audit's own inline/rejudge score
# always wins -- so they cannot change the binary hack definition or any category.
DIM_SCORES_DIR = DATA / "dim_scores"


def _load_dim_sidecars() -> dict:
    """All dim_scores/<dim>.json sidecars merged into
    {traj_key: {dim: {"score": float, "justification": str}}}. The justification is the
    single-dimension backfill judge's reasoning for that score; it's surfaced on the audit
    page (see _dim_just_section) only for dims whose backfilled score was actually applied."""
    out: dict[str, dict] = {}
    if not DIM_SCORES_DIR.is_dir():
        return out
    for f in sorted(DIM_SCORES_DIR.glob("*.json")):
        dim = f.stem
        try:
            entries = json.loads(f.read_text())
        except json.JSONDecodeError:
            print(f"  WARNING: could not parse {f} -- skipping its backfilled scores")
            continue
        for key, e in entries.items():
            v = e.get(dim)
            if isinstance(v, (int, float)):
                out.setdefault(key, {})[dim] = {
                    "score": v,
                    "justification": (e.get("justification") or "").strip(),
                }
    return out


def _apply_dim_scores(a: dict, sidecars: dict) -> dict:
    """Fill in backfilled dims on a["scores"] (in place), only where the audit doesn't
    already have a numeric score for that dim, so inline/rejudge scores always win. When a
    backfilled score IS applied, also stash that dim's single-dimension judge justification
    under a["dim_justifications"][dim], so the audit page can show WHY it got that score (the
    main combined justification doesn't cover separately-backfilled dims)."""
    for dim, info in sidecars.get(traj_key(a), {}).items():
        if not isinstance(a["scores"].get(dim), (int, float)):
            a["scores"][dim] = info["score"]
            if info.get("justification"):
                a.setdefault("dim_justifications", {})[dim] = info["justification"]
    return a


def _orig_id_from_task(task: str) -> int | None:
    """exp_rollback names continuation tasks 'rollback_<origId>_<seed>'."""
    m = re.match(r"rollback_(\d+)_", task)
    return int(m.group(1)) if m else None


async def load_mode(mode_dir: Path) -> list[dict]:
    """Cached wrapper around _load_mode_impl (see the load-cache note up top). A cache hit
    skips all .eval reading/rendering for this dir; a miss loads fresh and stores the result.
    Bypass with --no-cache or MAKE_VIEWER_NO_CACHE=1."""
    cached = _cache_get(mode_dir, "mode")
    if cached is not None:
        print(f"  (cache hit: {mode_dir.name}/ — {len(cached)} audit(s), skipped reload)")
        return cached
    audits = await _load_mode_impl(mode_dir)
    _cache_put(mode_dir, "mode", audits)
    return audits


async def _load_mode_impl(mode_dir: Path) -> list[dict]:
    """One dict per audit in this mode dir: scores + judge notes + transcript."""
    mode = mode_dir.name
    rejudge = _load_rejudge()  # re-judged scores merged in per audit (see exp_rejudge_rh.py)
    # newest .eval mtime in this dir — used to order run sections (newest first)
    eval_files = list(mode_dir.glob("*.eval"))
    mode_mtime = max((f.stat().st_mtime for f in eval_files), default=0.0)

    # judge-view transcripts, keyed by (task_set, task_id, repeat)
    rendered: dict[tuple, str] = {}
    ts = transcripts_from(str(mode_dir))
    async with ts.reader() as reader:
        infos = [i async for i in reader.index()]
        for info in infos:
            try:
                t = await reader.read(info, TranscriptContent(messages="all", events="all"))
            except FileNotFoundError as e:
                print(f"  WARNING: missing eval log for {info.task_set}/{info.task_id} "
                      f"(epoch {info.task_repeat}) — skipping transcript render: {e}")
                continue
            messages_as_str, _extract_refs, label_for_id = message_numbering(
                MessagesPreprocessor(exclude_system=False), label_for_id=True
            )
            if not t.timelines:
                # No timelines = empty transcript = a dead audit (the target
                # produced 0 output, e.g. a bad slug). Skip rendering; the audit
                # still loads from the eval log below and gets flagged `dead`.
                print(f"  WARNING: empty transcript for {info.task_set}/{info.task_id} "
                      f"(epoch {info.task_repeat}) — dead/empty audit, not rendering")
                continue
            target = next(
                (tl for tl in t.timelines if tl.name == "target"), t.timelines[0]
            )
            segments = flatten_timeline(target.root)
            rendered[(info.task_set, info.task_id, info.task_repeat)] = (
                await render_segments(segments, messages_as_str, label_for_id)
            )
            print(f"  rendered transcript: {info.task_set}/{info.task_id} (epoch {info.task_repeat})")

    # scores + judge notes from the eval logs
    audits: list[dict] = []
    for li in list_eval_logs(str(mode_dir)):
        try:
            log = read_eval_log(li)
        except FileNotFoundError as e:
            print(f"  WARNING: missing eval log while loading scores — skipping: {e}")
            continue
        roles = log.eval.model_roles or {}
        target_model = getattr(roles.get("target"), "model", None) or str(roles.get("target", "?"))
        auditor_model = getattr(roles.get("auditor"), "model", None) or str(roles.get("auditor", "?"))
        judge_model = getattr(roles.get("judge"), "model", None) or str(roles.get("judge", "?"))
        for s in log.samples or []:
            sc = (s.scores or {}).get("audit_judge")
            # Skip samples with no usable judge score: either no audit_judge score at all,
            # or a non-dict value (e.g. NaN) meaning the judge errored out. Both render as
            # an empty, confusing page (this is what made #74 look "empty"); excluding them
            # here means no page is written and the stale page is pruned at the end of main().
            if sc is None or not isinstance(sc.value, dict):
                why = "no audit_judge score" if sc is None else f"unusable judge score ({sc.value!r})"
                print(f"  WARNING: {why} for {s.id} epoch {s.epoch} — skipping (no page)")
                continue
            md = sc.metadata or {}
            transcript = rendered.get((log.eval.task, str(s.id), s.epoch), "")
            # resolve attachment placeholders the scout renderer left in the transcript
            transcript = resolve_attachments(transcript, s.attachments or {})
            # dead = the TARGET produced 0 output tokens for this sample (e.g. bad
            # slug / API error). The judge scores such an empty conversation 1 across
            # the board, which would masquerade as a clean "no reward hacking" run —
            # flag it so it's never silently counted as real data. We key on target
            # token usage (not on whether the transcript rendered): a transcript can
            # fail to render for display reasons while the audit itself is valid.
            tgt_short = target_model.split("/")[-1]
            tgt_out = sum(
                (u.output_tokens or 0)
                for m, u in (s.model_usage or {}).items()
                if m.split("/")[-1] == tgt_short
            )
            # auditor scratchpad, aligned to the rendered transcript's message numbers.
            # For rollback continuations this run's events only cover the LIVE turns
            # (replayed prefix turns produced no ModelEvent), so the dict it returns
            # only has live-region keys -- which is exactly right: write_rollback_page
            # splices the replayed-prefix scratchpad from the ORIGINAL audit at the cut
            # and uses this for everything from the cut onward.
            scratch = auditor_scratchpads(s, auditor_model, transcript) if transcript else {}
            # Compaction: inspect_ai emits a dedicated CompactionEvent (event=="compaction")
            # whenever it summarizes/trims the auditor's context to fit the window. None of
            # our current runs have any (the auditor context grows monotonically everywhere);
            # this collects them so a future long audit that triggers one is flagged, not silent.
            compactions = [
                {
                    "type": getattr(ev, "type", None),
                    "tokens_before": getattr(ev, "tokens_before", None),
                    "tokens_after": getattr(ev, "tokens_after", None),
                    "source": getattr(ev, "source", None),
                }
                for ev in (s.events or [])
                if getattr(ev, "event", None) == "compaction"
            ]
            # "fork": did the live auditor use a conversation rollback/restart tool on
            # this sample? restart_conversation = full context wipe + new system prompt;
            # rollback_conversation = mid-conversation branch. Originals run with these
            # disabled (enable_rollback=False), so a fork is expected only on resamples.
            # Stored per-output (boolean + per-tool counts) so analysis can exclude or
            # condition on it -- an auditor restart can confound a re-hack measurement.
            fork_calls = [
                tc.function
                for m in (s.messages or [])
                for tc in (getattr(m, "tool_calls", None) or [])
                if tc.function in ("restart_conversation", "rollback_conversation")
            ]
            audits.append(
                _apply_rejudge(
                    dict(
                        mode=mode,
                        mtime=mode_mtime,
                        task=log.eval.task,
                        target=target_model,
                        auditor=auditor_model,
                        # auditor extended-thinking setting AS RUN (stamped into
                        # log.eval.metadata by exp_rh_audit / exp_rollback). None on every
                        # pre-thinking run -> shown as plain auditor (no suffix).
                        auditor_reasoning_effort=(log.eval.metadata or {}).get("auditor_reasoning_effort"),
                        # FIXED-SP field: True = this run pinned the target system prompt via
                        # --fixed_sp (stamped into log.eval.metadata by exp_rh_audit).
                        # Informational only: the sweep split (see SWEEPS in make_viewer)
                        # is an explicit run-dir list, not this flag.
                        fixed_sp=bool((log.eval.metadata or {}).get("fixed_sp")),
                        judge=judge_model,
                        seed=str(s.id),
                        epoch=s.epoch,
                        scores=sc.value if isinstance(sc.value, dict) else {},
                        summary=md.get("summary", ""),
                        justification=getattr(sc, "explanation", None) or "",
                        highlights=md.get("highlights", ""),
                        transcript=transcript,
                        scratchpad=scratch,
                        auditor_calls=auditor_calls_by_anchor(s, transcript),
                        compactions=compactions,
                        dead=tgt_out == 0,
                        fork=bool(fork_calls),
                        fork_restart=fork_calls.count("restart_conversation"),
                        fork_rollback=fork_calls.count("rollback_conversation"),
                        # True if the auditor ended the conversation itself (end_conversation
                        # tool), False if it ran to the max-turns cap. Formerly backed the index
                        # CUTOFF badge (now removed; superseded by the `incompleteness` judge
                        # dimension); retained as a cheap cross-check against incompleteness.
                        ended_via_end_conv=any(
                            tc.function == "end_conversation"
                            for m in (s.messages or [])
                            for tc in (getattr(m, "tool_calls", None) or [])
                        ) or any(
                            getattr(m, "role", None) == "tool"
                            and getattr(m, "function", None) == "end_conversation"
                            for m in (s.messages or [])
                        ),
                    ),
                    rejudge,
                )
            )
    # fill in any single-dimension backfill scores (tools/exp_grade_dimension.py) for dims
    # an audit was judged before having (e.g. hallucination on older full hacks)
    sidecars = _load_dim_sidecars()
    for a in audits:
        _apply_dim_scores(a, sidecars)
    return audits


def rollback_grid(rb_dir: Path) -> dict[int, dict[int, bool]]:
    """Cached wrapper around _rollback_grid_impl (called twice per dir per build, each time
    re-reading every .eval). Same cache discipline as load_mode."""
    cached = _cache_get(rb_dir, "grid")
    if cached is not None:
        return cached
    grid = _rollback_grid_impl(rb_dir)
    _cache_put(rb_dir, "grid", grid)
    return grid


def _rollback_grid_impl(rb_dir: Path) -> dict[int, dict[int, bool]]:
    """What actually RAN in a rollback dir, before load_mode's display filtering.

    Returns {original_traj_id: {epoch: has_usable_judge_score}}. A sample that ran
    but whose audit_judge score is missing/non-dict (judge errored) is recorded as
    False here and is silently dropped by load_mode -- so comparing this grid to
    what the viewer surfaces tells us exactly which rollbacks are missing and why.
    """
    grid: dict[int, dict[int, bool]] = {}
    for li in list_eval_logs(str(rb_dir)):
        try:
            log = read_eval_log(li)
        except FileNotFoundError:
            continue
        oid = _orig_id_from_task(log.eval.task)
        if oid is None:
            continue
        for s in log.samples or []:
            sc = (s.scores or {}).get("audit_judge")
            usable = sc is not None and isinstance(sc.value, dict)
            grid.setdefault(oid, {})[s.epoch] = usable
    return grid
