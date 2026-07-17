"""Context-window and compaction visuals for confirmed PTB reward hacks.

The provider streams use different accounting formats. Claude records one usage
object per model message (with prompt-cache fields); OpenCode records one token
object per step. Codex records only cumulative turn totals and the Qwen trace's
per-message usage is zero, so those trajectories remain explicit coverage rows
but are never assigned estimated context sizes.

Event numbers always use the viewer's event index. A hack annotation on an
assistant event maps to the call containing it; an annotation on the following
tool-result event maps to the immediately preceding call that caused the tool.
"""
from __future__ import annotations

import io
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import runs, stream


# These are the capacities used by the run scaffolds, not today's model defaults.
# The normal Claude runs compacted under the 200k configuration; only run names
# explicitly marked `_1m` used the long-context variant.
OPENCODE_CONTEXT_WINDOWS = {
    "gemini-3.1-pro": 1_048_576,
    "gpt-5.1-codex-max": 400_000,
    "opencode/kimi-k2-thinking": 262_144,
    "opencode/kimi-k2.5": 262_144,
}

MODEL_COLORS = {
    "claude-opus-4-6": "#4C72B0",
    "claude-opus-4-6[1m]": "#64B5CD",
    "claude-sonnet-4-6": "#55A868",
    "gemini-3.1-pro": "#8172B3",
    "gpt-5.1-codex-max": "#937860",
    "opencode/kimi-k2-thinking": "#CCB974",
    "opencode/kimi-k2.5": "#DD8452",
}


@dataclass
class ContextCall:
    call_index: int
    event_start: int
    event_end: int
    context_tokens: int
    compactions_before: int = 0
    last_compaction_event: int | None = None
    calls_since_compaction: int | None = None
    hack_events: list[int] = field(default_factory=list)
    is_first_hack: bool = False


@dataclass
class ContextTrajectory:
    run_id: str
    viewer_number: int
    experiment: str
    benchmark: str
    model: str
    scaffold: str
    first_hack_event: int | None
    hack_events: list[int]
    context_window: int | None
    calls: list[ContextCall]
    compaction_events: list[int]
    coverage_status: str
    omission_reason: str | None = None
    unmapped_hack_events: list[int] = field(default_factory=list)

    @property
    def first_hack_call(self) -> ContextCall | None:
        return next((c for c in self.calls if c.is_first_hack), None)


def _context_window(traj: runs.Trajectory, model: str) -> int | None:
    if traj.scaffold == "claude":
        return 1_000_000 if "_1m" in traj.run_dir.name else 200_000
    if traj.scaffold == "opencode":
        return OPENCODE_CONTEXT_WINDOWS.get(model)
    return None


def _hack_events(highlight: dict) -> tuple[int | None, list[int]]:
    """Canonical first hack plus every explicitly hack-classified marked turn."""
    final = highlight.get("final") or {}
    first = final.get("first_hack_event")
    kinds = highlight.get("turn_kinds") or {}
    direct = [int(k) for k, v in kinds.items()
              if isinstance(v, dict) and v.get("kind") == "hack"]
    events = sorted(set(direct + ([int(first)] if first is not None else [])))
    return (int(first) if first is not None else None), events


def _claude_calls(traj: runs.Trajectory, events: list[dict]) -> list[ContextCall]:
    parsed = stream.parse(traj)
    alignment = stream.align(traj, parsed, events)
    if not alignment.ok:
        raise ValueError(f"stream alignment failed: {alignment.detail}")

    event_for_raw = {raw_i: event_i for event_i, (raw_i, _part)
                     in enumerate(alignment.ev_to_raw)}
    grouped: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    order: list[tuple] = []
    for raw_i, raw in enumerate(parsed.records):
        if raw.get("type") != "assistant" or raw.get("parent_tool_use_id"):
            continue
        message = raw.get("message") or {}
        message_id = message.get("id")
        # A message id is unique within a session. The fallback raw index keeps a
        # malformed/missing id from merging unrelated calls.
        key = (raw.get("session_id"), message_id if message_id else raw_i)
        if key not in grouped:
            order.append(key)
        grouped[key].append((raw_i, message.get("usage") or {}))

    calls = []
    for key in order:
        parts = grouped[key]
        usage = next((u for _i, u in reversed(parts) if u), {})
        total = sum(int(usage.get(k) or 0) for k in (
            "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        if total <= 0:
            continue
        evs = [event_for_raw[i] for i, _u in parts if i in event_for_raw]
        if not evs:
            continue
        calls.append(ContextCall(
            call_index=len(calls) + 1,
            event_start=min(evs), event_end=max(evs), context_tokens=total))
    return calls


def _opencode_calls(traj: runs.Trajectory, events: list[dict]) -> list[ContextCall]:
    parsed = stream.parse(traj)
    alignment = stream.align(traj, parsed, events)
    if not alignment.ok:
        raise ValueError(f"stream alignment failed: {alignment.detail}")

    events_for_raw: dict[int, list[int]] = defaultdict(list)
    for event_i, (raw_i, _part) in enumerate(alignment.ev_to_raw):
        events_for_raw[raw_i].append(event_i)

    active: dict[str, tuple[int, int]] = {}
    calls = []
    for raw_i, raw in enumerate(parsed.records):
        part = raw.get("part") or {}
        message_id = part.get("messageID")
        if raw.get("type") == "step_start" and message_id:
            start = min(events_for_raw.get(raw_i) or [0])
            active[message_id] = (raw_i, start)
        elif raw.get("type") == "step_finish" and message_id:
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            total = int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            if total <= 0:
                continue
            _raw_start, event_start = active.get(message_id, (raw_i, 0))
            end_events = events_for_raw.get(raw_i) or [event_start]
            calls.append(ContextCall(
                call_index=len(calls) + 1, event_start=event_start,
                event_end=max(end_events), context_tokens=total))
    return calls


def _map_annotations(calls: list[ContextCall], hack_events: list[int],
                     first_hack_event: int | None) -> list[int]:
    """Attach each hack event to its producing call; return unmapped events."""
    unmapped = []
    for event in hack_events:
        containing = next((c for c in calls if c.event_start <= event <= c.event_end), None)
        call = containing or next((c for c in reversed(calls) if c.event_start <= event), None)
        if call is None:
            unmapped.append(event)
            continue
        call.hack_events.append(event)
        if event == first_hack_event:
            call.is_first_hack = True
    return unmapped


def _map_compactions(calls: list[ContextCall], compactions: list[int]) -> None:
    for call in calls:
        prior = [e for e in compactions if e < call.event_start]
        call.compactions_before = len(prior)
        if not prior:
            continue
        last = prior[-1]
        call.last_compaction_event = last
        call.calls_since_compaction = sum(
            1 for candidate in calls
            if last < candidate.event_start <= call.event_start)


def load_context_trajectories(index_rows: list[dict]) -> list[ContextTrajectory]:
    """One coverage record per confirmed reward-hack trajectory, with no drops."""
    out = []
    for viewer_i, row in enumerate(index_rows, 1):
        run_id = row["run_id"]
        if row.get("is_rollback"):
            continue
        highlight_path = runs.HIGHLIGHTS / f"{run_id}.json"
        if not highlight_path.exists():
            continue
        try:
            highlight = json.loads(highlight_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not (highlight.get("final") or {}).get("is_reward_hack"):
            continue

        traj = runs.load(run_id)
        record = json.loads(traj.viewer_path.read_text())
        events = record.get("events") or []
        model = row.get("agent_model") or "?"
        first_hack, hack_events = _hack_events(highlight)
        compactions = [i for i, event in enumerate(events)
                       if event.get("type") == "system"
                       and event.get("subtype") == "compact_boundary"]
        calls: list[ContextCall] = []
        omission = None
        try:
            if traj.scaffold == "claude":
                calls = _claude_calls(traj, events)
            elif traj.scaffold == "opencode":
                calls = _opencode_calls(traj, events)
            elif traj.scaffold == "codex":
                omission = ("Codex saved cumulative turn usage, not the input size "
                            "of each model call")
            elif traj.scaffold == "qwen3max":
                omission = "Qwen saved zero input tokens on each model message"
            else:
                omission = f"no context extractor for {traj.scaffold}"
        except (OSError, ValueError, KeyError, IndexError) as exc:
            omission = f"usage extraction failed: {type(exc).__name__}: {exc}"
            calls = []

        window = _context_window(traj, model)
        if calls and window is None:
            omission = f"context-window capacity is unknown for {model}"
            calls = []
        if not calls and omission is None:
            omission = "no non-zero per-call context usage was recorded"

        _map_compactions(calls, compactions)
        unmapped = _map_annotations(calls, hack_events, first_hack)
        status = "complete" if calls and not unmapped else (
            "partial" if calls else "unavailable")
        out.append(ContextTrajectory(
            run_id=run_id, viewer_number=viewer_i,
            experiment=row.get("experiment") or run_id,
            benchmark=row.get("benchmark") or "?", model=model,
            scaffold=traj.scaffold, first_hack_event=first_hack,
            hack_events=hack_events, context_window=window, calls=calls,
            compaction_events=compactions, coverage_status=status,
            omission_reason=omission, unmapped_hack_events=unmapped))
    return out


def _setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#c9ccd4", "text.color": "#30333a",
        "axes.labelcolor": "#4b4f58", "xtick.color": "#60646d",
        "ytick.color": "#60646d", "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _fig_to_svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = buf.getvalue()
    return svg[svg.find("<svg"):]


def _short_model(model: str) -> str:
    return (model.replace("opencode/", "")
            .replace("claude-", "")
            .replace("gemini-", "gemini ")
            .replace("gpt-", "gpt "))


def _rank_percent(call: ContextCall, calls: list[ContextCall]) -> float:
    return 100 * sum(c.context_tokens <= call.context_tokens for c in calls) / len(calls)


def fig_first_hack_context(trajectories: list[ContextTrajectory]) -> str:
    """Raw tokens, capacity fraction, and within-trajectory rank for first hacks."""
    rows = [t for t in trajectories if t.first_hack_call and t.context_window]
    rows.sort(key=lambda t: t.first_hack_call.context_tokens / t.context_window)
    if not rows:
        return ""
    _setup_style()
    height = max(5.0, 0.36 * len(rows) + 1.8)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, height), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    y = np.arange(len(rows))
    labels = [f"#{t.viewer_number}  {t.benchmark} · {_short_model(t.model)}" for t in rows]
    raw = np.array([t.first_hack_call.context_tokens / 1000 for t in rows])
    full = np.array([100 * t.first_hack_call.context_tokens / t.context_window for t in rows])
    ranks = np.array([_rank_percent(t.first_hack_call, t.calls) for t in rows])
    colors = [MODEL_COLORS.get(t.model, "#6b7280") for t in rows]

    for ax, values, title, xlabel, xmax in (
        (axes[0], raw, "Raw context", "input tokens (thousands)", None),
        (axes[1], full, "Window fullness", "% of context window", 100),
        (axes[2], ranks, "Position within same trajectory",
         "percentile among that run's calls", 100),
    ):
        ax.hlines(y, 0, values, color="#e1e3e8", lw=1.1, zorder=1)
        ax.scatter(values, y, c=colors, s=38, edgecolor="white", lw=0.6, zorder=2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#eceef2", lw=0.8)
        if xmax:
            ax.set_xlim(0, xmax)
    axes[0].set_yticks(y, labels)
    axes[1].axvline(50, color="#bfc3cc", lw=0.8, ls="--")
    axes[2].axvline(50, color="#bfc3cc", lw=0.8, ls="--")
    axes[0].set_ylim(-0.8, len(rows) - 0.2)
    fig.suptitle("Context at the canonical first reward hack", x=0.07, ha="left",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.18, top=0.93)
    return _fig_to_svg(fig)


def fig_compaction_distance(trajectories: list[ContextTrajectory]) -> str:
    """First-hack distance from explicit Claude compactions."""
    compacting = [t for t in trajectories if t.calls and t.compaction_events]
    if not compacting:
        return ""
    after = [t for t in compacting
             if t.first_hack_call and t.first_hack_call.last_compaction_event is not None]
    before = [t for t in compacting
              if t.first_hack_call and t.first_hack_call.last_compaction_event is None]
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1),
                             gridspec_kw={"width_ratios": [1.65, 1]})
    ax = axes[0]
    if after:
        after.sort(key=lambda t: t.first_hack_call.calls_since_compaction)
        y = np.arange(len(after))
        x = [t.first_hack_call.calls_since_compaction for t in after]
        ax.hlines(y, 0, x, color="#f0c7c4", lw=1.2)
        ax.scatter(x, y, color="#C44E52", s=42, edgecolor="white", lw=0.5)
        ax.set_yticks(y, [f"#{t.viewer_number}  {t.benchmark}" for t in after])
        ax.set_ylim(-0.7, len(after) - 0.3)
    ax.set_xlabel("model calls since latest compaction (first call after = 1)")
    ax.set_title("First hacks after a compaction", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#eceef2", lw=0.8)

    axes[1].bar(["before any\ncompaction", "after a\ncompaction"],
                [len(before), len(after)], color=["#b9bcc6", "#C44E52"], width=0.58)
    for i, n in enumerate((len(before), len(after))):
        axes[1].text(i, n + 0.12, str(n), ha="center", fontweight="bold")
    axes[1].set_ylabel("trajectories")
    axes[1].set_title("Among runs that compacted", loc="left", fontweight="bold")
    axes[1].set_ylim(0, max(1, len(compacting)) * 1.15)
    axes[1].grid(axis="y", color="#eceef2", lw=0.8)
    fig.suptitle("Canonical first hack relative to recorded compaction boundaries",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.subplots_adjust(wspace=0.38, top=0.82)
    return _fig_to_svg(fig)


def fig_context_at_hack_actions(trajectories: list[ContextTrajectory]) -> str:
    """All mapped hack actions, preserving run-level grouping rather than pooling calls."""
    rows = []
    for t in trajectories:
        hack_calls = [c for c in t.calls if c.hack_events]
        if hack_calls and t.context_window:
            rows.append((t, hack_calls))
    if not rows:
        return ""
    rows.sort(key=lambda item: np.median([
        100 * c.context_tokens / item[0].context_window for c in item[1]]))
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.36 * len(rows) + 1.7)))
    for y, (t, calls) in enumerate(rows):
        all_full = [100 * c.context_tokens / t.context_window for c in t.calls]
        hack_full = [100 * c.context_tokens / t.context_window for c in calls]
        ax.hlines(y, min(all_full), max(all_full), color="#d9dce3", lw=2.2)
        ax.scatter(hack_full, [y] * len(hack_full), s=30,
                   color=MODEL_COLORS.get(t.model, "#C44E52"),
                   edgecolor="white", lw=0.5, zorder=3)
        first = t.first_hack_call
        if first:
            ax.scatter([100 * first.context_tokens / t.context_window], [y],
                       marker="*", s=105, color="#B3261E", edgecolor="white",
                       lw=0.5, zorder=4)
    ax.set_yticks(np.arange(len(rows)), [
        f"#{t.viewer_number}  {t.benchmark} · {_short_model(t.model)}" for t, _ in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of context window")
    ax.set_title("Every annotated hack action", loc="left", fontsize=13, fontweight="bold")
    ax.text(0, 1.012, "gray line = that trajectory's full observed range  ·  star = canonical first hack",
            transform=ax.transAxes, color="#6b7280", fontsize=8)
    ax.grid(axis="x", color="#eceef2", lw=0.8)
    return _fig_to_svg(fig)


def fig_timeline(trajectory: ContextTrajectory) -> str:
    """One trajectory's complete context curve with every hack and compaction."""
    if not trajectory.calls or not trajectory.context_window:
        return ""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.2, 2.35))
    x = np.array([c.call_index for c in trajectory.calls])
    y = np.array([100 * c.context_tokens / trajectory.context_window
                  for c in trajectory.calls])
    color = MODEL_COLORS.get(trajectory.model, "#4C72B0")
    ax.plot(x, y, color=color, lw=1.35)
    ax.fill_between(x, y, color=color, alpha=0.08)
    hack_calls = [c for c in trajectory.calls if c.hack_events]
    if hack_calls:
        hx = [c.call_index for c in hack_calls]
        hy = [100 * c.context_tokens / trajectory.context_window for c in hack_calls]
        ax.scatter(hx, hy, color="#C44E52", s=36, edgecolor="white", lw=0.5, zorder=4)
    first = trajectory.first_hack_call
    if first:
        fy = 100 * first.context_tokens / trajectory.context_window
        ax.scatter([first.call_index], [fy], marker="*", s=125, color="#B3261E",
                   edgecolor="white", lw=0.5, zorder=5)
        ax.annotate(f"{first.context_tokens/1000:.0f}k · {fy:.0f}%",
                    (first.call_index, fy), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color="#8f1d1d")
    for event in trajectory.compaction_events:
        after = next((c for c in trajectory.calls if c.event_start > event), None)
        if after:
            ax.axvline(after.call_index - 0.5, color="#555b66", lw=0.9, ls="--")
    ax.set_xlim(1, max(x))
    ax.set_ylim(0, max(100, float(max(y)) * 1.12))
    ax.set_xlabel("model call")
    ax.set_ylabel("context window full")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0f}%")
    ax.grid(axis="y", color="#eceef2", lw=0.8)
    ax.set_title(f"#{trajectory.viewer_number} · {trajectory.benchmark} · {_short_model(trajectory.model)}",
                 loc="left", fontweight="bold")
    return _fig_to_svg(fig)


def first_hack_table_rows(trajectories: list[ContextTrajectory]) -> list[dict]:
    rows = []
    for t in trajectories:
        call = t.first_hack_call
        if call and t.context_window:
            if call.last_compaction_event is not None:
                relative = f"{call.calls_since_compaction} calls after event {call.last_compaction_event}"
            elif t.compaction_events:
                relative = "before first compaction"
            else:
                relative = "no compaction in run"
            rows.append({
                "trajectory": t, "call": call,
                "fullness": 100 * call.context_tokens / t.context_window,
                "rank": _rank_percent(call, t.calls), "relative": relative,
            })
    return rows


def grouped_timelines(trajectories: list[ContextTrajectory]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        if t.calls:
            groups[t.model].append({"trajectory": t, "svg": fig_timeline(t)})
    return [{"model": model, "items": groups[model]}
            for model in sorted(groups, key=lambda m: (
                min(i["trajectory"].viewer_number for i in groups[m]), m))]


def capacity_rows(trajectories: list[ContextTrajectory]) -> list[dict]:
    seen = {}
    for t in trajectories:
        if t.context_window:
            seen[(t.model, t.context_window)] = {
                "model": t.model, "tokens": t.context_window,
                "source": ("PTB run name (`_1m` only for the long-context variant)"
                           if t.scaffold == "claude" else
                           "OpenCode v1.1.59 model context limit"),
            }
    return sorted(seen.values(), key=lambda r: (r["tokens"], r["model"]))
