"""Reconstruct how much wall-clock time had elapsed at the cut point, so the
resumed agent's `timer.sh` reports a faithful "time remaining".

The Claude stream-json traces in this dataset carry NO reliable per-event
timestamps (the few that appear are incidental — the agent `cat`-ing its own
session file). So we approximate from the events, calibrated to the known total
wall time in `time_taken.txt`.

Model:
  total = HARD waits (explicit `sleep N` the agent issues while polling its
          background training/eval jobs — the real time sinks) + SOFT overhead
          (model latency + quick tool execs spread across all the other steps).

  elapsed(cut) = (hard sleeps before cut)
               + soft_overhead * (soft weight before cut / total soft weight)

For a pre-training cut (before any long sleeps) this is dominated by the soft
term and lands at a small fraction of the budget — which is correct: the agent
has done only fast reads/bash/edits so far.

If a trajectory DOES have genuine per-event timestamps, `elapsed_from_timestamps`
is used instead.
"""
from __future__ import annotations

import re

from . import config, ptbio

_SLEEP_RE = re.compile(r"\bsleep\s+(\d+(?:\.\d+)?)")
# Soft per-step baseline (seconds) — a rough model-latency + quick-exec cost,
# only used to DISTRIBUTE the non-sleep remainder, so its absolute value washes
# out in calibration. Assistant turns cost more than tool results.
_SOFT_ASSISTANT = 8.0
_SOFT_OTHER = 1.0


def parse_time_taken(traj: config.Trajectory) -> int | None:
    """Total wall seconds from time_taken.txt (HH:MM:SS), or None."""
    p = traj.raw_dir / "time_taken.txt"
    if not p.exists():
        return None
    parts = p.read_text().strip().split(":")
    if len(parts) != 3:
        return None
    h, m, s = (int(x) for x in parts)
    return h * 3600 + m * 60 + s


def event_sleep_seconds(events: list[dict]) -> list[float]:
    """Per-event seconds of explicit `sleep` inside that event's bash commands."""
    out = [0.0] * len(events)
    for tc in ptbio.iter_tool_calls(events):
        if tc.name.lower() != "bash":
            continue
        cmd = tc.input.get("command")
        if isinstance(cmd, str):
            out[tc.idx] += sum(float(m) for m in _SLEEP_RE.findall(cmd))
    return out


def _soft_weight(ev: dict) -> float:
    return _SOFT_ASSISTANT if ev.get("type") == "assistant" else _SOFT_OTHER


def elapsed_at(events: list[dict], cut: int, total_seconds: int | None) -> float:
    """Estimated wall seconds elapsed when execution reaches event `cut`.

    Counts events [0, cut). `total_seconds` calibrates the soft overhead; if
    None, falls back to raw soft baseline + hard sleeps (uncalibrated)."""
    sleeps = event_sleep_seconds(events)
    hard_before = sum(sleeps[:cut])
    hard_total = sum(sleeps)
    soft_before = sum(_soft_weight(events[i]) for i in range(min(cut, len(events))))
    soft_total = sum(_soft_weight(e) for e in events)

    if total_seconds is None:
        return hard_before + soft_before  # uncalibrated best effort

    soft_overhead = max(0.0, total_seconds - hard_total)
    soft_frac = (soft_before / soft_total) if soft_total else 0.0
    return hard_before + soft_overhead * soft_frac


def elapsed_from_timestamps(events: list[dict], cut: int) -> float | None:
    """If events carry usable ISO/epoch timestamps, return elapsed time at the
    cut: latest ts among the KEPT events [0, cut) minus ts[0]. (Anchoring on
    events[cut] would wrongly include the dropped step's own duration.)"""
    def ts(ev):
        v = ev.get("ts") or ev.get("timestamp")
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    t0 = next((ts(e) for e in events if ts(e) is not None), None)
    kept = [t for t in (ts(e) for e in events[:cut]) if t is not None]
    if t0 is None or not kept:
        return None
    return max(0.0, max(kept) - t0)


def cut_epoch(events: list[dict], cut: int) -> float | None:
    """Absolute wall-clock (unix epoch) of the cut: latest timestamp among the
    kept events [0, cut). None when the trace has no usable timestamps."""
    def ts(ev):
        v = ev.get("ts") or ev.get("timestamp")
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None
    kept = [t for t in (ts(e) for e in events[:cut]) if t is not None]
    return max(kept) if kept else None


def _timer_remaining(text: str) -> float | None:
    """Seconds remaining parsed from a timer.sh result ('H:MM', or expired)."""
    if not text:
        return None
    if "expired" in text.lower():
        return 0.0
    m = re.search(r"(\d{1,2}):(\d{2})\b", text)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60


def timer_anchors(events: list[dict], budget: int) -> list[tuple[int, float]]:
    """(event_index, elapsed_seconds) for every `timer.sh` call, derived from the
    time-remaining it reported. These are REAL wall-clock anchors — far more
    accurate than step-count when the trace lacks timestamps. timer.sh truncates
    to whole minutes, so each reading over-states elapsed by 0-60s; we bias-correct
    by the 30s mean."""
    res: dict = {}
    for ev in events:
        for b in ev.get("blocks") or []:
            if b.get("type") == "tool_result":
                res[b.get("tool_use_id")] = str(b.get("content") or "")
    anchors = []
    for tc in ptbio.iter_tool_calls(events):
        cmd = tc.input.get("command") if isinstance(tc.input, dict) else None
        if not (isinstance(cmd, str) and "timer.sh" in cmd):
            continue
        rem = _timer_remaining(res.get(tc.tool_id, ""))
        if rem is None:
            continue
        anchors.append((tc.idx, max(0.0, (budget - rem) - 30.0)))
    return sorted(anchors)


def elapsed_from_timer(events: list[dict], cut: int, budget: int,
                       total: int | None) -> float | None:
    """Elapsed at the cut, anchored on the run's own timer.sh readings and
    distributed WITHIN an anchor bracket by the step-count shape. None if the run
    never called timer.sh (then `reconstruct` falls back to the approximation)."""
    anchors = timer_anchors(events, budget)
    if not anchors:
        return None
    h = lambda i: elapsed_at(events, i, total)   # heuristic shape between anchors
    before = [a for a in anchors if a[0] <= cut]
    after = [a for a in anchors if a[0] >= cut]
    if before and before[-1][0] == cut:
        return before[-1][1]
    A = before[-1] if before else None
    B = after[0] if after else None
    if A and B:                       # interpolate between bracketing anchors
        ha, hb, hc = h(A[0]), h(B[0]), h(cut)
        frac = (hc - ha) / (hb - ha) if hb > ha else (cut - A[0]) / max(B[0] - A[0], 1)
        return A[1] + frac * (B[1] - A[1])
    if A:                             # cut after last anchor -> extrapolate to total
        end = len(events)
        ha, he, hc = h(A[0]), h(end), h(cut)
        ceil = total if (total and total > A[1]) else budget
        frac = (hc - ha) / (he - ha) if he > ha else 0.0
        return min(A[1] + frac * (ceil - A[1]), budget)
    hb, hc = h(B[0]), h(cut)          # cut before first anchor
    frac = (hc / hb) if hb > 0 else (cut / max(B[0], 1))
    return frac * B[1]


def reconstruct(traj: config.Trajectory, cut: int) -> dict:
    """Everything the runner needs to write a faithful timer.sh at the cut.

    Returns elapsed/remaining seconds and which method was used. The runner sets
    timer CREATION_DATE = (resume wall-clock now) - elapsed, so timer.sh reports
    `remaining` correctly going forward."""
    events = ptbio.load_events(traj)
    total = parse_time_taken(traj)
    budget = traj.num_hours * 3600

    # priority: real per-event timestamps > the run's own timer.sh readings
    # (real wall-clock anchors) > step-count approximation (last resort).
    ts_elapsed = elapsed_from_timestamps(events, cut)
    if ts_elapsed is not None:
        elapsed, method = ts_elapsed, "timestamps"
    else:
        tmr = elapsed_from_timer(events, cut, budget, total)
        if tmr is not None:
            elapsed, method = tmr, "timer_anchored"
        else:
            elapsed, method = elapsed_at(events, cut, total), "approximation"

    elapsed = min(elapsed, budget)
    return {
        "method": method,
        "elapsed_seconds": round(elapsed),
        "remaining_seconds": round(budget - elapsed),
        "budget_seconds": budget,
        "total_wall_seconds": total,
        "cut_before_event": cut,
    }
