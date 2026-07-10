"""Codex context reconstruction — REPORT ONLY in v1, with hard fidelity flags.

The codex `--json` stream records reasoning SUMMARIES, agent messages, and
command executions with output, but:
  - `file_change` items carry only {path, kind} — the apply_patch diff bodies
    that entered the model's context are NOT saved anywhere;
  - raw reasoning items (Responses API) were never exported, and the rollout
    file `codex exec resume` would need was not archived.
So a faithful native resume is impossible; anything built from this data is
an approximation and every output carries error-severity flags. v1 exports
the kept items verbatim so the approximate path can be built later.
"""
from __future__ import annotations

from dataclasses import dataclass

from .runs import Trajectory
from .stream import CutPlan, Parsed


@dataclass
class CodexBundle:
    traj: Trajectory
    plan: CutPlan
    items: list[dict]             # verbatim item.completed payloads before cut
    flags: list[dict]
    stats: dict


def _flag(code: str, detail: str, severity: str = "warn") -> dict:
    return {"code": code, "severity": severity, "detail": detail}


def build(traj: Trajectory, parsed: Parsed, plan: CutPlan) -> CodexBundle:
    kept = [r["item"] for r in parsed.records[: plan.raw_cut]
            if r.get("type") == "item.completed"]
    n_file_change = sum(1 for it in kept if it.get("type") == "file_change")
    flags = [
        _flag("codex_no_rollout_archived",
              "the rollout file `codex exec resume` needs was not archived; native "
              "resume impossible, reconstruction is approximate at best", "error"),
        _flag("codex_reasoning_summaries_only",
              "reasoning items are summaries (model_reasoning_summary=detailed), not "
              "the raw reasoning the model produced", "error"),
    ]
    if n_file_change:
        flags.append(_flag(
            "codex_patch_bodies_missing",
            f"{n_file_change} file_change item(s) before the cut carry only "
            f"{{path, kind}}; the patch bodies that entered the model's context are "
            f"lost", "error"))
    stats = {"items_kept": len(kept),
             "items_dropped": sum(1 for r in parsed.records[plan.raw_cut:]
                                  if r.get("type") == "item.completed"),
             "file_changes_before_cut": n_file_change,
             "n_verbatim": len(kept), "n_synthesized": 0}
    return CodexBundle(traj, plan, kept, flags, stats)
