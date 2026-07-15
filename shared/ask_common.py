"""Passive plumbing shared by shared/exp_ask_questions.py and its environment
adapters (petri/lib/exp_ask_env.py, posttrainbench/lib/exp_ask_env.py).

This module makes no API calls and launches nothing; it exists so the driver
and the adapters can share the AskUnit type without importing each other.

The adapter interface (what shared/exp_ask_questions.py expects each
exp_ask_env module to provide):

  constants
    NAME                 "petri" / "posttrainbench"
    TRANSITION           env wording of the task-is-over prefix (question sets
                         that carry a transition use this text)
    OUT_ROOT             Path; campaign output dirs go under it
    SUBJECT              "target" / "agent" -- noun used in shared md text
    TRACKS_EXACT_COST    True when per-ask cost_source can be "exact ..."
                         (adds cost_exact_for to summaries + md)
    EMPTY_RECORD_EXTRA   env-specific None-valued record keys for the
                         catch-all error record (e.g. {"exit_code": None})
    MD_NO_ANSWER         probe_result.md fallback text when an ask has no answer
    BASELINE_MD_NOTE     results.md line explaining what this env's baseline is
    TRAJECTORY_HELP / TURN_HELP / BASELINE_HELP   env-specific CLI help text

  functions
    add_cli_args(ap)                     env-only flags (may be a no-op)
    build_units(args) -> list[AskUnit]   resolve --trajectory, build one unit
                                         per resumable context; print per-unit
                                         info + refusals (free)
    add_baselines(units, args) -> list[AskUnit]   env-specific baseline units
    dry_run_report(units, n_per_unit, n_total)    print per-unit input-cost
                                         estimates (free; no other setup ran)
    prepare(units, args)                 paid-run setup after out_dirs exist
                                         (models/cost hooks/CLI/config); may
                                         write per-unit context files and must
                                         finalize md_info for every unit
    async ask(unit, q, sent, i, ask_dir, fidelity, args) -> record dict
                                         one independent ask; mutates
                                         fidelity["flags"]/["cost"] and writes
                                         env artifacts; NEVER judges and NEVER
                                         writes fidelity.json/probe_result.md
                                         (the driver does, after judging)
    summary_fields(unit, args) -> dict   env-schema identity head for the
                                         results.json summary (run_id/... or
                                         trajectory_id/...), incl. requested_turn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AskUnit:
    """One resumable context (a trajectory at a cut, or a baseline config).
    Every (unit, question, sample) triple becomes one independent ask."""
    label: str                # console progress label ("id 1561" / run_id[:44])
    dirname: str              # output dir name (the driver uniquifies collisions)
    ask_who: str              # short header for each ask's probe_result.md
    md_who: str               # header for the unit's results.md
    md_info: str              # per-ask info line for probe_result.md
                              # (adapters may finalize this in prepare())
    is_baseline: bool = False
    baseline_for: list = field(default_factory=list)  # anchor ids (baseline only)
    fidelity_base: dict = field(default_factory=dict) # per-ask fidelity seed
                              # (deep-copied per ask; must contain "flags")
    env: dict = field(default_factory=dict)           # opaque adapter payload
    out_dir: Path | None = None                       # set by the driver
