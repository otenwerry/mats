"""Resolve a trajectory selector to a concrete launch list, with runnability and
done-status, for the fan-out launcher (exp_fanout.sh).

Selector grammar (case-insensitive):
  runnable            every credential-ready RH trajectory (opencode + claude-API)
  all                 runnable PLUS codex (only if codex recon is validated)
  opencode|claude|codex[,...]   by scaffold (claude => the API claude, not non_api)
  <nick>|<run_id>[,...]         explicit (nicknames from config.TRAJECTORIES, or run_ids)

Runnability (what we actually have credentials/validation for right now):
  - opencode  -> OPENROUTER_API_KEY            (ready)
  - claude (agent=='claude') -> ANTHROPIC_API_KEY   (ready)
  - codex     -> ChatGPT auth.json + recon validation seam (PTB_CODEX_ROLLOUT_VALIDATED)
  - claude_non_api (auth=='oauth') -> needs CLAUDE_CODE_OAUTH_TOKEN   (NOT set -> blocked)
  - qwen3max  -> needs DASHSCOPE key                                  (NOT set -> blocked)

This module makes NO network/box calls — pure metadata + a scan of the local
viewer_data dir for how many prompt conditions already exist per trajectory.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from .. import config

# Prompt conditions that constitute a "complete" trajectory (the 3 experiments).
ALL_PROMPTS = ("prompt1", "prompt2", "prompt3")
_VIEWER_RE = re.compile(r"rollback_(prompt\d)_run\d+__(.+)\.json$")


def _done_prompts_by_run_name() -> dict[str, set[str]]:
    """run_name -> set of prompt ids already present in viewer_data (COMPLETED)."""
    out: dict[str, set[str]] = {}
    d = config.ROLLBACK_VIEWER_DATA
    if not d.exists():
        return out
    for p in d.glob("rollback_*.json"):
        m = _VIEWER_RE.search(p.name)
        if m:
            out.setdefault(m.group(2), set()).add(m.group(1))
    return out


def _inflight_prompts_by_run_name() -> dict[str, set[str]]:
    """run_name -> prompt ids currently IN-FLIGHT (running), from the running/ dir
    + the legacy running_rollouts.json. We must treat these as already-covered so a
    gap-fill never relaunches a condition that's already running — INCLUDING runs
    started in a previous session. (Missing this is what caused duplicate cut67 /
    gemini launches.)"""
    out: dict[str, set[str]] = {}
    rb = config.ROLLBACK_LOCAL
    sources: list[dict] = []
    legacy = rb / "running_rollouts.json"
    if legacy.exists():
        try:
            v = json.loads(legacy.read_text())
            sources += v if isinstance(v, list) else []
        except (json.JSONDecodeError, OSError):
            pass
    rdir = rb / "running"
    if rdir.exists():
        for p in rdir.glob("*.json"):
            try:
                sources.append(json.loads(p.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
    for e in sources:
        if not isinstance(e, dict):
            continue
        cond = e.get("condition")
        if cond not in ALL_PROMPTS:
            continue
        rn = e.get("run_name")
        if not rn:
            rid = e.get("source_trajectory") or e.get("run_id") or ""
            rn = rid.split("__", 1)[1] if "__" in rid else rid
        if rn:
            out.setdefault(rn, set()).add(cond)
    return out


def runnable_reason(traj) -> str:
    """'' if runnable now, else a short reason it's blocked."""
    # Manually parked (a deliberate choice, e.g. too-heavy preps) — overrides the
    # credential-derived reasons below. Single source: rollback/manual_defer.json.
    if traj.run_id in config.MANUAL_DEFER:
        return config.MANUAL_DEFER[traj.run_id]
    if traj.scaffold == "opencode":
        return ""
    if traj.scaffold == "claude":
        if traj.agent == "qwen3max":
            return "needs DASHSCOPE key"
        if traj.auth == "oauth":
            return "claude_non_api needs CLAUDE_CODE_OAUTH_TOKEN"
        return ""  # claude-API
    if traj.scaffold == "codex":
        # recon_codex's validation gate is lifted (0.140 exec_command schema); codex
        # runs via the ChatGPT-subscription auth.json. Runnable iff that file is
        # deployed. (The 0.139->0.140 resume itself is confirmed by the codex canary.)
        auth = os.environ.get("PTB_CODEX_AUTH") or os.path.expanduser("~/.config/ptb/codex_auth.json")
        return "" if os.path.exists(auth) else "codex needs ChatGPT auth.json at ~/.config/ptb/codex_auth.json"
    return f"unknown scaffold {traj.scaffold}"


def _all_rh_run_ids() -> list[str]:
    return list(config.ALL_TRAJECTORIES)


def resolve_run_ids(selector: str) -> list[str]:
    sel = selector.strip().lower()
    allids = _all_rh_run_ids()
    by_scaffold = lambda sc: [rid for rid in allids
                              if config.ALL_TRAJECTORIES[rid].scaffold == sc]

    if sel in ("runnable", "all"):
        ids = allids  # filtered by runnability later; 'all' just allows codex via env
    elif all(tok.strip() in ("opencode", "claude", "codex") for tok in sel.split(",")):
        ids = []
        for tok in sel.split(","):
            ids += by_scaffold(tok.strip())
    else:
        # explicit nicknames / run_ids
        ids = []
        for tok in selector.split(","):
            tok = tok.strip()
            if not tok:
                continue
            ids.append(config.get_trajectory(tok).run_id)
    # stable order: opencode, claude, codex, then by run_id
    order = {"opencode": 0, "claude": 1, "codex": 2}
    return sorted(set(ids), key=lambda r: (order.get(config.ALL_TRAJECTORIES[r].scaffold, 9), r))


def plan(selector: str, include_done: bool) -> list[dict]:
    done = _done_prompts_by_run_name()
    inflight = _inflight_prompts_by_run_name()
    rows = []
    for rid in resolve_run_ids(selector):
        t = config.ALL_TRAJECTORIES[rid]
        reason = runnable_reason(t)
        have = done.get(t.run_name, set())          # completed
        running = inflight.get(t.run_name, set())   # in-flight (incl prior sessions)
        covered = have | running                    # don't relaunch either
        missing = [p for p in ALL_PROMPTS if p not in covered]
        fully_done = not [p for p in ALL_PROMPTS if p not in have]
        # launch only the genuinely-uncovered conditions of a runnable trajectory
        launch = (reason == "") and bool(missing)
        if include_done:
            launch = (reason == "")
        rows.append({
            "run_id": rid, "run_name": t.run_name, "scaffold": t.scaffold,
            "agent": t.agent, "auth": t.auth, "needs_prep": t.needs_prep,
            "cut": t.default_cut, "have": sorted(have), "running": sorted(running),
            "missing": missing, "fully_done": fully_done,
            "blocked_reason": reason, "launch": launch,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selector")
    ap.add_argument("--ids", action="store_true",
                    help="print only the run_ids we'd launch (for the shell)")
    ap.add_argument("--with-conds", action="store_true",
                    help="with --ids: print 'run_id<TAB>missing_prompts_csv' (gap-fill)")
    ap.add_argument("--include-done", action="store_true",
                    help="don't skip trajectories that already have all 3 prompts")
    args = ap.parse_args()
    rows = plan(args.selector, args.include_done)

    if args.ids:
        for r in rows:
            if not r["launch"]:
                continue
            if args.with_conds:
                # gap-fill: the prompts this trajectory still lacks (all 3 if untouched)
                conds = r["missing"] or list(ALL_PROMPTS)
                print(f"{r['run_id']}\t{','.join(conds)}")
            else:
                print(r["run_id"])
        return

    print(f"{'scaffold':9} {'auth':5} {'prep':4} {'cut':>4} {'have':18} "
          f"{'LAUNCH?':8} run_id / reason")
    print("-" * 110)
    for r in rows:
        prep = "yes" if r["needs_prep"] else "no"
        have = ",".join(p[-1] for p in r["have"]) or "-"   # "1,2,3"
        tag = "LAUNCH" if r["launch"] else ("done" if r["fully_done"] and not r["blocked_reason"]
                                            else "skip")
        note = r["run_id"] if r["launch"] else f"{r['run_id']}  [{r['blocked_reason'] or 'already complete'}]"
        print(f"{r['scaffold']:9} {r['auth']:5} {prep:4} {r['cut']:>4} {have:18} {tag:8} {note}")
    n = sum(1 for r in rows if r["launch"])
    print("-" * 110)
    print(f"would launch {n} trajectory batch(es) x {len(ALL_PROMPTS)} prompts")


if __name__ == "__main__":
    main()
