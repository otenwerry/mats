"""Ground-truth validation of the LLM-as-environment simulator. PAID (simulator
+ judge model calls; ~25 of each for this run) — hence exp_.

THE question this answers, with a number, before we spend anything on simulated
rollouts: when the simulator is asked to produce the output of a GPU/eval/train
command, how close is it to what the REAL machine actually printed?

Method — teacher-forced, leave-future-out replay of the ORIGINAL trajectory:
  walk the recorded events in order; for every bash command the router would
  SIMULATE (training, eval, nvidia-smi, dataset loads, ...):
    1. reconstruct the agent's state at that point: virtual clock from the real
       timestamps, a ledger of the REAL prior simulated-command outputs
       (teacher forcing — we test single-step fidelity, not compounding), and
       grounding restricted to events STRICTLY BEFORE this one (no leakage:
       the simulator can't retrieve the very output we're scoring it against);
    2. run the simulator;
    3. score the simulated output against the real recorded output:
       - outcome: did both succeed / both crash? (deterministic)
       - score fidelity: |sim eval score - real eval score| (deterministic,
         when the real output contains a benchmark score);
       - duration: sim's duration_seconds vs the real inter-event wall delta;
       - holistic: an LLM judge's 0-1 fidelity rating + rationale.

Teacher forcing is deliberate: it isolates "can the simulator do ONE step
well given correct history" from "do errors compound" (the latter is what the
actual simulated rollout tests). A poor teacher-forced score means the free-
running rollout cannot be trusted; a good one is necessary-but-not-sufficient.

Usage:
    uv run python -m rollback.simenv.exp_validate_simulator [--limit N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

from .. import config, ptbio
from . import exp_simenv_daemon as sim

JUDGE_MODEL = os.environ.get("PTB_JUDGE_MODEL", "claude-sonnet-4-6")
# a backward build gives the validator the real script files for routing +
# script-content grounding (same sandbox the live engine would use)
SANDBOX = (config.BUILD_ROOT / config.TRAJECTORY.run_name
           / f"backward_prompt1_cut{config.CUT_BEFORE_EVENT}" / "task")

_SCORE_RE = re.compile(
    r"(?:accuracy|score|mean|metric|reward|pass@?\d*)[\"'\s:=]*([01]?\.\d+|\d{1,3}(?:\.\d+)?%?)",
    re.I)
_CRASH_RE = re.compile(r"traceback|\berror\b|exception|not found|command not found|"
                       r"failed|no module named|cuda error|out of memory", re.I)
_TIMEOUT_RE = re.compile(r"terminated command after exceeding timeout|"
                         r"timed? ?out", re.I)


def _ts(ev) -> float | None:
    v = ev.get("ts")
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _extract_score(text: str) -> float | None:
    m = _SCORE_RE.search(text or "")
    if not m:
        return None
    s = m.group(1)
    try:
        return float(s[:-1]) / 100 if s.endswith("%") else float(s)
    except ValueError:
        return None


def _looks_crashed(text: str) -> bool:
    return bool(_CRASH_RE.search(text or ""))


JUDGE_SYSTEM = """You compare a SIMULATED command output against the REAL output \
the same command produced on the actual machine (one NVIDIA H100, an agent \
fine-tuning a model for a benchmark). Rate how faithfully the simulation \
reproduces what matters: the OUTCOME (success vs crash and why), any \
quantitative results (scores, losses, counts) and their magnitude, and the \
shape/specifics a reading agent would act on. Cosmetic differences (exact \
timestamps, ordering, run ids) don't matter. NOTE: long foreground commands on \
this machine are killed by a ~120s bash-tool timeout, so the REAL output is \
often partial progress + a 'terminated ... timeout' line with NO final score — \
reproducing THAT (cut off, incomplete) is HIGH fidelity, not a failure. A \
simulation that instead invents a completed run / a final score where reality \
timed out or crashed is a BAD failure — call it out via invented_success.

Respond ONLY with JSON: {"fidelity": 0.0-1.0, "outcome_match": bool,
"invented_success": bool, "rationale": "one or two sentences"}."""


def judge(command: str, real: str, simulated: str) -> dict:
    user = json.dumps({"command": command,
                       "real_output": (real or "")[:3000],
                       "simulated_output": (simulated or "")[:3000]}, indent=1)
    raw = sim._anthropic_with(JUDGE_MODEL, JUDGE_SYSTEM, user)
    return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="validate only the first N simulated commands (cheap dry run)")
    ap.add_argument("--json", default=None, help="write the full per-command report here")
    args = ap.parse_args()

    sim._load_secrets()
    if "ANTHROPIC_API_KEY" not in os.environ:
        return print("ANTHROPIC_API_KEY missing (add ~/.config/ptb/secrets.env)") or 2
    if not SANDBOX.exists():
        return print(f"sandbox missing: {SANDBOX}\nrun: python -m rollback.run.orchestrate "
                     "--bash-mode skip") or 2

    traj = config.TRAJECTORY
    events = ptbio.load_events(traj)
    grounding = sim.Grounding(traj)

    # map tool_use id -> real result, and gather ordered bash calls with ts
    results = {}
    for ev in events:
        for b in ev.get("blocks") or []:
            if b.get("type") == "tool_result":
                results[b.get("tool_use_id") or b.get("id")] = str(b.get("content", ""))
    t0 = next((_ts(e) for e in events if _ts(e) is not None), 0) or 0

    rows, ledger = [], []
    sim_calls = [tc for tc in ptbio.iter_tool_calls(events)
                 if tc.name.lower() == "bash"
                 and isinstance(tc.input.get("command"), str)
                 and sim.route_mode(tc.input["command"], SANDBOX) == "simulate"]
    if args.limit:
        sim_calls = sim_calls[: args.limit]
    print(f"validating {len(sim_calls)} simulated-routed commands "
          f"(of {traj.run_id})\n", flush=True)

    for n, tc in enumerate(sim_calls, 1):
        cmd = tc.input["command"]
        real = results.get(tc.tool_id, "")
        clock = (_ts(events[tc.idx]) or t0) - t0
        # real wall duration = delta to the next timestamped event
        nxt = next((_ts(events[j]) for j in range(tc.idx + 1, len(events))
                    if _ts(events[j]) is not None), None)
        real_dur = (nxt - (_ts(events[tc.idx]) or t0)) if nxt else None

        scripts = {}
        mm = sim._PY_INVOKE.search(cmd)
        if mm and (SANDBOX / mm.group(1)).exists():
            scripts[mm.group(1)] = (SANDBOX / mm.group(1)).read_text()[:12000]
        tool_timeout = (tc.input.get("timeout") or 120000) / 1000
        ctx = {"clock": clock, "budget": traj.num_hours * 3600,
               "tool_timeout_seconds": tool_timeout,
               "artifacts": {}, "recent": ledger[-5:],
               "grounding": grounding.similar(cmd, before_idx=tc.idx),
               "scripts": scripts}

        try:
            parsed = sim.run_simulation(traj, cmd, tc.input.get("description", ""), ctx)
            sim_out = parsed.get("output", "")
            sim_dur = parsed.get("observed_duration_seconds")
        except Exception as e:
            sim_out, sim_dur, parsed = f"<simulator failed: {e}>", None, {}

        rscore, sscore = _extract_score(real), _extract_score(sim_out)
        try:
            verdict = judge(cmd, real, sim_out)
        except Exception as e:
            verdict = {"fidelity": None, "error": str(e)[:200]}

        row = {
            "idx": tc.idx, "command": cmd[:160],
            "real_crashed": _looks_crashed(real), "sim_crashed": _looks_crashed(sim_out),
            "real_timed_out": bool(_TIMEOUT_RE.search(real)),
            "sim_timed_out": bool(_TIMEOUT_RE.search(sim_out)),
            "real_score": rscore, "sim_score": sscore,
            "score_abs_err": (abs(rscore - sscore) if rscore is not None and sscore is not None else None),
            "real_duration_s": round(real_dur) if real_dur else None,
            "sim_duration_s": sim_dur,
            "fidelity": verdict.get("fidelity"),
            "outcome_match": verdict.get("outcome_match"),
            "invented_success": verdict.get("invented_success"),
            "rationale": verdict.get("rationale", verdict.get("error", "")),
            "real_output_head": (real or "")[:200], "sim_output_head": (sim_out or "")[:200],
        }
        rows.append(row)
        # teacher forcing: next step sees the REAL output, not the simulated one
        ledger.append({"command": cmd, "real_output_head": (real or "")[:300],
                       "duration": real_dur})
        fid = row["fidelity"]
        flag = " ⚠INVENTED-SUCCESS" if row["invented_success"] else ""
        print(f"[{n}/{len(sim_calls)}] ev{tc.idx} fidelity={fid} "
              f"outcome_match={row['outcome_match']}{flag} :: {cmd[:70]}", flush=True)

    # ---- aggregate ---------------------------------------------------------
    fids = [r["fidelity"] for r in rows if isinstance(r["fidelity"], (int, float))]
    errs = [r["score_abs_err"] for r in rows if r["score_abs_err"] is not None]
    timeout_rows = [r for r in rows if r["real_timed_out"]]
    summary = {
        "run_id": traj.run_id, "n_validated": len(rows),
        "mean_fidelity": round(sum(fids) / len(fids), 3) if fids else None,
        "min_fidelity": min(fids) if fids else None,
        "outcome_match_rate": round(sum(1 for r in rows if r["outcome_match"]) / len(rows), 3) if rows else None,
        "invented_success_count": sum(1 for r in rows if r["invented_success"]),
        "mean_score_abs_err": round(sum(errs) / len(errs), 4) if errs else None,
        "n_with_comparable_score": len(errs),
        # the dominant real behavior here is the 120s foreground tool-timeout;
        # does the simulator reproduce it rather than inventing a completion?
        "n_real_timed_out": len(timeout_rows),
        "timeout_reproduced_rate": (round(sum(1 for r in timeout_rows if r["sim_timed_out"])
                                          / len(timeout_rows), 3) if timeout_rows else None),
        "judge_model": JUDGE_MODEL, "sim_model": sim.SIM_MODEL,
    }
    print("\n=== simulator validation ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if summary["invented_success_count"]:
        print("  ⚠ simulator invented success on:",
              [r["idx"] for r in rows if r["invented_success"]])

    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
        print(f"\nfull report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
