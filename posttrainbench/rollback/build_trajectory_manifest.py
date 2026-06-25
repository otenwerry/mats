"""Auto-enumerate ALL reward-hack trajectories into a small manifest, so the
rollback infra can target all ~30 without hand-writing each Trajectory.

FREE + offline. Scans the adjudication blocks (posttrainbench/highlights/*.json,
the canonical `final.is_reward_hack` set) and the originals' viewer_data, and
derives every baseline field we CAN derive mechanically:

  run_id, experiment, run_name, agent (experiment family), scaffold (which
  resume engine), auth (api|oauth|none), policy_model_recorded, model_to_train,
  benchmark_id, num_hours, default_cut (= final.first_hack_event), needs_prep
  (= final.trainings_before_first_hack > 0), trainings_before_first_hack,
  trace_format.

Writes rollback/trajectory_manifest.json. config.py loads this (fast) and merges
the hand-curated TRAJECTORIES on top for fields we CANNOT derive (the exact
policy_model_continuation route when a gateway model id drifted, the precise
prep_commands, eval_model_dir, eval_limit, hand-validated cut boundaries).

Re-run whenever the highlights set changes:
    uv run python -m rollback.build_trajectory_manifest
    uv run python -m rollback.build_trajectory_manifest --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
PTB_TOOLING = PKG_DIR.parent
SUPERMATS = PKG_DIR.parents[2]
HIGHLIGHTS = PTB_TOOLING / "highlights"
VIEWER_DATA = Path(os.environ.get(
    "PTB_DATA", SUPERMATS / "mats-local" / "posttrainbench" / "viewer_data"))
MANIFEST = PKG_DIR / "trajectory_manifest.json"

# trace_format -> the resume ENGINE family. claude_code covers claude (API),
# claude_non_api (OAuth) AND qwen3max — they all persist a Claude-Code-style
# session, so one recon/engine path handles them (auth/prompt differ per family).
_ENGINE_BY_TRACE = {"opencode": "opencode", "claude_code": "claude", "codex": "codex"}


def _family(experiment: str) -> str:
    """Coarse agent family from the experiment dir name (drives auth + which
    base scaffold actually generated the run)."""
    if experiment.startswith("opencode"):
        return "opencode"
    if experiment.startswith("codex"):
        return "codex"
    if experiment.startswith("claude_non_api"):
        return "claude_non_api"
    if experiment.startswith("claude"):
        return "claude"
    if experiment.startswith("qwen3max"):
        return "qwen3max"
    return experiment.split("_", 1)[0]


def _auth(family: str) -> str:
    # how the ORIGINAL run authenticated the policy — the faithful continuation
    # uses the same. opencode -> our OpenRouter key; *_non_api -> subscription
    # (Claude OAuth / ChatGPT auth.json); plain claude -> Anthropic API key.
    return {
        "opencode": "api",          # OpenRouter API key (gateway)
        "claude": "api",            # ANTHROPIC_API_KEY
        "claude_non_api": "oauth",  # CLAUDE_CODE_OAUTH_TOKEN
        "codex": "oauth",           # ChatGPT auth.json (all 7 RH codex are non_api)
        "qwen3max": "api",          # qwen api (DashScope/OpenRouter)
    }.get(family, "api")


def _parse_run_name(run_name: str) -> tuple[str, str]:
    """run_name = '<benchmark>_<hf/org_model>_<jobid>'. Returns (benchmark, model).
    The model's org/name '/' was flattened to '_' — restore the FIRST '_'."""
    body = run_name.rsplit("_", 1)[0]            # strip trailing numeric job id
    benchmark, _, model_flat = body.partition("_")
    model = model_flat.replace("_", "/", 1)      # google_gemma-3-4b-pt -> google/gemma-3-4b-pt
    return benchmark, model


def _viewer_index_row(run_id: str) -> dict:
    p = VIEWER_DATA / f"{run_id}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("index_row") or {}
    except (json.JSONDecodeError, OSError):
        return {}


def build() -> list[dict]:
    rows = []
    for f in sorted(HIGHLIGHTS.glob("*.json")):
        if f.name.startswith("rollback_"):
            continue
        try:
            fin = (json.loads(f.read_text()).get("final") or {})
        except json.JSONDecodeError:
            continue
        if not fin.get("is_reward_hack"):
            continue
        run_id = f.stem
        # split on the LAST '__': some experiment names themselves contain '__'
        # (e.g. claude_non_api_claude-opus-4-6_1m__10h_run1), while run_name
        # (<benchmark>_<model>_<jobid>) never does.
        experiment, _, run_name = run_id.rpartition("__")
        if not run_name:
            continue
        ir = _viewer_index_row(run_id)
        bench_rn, model_rn = _parse_run_name(run_name)
        family = _family(experiment)
        trace_format = ir.get("trace_format") or ""
        scaffold = _ENGINE_BY_TRACE.get(trace_format, family)
        tbh = fin.get("trainings_before_first_hack")
        rows.append({
            "run_id": run_id,
            "experiment": experiment,
            "run_name": run_name,
            "agent": family,
            "scaffold": scaffold,
            "auth": _auth(family),
            "trace_format": trace_format,
            "policy_model_recorded": ir.get("agent_model") or "",
            # index_row.trained_model is FLATTENED (google_gemma-3-4b-pt); the
            # run_name parse restores the org '/' for a valid HF id.
            "model_to_train": model_rn,
            "benchmark_id": ir.get("benchmark") or bench_rn,
            "num_hours": int(ir.get("time_budget_h") or 10),
            "default_cut": fin.get("first_hack_event"),
            "trainings_before_first_hack": tbh,
            "needs_prep": bool(tbh and tbh > 0),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = build()
    # surface anything we couldn't fully derive (a flag, not a silent gap)
    incomplete = [r["run_id"] for r in rows
                  if r["default_cut"] is None or not r["policy_model_recorded"]]
    print(f"enumerated {len(rows)} reward-hack trajectories")
    import collections
    print("  by scaffold:", dict(collections.Counter(r["scaffold"] for r in rows)))
    print("  needs_prep:", sum(r["needs_prep"] for r in rows),
          "/ prep-free:", sum(not r["needs_prep"] for r in rows))
    if incomplete:
        print(f"  WARNING: {len(incomplete)} with missing cut/model (need a "
              f"curated override): {incomplete}")
    if args.dry_run:
        print("(dry-run; manifest not written)")
        return
    MANIFEST.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
