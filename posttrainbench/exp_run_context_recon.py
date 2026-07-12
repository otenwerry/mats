"""Run the context-reconstruction continuation on every reward-hack trajectory.

PAID: each trajectory launches the scaffold CLI (claude), which calls model APIs
(~$0.3-0.5 per probe, from the two existing probes).

One continuation per trajectory, cut at the END (--turn last), asking the model
to report its own context (exp_probe_context.DEFAULT_PROBE). Outputs land in the
"context reconstruction tests" campaign dir (probes_context_recon/), which the
viewer's context-reconstruction window reads.

Coverage: only claude / claude_non_api (and opencode, if its CLI is installed)
can be resumed. codex and qwen3max are refused (lossy data / no CLI) and are
skipped with a printed reason — those rows stay greyed in the viewer with a
"can't reconstruct" caveat. Already-run trajectories are skipped (idempotent),
so re-running only fills gaps.

  uv run python posttrainbench/exp_run_context_recon.py            # run all supported
  uv run python posttrainbench/exp_run_context_recon.py --dry-run  # just show the plan
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import runs  # noqa: E402

CAMPAIGN = "probes_context_recon"
PROBE_SCRIPT = Path(__file__).resolve().parent / "exp_probe_context.py"
CAMPAIGN_DIR = runs.OUT_ROOT / CAMPAIGN


def _support(traj) -> tuple[bool, str]:
    """(runnable, reason-if-not) — mirrors viewer._probe_support."""
    if traj.scaffold == "claude":
        return True, ""
    if traj.scaffold == "opencode":
        if shutil.which("opencode"):
            return True, ""
        return False, "opencode CLI not installed (needs opencode-ai + provider auth)"
    if traj.scaffold == "codex":
        return False, "codex traces lossy (no rollout/patch bodies) — can't reconstruct"
    if traj.scaffold == "qwen3max":
        return False, "no qwen CLI to resume"
    return False, f"unsupported scaffold {traj.scaffold}"


def _already_run(run_id: str) -> bool:
    if not CAMPAIGN_DIR.exists():
        return False
    return any(d.name.startswith(run_id + "__ev") and (d / "fidelity.json").exists()
               for d in CAMPAIGN_DIR.iterdir() if d.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (what would run / skip) and exit")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    trajs = sorted(runs.reward_hacks(), key=lambda t: t.run_id)
    runnable, unsupported, done = [], [], []
    for t in trajs:
        ok, reason = _support(t)
        if not ok:
            unsupported.append((t, reason))
        elif _already_run(t.run_id):
            done.append(t)
        else:
            runnable.append(t)

    print(f"{len(trajs)} reward-hack trajectories total")
    print(f"  {len(runnable)} to run · {len(done)} already done · "
          f"{len(unsupported)} can't run")
    for t, reason in unsupported:
        print(f"    SKIP  {t.scaffold:9} {t.run_id[:52]:52}  ({reason})")
    if args.dry_run or not runnable:
        if not runnable:
            print("nothing to run.")
        return

    print(f"\nrunning {len(runnable)} continuations into {CAMPAIGN}/ "
          f"(cut at end, --turn last)\n")
    ok_n = fail_n = 0
    for i, t in enumerate(runnable, 1):
        print(f"[{i}/{len(runnable)}] {t.scaffold} · {t.run_id}")
        proc = subprocess.run(
            [sys.executable, str(PROBE_SCRIPT), "--trajectory", t.run_id,
             "--turn", "last", "--campaign", CAMPAIGN, "--timeout", str(args.timeout)],
            cwd=PROBE_SCRIPT.parent)
        if proc.returncode == 0:
            ok_n += 1
            print(f"    done ({ok_n} ok so far)")
        else:
            fail_n += 1
            print(f"    FAILED exit {proc.returncode} ({fail_n} failed so far)")

    print(f"\nfinished: {ok_n} ok, {fail_n} failed, {len(done)} pre-existing, "
          f"{len(unsupported)} unsupported. Results in {CAMPAIGN_DIR}")


if __name__ == "__main__":
    main()
