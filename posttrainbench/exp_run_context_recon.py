"""Run the context-reconstruction continuation on every reward-hack trajectory.

PAID: each trajectory launches the scaffold CLI (claude), which calls model APIs
(~$0.3-0.5 per probe, from the two existing probes).

One continuation per trajectory, cut at the END (--turn end: the full completed
trajectory incl. the final assistant message). Clean-ending trajectories need
no resume bridge; trajectories that genuinely died on an unanswered tool call
snap before that dangling action and retain the structural CLI bridge. The probe
asks the model to report its own context
(exp_probe_context.DEFAULT_PROBE). Outputs land in the "context reconstruction
tests" campaign dir (the current campaign named in lib/runs.py), which the viewer's
context-reconstruction window reads. --rerun re-probes already-probed
trajectories; the viewer shows the latest cut per run, so re-runs supersede
old probes (and clear the green fixed-in-code issue tags).

Coverage: only claude / claude_non_api (and opencode, if its CLI is installed)
can be resumed. codex and qwen3max are refused (lossy data / no CLI) and are
skipped with a printed reason — those rows stay greyed in the viewer with a
"can't reconstruct" caveat. Already-run trajectories are skipped (idempotent),
so re-running only fills gaps.

Probes run in parallel (4 workers by default). Use --workers N to control API
concurrency, or --workers 1 for the old sequential behavior. Child probe output
streams directly to the terminal, so lines from concurrent probes may interleave.

  uv run python posttrainbench/exp_run_context_recon.py            # run all supported
  uv run python posttrainbench/exp_run_context_recon.py --scaffold claude
  uv run python posttrainbench/exp_run_context_recon.py --scaffold claude --workers 8
  uv run python posttrainbench/exp_run_context_recon.py --dry-run  # just show the plan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import runs  # noqa: E402

CAMPAIGN = runs.CONTEXT_RECON_CAMPAIGN
PROBE_SCRIPT = Path(__file__).resolve().parent / "exp_probe_context.py"
CAMPAIGN_DIR = runs.OUT_ROOT / CAMPAIGN


def _support(traj) -> tuple[bool, str]:
    """(runnable, reason-if-not), including campaign-level deferrals."""
    if traj.run_id in runs.CONTEXT_RECON_DEFERRED_RUN_IDS:
        return False, "manually deferred (see the context-reconstruction viewer)"
    if traj.scaffold == "claude":
        return True, ""
    if traj.scaffold == "opencode":
        if not runs.opencode_cmd():
            return False, "opencode CLI not installed (pinned install expected in mats-local/tools/)"
        if not runs.env_has("OPENCODE_API_KEY"):
            return False, "zen gateway key missing (add OPENCODE_API_KEY to mats/.env)"
        return True, ""
    if traj.scaffold == "codex":
        return False, "codex traces lossy (no rollout/patch bodies) — can't reconstruct"
    if traj.scaffold == "qwen3max":
        return False, ("qwen3max = Claude Code + DashScope endpoint; resumable in "
                       "principle but needs a DashScope key + env overrides (not set up)")
    return False, f"unsupported scaffold {traj.scaffold}"


def _already_run(run_id: str) -> bool:
    if not CAMPAIGN_DIR.exists():
        return False
    for d in CAMPAIGN_DIR.iterdir():
        fp = d / "fidelity.json"
        if not (d.is_dir() and d.name.startswith(run_id + "__ev") and fp.exists()):
            continue
        try:
            if json.loads(fp.read_text()).get("probe_answer"):
                return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _run_one(index: int, total: int, traj, timeout: int) -> tuple[str, int]:
    """Launch one independent probe; its child output streams to the terminal."""
    print(f"[{index}/{total}] START {traj.scaffold} · {traj.run_id}", flush=True)
    proc = subprocess.run(
        [sys.executable, str(PROBE_SCRIPT), "--trajectory", traj.run_id,
         "--turn", "end", "--campaign", CAMPAIGN, "--timeout", str(timeout)],
        cwd=PROBE_SCRIPT.parent)
    return traj.run_id, proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (what would run / skip) and exit")
    ap.add_argument("--rerun", action="store_true",
                    help="also re-probe trajectories that already have a probe "
                         "(new probes supersede old ones in the viewer)")
    ap.add_argument("--scaffold", default="all",
                    choices=("all", "claude", "opencode", "codex", "qwen3max"),
                    help="limit the campaign to one scaffold (default: all)")
    ap.add_argument("--workers", type=int, default=4,
                    help="number of probes to run concurrently (default: 4; "
                         "use 1 for sequential execution)")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("--workers must be at least 1")

    trajs = sorted(runs.reward_hacks(), key=lambda t: t.run_id)
    if args.scaffold != "all":
        trajs = [t for t in trajs if t.scaffold == args.scaffold]
    runnable, unsupported, done = [], [], []
    for t in trajs:
        ok, reason = _support(t)
        if not ok:
            unsupported.append((t, reason))
        elif _already_run(t.run_id) and not args.rerun:
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

    workers = min(args.workers, len(runnable))
    print(f"\nrunning {len(runnable)} continuations into {CAMPAIGN}/ "
          f"with {workers} parallel worker(s) "
          f"(cut at the end of the trajectory, --turn end)\n", flush=True)
    ok_n = fail_n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, i, len(runnable), t, args.timeout)
                   for i, t in enumerate(runnable, 1)]
        for future in as_completed(futures):
            try:
                run_id, returncode = future.result()
            except Exception as exc:
                fail_n += 1
                print(f"    FAILED launcher exception: {exc} "
                      f"({fail_n} failed so far)", flush=True)
                continue
            if returncode == 0:
                ok_n += 1
                print(f"    DONE {run_id} ({ok_n} ok, {fail_n} failed)", flush=True)
            else:
                fail_n += 1
                print(f"    FAILED {run_id} exit {returncode} "
                      f"({ok_n} ok, {fail_n} failed)", flush=True)

    print(f"\nfinished: {ok_n} ok, {fail_n} failed, {len(done)} pre-existing, "
          f"{len(unsupported)} unsupported. Results in {CAMPAIGN_DIR}")


if __name__ == "__main__":
    main()
