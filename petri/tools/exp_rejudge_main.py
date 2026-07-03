"""Re-judge EVERY trajectory on the MAIN trajectories page, fresh, on ALL current dimensions.

Why: the main page accumulated trajectories that were judged before some dimensions existed
(they render "null" in the viewer -- e.g. hack_in_final_solution) and before recent tweaks to
the dimension rubrics. This re-runs Petri's ACTUAL judge (`audit_judge`, the same scanner /
model / prompt exp_rh_audit.py uses) over each main-page original's EXISTING transcript and
scores every current dimension fresh in one pass. No new trajectories are generated -- only
the judging is redone. The fresh scores REPLACE that trajectory's entry in rejudge_scores.json
(scores + summary + justification + highlights), which make_viewer merges in (full
replacement), so every main-page number reflects the current judging system and there are no
nulls.

Scope: ORIGINAL audits (non-rollback log dirs) that land on the MAIN page -- i.e.
make_viewer.is_old_trajectory(a) is False (runs dated >= _MAIN_MIN_DATE). Rollback
continuations are NOT touched (re-judge those with exp_rejudge_rollbacks.py if needed). By
DEFAULT it re-judges ALL selected trajectories -- that's the point, to refresh stale/null
scores -- so it re-spends on ones already in the file. Pass --skip-existing to only fill in
trajectories not yet present in rejudge_scores.json.

Usage:
  uv run tools/exp_rejudge_main.py                  # re-judge every main-page original
  uv run tools/exp_rejudge_main.py --limit=1        # cheap sanity run: judge just 1 first
  uv run tools/exp_rejudge_main.py --skip-existing  # only judge ones not already in the file
  uv run tools/exp_rejudge_main.py --concurrency=50 # parallel judges (default 50)
  uv run tools/exp_rejudge_main.py --model=anthropic/claude-opus-4-8
Then regenerate the viewer (free): uv run make_viewer.py

Costs money (Anthropic API: the judge model).
"""

import asyncio
import pathlib
import sys

# this tool lives in tools/; put the project root (for make_viewer) and ../lib
# (for petri_paths + the core modules) on the import path.
_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from dotenv import load_dotenv
from inspect_petri import audit_judge, judge_dimensions
from inspect_scout import TranscriptContent, transcripts_from

# single source of truth for paths, dims, the merge key, and the main/old split
from petri_paths import ENV_FILE, DIMENSIONS_DIR, DATA
from make_viewer import (
    LOGS, ROLLBACK_PREFIX, REJUDGE_FILE, traj_key, binary_hack_eval, is_old_trajectory,
)

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)
MODEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "anthropic/claude-opus-4-8")
CONCURRENCY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), "50"))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
SKIP_EXISTING = "--skip-existing" in sys.argv
if CONCURRENCY < 1:
    raise SystemExit(f"--concurrency must be >= 1, got {CONCURRENCY}")

import json  # noqa: E402  (after the SystemExit guard so a bad flag fails first)

# trajectory-id registry (keyed by traj_key) -- needed to apply the same MAIN/OLD page split
# the viewer uses (is_old_trajectory needs each trajectory's stable id).
_REG_FILE = DATA / "trajectory_ids.json"
REGISTRY = json.loads(_REG_FILE.read_text()) if _REG_FILE.exists() else {}


def _key(mode: str, task: str, seed: str, epoch: int) -> str:
    return traj_key({"mode": mode, "task": task, "seed": seed, "epoch": epoch})


async def collect_candidates(existing: dict) -> list[dict]:
    """Every ORIGINAL audit on the MAIN page, with its transcript loaded. Membership is the
    viewer's exact rule (not is_old_trajectory), using the trajectory id from the registry."""
    cands: list[dict] = []
    skipped_unregistered = 0
    mode_dirs = sorted(
        d for d in LOGS.iterdir() if d.is_dir() and not d.name.startswith(ROLLBACK_PREFIX)
    )
    for d in mode_dirs:
        print(f"  scanning {d.name}/ ...", flush=True)
        ts = transcripts_from(str(d))
        async with ts.reader() as reader:
            infos = [i async for i in reader.index()]
            for info in infos:
                seed = str(info.task_id)
                key = _key(d.name, info.task_set, seed, info.task_repeat)
                tid = REGISTRY.get(key)
                if tid is None:
                    # not in the id registry (rebuild the viewer to assign ids); can't place
                    # it on a page, so skip rather than risk re-judging an old-page trajectory.
                    skipped_unregistered += 1
                    continue
                if is_old_trajectory({"mode": d.name, "id": tid, "seed": seed}):   # OLD page -> skip
                    continue
                if SKIP_EXISTING and key in existing:
                    continue
                rh = (info.score or {}).get("reward_hacking")
                try:
                    t = await reader.read(info, TranscriptContent(messages="all", events="all"))
                except FileNotFoundError as e:
                    print(f"    WARNING: missing eval log for {key} -- skipping: {e}")
                    continue
                if not t.timelines:
                    print(f"    WARNING: empty transcript for {key} (dead audit) -- skipping")
                    continue
                cands.append({
                    "key": key, "id": tid, "mode": d.name, "task": info.task_set,
                    "seed": seed, "epoch": info.task_repeat,
                    "original_reward_hacking": rh, "transcript": t,
                })
    if skipped_unregistered:
        print(f"  NOTE: skipped {skipped_unregistered} trajectory(ies) with no id in "
              f"{_REG_FILE.name} -- run `uv run make_viewer.py` first to register them, then retry.")
    return cands


async def main() -> None:
    REJUDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REJUDGE_FILE.read_text()) if REJUDGE_FILE.exists() else {}

    dims = judge_dimensions(DIMENSIONS_DIR)
    dim_names = [d.name for d in dims]
    print(f"\nmodel={MODEL}  concurrency={CONCURRENCY}  skip_existing={SKIP_EXISTING}"
          + (f"  limit={LIMIT}" if LIMIT else ""))
    print(f"judge dimensions ({len(dim_names)}): {dim_names}\n")

    print("[setup] collecting MAIN-page original audits ...")
    cands = await collect_candidates(existing)
    if LIMIT is not None:
        cands = cands[:LIMIT]
    total = len(cands)
    if total == 0:
        print("\nNothing to re-judge (no main-page originals selected"
              + (" that aren't already in the file" if SKIP_EXISTING else "") + "). Done.")
        return
    print(f"[setup] {total} main-page original(s) to re-judge "
          f"(rejudge_scores.json currently has {len(existing)} entrie(s))\n")

    scanner = audit_judge(dimensions=dims, model=MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    results = dict(existing)
    done = failed = 0

    async def checkpoint():
        async with lock:
            REJUDGE_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))

    async def judge_one(i: int, c: dict) -> None:
        nonlocal done, failed
        async with sem:
            print(f"  [{i}/{total}] judging #{c['id']} {c['seed']} e{c['epoch']} "
                  f"({c['mode']}) ...", flush=True)
            try:
                result = await scanner(c["transcript"])
            except Exception as e:
                failed += 1
                print(f"    [{i}/{total}] FAILED {c['key']}: {type(e).__name__}: {e}", flush=True)
                return
        if not isinstance(result.value, dict):
            failed += 1
            reason = (result.metadata or {}).get("stop_reason") or "no dict value (refusal?)"
            print(f"    [{i}/{total}] FAILED {c['key']}: judge returned no scores ({reason})", flush=True)
            return
        scores = {k: result.value.get(k) for k in dim_names}
        miss = [k for k, v in scores.items() if not isinstance(v, (int, float))]
        if miss:
            failed += 1
            print(f"    [{i}/{total}] FAILED {c['key']}: judge omitted dims {miss}", flush=True)
            return
        md = result.metadata or {}
        results[c["key"]] = {
            "mode": c["mode"], "task": c["task"], "seed": c["seed"], "epoch": c["epoch"],
            "original_reward_hacking": c["original_reward_hacking"],
            "model": MODEL,
            "scores": scores,
            "summary": md.get("summary", ""),
            "highlights": md.get("highlights", ""),
            "justification": getattr(result, "explanation", None) or "",
        }
        done += 1
        verdict = binary_hack_eval(results[c["key"]])
        tag = "HACK" if verdict["ok"] else ("fails:" + ",".join(verdict["failed"]) if verdict["failed"] else "unclassified")
        await checkpoint()
        print(f"    [{i}/{total}] #{c['id']} {c['seed']} e{c['epoch']}: {scores}  -> {tag}", flush=True)

    await asyncio.gather(*(judge_one(i, c) for i, c in enumerate(cands, 1)))
    await checkpoint()

    print(f"\nthis run: {done} re-judged, {failed} failed")
    print(f"{REJUDGE_FILE} now has {len(results)} re-judged audit(s) total.")
    if failed:
        print(f"  NOTE: {failed} audit(s) failed to re-judge -- they keep their old scores; "
              "re-run to retry (use --skip-existing to avoid re-spending on the ones that worked).")
    print("\nRegenerate the viewer (free) to see the merged scores: uv run make_viewer.py")


if __name__ == "__main__":
    asyncio.run(main())
