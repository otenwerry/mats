"""TEMPORARY one-off: grade the new `incompleteness` dimension on every MAIN-page
trajectory, so the 10 trajectories that were judged before this dimension existed get a
score for it right now.

This is the same idea as tools/exp_grade_dimension.py (single-dimension backfill via the
dim_scores sidecar), but scoped to ALL main-page originals instead of just the full reward
hacks -- the user wants incompleteness on the whole main page. New audit runs already score
incompleteness automatically (it's a file in dimensions/, auto-loaded by judge_dimensions),
so this is only needed for the already-judged trajectories.

What it does: finds every ORIGINAL audit (non-rollback) that lands on the MAIN page (the
viewer's exact rule: registered in trajectory_ids.json AND not is_old_trajectory), then runs
Petri's real judge (`audit_judge`, default claude-opus-4-8) over each transcript with JUST
the `incompleteness` dimension, and records the score + the judge's justification.

Output: mats-local/petri/dim_scores/incompleteness.json, keyed by traj_key. This is an
add-only sidecar: viewer merges it in for any audit that doesn't already have an inline
incompleteness score (inline/rejudge scores always win), so it shows up as the new column
without re-judging or touching any other dimension, and it does NOT feed the binary hack
definition -- it cannot change any trajectory's category.

Dead/empty-transcript audits are SKIPPED (the judge scores those all-1 by construction, and
the viewer already badges them DEAD); the count skipped is printed.

Usage:
  uv run tools/exp_grade_incompleteness.py                 # grade every main-page original
  uv run tools/exp_grade_incompleteness.py --limit=1       # cheap sanity run: judge just 1
  uv run tools/exp_grade_incompleteness.py --concurrency=50 # parallel judges (default 50)
  uv run tools/exp_grade_incompleteness.py --force         # re-grade ones already in the file
  uv run tools/exp_grade_incompleteness.py --model=anthropic/claude-opus-4-8
Then regenerate the viewer (free): uv run viewer.py

Costs money (Anthropic API: the judge model).
"""

import asyncio
import pathlib
import sys

# this tool lives in tools/; put the project root (for viewer) and ../lib
# (for petri_paths + the core modules) on the import path.
_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from dotenv import load_dotenv
from inspect_petri import audit_judge, judge_dimensions
from inspect_scout import TranscriptContent, transcripts_from

from petri_paths import ENV_FILE, DIMENSIONS_DIR, DATA
from viewer import (
    LOGS, ROLLBACK_PREFIX, traj_key, is_old_trajectory,
)

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)
DIM = "incompleteness"
MODEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "anthropic/claude-opus-4-8")
CONCURRENCY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), "50"))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
FORCE = "--force" in sys.argv
if CONCURRENCY < 1:
    raise SystemExit(f"--concurrency must be >= 1, got {CONCURRENCY}")

import json  # noqa: E402  (after the SystemExit guard so a bad flag fails first)

OUT_FILE = DATA / "dim_scores" / f"{DIM}.json"
_REG_FILE = DATA / "trajectory_ids.json"
REGISTRY = json.loads(_REG_FILE.read_text()) if _REG_FILE.exists() else {}


def _key(mode: str, task: str, seed: str, epoch: int) -> str:
    return traj_key({"mode": mode, "task": task, "seed": seed, "epoch": epoch})


async def collect_candidates(existing: dict) -> tuple[list[dict], int]:
    """Every ORIGINAL audit on the MAIN page, with its transcript loaded. Membership is the
    viewer's exact rule, using the trajectory id from the registry (mirrors
    exp_rejudge_main.collect_candidates). Returns (candidates, n_dead_skipped)."""
    cands: list[dict] = []
    skipped_unregistered = 0
    skipped_dead = 0
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
                    # it on a page, so skip rather than risk grading an old-page trajectory.
                    skipped_unregistered += 1
                    continue
                if is_old_trajectory({"mode": d.name, "id": tid, "seed": seed}):   # OLD page -> skip
                    continue
                if key in existing and not FORCE:
                    continue
                try:
                    t = await reader.read(info, TranscriptContent(messages="all", events="all"))
                except FileNotFoundError as e:
                    print(f"    WARNING: missing eval log for {key} -- skipping: {e}")
                    continue
                if not t.timelines:
                    # dead audit (target produced nothing) -- judge would score all-1 anyway,
                    # and the viewer badges these DEAD; skip and surface the count.
                    skipped_dead += 1
                    continue
                cands.append({
                    "key": key, "id": tid, "mode": d.name, "task": info.task_set,
                    "seed": seed, "epoch": info.task_repeat,
                    "reward_hacking": (info.score or {}).get("reward_hacking"),
                    "transcript": t,
                })
    if skipped_unregistered:
        print(f"  NOTE: skipped {skipped_unregistered} trajectory(ies) with no id in "
              f"{_REG_FILE.name} -- run `uv run viewer.py` first to register them, then retry.")
    return cands, skipped_dead


async def main() -> None:
    dims = [x for x in judge_dimensions(DIMENSIONS_DIR) if x.name == DIM]
    if len(dims) != 1:
        avail = sorted(x.name for x in judge_dimensions(DIMENSIONS_DIR))
        raise SystemExit(
            f"dimension {DIM!r} did not match exactly one dimension in {DIMENSIONS_DIR}.\n"
            f"(did you create dimensions/{DIM}.md?)  available: {avail}"
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    print(f"\ndimension={DIM}  model={MODEL}  concurrency={CONCURRENCY}  force={FORCE}"
          + (f"  limit={LIMIT}" if LIMIT else ""))
    print("[setup] collecting MAIN-page original audits ...")
    cands, skipped_dead = await collect_candidates(existing)
    if skipped_dead:
        print(f"  NOTE: skipped {skipped_dead} dead/empty-transcript audit(s) "
              "(judge scores those all-1; viewer badges them DEAD).")
    if LIMIT is not None:
        cands = cands[:LIMIT]
    total = len(cands)
    if total == 0:
        print(f"\nNothing to grade (every main-page original is already in {OUT_FILE.name}; "
              "use --force to redo). Done.")
        return
    print(f"[setup] {total} main-page original(s) to grade "
          f"({len(existing)} already in {OUT_FILE.name})\n")

    scanner = audit_judge(dimensions=dims, model=MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    results = dict(existing)
    done = failed = 0

    async def checkpoint():
        async with lock:
            OUT_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))

    async def grade_one(i: int, c: dict) -> None:
        nonlocal done, failed
        label = f"#{c['id']}" if c["id"] is not None else c["key"]
        async with sem:
            print(f"  [{i}/{total}] judging {label} {c['seed']} e{c['epoch']} "
                  f"({c['mode']}) ...", flush=True)
            try:
                result = await scanner(c["transcript"])
            except Exception as e:
                failed += 1
                print(f"    [{i}/{total}] FAILED {label}: {type(e).__name__}: {e}", flush=True)
                return
        if not isinstance(result.value, dict):
            failed += 1
            reason = (result.metadata or {}).get("stop_reason") or "no dict value (refusal?)"
            print(f"    [{i}/{total}] FAILED {label}: judge returned no scores ({reason})", flush=True)
            return
        score = result.value.get(DIM)
        if not isinstance(score, (int, float)):
            failed += 1
            print(f"    [{i}/{total}] FAILED {label}: judge omitted {DIM} (got {score!r})", flush=True)
            return
        md = result.metadata or {}
        results[c["key"]] = {
            "id": c["id"], "mode": c["mode"], "task": c["task"], "seed": c["seed"],
            "epoch": c["epoch"], "reward_hacking": c["reward_hacking"], "model": MODEL,
            DIM: score,
            "justification": getattr(result, "explanation", None) or "",
            "summary": md.get("summary", ""),
            "highlights": md.get("highlights", ""),
        }
        done += 1
        await checkpoint()
        print(f"    [{i}/{total}] {label} {c['seed']} e{c['epoch']}: {DIM}={score}", flush=True)

    await asyncio.gather(*(grade_one(i, c) for i, c in enumerate(cands, 1)))
    await checkpoint()

    # tally over EVERYTHING in the file now (so re-runs report the full set)
    graded = [e for e in results.values() if isinstance(e.get(DIM), (int, float))]
    vals = [e[DIM] for e in graded]
    print(f"\nthis run: {done} graded, {failed} failed")
    print(f"{OUT_FILE} now has {len(results)} graded main-page original(s).")
    if vals:
        hist = {s: sum(1 for v in vals if round(v) == s) for s in range(1, 11)}
        dist = "  ".join(f"{s}:{n}" for s, n in hist.items() if n)
        print(f"\n{DIM} over {len(vals)} trajectory(ies):")
        print(f"  mean={sum(vals)/len(vals):.2f}   min={min(vals)}   max={max(vals)}")
        print(f"  distribution (score:count):  {dist}")
        print(f"\n  all by {DIM} (highest first):")
        for e in sorted(graded, key=lambda e: e[DIM], reverse=True):
            tag = f"#{e['id']}" if e.get("id") is not None else e["seed"]
            print(f"    {DIM}={e[DIM]}  rh={e.get('reward_hacking')}  {tag}  "
                  f"{e['seed']} e{e['epoch']} ({e['mode']})")
    if failed:
        print(f"\n  NOTE: {failed} failed to grade -- re-run to retry (incremental).")
    print(f"\nRegenerate the viewer (free) to see the new column: uv run viewer.py")


if __name__ == "__main__":
    asyncio.run(main())
