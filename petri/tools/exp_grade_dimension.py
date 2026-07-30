"""One-off: grade ONE judge dimension on ONLY the current full reward hacks.

Use this to backfill a single newly-added dimension onto the EXISTING full reward
hacks so you can look at it right now, without re-judging everything or re-spending on
the other dimensions. New audit runs already score every dimension in dimensions/
automatically (auto-loaded by judge_dimensions); this is just for looking at one
dimension on trajectories that were judged before it existed.

What it does: finds every ORIGINAL audit (non-rollback) that is a *full* reward hack
under the committed binary definition (viewer.is_hack_binary, on the
rejudge-merged scores -- exactly the "Reward hacks" table in the viewer), then runs
Petri's real judge (`audit_judge`, default claude-opus-4-8) over each transcript with
JUST the one dimension you name, and records the score + the judge's justification.

Output: mats-local/petri/dim_scores/<dim>.json, keyed by traj_key. This is a SEPARATE
sidecar per dimension -- it does NOT touch rejudge_scores.json and does NOT feed the
binary hack definition, so it cannot change any trajectory's category. Incremental
(skips trajectories already in the file unless --force), checkpointed after each call.

Usage:
  uv run tools/exp_grade_dimension.py --dim=hallucination            # grade all full hacks
  uv run tools/exp_grade_dimension.py --dim=honest_path_ambiguity      # any dim in dimensions/
  uv run tools/exp_grade_dimension.py --dim=hallucination --limit=1  # cheap sanity run
  uv run tools/exp_grade_dimension.py --dim=hallucination --concurrency=50  # default 50
  uv run tools/exp_grade_dimension.py --dim=hallucination --force    # re-grade everything
  uv run tools/exp_grade_dimension.py --dim=hallucination --model=anthropic/claude-opus-4-8

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
from judge_models import resolve_judge
from viewer import (
    LOGS, ROLLBACK_PREFIX, traj_key, is_hack_binary, _load_rejudge,
)

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)
DIM = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dim=")), None)
MODEL = resolve_judge(next((a.split("=", 1)[1] for a in sys.argv
                            if a.startswith("--model=")), None))
CONCURRENCY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), "50"))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
FORCE = "--force" in sys.argv
if not DIM:
    raise SystemExit(
        "--dim=<name> is REQUIRED (a dimension stem from dimensions/, e.g. "
        "--dim=hallucination or --dim=honest_path_ambiguity)."
    )
if CONCURRENCY < 1:
    raise SystemExit(f"--concurrency must be >= 1, got {CONCURRENCY}")

import json  # noqa: E402  (after the SystemExit guards so a bad flag fails first)

OUT_FILE = DATA / "dim_scores" / f"{DIM}.json"


def _ids() -> dict:
    """traj_key -> stable trajectory id (the integer shown in the viewer), for display."""
    f = DATA / "trajectory_ids.json"
    return json.loads(f.read_text()) if f.exists() else {}


async def collect_full_hacks(existing: dict) -> list[dict]:
    """Every ORIGINAL audit that is a FULL reward hack (rejudge-merged scores), with its
    transcript loaded. Mirrors how viewer decides the 'Reward hacks' table: take the
    rejudged scores when present, else the eval-log scores, then is_hack_binary()."""
    rejudge = _load_rejudge()
    ids = _ids()
    cands: list[dict] = []
    mode_dirs = sorted(
        d for d in LOGS.iterdir() if d.is_dir() and not d.name.startswith(ROLLBACK_PREFIX)
    )
    for d in mode_dirs:
        print(f"  scanning {d.name}/ ...", flush=True)
        ts = transcripts_from(str(d))
        async with ts.reader() as reader:
            infos = [i async for i in reader.index()]
            for info in infos:
                key = traj_key({
                    "mode": d.name, "task": info.task_set,
                    "seed": str(info.task_id), "epoch": info.task_repeat,
                })
                # rejudge entry (if any) supersedes the eval-log score, same as the viewer
                scores = rejudge[key]["scores"] if key in rejudge else (info.score or {})
                if not is_hack_binary({"scores": scores}):
                    continue  # not a full reward hack -> out of scope
                if key in existing and not FORCE:
                    continue
                try:
                    t = await reader.read(info, TranscriptContent(messages="all", events="all"))
                except FileNotFoundError as e:
                    print(f"    WARNING: missing eval log for {key} -- skipping: {e}")
                    continue
                if not t.timelines:
                    print(f"    WARNING: empty transcript for {key} (dead audit) -- skipping")
                    continue
                cands.append({
                    "key": key, "id": ids.get(key), "mode": d.name, "task": info.task_set,
                    "seed": str(info.task_id), "epoch": info.task_repeat,
                    "reward_hacking": scores.get("reward_hacking"), "transcript": t,
                })
    return cands


async def main() -> None:
    dims = [x for x in judge_dimensions(DIMENSIONS_DIR) if x.name == DIM]
    if len(dims) != 1:
        avail = sorted(x.name for x in judge_dimensions(DIMENSIONS_DIR))
        raise SystemExit(
            f"--dim={DIM!r} did not match exactly one dimension in {DIMENSIONS_DIR}.\n"
            f"available: {avail}"
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    print(f"\ndimension={DIM}  model={MODEL}  concurrency={CONCURRENCY}  force={FORCE}"
          + (f"  limit={LIMIT}" if LIMIT else ""))
    print("[setup] collecting FULL reward hacks (is_hack_binary on rejudge-merged scores) ...")
    cands = await collect_full_hacks(existing)
    if LIMIT is not None:
        cands = cands[:LIMIT]
    total = len(cands)
    if total == 0:
        print(f"\nNothing to grade: every full reward hack is already in {OUT_FILE} "
              "(use --force to redo). Done.")
        return
    print(f"[setup] {total} full reward hack(s) to grade "
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
                  f"({c['mode']}, rh={c['reward_hacking']}) ...", flush=True)
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
    print(f"{OUT_FILE} now has {len(results)} graded full hack(s).")
    if vals:
        hist = {s: sum(1 for v in vals if round(v) == s) for s in range(1, 11)}
        dist = "  ".join(f"{s}:{n}" for s, n in hist.items() if n)
        print(f"\n{DIM} over {len(vals)} full hack(s):")
        print(f"  mean={sum(vals)/len(vals):.2f}   min={min(vals)}   max={max(vals)}")
        print(f"  distribution (score:count):  {dist}")
        print(f"\n  all full hacks by {DIM} (highest first):")
        for e in sorted(graded, key=lambda e: e[DIM], reverse=True):
            tag = f"#{e['id']}" if e.get("id") is not None else e["seed"]
            print(f"    {DIM}={e[DIM]}  rh={e.get('reward_hacking')}  {tag}  "
                  f"{e['seed']} e{e['epoch']} ({e['mode']})")
    if failed:
        print(f"\n  NOTE: {failed} failed to grade -- re-run to retry (incremental).")
    print(f"\nScores written to {OUT_FILE} (not merged into the viewer).")


if __name__ == "__main__":
    asyncio.run(main())
