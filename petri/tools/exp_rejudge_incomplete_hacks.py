"""Re-judge the degenerate/nudged hacks that are missing at least one current dimension.

Why this exists: we re-judged the FULL hacks on the current 7-dim set (exp_rejudge_rh.py
--full-hacks-only), but the degenerate and nudged hacks were judged earlier on a smaller
set, so they're missing the newer dims (hallucination, honest_path_ambiguity) and show "null"
in the viewer. This re-runs Petri's real judge (`audit_judge`, same model claude-opus-4-8)
over each such trajectory with ALL current dimensions, so their scores are complete.

Scope: every ORIGINAL audit (non-rollback) that is currently categorized `degenerate` or
`nudged` (make_viewer.hack_category on the rejudge-merged scores -- exactly those viewer
sections) AND is missing >= 1 of the current judge dimensions. Trajectories already complete
are skipped, so this is naturally idempotent: re-running only picks up whatever is still
incomplete (e.g. anything that failed to judge last time).

Output: merged into mats-local/petri/rejudge_scores.json (same file + format as
exp_rejudge_rh.py: a full-replacement entry per trajectory with all current scores +
summary/justification/highlights). make_viewer's load_mode merges it in, so the viewer,
exp_annotate_hacks, and rollback selection all see one consistent judge pass. Incremental
and checkpointed after every trajectory. NOTE: re-judging is a fresh all-dims pass, so a
trajectory may change category (e.g. a nudged run whose validity dim now passes can become
a full hack) -- that's expected.

Usage:
  uv run tools/exp_rejudge_incomplete_hacks.py --dry-run   # FREE: list what would re-judge
  uv run tools/exp_rejudge_incomplete_hacks.py             # re-judge them (all current dims)
  uv run tools/exp_rejudge_incomplete_hacks.py --limit=1   # cheap sanity run: judge just 1
  uv run tools/exp_rejudge_incomplete_hacks.py --concurrency=50
  uv run tools/exp_rejudge_incomplete_hacks.py --model=anthropic/claude-opus-4-8
Then regenerate the viewer (free): uv run make_viewer.py

Costs money (Anthropic API: the judge model), except --dry-run.
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

from petri_paths import ENV_FILE, DIMENSIONS_DIR
from make_viewer import (
    LOGS, ROLLBACK_PREFIX, REJUDGE_FILE, traj_key, hack_category, _load_rejudge,
)

# the categories we backfill (the viewer's degenerate + nudged hack sections)
TARGET_CATEGORIES = {"degenerate", "nudged"}

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)
MODEL = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "anthropic/claude-opus-4-8")
CONCURRENCY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), "50"))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
DRY_RUN = "--dry-run" in sys.argv
if CONCURRENCY < 1:
    raise SystemExit(f"--concurrency must be >= 1, got {CONCURRENCY}")

import json  # noqa: E402  (after the SystemExit guard so a bad flag fails first)

_ids_cache = None


def _ids() -> dict:
    """traj_key -> stable trajectory id (the integer shown in the viewer), for display."""
    global _ids_cache
    if _ids_cache is None:
        f = (REJUDGE_FILE.parent / "trajectory_ids.json")
        _ids_cache = json.loads(f.read_text()) if f.exists() else {}
    return _ids_cache


async def collect_candidates(rejudge: dict, dim_names: list[str], load_transcripts: bool) -> list[dict]:
    """Every ORIGINAL audit currently in a TARGET_CATEGORIES section that is missing >= 1
    current dimension. Category + missing-check both use the rejudge-merged scores, exactly
    like the viewer. Transcripts are only read when we're actually going to judge."""
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
                scores = rejudge[key]["scores"] if key in rejudge else (info.score or {})
                cat = hack_category({"scores": scores})
                if cat not in TARGET_CATEGORIES:
                    continue
                missing = [d_ for d_ in dim_names if not isinstance(scores.get(d_), (int, float))]
                if not missing:
                    continue  # already complete -> nothing to fill
                cand = {
                    "key": key, "id": ids.get(key), "mode": d.name, "task": info.task_set,
                    "seed": str(info.task_id), "epoch": info.task_repeat,
                    "category": cat, "missing": missing,
                    "reward_hacking": scores.get("reward_hacking"),
                }
                if load_transcripts:
                    try:
                        t = await reader.read(info, TranscriptContent(messages="all", events="all"))
                    except FileNotFoundError as e:
                        print(f"    WARNING: missing eval log for {key} -- skipping: {e}")
                        continue
                    if not t.timelines:
                        print(f"    WARNING: empty transcript for {key} (dead audit) -- skipping")
                        continue
                    cand["transcript"] = t
                cands.append(cand)
    return cands


async def main() -> None:
    dims = judge_dimensions(DIMENSIONS_DIR)
    dim_names = [d.name for d in dims]
    print(f"\nmodel={MODEL}  concurrency={CONCURRENCY}  dry_run={DRY_RUN}"
          + (f"  limit={LIMIT}" if LIMIT else ""))
    print(f"current judge dimensions ({len(dim_names)}): {dim_names}")
    print(f"target sections: {sorted(TARGET_CATEGORIES)}\n")

    rejudge = _load_rejudge()
    print("[setup] finding degenerate/nudged hacks missing >= 1 current dimension ...")
    cands = await collect_candidates(rejudge, dim_names, load_transcripts=not DRY_RUN)
    if LIMIT is not None:
        cands = cands[:LIMIT]
    total = len(cands)

    if DRY_RUN or total == 0:
        by_cat = {c: sum(1 for x in cands if x["category"] == c) for c in sorted(TARGET_CATEGORIES)}
        print(f"\n{total} trajector(ies) would be re-judged  {by_cat}")
        for c in sorted(cands, key=lambda x: (x["category"], x["seed"])):
            tag = f"#{c['id']}" if c["id"] is not None else c["key"]
            print(f"    [{c['category']:10s}] {tag}  {c['seed']} e{c['epoch']} "
                  f"(rh={c['reward_hacking']}, missing: {c['missing']})")
        if DRY_RUN:
            print("\n(dry run -- nothing judged, no cost. Drop --dry-run to run it.)")
        return

    print(f"[setup] re-judging {total} trajector(ies) on all {len(dim_names)} dims\n")
    scanner = audit_judge(dimensions=dims, model=MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    results = dict(rejudge)
    done = failed = 0

    async def checkpoint():
        async with lock:
            REJUDGE_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))

    async def judge_one(i: int, c: dict) -> None:
        nonlocal done, failed
        label = f"#{c['id']}" if c["id"] is not None else c["key"]
        async with sem:
            print(f"  [{i}/{total}] judging {label} {c['seed']} e{c['epoch']} "
                  f"({c['category']}, missing {len(c['missing'])}) ...", flush=True)
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
        scores = {k: result.value.get(k) for k in dim_names}
        miss = [k for k, v in scores.items() if not isinstance(v, (int, float))]
        if miss:
            failed += 1
            print(f"    [{i}/{total}] FAILED {label}: judge omitted dims {miss}", flush=True)
            return
        md = result.metadata or {}
        results[c["key"]] = {
            "mode": c["mode"], "task": c["task"], "seed": c["seed"], "epoch": c["epoch"],
            "original_reward_hacking": c["reward_hacking"],
            "model": MODEL,
            "scores": scores,
            "summary": md.get("summary", ""),
            "highlights": md.get("highlights", ""),
            "justification": getattr(result, "explanation", None) or "",
        }
        done += 1
        new_cat = hack_category({"scores": scores})
        moved = "" if new_cat == c["category"] else f"  -> now {new_cat.upper()}"
        await checkpoint()
        print(f"    [{i}/{total}] {label}: {scores}{moved}", flush=True)

    await asyncio.gather(*(judge_one(i, c) for i, c in enumerate(cands, 1)))
    await checkpoint()

    print(f"\nthis run: {done} re-judged, {failed} failed")
    if failed:
        print(f"  NOTE: {failed} failed -- they keep their old (incomplete) scores and will "
              "be picked up again on a re-run (incremental).")
    print("\nRegenerate the viewer (free) to see the merged scores: uv run make_viewer.py")


if __name__ == "__main__":
    asyncio.run(main())
