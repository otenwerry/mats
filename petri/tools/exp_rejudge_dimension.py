"""Re-judge ONE dimension on EVERY trajectory of one run dir and OVERWRITE its score.

Sibling of tools/exp_grade_dimension.py with the opposite write semantics: that tool
BACKFILLS a dimension onto audits never scored on it and never touches an existing
score; this one re-judges a dimension you believe was mis-calibrated -- typically after
rewriting its rubric in dimensions/<dim>.md -- and REPLACES the inline judge's score +
reasoning for that dimension. All other dimensions keep their original inline scores.

Mechanics: runs Petri's real judge (audit_judge) over each transcript in the run dir
with JUST the named dimension, whose rubric is read fresh from dimensions/<dim>.md at
launch -- so edit the rubric first, then run this. Results are written as entries with
"overwrite": true into mats-local/petri/dim_scores/<dim>.json; the viewer load layer
(lib/viewer_load._apply_dim_scores) applies overwrite entries ON TOP of inline/rejudged
scores, so the new score + justification feed every consumer: the index tables, the
binary hack criteria, and the annotate gate. The .eval logs are NEVER modified, so the
original scores stay recoverable (delete the sidecar entries to revert). Re-running
after another rubric edit simply overwrites again -- the intended iteration loop. Each
entry records previous_score (the inline score it replaced) and graded_at for tracking
across rubric iterations.

Usage:
  uv run tools/exp_rejudge_dimension.py --dim=auditor_nudging --run-dir=v2-4targets-allow-5ep-20260703-160853
  ... --limit=2       # cheap sanity run on the first 2 trajectories
  ... --model=anthropic/claude-opus-4-8 --concurrency=50   # the defaults
  ... --skip-viewer   # don't rebuild the viewer at the end

Costs money (Anthropic API: the judge model); the viewer rebuild at the end is free.
"""

import asyncio
import json
import pathlib
import sys
import traceback
from datetime import datetime

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
import viewer
from viewer import LOGS, traj_key, _load_rejudge

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)
DIM = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dim=")), None)
RUN_DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--run-dir=")), None)
MODEL = resolve_judge(next((a.split("=", 1)[1] for a in sys.argv
                            if a.startswith("--model=")), None))
CONCURRENCY = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--concurrency=")), "50"))
LIMIT = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--limit=")), None)
SKIP_VIEWER = "--skip-viewer" in sys.argv
if not DIM:
    raise SystemExit("--dim=<name> is REQUIRED (a dimension stem from dimensions/, "
                     "e.g. --dim=auditor_nudging).")
if not RUN_DIR:
    dirs = sorted(d.name for d in LOGS.iterdir() if d.is_dir()) if LOGS.exists() else []
    raise SystemExit(f"--run-dir=<name> is REQUIRED (a dir under {LOGS}). Available: {dirs}")
if CONCURRENCY < 1:
    raise SystemExit(f"--concurrency must be >= 1, got {CONCURRENCY}")

MODE_DIR = LOGS / RUN_DIR
if not MODE_DIR.is_dir():
    dirs = sorted(d.name for d in LOGS.iterdir() if d.is_dir())
    raise SystemExit(f"--run-dir={RUN_DIR!r} not found under {LOGS}. Available: {dirs}")

OUT_FILE = DATA / "dim_scores" / f"{DIM}.json"


def _ids() -> dict:
    """traj_key -> stable trajectory id (the integer shown in the viewer), for display."""
    f = DATA / "trajectory_ids.json"
    return json.loads(f.read_text()) if f.exists() else {}


async def collect_run_trajectories() -> tuple[list[dict], int]:
    """EVERY trajectory in the run dir with its transcript loaded, plus the count of
    skipped (empty-transcript / unreadable) ones. previous_score is the score this
    overwrite will replace: the rejudged score when a rejudge entry exists, else the
    inline eval-log score -- mirroring what the viewer showed before the overwrite."""
    rejudge = _load_rejudge()
    ids = _ids()
    cands: list[dict] = []
    skipped = 0
    ts = transcripts_from(str(MODE_DIR))
    async with ts.reader() as reader:
        infos = [i async for i in reader.index()]
        for info in infos:
            key = traj_key({
                "mode": RUN_DIR, "task": info.task_set,
                "seed": str(info.task_id), "epoch": info.task_repeat,
            })
            scores = rejudge[key]["scores"] if key in rejudge else (info.score or {})
            try:
                t = await reader.read(info, TranscriptContent(messages="all", events="all"))
            except FileNotFoundError as e:
                skipped += 1
                print(f"  WARNING: missing eval log for {key} -- SKIPPED (keeps old score): {e}")
                continue
            if not t.timelines:
                skipped += 1
                print(f"  WARNING: empty transcript for {key} (dead audit) -- SKIPPED (keeps old score)")
                continue
            cands.append({
                "key": key, "id": ids.get(key), "task": info.task_set,
                "seed": str(info.task_id), "epoch": info.task_repeat,
                "previous": scores.get(DIM), "transcript": t,
            })
    return cands, skipped


async def main() -> None:
    dims = [x for x in judge_dimensions(DIMENSIONS_DIR) if x.name == DIM]
    if len(dims) != 1:
        avail = sorted(x.name for x in judge_dimensions(DIMENSIONS_DIR))
        raise SystemExit(f"--dim={DIM!r} did not match exactly one dimension in "
                         f"{DIMENSIONS_DIR}.\navailable: {avail}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else {}

    print(f"\nRE-JUDGE + OVERWRITE  dim={DIM}  run-dir={RUN_DIR}")
    print(f"  model={MODEL}  concurrency={CONCURRENCY}"
          + (f"  limit={LIMIT}" if LIMIT else ""))
    print(f"  rubric read fresh from {DIMENSIONS_DIR / (DIM + '.md')}")
    print(f"[setup] collecting every trajectory in {RUN_DIR}/ ...")
    cands, skipped = await collect_run_trajectories()
    if LIMIT is not None:
        cands = cands[:LIMIT]
    total = len(cands)
    if total == 0:
        raise SystemExit(f"no judgeable trajectories found in {MODE_DIR}")
    print(f"[setup] {total} trajectory(ies) to re-judge"
          + (f"  ({skipped} skipped -- they KEEP their old {DIM} score)" if skipped else "")
          + "\n")

    scanner = audit_judge(dimensions=dims, model=MODEL)
    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    results = dict(existing)
    changes: list[dict] = []
    done = failed = 0

    async def checkpoint():
        async with lock:
            OUT_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))

    async def grade_one(i: int, c: dict) -> None:
        nonlocal done, failed
        label = f"#{c['id']}" if c["id"] is not None else c["key"]
        async with sem:
            print(f"  [{i}/{total}] judging {label} {c['seed']} e{c['epoch']} "
                  f"(old {DIM}={c['previous']}) ...", flush=True)
            try:
                result = await scanner(c["transcript"])
            except Exception as e:
                failed += 1
                print(f"    [{i}/{total}] FAILED {label}: {type(e).__name__}: {e} -- "
                      "keeps old score; re-run to retry", flush=True)
                return
        score = result.value.get(DIM) if isinstance(result.value, dict) else None
        if not isinstance(score, (int, float)):
            failed += 1
            reason = ((result.metadata or {}).get("stop_reason")
                      if isinstance(result.value, dict) else "no dict value (refusal?)")
            print(f"    [{i}/{total}] FAILED {label}: judge returned no {DIM} score "
                  f"({reason}) -- keeps old score; re-run to retry", flush=True)
            return
        results[c["key"]] = {
            "id": c["id"], "mode": RUN_DIR, "task": c["task"], "seed": c["seed"],
            "epoch": c["epoch"], "model": MODEL,
            DIM: score,
            "previous_score": c["previous"],
            # overwrite: the viewer load layer REPLACES the inline score with this one
            # (backfill entries from exp_grade_dimension.py lack this field).
            "overwrite": True,
            "graded_at": datetime.now().isoformat(timespec="seconds"),
            "justification": getattr(result, "explanation", None) or "",
        }
        changes.append({"label": label, "seed": c["seed"], "epoch": c["epoch"],
                        "old": c["previous"], "new": score})
        done += 1
        await checkpoint()
        arrow = "==" if score == c["previous"] else "->"
        print(f"    [{i}/{total}] {label} {c['seed']} e{c['epoch']}: "
              f"{DIM} {c['previous']} {arrow} {score}", flush=True)

    await asyncio.gather(*(grade_one(i, c) for i, c in enumerate(cands, 1)))
    await checkpoint()

    print(f"\nthis run: {done} re-judged, {failed} failed"
          + (f", {skipped} skipped (kept old scores)" if skipped else ""))
    if changes:
        vals = [c["new"] for c in changes]
        hist = {s: sum(1 for v in vals if round(v) == s) for s in range(1, 11)}
        print(f"\nnew {DIM} distribution over {len(vals)} trajectory(ies):  "
              + "  ".join(f"{s}:{n}" for s, n in hist.items() if n))
        moved = [c for c in changes if c["new"] != c["old"]]
        print(f"changed: {len(moved)}/{len(changes)}  (old -> new, biggest moves first)")
        for c in sorted(moved, key=lambda c: -abs((c["new"] or 0) - (c["old"] or 0))):
            print(f"  {c['label']}  {c['seed']} e{c['epoch']}:  {c['old']} -> {c['new']}")
    if failed:
        print(f"\nNOTE: {failed} trajectory(ies) failed and KEEP their old {DIM} score -- "
              "re-run to retry (already-overwritten ones just get overwritten again).")
    print(f"\noverwrite entries written to {OUT_FILE}")

    if SKIP_VIEWER:
        print("(skipping viewer rebuild: --skip-viewer)")
        return
    print("\nrebuilding the viewer (free) so the overwritten scores show everywhere ...")
    try:
        await viewer.main()
    except Exception as e:
        print(f"!! viewer rebuild failed (scores are safely written; run "
              f"`uv run viewer.py` manually): {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
