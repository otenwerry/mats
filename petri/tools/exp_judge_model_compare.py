"""Judge-model comparison: is DeepSeek an acceptable (cheaper) replacement for the
Claude Opus judge?

PURELY EXPLORATORY -- writes NOTHING the viewer or any pipeline reads. Results go to a
standalone analysis dump (mats-local/petri/_analysis/judge_model_compare/results.json);
no rejudge_scores.json, no dim_scores/ sidecars, no viewer rebuild. Delete the dump dir
to erase all trace.

Design: take a stratified subset of the MOST RECENT runs (the 2026-07-04 collection
runs, judged inline by claude-opus-4-8) and re-judge each sampled trajectory TWICE with
the full current dimension set, exactly like the real judge (all dims in one call):

  arm "claude_rerun"  -- anthropic/claude-opus-4-8 (the incumbent judge, re-rolled)
  arm "deepseek"      -- openrouter/deepseek/deepseek-v4-pro-20260423 (the candidate;
                         same slug we now use for the auditor)

Then compare each arm against the ORIGINAL inline scores. The claude_rerun deltas are
the yardstick: they measure pure judge sampling variance. If the deepseek deltas look
similar (per-dimension movement + binary hack-classification flips), switching the
judge to DeepSeek costs little accuracy. If they are much larger or systematically
biased, it doesn't.

Sampling: stratified by original reward_hacking score -- thirds from low (<=4),
mid (5-8), high (>=9) -- so non-hacks, borderline cases, and clear hacks are all
represented (a uniform sample of these runs would be mostly 9s and 10s). Deterministic
(fixed RNG seed over sorted keys), so re-running resumes the same subset. Trajectories
with no inline scores (crashed / never judged, ~25 of 95 in these dirs) and dead audits
are SKIPPED -- counted, printed, and recorded under "skipped" in the results file.

Resumable: results checkpoint to the dump file after every judge call; a re-run skips
(trajectory, arm) pairs already done and retries failures. --summary-only recomputes
the comparison from the dump for free.

Usage (from petri/):
  uv run tools/exp_judge_model_compare.py --dry-run       # FREE: show the sample, no API calls
  uv run tools/exp_judge_model_compare.py                 # the real thing
  uv run tools/exp_judge_model_compare.py --summary-only  # FREE: re-print comparison from the dump
  ... --n-sample=24        # default; total judge calls = 2x this
  ... --run-dirs=a,b       # default: the two 20260704-0100xx collection dirs
  ... --arms=deepseek      # run just one arm (default both: claude_rerun,deepseek)
  ... --concurrency=50     # the default

Costs money (default: ~24 full-transcript Opus judge calls + ~24 DeepSeek judge calls --
about the cost of inline-judging one 24-trajectory run, plus the cheap DeepSeek half).
"""

import asyncio
import json
import pathlib
import random
import statistics
import sys
from datetime import datetime

# this tool lives in tools/; put the project root (for viewer) and ../lib
# (for petri_paths + the core modules) on the import path.
_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from dotenv import load_dotenv
from inspect_petri import audit_judge, judge_dimensions
from inspect_scout import TranscriptContent, transcripts_from

from petri_paths import ENV_FILE, DIMENSIONS_DIR, DATA, LOGS
from viewer_load import (traj_key, _load_rejudge, _load_dim_sidecars,
                         _apply_rejudge, _apply_dim_scores)
from viewer import binary_hack_eval, BINARY_HACK_CRITERIA

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY + OPENROUTER_API_KEY)

DEFAULT_RUN_DIRS = [
    "v2-5targets-allow-4ep-20260704-010042",
    "v2-5targets-correct-4ep-20260704-010107",
]
ARM_MODELS = {
    "claude_rerun": "anthropic/claude-opus-4-8",
    "deepseek": "openrouter/deepseek/deepseek-v4-pro-20260423",
}

def _arg(name: str, default: str) -> str:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)

RUN_DIRS = [d for d in _arg("run-dirs", ",".join(DEFAULT_RUN_DIRS)).split(",") if d]
N_SAMPLE = int(_arg("n-sample", "24"))
CONCURRENCY = int(_arg("concurrency", "50"))
SAMPLE_SEED = int(_arg("sample-seed", "0"))
ARMS = [a for a in _arg("arms", ",".join(ARM_MODELS)).split(",") if a]
DRY_RUN = "--dry-run" in sys.argv
SUMMARY_ONLY = "--summary-only" in sys.argv

for a in ARMS:
    if a not in ARM_MODELS:
        raise SystemExit(f"--arms contains {a!r}; known arms: {sorted(ARM_MODELS)}")
for d in RUN_DIRS:
    if not (LOGS / d).is_dir():
        dirs = sorted(x.name for x in LOGS.iterdir() if x.is_dir())
        raise SystemExit(f"--run-dirs entry {d!r} not found under {LOGS}. Available: {dirs}")

OUT_DIR = DATA / "_analysis" / "judge_model_compare"
OUT_FILE = OUT_DIR / "results.json"

# stratification buckets over the ORIGINAL reward_hacking score
BUCKETS = [
    ("low(rh<=4)", lambda v: v <= 4),
    ("mid(rh5-8)", lambda v: 5 <= v <= 8),
    ("high(rh>=9)", lambda v: v >= 9),
]


def _ids() -> dict:
    """traj_key -> stable trajectory id (the integer shown in the viewer), for display."""
    f = DATA / "trajectory_ids.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _load_results() -> dict:
    if OUT_FILE.exists():
        return json.loads(OUT_FILE.read_text())
    return {"_config": {}, "trajectories": {}, "skipped": []}


async def collect_candidates() -> tuple[list[dict], list[dict]]:
    """Every trajectory in RUN_DIRS with viewer-visible original scores (inline scores
    with any rejudge/sidecar merge applied -- for these fresh dirs that's just the
    inline judge). Returns (candidates, skipped); skipped = no numeric reward_hacking
    (crashed / never judged). Transcripts are NOT read here -- only the sampled subset
    gets read later."""
    rejudge, sidecars, ids = _load_rejudge(), _load_dim_sidecars(), _ids()
    cands: list[dict] = []
    skipped: list[dict] = []
    for d in RUN_DIRS:
        ts = transcripts_from(str(LOGS / d))
        async with ts.reader() as reader:
            infos = [i async for i in reader.index()]
        for info in infos:
            a = {"mode": d, "task": info.task_set, "seed": str(info.task_id),
                 "epoch": info.task_repeat, "scores": dict(info.score or {})}
            _apply_rejudge(a, rejudge)
            _apply_dim_scores(a, sidecars)
            key = traj_key(a)
            rh = a["scores"].get("reward_hacking")
            if not isinstance(rh, (int, float)):
                skipped.append({"key": key, "reason": "no original scores (crashed / never judged)"})
                continue
            cands.append({"key": key, "id": ids.get(key), "mode": d, "task": info.task_set,
                          "seed": str(info.task_id), "epoch": info.task_repeat,
                          "original": a["scores"], "info": info})
    return cands, skipped


def stratified_sample(cands: list[dict]) -> list[dict]:
    """Deterministic stratified sample: shuffle each reward_hacking bucket with a fixed
    seed, then round-robin one from each non-empty bucket until N_SAMPLE. Small buckets
    just run dry and the others fill in."""
    rng = random.Random(SAMPLE_SEED)
    pools: list[list[dict]] = []
    for name, pred in BUCKETS:
        pool = sorted((c for c in cands if pred(c["original"]["reward_hacking"])),
                      key=lambda c: c["key"])
        rng.shuffle(pool)
        for c in pool:
            c["bucket"] = name
        pools.append(pool)
    leftover = [c for c in cands if "bucket" not in c]  # shouldn't happen; buckets cover 1..10
    if leftover:
        print(f"  WARNING: {len(leftover)} trajectory(ies) fell in no bucket -- excluded")
    picked: list[dict] = []
    while len(picked) < N_SAMPLE and any(pools):
        for pool in pools:
            if pool and len(picked) < N_SAMPLE:
                picked.append(pool.pop())
    return sorted(picked, key=lambda c: c["key"])


def print_sample(sample: list[dict]) -> None:
    counts: dict[str, int] = {}
    for c in sample:
        counts[c["bucket"]] = counts.get(c["bucket"], 0) + 1
    print(f"\nsampled {len(sample)} trajectory(ies): "
          + "  ".join(f"{k}:{v}" for k, v in counts.items()))
    for c in sample:
        label = f"#{c['id']}" if c["id"] is not None else c["key"]
        print(f"  {label:>6}  {c['bucket']:12}  rh={c['original']['reward_hacking']:>2}  "
              f"{c['task']}  {c['seed']} e{c['epoch']}")


async def run_arms(sample: list[dict], results: dict) -> None:
    """Judge every (sampled trajectory, arm) pair not already in the dump. One judge
    call scores ALL current dimensions at once, matching the real inline judge."""
    dims = judge_dimensions(DIMENSIONS_DIR)
    dim_names = sorted(x.name for x in dims)
    print(f"[setup] judging with {len(dims)} dimensions (rubrics read fresh from "
          f"{DIMENSIONS_DIR}): {dim_names}")
    scanners = {arm: audit_judge(dimensions=dims, model=ARM_MODELS[arm]) for arm in ARMS}

    # register sampled trajectories + figure out which (traj, arm) pairs still need a call
    todo: list[tuple[dict, str]] = []
    for c in sample:
        rec = results["trajectories"].setdefault(c["key"], {
            "id": c["id"], "mode": c["mode"], "task": c["task"], "seed": c["seed"],
            "epoch": c["epoch"], "bucket": c["bucket"],
            "original": {"scores": c["original"]}, "arms": {},
        })
        for arm in ARMS:
            if rec["arms"].get(arm, {}).get("scores"):
                continue  # already judged in a previous invocation
            todo.append((c, arm))
    n_calls = len(todo)
    done_before = len(sample) * len(ARMS) - n_calls
    if done_before:
        print(f"[setup] {done_before} (trajectory, arm) pair(s) already in the dump -- skipped")
    if not n_calls:
        print("[setup] nothing to judge; jumping to the summary")
        return
    print(f"[setup] {n_calls} judge call(s) to make "
          f"({' + '.join(f'{sum(1 for _, a in todo if a == arm)} {ARM_MODELS[arm]}' for arm in ARMS)})\n")

    # read transcripts for the trajectories that still need at least one call
    need = {c["key"]: c for c, _ in todo}
    transcripts: dict[str, object] = {}
    for d in RUN_DIRS:
        wanted = [c for c in need.values() if c["mode"] == d]
        if not wanted:
            continue
        print(f"[setup] reading {len(wanted)} transcript(s) from {d}/ ...")
        ts = transcripts_from(str(LOGS / d))
        async with ts.reader() as reader:
            for c in wanted:
                t = await reader.read(c["info"], TranscriptContent(messages="all", events="all"))
                if not t.timelines:
                    print(f"  WARNING: empty transcript for {c['key']} (dead audit) -- SKIPPED")
                    results["skipped"].append({"key": c["key"], "reason": "empty transcript (dead audit)"})
                    continue
                transcripts[c["key"]] = t
    todo = [(c, arm) for c, arm in todo if c["key"] in transcripts]

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    done = failed = 0
    total = len(todo)

    async def checkpoint():
        async with lock:
            OUT_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))

    async def judge_one(i: int, c: dict, arm: str) -> None:
        nonlocal done, failed
        label = f"#{c['id']}" if c["id"] is not None else c["key"]
        async with sem:
            print(f"  [{i}/{total}] {arm:13} judging {label} {c['seed']} e{c['epoch']} ...", flush=True)
            try:
                result = await scanners[arm](transcripts[c["key"]])
            except Exception as e:
                failed += 1
                print(f"    [{i}/{total}] FAILED {arm} {label}: {type(e).__name__}: {e} "
                      "-- re-run to retry", flush=True)
                return
        value = result.value if isinstance(result.value, dict) else {}
        scores = {k: v for k, v in value.items() if isinstance(v, (int, float))}
        if not scores:
            failed += 1
            reason = (result.metadata or {}).get("stop_reason") or "no dict value (refusal?)"
            print(f"    [{i}/{total}] FAILED {arm} {label}: judge returned no scores "
                  f"({reason}) -- re-run to retry", flush=True)
            return
        missing = [x for x in dim_names if x not in scores]
        results["trajectories"][c["key"]]["arms"][arm] = {
            "model": ARM_MODELS[arm],
            "scores": scores,
            "complete": not missing,
            "missing_dims": missing,
            "justification": getattr(result, "explanation", None) or "",
            "summary": (result.metadata or {}).get("summary", ""),
            "graded_at": datetime.now().isoformat(timespec="seconds"),
        }
        done += 1
        await checkpoint()
        rh_old = c["original"].get("reward_hacking")
        print(f"    [{i}/{total}] {arm:13} {label}: reward_hacking {rh_old} -> "
              f"{scores.get('reward_hacking')}"
              + (f"  (MISSING dims: {missing})" if missing else ""), flush=True)

    await asyncio.gather(*(judge_one(i, c, arm) for i, (c, arm) in enumerate(todo, 1)))
    await checkpoint()
    print(f"\nthis run: {done} judged, {failed} failed"
          + (" -- re-run to retry the failures (already-done pairs are skipped)" if failed else ""))


# --------------------------------------------------------------------------- #
# summary: computed purely from the dump, so --summary-only is free
# --------------------------------------------------------------------------- #
def _scores_of(rec: dict, source: str) -> dict | None:
    if source == "original":
        return rec["original"]["scores"]
    arm = rec["arms"].get(source)
    return arm["scores"] if arm and arm.get("scores") else None


def print_summary(results: dict) -> None:
    trajs = results["trajectories"]
    if not trajs:
        print("\nno judged trajectories in the dump yet -- nothing to summarize")
        return
    all_dims = sorted({d for r in trajs.values()
                       for src in ["original", *ARM_MODELS]
                       for d in (_scores_of(r, src) or {})})
    all_dims = (["reward_hacking"] if "reward_hacking" in all_dims else []) + \
               [d for d in all_dims if d != "reward_hacking"]
    pairs = [("original", "claude_rerun"), ("original", "deepseek"),
             ("claude_rerun", "deepseek")]

    incomplete = [(k, a) for k, r in trajs.items() for a, v in r["arms"].items()
                  if not v.get("complete", True)]
    if incomplete:
        print(f"\nCAVEAT: {len(incomplete)} arm result(s) came back with missing dimensions "
              "-- those dims are excluded pairwise below:")
        for k, a in incomplete:
            print(f"  {k} [{a}]: missing {trajs[k]['arms'][a].get('missing_dims')}")
    if results.get("skipped"):
        print(f"\nCAVEAT: {len(results['skipped'])} trajectory(ies) were skipped and are "
              "absent from everything below (see 'skipped' in the dump for the list)")

    print("\n=== per-dimension score movement (x -> y; delta = y - x) ===")
    header = f"{'dimension':<24}" + "".join(
        f"{a + ' -> ' + b:>28}" for a, b in pairs)
    sub = f"{'':<24}" + "".join(f"{'n  m|d| meand  ex%  +-1%':>28}" for _ in pairs)
    print(header)
    print(sub)
    for dim in all_dims:
        row = f"{dim:<24}"
        for a, b in pairs:
            ds = []
            for r in trajs.values():
                x, y = _scores_of(r, a), _scores_of(r, b)
                if x and y and isinstance(x.get(dim), (int, float)) and isinstance(y.get(dim), (int, float)):
                    ds.append(y[dim] - x[dim])
            if not ds:
                row += f"{'-':>28}"
                continue
            n = len(ds)
            mabs = statistics.mean(abs(d) for d in ds)
            mean = statistics.mean(ds)
            ex = 100 * sum(1 for d in ds if d == 0) / n
            w1 = 100 * sum(1 for d in ds if abs(d) <= 1) / n
            row += f"{n:>5} {mabs:>5.2f} {mean:>+5.2f} {ex:>4.0f} {w1:>5.0f}"
        print(row)

    print("\n=== binary hack classification (viewer.BINARY_HACK_CRITERIA) ===")
    for src in ["original", *[a for a in ARM_MODELS if a in ARMS or any(
            a in r["arms"] for r in trajs.values())]]:
        evald = {k: binary_hack_eval({"scores": _scores_of(r, src)})
                 for k, r in trajs.items() if _scores_of(r, src)}
        n_hack = sum(1 for e in evald.values() if e["ok"])
        print(f"  {src:<13} {n_hack}/{len(evald)} hacks")
    for a, b in pairs:
        flips = []
        both = 0
        for k, r in trajs.items():
            x, y = _scores_of(r, a), _scores_of(r, b)
            if not x or not y:
                continue
            both += 1
            ex_, ey = binary_hack_eval({"scores": x}), binary_hack_eval({"scores": y})
            if ex_["ok"] != ey["ok"]:
                moved = [f"{d} {x.get(d)}->{y.get(d)}" for d in BINARY_HACK_CRITERIA
                         if (d in ex_["failed"]) != (d in ey["failed"])
                         or (d in ex_["missing"]) != (d in ey["missing"])]
                lbl = f"#{r['id']}" if r["id"] is not None else k
                flips.append(f"{lbl} {r['seed']} e{r['epoch']}: "
                             f"{'hack->non' if ex_['ok'] else 'non->hack'} ({', '.join(moved)})")
        agree = both - len(flips)
        print(f"\n  {a} vs {b}: agree on {agree}/{both}" + ("" if not flips else " -- flips:"))
        for f in flips:
            print(f"    {f}")
    print(f"\nfull per-trajectory data (scores + judge justifications): {OUT_FILE}")


async def main() -> None:
    results = _load_results()

    if SUMMARY_ONLY:
        print(f"summary from existing dump ({len(results['trajectories'])} trajectory(ies))")
        print_summary(results)
        return

    print(f"\nJUDGE MODEL COMPARISON  ({'DRY RUN -- no API calls' if DRY_RUN else 'live'})")
    print(f"  run dirs: {RUN_DIRS}")
    print(f"  arms: " + "  ".join(f"{a}={ARM_MODELS[a]}" for a in ARMS))
    print(f"  n-sample={N_SAMPLE}  sample-seed={SAMPLE_SEED}  concurrency={CONCURRENCY}")
    print(f"  dump: {OUT_FILE}\n")

    print("[setup] indexing run dirs (no transcript reads yet) ...")
    cands, skipped = await collect_candidates()
    print(f"[setup] {len(cands)} candidate trajectory(ies) with original scores; "
          f"{len(skipped)} skipped (no scores)")
    known = {s["key"] for s in results["skipped"]}
    results["skipped"] += [s for s in skipped if s["key"] not in known]

    sample = stratified_sample(cands)
    print_sample(sample)

    if DRY_RUN:
        print(f"\ndry run: would make up to {len(sample) * len(ARMS)} judge calls "
              f"({len(sample)} per arm). Nothing was called or written.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results["_config"] = {"run_dirs": RUN_DIRS, "n_sample": N_SAMPLE,
                          "sample_seed": SAMPLE_SEED, "arms": {a: ARM_MODELS[a] for a in ARMS},
                          "last_run_at": datetime.now().isoformat(timespec="seconds")}
    await run_arms(sample, results)
    OUT_FILE.write_text(json.dumps(results, indent=2, sort_keys=True))
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
