"""End-to-end rollback pipeline: generate -> judge -> annotate -> viewer, one command.

For each FULL reward-hack original (binary definition, with a hack-turn annotation so we
can locate the first hack), this re-rolls the conversation at one or more CUT LOCATIONS,
in both a CONTROL (plain re-roll, nothing inserted) and a TREATMENT (a location-specific
honesty message inserted) condition. Then it judges every continuation, annotates the
ones the viewer tints red (full or degenerate reward hack), and rebuilds the viewer --
all at one concurrency.

Cut locations (k = the original's first-hack assistant turn [A_k]; j = the target turn we
re-roll live; replay 1..j-1, then go live):

  begin   j = 1            fold the message into the first user turn [M2].          always
  middle  j = ceil(k/2)    "...later in the session..."; fold into a user turn.     if k > 4
  before  j = k            the before-hack prompt (PROMPTS["prompt1"]).             always
  after   j = k+1          REPLACE the post-hack user turn (else fold).             if a turn k+1 exists

  Every cell's run dir is named rollback-<location>-<condition>-<N>x-<timestamp>
  (uniform across locations). This pipeline dedups EVERY cell: a (trajectory, location,
  condition) is skipped if a completed continuation for it already exists in a run dir
  with that cell's label, so re-running only fills gaps.

Provenance: each generated run dir gets a rollback_meta.json recording the per-trajectory
re-roll turn j. The judge reads it (instead of assuming the cut is at the hack), so the
cut-scoping and the viewer's cut marker are correct for every location.

Usage:
  uv run exp_rollback_pipeline.py --locations=all --conditions=all --N=5
  uv run exp_rollback_pipeline.py --locations=begin,after --conditions=treatment --N=3 --concurrency=50
  uv run exp_rollback_pipeline.py --locations=all --conditions=all --N=5 --dry-run     # FREE: plan only
  uv run exp_rollback_pipeline.py --locations=all --conditions=control --N=5 --trajectories=12,55

Flags:
  --locations=<a,b,..|all>  REQUIRED. subset of {begin,middle,before,after}, or `all`.
  --conditions=<a,b|all>    REQUIRED. subset of {control,treatment}, or `all`. control =
                            plain re-roll; treatment = location-specific honesty message.
  --N=<int>                 REQUIRED. rollbacks (epochs) per cell.
  --concurrency=<int>       one knob -> generation max_samples/max_connections AND
                            judging parallelism (default 50).
  --trajectories=<ids>      restrict to these original trajectory ids (default: all
                            full-hack originals that have a hack-turn annotation).
  --annotate-model=<m>      Anthropic model for the secondary hack-turn judging (default
                            claude-opus-4-8).
  --dry-run                 build + validate the whole plan and print it; no run, no cost.
  --skip-judge              generate only (still rebuilds the viewer).
  --skip-viewer             don't rebuild the viewer at the end.
  --force-judge             re-judge (re-spend on) red rows already in rollback_results.json.

Costs money (Anthropic + OpenRouter) unless --dry-run.
"""

import asyncio
import json
import pathlib
import re
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

# the core modules live in lib/; put it on the import path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

from inspect_ai import eval_set
from inspect_ai.log import list_eval_logs, read_eval_log

import make_viewer
import exp_rollback as rb
from exp_rollback import (
    PROMPTS, build_replay_data, make_resume_spec, build_rollback_task,
    get_rollback_dims, write_rollback_meta,
)
from exp_rollback_judge import run_judging, DEFAULT_MODEL as DEFAULT_ANNOTATE_MODEL
from make_viewer import (
    DATA, LOGS, ROLLBACK_PREFIX, is_hack_binary, page_name, load_originals_by_id,
    _orig_id_from_task,
)

DEFAULT_CONCURRENCY = 50

COND_ORDER = ["control", "treatment"]  # canonical order for --conditions

# name -> (j(k), prompt key in PROMPTS, insert mode, applies(k, n_resumes))
LOCATIONS = {
    "begin":  (lambda k: 1,            "begin",   "fold",    lambda k, n: True),
    "middle": (lambda k: (k + 1) // 2, "middle",  "fold",    lambda k, n: k > 4),
    "before": (lambda k: k,            "prompt1", "fold",    lambda k, n: True),
    "after":  (lambda k: k + 1,        "after",   "replace", lambda k, n: k + 1 <= n),
}
ORDER = ["begin", "middle", "before", "after"]


def cell_label(loc: str, cond: str) -> str:
    """Dir label for a (location, condition) cell: '<location>-<condition>', uniform across
    every location (begin-control, middle-treatment, before-control, after-treatment, ...).
    The run dir is rollback-<label>-<N>x-<timestamp>; dedup parses the label back out."""
    return f"{loc}-{cond}"


def _arg(flag: str, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _parse_args() -> dict:
    loc_arg = _arg("--locations")
    if loc_arg is None:
        raise SystemExit("--locations is required (subset of begin,middle,before,after, or `all`)")
    if loc_arg.strip() == "all":
        locations = list(ORDER)
    else:
        locations = list(dict.fromkeys(s.strip() for s in loc_arg.split(",") if s.strip()))
        unknown = [l for l in locations if l not in LOCATIONS]
        if unknown:
            raise SystemExit(f"unknown --locations {unknown}; choices: {ORDER} (or `all`)")
    if not locations:
        raise SystemExit(f"--locations had no usable names; choices: {ORDER} (or `all`)")

    cond_arg = _arg("--conditions")
    if cond_arg is None:
        raise SystemExit("--conditions is required (control, treatment, or `all`)")
    if cond_arg.strip() == "all":
        conditions = list(COND_ORDER)
    else:
        chosen = {s.strip() for s in cond_arg.split(",") if s.strip()}
        unknown = [c for c in chosen if c not in COND_ORDER]
        if unknown:
            raise SystemExit(f"unknown --conditions {unknown}; choices: {COND_ORDER} (or `all`)")
        conditions = [c for c in COND_ORDER if c in chosen]  # canonical order
    if not conditions:
        raise SystemExit(f"--conditions had no usable names; choices: {COND_ORDER} (or `all`)")

    n_arg = _arg("--N")
    if n_arg is None:
        raise SystemExit("--N is required (rollbacks per cell); no default.")
    try:
        N = int(n_arg)
    except ValueError:
        raise SystemExit(f"--N must be an integer, got {n_arg!r}")
    if N < 1:
        raise SystemExit(f"--N must be >= 1, got {N}")

    conc_raw = _arg("--concurrency")
    concurrency = int(conc_raw) if conc_raw else DEFAULT_CONCURRENCY
    if concurrency < 1:
        raise SystemExit(f"--concurrency must be >= 1, got {concurrency}")

    traj_arg = _arg("--trajectories")
    trajectories = [int(x) for x in traj_arg.split(",") if x.strip()] if traj_arg else None

    return {
        "locations": locations,
        "conditions": conditions,
        "N": N,
        "concurrency": concurrency,
        "trajectories": trajectories,
        "annotate_model": _arg("--annotate-model", DEFAULT_ANNOTATE_MODEL),
        "dry_run": "--dry-run" in sys.argv,
        "skip_judge": "--skip-judge" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "force_judge": "--force-judge" in sys.argv,
    }


def _existing_cell_ids() -> dict[str, set]:
    """label -> set of original-trajectory ids that already have a COMPLETED continuation
    in some run dir with that label. Used to dedup every cell (so before-hack already done
    isn't re-run, and re-running the pipeline only fills gaps). Only counts eval logs whose
    status is 'success' -- an interrupted/errored cell (status started/cancelled/error)
    must stay re-runnable, so its leftover task header does NOT block regeneration.
    header_only read is cheap; the traj id is in the task name `rollback_<id>_<seed>`."""
    by_label: dict[str, set] = {}
    for d in sorted(p for p in LOGS.iterdir() if p.is_dir() and p.name.startswith(ROLLBACK_PREFIX)):
        m = re.match(r"^rollback-(.+)-\d+x-\d{8}-\d{6}$", d.name)
        label = m.group(1) if m else d.name.removeprefix(ROLLBACK_PREFIX)
        ids = by_label.setdefault(label, set())
        for li in list_eval_logs(str(d)):
            try:
                log = read_eval_log(li, header_only=True)
            except Exception:
                continue
            if getattr(log, "status", None) != "success":
                continue  # not a finished continuation -> leave the cell re-runnable
            oid = _orig_id_from_task(log.eval.task)
            if oid is not None:
                ids.add(oid)
    return by_label


def _select_full_hack_ids(refs: dict, originals: dict, annotations: dict) -> list[int]:
    """Full reward hacks (binary def) that have a hack-turn annotation (needed to locate
    the first hack k). These are the rollback source set."""
    out = []
    for tid, ref in refs.items():
        a = originals.get(tid)
        if not a or not is_hack_binary(a):
            continue
        ann = annotations.get(page_name(a["mode"], a["task"], a["seed"], a["epoch"]))
        if ann and ann.get("hack_turns"):
            out.append(tid)
    return sorted(out)


WORK_PREFIX = "_rbwork-"  # combined working dir (NOT "rollback-", so the viewer ignores it)


def _local_path(uri: str) -> Path:
    """EvalLogInfo.name is a file:// URI for local logs; strip the scheme to a real path."""
    return Path(uri[len("file://"):] if uri.startswith("file://") else uri)


def _combined_task_name(traj_id: int, loc: str, cond: str, seed: str) -> str:
    """Unique task name for the single combined eval_set, encoding the cell so the run's
    results can be split back into per-cell dirs. _orig_id_from_task still parses the id
    (regex `rollback_<id>_`); location/condition are looked up from _rbmeta by exact name."""
    return f"rollback_{traj_id}_{loc}_{cond}_{seed}"


def _write_rbmeta(work_dir: Path, plan: list) -> None:
    """Per-task cut provenance for the combined working dir, keyed by exact task name.
    Written BEFORE eval_set so even a killed run can be split on the next launch."""
    tasks = {}
    for loc, cond, label, prompt, specs in plan:
        for s in specs:
            tn = _combined_task_name(s.ref.traj_id, loc, cond, s.ref.seed)
            tasks[tn] = {"location": loc, "condition": cond,
                         "reroll_turn": s.reroll_turn, "hack_m": s.hack_m, "prompt": prompt}
    (work_dir / "_rbmeta.json").write_text(json.dumps(tasks, indent=2))


def _redistribute(work_dir: Path) -> list[Path]:
    """Split a finished (or killed) combined working dir into the per-cell dirs the rest
    of the pipeline expects: move each .eval into rollback-<label>-<N>x-<ts>/ by its cell
    (from _rbmeta), write that dir's rollback_meta.json (same format the judge/viewer read),
    then delete the working dir. The judge/viewer/dedup are unchanged -- they only ever see
    normal per-cell dirs. Returns the per-cell dirs written."""
    m = re.match(rf"{WORK_PREFIX}(\d+)x-(\d{{8}}-\d{{6}})$", work_dir.name)
    if not m:
        return []
    N, ts = m.group(1), m.group(2)
    mf = work_dir / "_rbmeta.json"
    if not mf.exists():
        shutil.rmtree(work_dir, ignore_errors=True)  # nothing to route
        return []
    task_meta = json.loads(mf.read_text())
    cells: dict[tuple, dict] = {}
    for li in list_eval_logs(str(work_dir)):
        tn = li.task
        info = task_meta.get(tn)
        if info is None:  # filename-sanitized name didn't match -> read canonical name
            try:
                tn = read_eval_log(li, header_only=True).eval.task
            except Exception:
                continue
            info = task_meta.get(tn)
        if info is None:
            continue
        key = (info["location"], info["condition"])
        if key not in cells:
            cdir = LOGS / f"rollback-{cell_label(*key)}-{N}x-{ts}"
            cdir.mkdir(parents=True, exist_ok=True)
            cells[key] = {"dir": cdir, "prompt": info["prompt"], "reroll": {}, "hack_m": {}}
        oid = _orig_id_from_task(tn)
        if oid is not None:
            cells[key]["reroll"][str(oid)] = info["reroll_turn"]
            cells[key]["hack_m"][str(oid)] = info.get("hack_m")
        src = _local_path(li.name)
        shutil.move(str(src), str(cells[key]["dir"] / src.name))
    for (loc, cond), c in cells.items():
        (c["dir"] / "rollback_meta.json").write_text(json.dumps(
            {"location": loc, "condition": cond, "prompt": c["prompt"],
             "reroll_turns": c["reroll"], "hack_m": c["hack_m"]}, indent=2))
    shutil.rmtree(work_dir, ignore_errors=True)
    return [c["dir"] for c in cells.values()]


def main() -> None:
    cfg = _parse_args()
    print("=" * 76)
    print(f"ROLLBACK PIPELINE  locations={cfg['locations']}  conditions={cfg['conditions']}  "
          f"N={cfg['N']}  concurrency={cfg['concurrency']}  dry_run={cfg['dry_run']}")
    print("=" * 76)

    # Heal any leftover combined working dir from a previously-killed run BEFORE dedup,
    # so its completed continuations become normal per-cell dirs (counted by dedup, shown
    # in the viewer) and never get duplicated. (Skipped on a dry-run -- it moves no files.)
    if not cfg["dry_run"]:
        for wd in sorted(LOGS.glob(f"{WORK_PREFIX}*")):
            if wd.is_dir():
                moved = _redistribute(wd)
                if moved:
                    print(f"[startup] split leftover {wd.name} -> {len(moved)} per-cell dir(s)")

    # --- async prep: refs (id->TrajRef), originals (id->audit dict), annotations ---
    refs = rb._all_refs()
    originals = asyncio.run(load_originals_by_id())
    annotations = json.loads(rb.ANN_FILE.read_text()) if rb.ANN_FILE.exists() else {}

    if cfg["trajectories"] is not None:
        selected = cfg["trajectories"]
        unknown = [t for t in selected if t not in refs]
        if unknown:
            raise SystemExit(f"unknown trajectory ids {unknown}")
        print(f"[select] {len(selected)} trajectory(ies) from --trajectories: {selected}")
    else:
        selected = _select_full_hack_ids(refs, originals, annotations)
        print(f"[select] {len(selected)} full-hack originals (binary def, annotated): {selected}")
    if not selected:
        raise SystemExit("no trajectories selected (need full hacks with hack-turn annotations).")

    # --- build replay data per trajectory (reads each sample once; free) ---
    print(f"\n[build] reconstructing replay data + first-hack turn for {len(selected)} trajectory(ies) ...")
    rds: dict[int, object] = {}
    for tid in selected:
        rds[tid] = build_replay_data(refs[tid], originals[tid]["transcript"])

    # --- per-trajectory location map (k, n_resumes, j per location) ---
    print("\n[plan] per-trajectory cut turns (k = first-hack turn; '-' = location skipped):")
    print(f"  {'id':>4}  {'target':<22} {'seed':<30} {'k':>2} {'res':>3}  "
          f"{'begin':>5} {'mid':>4} {'before':>6} {'after':>5}")
    for tid in selected:
        rd = rds[tid]
        n = len(rd.resume_auditor_turns)
        cells = {loc: (LOCATIONS[loc][0](rd.k) if LOCATIONS[loc][3](rd.k, n) else None) for loc in ORDER}
        fmt = lambda v: (str(v) if v is not None else "-")
        print(f"  {tid:>4}  {refs[tid].target_model.split('/')[-1]:<22} {refs[tid].seed:<30} "
              f"{rd.k:>2} {n:>3}  {fmt(cells['begin']):>5} {fmt(cells['middle']):>4} "
              f"{fmt(cells['before']):>6} {fmt(cells['after']):>5}")

    # --- build the cell plan with applicability + dedup ---
    existing = _existing_cell_ids()
    plan = []  # (loc, cond, label, prompt, specs)
    print("\n[plan] cells (location x condition):")
    for loc in cfg["locations"]:
        jf, prompt_key, insert_mode, applies = LOCATIONS[loc]
        for cond in cfg["conditions"]:
            label = cell_label(loc, cond)
            prompt = None if cond == "control" else PROMPTS[prompt_key]
            already = existing.get(label, set())
            specs, skip_applic, skip_dedup, skip_err = [], 0, 0, 0
            for tid in selected:
                rd = rds[tid]
                n = len(rd.resume_auditor_turns)
                if not applies(rd.k, n):
                    skip_applic += 1
                    continue
                if tid in already:
                    skip_dedup += 1
                    continue
                try:
                    specs.append(make_resume_spec(rd, jf(rd.k), prompt, insert_mode))
                except ValueError as e:
                    skip_err += 1
                    print(f"    (skip id={tid} {loc}/{cond}: {e})")
            plan.append((loc, cond, label, prompt, specs))
            print(f"  {loc:>6}/{cond:<9} label={label:<16} -> {len(specs):>3} to run "
                  f"({skip_applic} n/a, {skip_dedup} already done"
                  + (f", {skip_err} errored" if skip_err else "") + ")")

    total = sum(len(specs) for *_, specs in plan) * cfg["N"]
    print(f"\n[plan] total continuations to generate: {total} "
          f"({sum(len(s) for *_, s in plan)} cells-trajectories x N={cfg['N']})")

    if cfg["dry_run"]:
        print("\n[dry-run] plan validated (cuts + applicability + dedup). No run, no cost.")
        return
    if total == 0:
        print("\nNothing to generate (everything deduped or n/a). Running judge+viewer on existing runs.")

    # --- STAGE 1: ONE combined eval_set over every cell's tasks (full concurrency, so a
    # stalled sample never blocks other cells), then SPLIT the results back into per-cell
    # dirs the judge/viewer already understand. Task names encode the cell for the split. ---
    dims = get_rollback_dims()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_tasks = []
    for loc, cond, label, prompt, specs in plan:
        for s in specs:
            all_tasks.append(build_rollback_task(
                s, dims, task_name=_combined_task_name(s.ref.traj_id, loc, cond, s.ref.seed)))

    new_dirs: list[Path] = []
    if all_tasks:
        work_dir = LOGS / f"{WORK_PREFIX}{cfg['N']}x-{ts}"
        work_dir.mkdir(parents=True, exist_ok=True)
        _write_rbmeta(work_dir, plan)  # before eval_set: a killed run can still be split
        # How many TASKS to keep active at once. The real throughput ceiling is the shared
        # auditor connection pool (max_connections=concurrency), so we keep exactly enough
        # tasks open to keep that pool fed: with `concurrency` tasks active, even in the
        # worst case where every open task is down to its last slow sample, there are still
        # `concurrency` samples ready -- exactly enough to saturate the pool. Fewer (the old
        # ceil(concurrency/N) formula, ~18) let the pool starve MID-RUN as active tasks
        # dwindled to stragglers (the slowness we were chasing); more buys nothing -- the
        # pool is the ceiling -- and just wastes memory + clutters the display. Decoupled
        # from N on purpose. (Open log files are not the constraint: ulimit -n is ~1M here.)
        max_tasks = min(len(all_tasks), cfg["concurrency"])
        print("\n" + "=" * 76)
        print(f"STAGE 1 GENERATE  {len(all_tasks)} tasks x N={cfg['N']} = {total} continuations "
              f"(ONE pool, concurrency={cfg['concurrency']}, max_tasks={max_tasks})  -> {work_dir.name}")
        print("=" * 76)
        try:
            success, logs = eval_set(
                all_tasks, epochs=cfg["N"], max_tasks=max_tasks,
                max_samples=cfg["concurrency"], max_connections=cfg["concurrency"],
                log_dir=str(work_dir),
            )
            print(f"  eval_set finished, success={success}")
        except Exception as e:
            print(f"  !! GENERATE crashed (will still split whatever completed): {type(e).__name__}: {e}")
            traceback.print_exc()
        new_dirs = _redistribute(work_dir)
        print(f"  split into {len(new_dirs)} per-cell dir(s)")

    # --- STAGE 2: judge + annotate red rows. Judge ALL rollback dirs (incremental: dirs
    # already judged are skipped) so the new cells, plus any leftover/unjudged dirs from a
    # killed prior run, all get covered -- nothing orphaned. ---
    if not cfg["skip_judge"]:
        all_rb_dirs = sorted(d for d in LOGS.iterdir()
                             if d.is_dir() and d.name.startswith(ROLLBACK_PREFIX))
        print("\n" + "=" * 76)
        print(f"STAGE 2 JUDGE + ANNOTATE  ({len(all_rb_dirs)} rollback dir(s), red rows only, incremental)")
        print("=" * 76)
        try:
            run_judging(all_rb_dirs, model=cfg["annotate_model"],
                        concurrency=cfg["concurrency"], force=cfg["force_judge"], red_only=True)
        except Exception as e:
            print(f"  !! JUDGE STAGE FAILED (continuing to viewer): {type(e).__name__}: {e}")
            traceback.print_exc()
    elif cfg["skip_judge"]:
        print("\n(skipping STAGE 2 judge: --skip-judge)")

    # --- STAGE 3: rebuild the viewer ---
    if not cfg["skip_viewer"]:
        print("\n" + "=" * 76)
        print("STAGE 3 VIEWER (rebuild, free)")
        print("=" * 76)
        try:
            asyncio.run(make_viewer.main())
        except Exception as e:
            print(f"  !! VIEWER STAGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 3 viewer: --skip-viewer)")

    print("\n" + "=" * 76)
    print(f"ROLLBACK PIPELINE DONE.  open {DATA / 'viewer' / 'index.html'} "
          "(rollbacks fold into the full-hack rows; see the Visuals tab for charts)")
    print("=" * 76)


if __name__ == "__main__":
    main()
