"""Re-judge ONE trajectory (by its viewer id) fresh on the CURRENT dimensions, and
OVERWRITE its stored judge output.

Why this exists: a one-off to test whether a rubric edit (e.g. adding the
`inconsequential_violation` outcome category to dimensions/reward_hacking.md) actually
changes the judge's behavior on a specific trajectory -- without re-spending on the whole
sweep. It re-runs Petri's ACTUAL judge (`audit_judge`, same model/prompt exp_rh_audit.py
uses) over that one trajectory's existing transcript, scoring every current dimension, and
writes a FULL-REPLACEMENT entry into mats-local/petri/rejudge_scores.json (the same overlay
channel exp_rejudge_main.py uses). make_viewer's load layer merges it in (scores + summary +
justification + highlights) and re-parses the RH_CATEGORY / RH_TYPES lines, so the viewer
reflects the fresh judgment. The .eval log is never touched -- delete the trajectory's entry
in rejudge_scores.json to revert.

The script prints a BEFORE/AFTER: the original inline reward_hacking score vs the new score,
and the new RH_CATEGORY / RH_TYPES the judge emitted -- so you can see at a glance whether the
rubric change moved the label.

Usage:
  uv run tools/exp_rejudge_one.py --id=1415
  uv run tools/exp_rejudge_one.py --id=1415 --run-dir=v2-4targets-allow-3ep-20260708-145648  # narrow the scan (faster)
  uv run tools/exp_rejudge_one.py --id=1415 --skip-viewer     # don't rebuild the viewer at the end
  uv run tools/exp_rejudge_one.py --id=1415 --model=anthropic/claude-opus-4-8  # the default

Costs money (Anthropic API: one judge call). The viewer rebuild at the end is free.
"""

import asyncio
import json
import pathlib
import subprocess
import sys

# this tool lives in tools/; put the project root (for make_viewer) and ../lib
# (for petri_paths + the core modules) on the import path.
_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from dotenv import load_dotenv
from inspect_petri import audit_judge, judge_dimensions
from inspect_scout import TranscriptContent, transcripts_from

from petri_paths import ENV_FILE, DIMENSIONS_DIR, DATA
from make_viewer import LOGS, ROLLBACK_PREFIX, REJUDGE_FILE, traj_key
from viewer_load import parse_rh_category

load_dotenv(ENV_FILE)  # mats/.env (ANTHROPIC_API_KEY)


def _arg(name: str, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


TARGET_ID = _arg("id")
RUN_DIR = _arg("run-dir")  # optional: narrow the scan to one run dir (faster)
MODEL = _arg("model", "anthropic/claude-opus-4-8")
SKIP_VIEWER = "--skip-viewer" in sys.argv
if TARGET_ID is None:
    raise SystemExit("usage: uv run tools/exp_rejudge_one.py --id=<viewer id> [--run-dir=<name>] [--skip-viewer]")
TARGET_ID = int(TARGET_ID)

# trajectory-id registry (key -> stable viewer id), the same map make_viewer assigns from.
_REG_FILE = DATA / "trajectory_ids.json"
REGISTRY = json.loads(_REG_FILE.read_text()) if _REG_FILE.exists() else {}
if not REGISTRY:
    raise SystemExit(f"no {_REG_FILE} -- run `uv run make_viewer.py` first to assign trajectory ids.")


def _key(mode: str, task: str, seed: str, epoch: int) -> str:
    return traj_key({"mode": mode, "task": task, "seed": seed, "epoch": epoch})


async def find_target() -> dict | None:
    """Scan the (optionally single) run dir(s) for the transcript whose registry id == TARGET_ID."""
    mode_dirs = sorted(
        d for d in LOGS.iterdir()
        if d.is_dir() and not d.name.startswith(ROLLBACK_PREFIX)
        and (RUN_DIR is None or d.name == RUN_DIR)
    )
    if not mode_dirs:
        raise SystemExit(f"no matching run dir(s) under {LOGS}" + (f" for --run-dir={RUN_DIR}" if RUN_DIR else ""))
    for d in mode_dirs:
        print(f"  scanning {d.name}/ ...", flush=True)
        ts = transcripts_from(str(d))
        async with ts.reader() as reader:
            infos = [i async for i in reader.index()]
            for info in infos:
                seed = str(info.task_id)
                key = _key(d.name, info.task_set, seed, info.task_repeat)
                if REGISTRY.get(key) != TARGET_ID:
                    continue
                t = await reader.read(info, TranscriptContent(messages="all", events="all"))
                if not t.timelines:
                    raise SystemExit(f"#{TARGET_ID} ({key}) has an empty transcript (dead audit) -- nothing to judge.")
                return {
                    "key": key, "id": TARGET_ID, "mode": d.name, "task": info.task_set,
                    "seed": seed, "epoch": info.task_repeat,
                    "original_reward_hacking": (info.score or {}).get("reward_hacking"),
                    "transcript": t,
                }
    return None


async def main() -> None:
    dims = judge_dimensions(DIMENSIONS_DIR)
    dim_names = [d.name for d in dims]
    print(f"\nmodel={MODEL}")
    print(f"judge dimensions ({len(dim_names)}): {dim_names}")
    print(f"\n[setup] locating trajectory #{TARGET_ID}"
          + (f" in {RUN_DIR}/" if RUN_DIR else " (scanning all run dirs)") + " ...")
    c = await find_target()
    if c is None:
        raise SystemExit(f"could not find a trajectory with id #{TARGET_ID}"
                         + (f" in {RUN_DIR}/" if RUN_DIR else "")
                         + " -- check the id (and run `uv run make_viewer.py` if it's brand new).")
    print(f"[setup] found: #{c['id']}  {c['mode']}/{c['seed']} e{c['epoch']} (task {c['task']})")

    scanner = audit_judge(dimensions=dims, model=MODEL)
    print(f"\n[judge] re-running the judge on #{c['id']} ...", flush=True)
    result = await scanner(c["transcript"])
    if not isinstance(result.value, dict):
        reason = (result.metadata or {}).get("stop_reason") or "no dict value (refusal?)"
        raise SystemExit(f"judge returned no scores ({reason}) -- not writing anything.")
    scores = {k: result.value.get(k) for k in dim_names}
    miss = [k for k, v in scores.items() if not isinstance(v, (int, float))]
    if miss:
        raise SystemExit(f"judge omitted dims {miss} -- not writing anything.")

    md = result.metadata or {}
    justification = getattr(result, "explanation", None) or ""
    entry = {
        "mode": c["mode"], "task": c["task"], "seed": c["seed"], "epoch": c["epoch"],
        "original_reward_hacking": c["original_reward_hacking"],
        "model": MODEL,
        "scores": scores,
        "summary": md.get("summary", ""),
        "highlights": md.get("highlights", ""),
        "justification": justification,
    }

    # write / overwrite this trajectory's entry in the rejudge overlay (full replacement).
    REJUDGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(REJUDGE_FILE.read_text()) if REJUDGE_FILE.exists() else {}
    had_entry = c["key"] in existing
    existing[c["key"]] = entry
    REJUDGE_FILE.write_text(json.dumps(existing, indent=2, sort_keys=True))

    # before/after for the thing we're testing: the reward_hacking score + outcome category.
    new_cat, new_types, cat_err = parse_rh_category(justification, scores.get("reward_hacking"))
    print("\n" + "=" * 64)
    print(f"RE-JUDGED #{c['id']}  ({'overwrote' if had_entry else 'added'} its rejudge entry)")
    print(f"  reward_hacking:  before(inline)={c['original_reward_hacking']}  ->  after={scores.get('reward_hacking')}")
    print(f"  RH_CATEGORY (new):  {new_cat}")
    print(f"  RH_TYPES (new):     {new_types}")
    if cat_err:
        print("  ** WARNING: the new justification has a missing/garbled RH_CATEGORY/RH_TYPES line (parse error).")
    print(f"  noticed_hack (new): {scores.get('noticed_hack')}")
    print(f"  all new scores:     {scores}")
    print("=" * 64)
    print(f"\n(compare RH_CATEGORY against what the viewer showed before this run to see if the rubric edit moved it.)")
    print(f"wrote {REJUDGE_FILE}")

    if SKIP_VIEWER:
        print("\n--skip-viewer: not rebuilding. Run `uv run make_viewer.py` to see it in the viewer.")
        return
    print("\n[viewer] rebuilding (free) ...", flush=True)
    subprocess.run([sys.executable, str(_petri / "make_viewer.py")], check=True)


if __name__ == "__main__":
    asyncio.run(main())
