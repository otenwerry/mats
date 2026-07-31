"""Hack-turn annotation for the environments project's own runs.

The annotation ENGINE is petri's (annotate_agent's agentic loop + exp_annotate_hacks'
run_annotation, which already takes ann_file as a parameter and already switches its
prompt on target_tools_mode == "real"). This module only supplies what differs: which log
dirs to load, and where the annotations file lives.

Free to import; run_annotation costs money (one short agentic loop per annotated hack).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENVIRONMENTS = Path(__file__).resolve().parents[1]
_PETRI = _ENVIRONMENTS.parent / "petri"
for _p in (str(_ENVIRONMENTS / "lib"), str(_PETRI / "lib"), str(_PETRI)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import viewer_load  # noqa: E402
from env_paths import ANN_FILE, DATA, LOGS, annotation_file_lock  # noqa: E402

# The load layer derives its cache + overlay paths from a data root at import; point it at
# ours before anything loads (see viewer_load.set_data_root for why this matters).
viewer_load.set_data_root(DATA)

from exp_annotate_hacks import DEFAULT_CONCURRENCY, DEFAULT_MODEL, run_annotation  # noqa: E402,F401


async def load_all_real_audits() -> list[dict]:
    """Every trajectory across this project's run dirs. An empty/missing logs dir is NOT
    an error (petri's equivalent raises): a project with no runs yet is a normal state."""
    if not LOGS.exists():
        print(f"  (no logs dir yet at {LOGS})")
        return []
    mode_dirs = sorted(d for d in LOGS.iterdir() if d.is_dir())
    if not mode_dirs:
        print(f"  (no run directories under {LOGS})")
        return []
    audits: list[dict] = []
    skipped_dirs = 0
    for mode_dir in mode_dirs:
        print(f"loading {mode_dir.name}/ ...")
        try:
            audits.extend(await viewer_load.load_mode(mode_dir))
        except Exception as ex:  # noqa: BLE001
            # A concurrently running or abandoned eval can contain only its start
            # journal. The shared transcript reader cannot query that archive yet, but
            # it must not block annotations for every completed directory.
            skipped_dirs += 1
            print(
                f"  WARNING: skipping unreadable/incomplete run dir "
                f"{mode_dir.name}/ during annotation load "
                f"({type(ex).__name__}: {ex})"
            )
    if skipped_dirs:
        print(
            f"  NOTE: skipped {skipped_dirs} run dir(s); completed readable runs "
            "will still be considered for annotation"
        )
    return audits


async def annotate_real_hacks(model: str = DEFAULT_MODEL,
                              concurrency: int = DEFAULT_CONCURRENCY,
                              force: bool = False) -> dict:
    # Keep the lock through load + annotation + checkpoints. Without this, two
    # pipelines finishing together can both start from the same annotations.json,
    # duplicate paid work, and let the last writer discard the other's results.
    print("  waiting for the environments annotation lock ...")
    with annotation_file_lock(DATA):
        print("  annotation lock acquired")
        audits = await load_all_real_audits()
        if not audits:
            print("  (nothing to annotate yet)")
            return {"done": 0, "failed": 0, "candidates": 0}
        return await run_annotation(
            audits,
            model=model,
            concurrency=concurrency,
            force=force,
            ann_file=ANN_FILE,
        )
