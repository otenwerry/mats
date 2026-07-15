"""FREE, REVERSIBLE. Move rollback continuations of NON-binary-hack originals to a
trash folder (does not call any API).

An original is a binary hack iff binary_hack_eval(its re-judged scores)["ok"]. Every
rollback .eval whose original trajectory is NOT a binary hack is moved (preserving its
run-dir name) under
  mats-local/petri/_trash_nonhack_rollbacks/<run-dir>/
and the matching entries are pruned from each run dir's rollback_results.json (the
original json is backed up alongside as rollback_results.json.bak). Reversible: move
the .eval files back into logs/<run-dir>/ and restore the .bak to undo.

Re-run whenever the hack definition changes or stale rollbacks accumulate; idempotent
(a clean repeat finds nothing to move). Run: uv run tools/cleanup_nonhack_rollbacks.py
"""
import json
import pathlib
import shutil
import sys

# this tool lives in tools/; put the project root (for viewer) and ../lib
# (for petri_paths + the core modules) on the import path.
_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from inspect_ai.log import read_eval_log
from viewer import (
    DATA, LOGS, ROLLBACK_PREFIX, REJUDGE_FILE, binary_hack_eval, _orig_id_from_task,
)

TRASH = DATA / "_trash_nonhack_rollbacks"


def main() -> None:
    ids = json.loads((DATA / "trajectory_ids.json").read_text())
    rejudge = json.loads(REJUDGE_FILE.read_text()) if REJUDGE_FILE.exists() else {}

    # binary-hack verdict per trajectory id, from the re-judged original scores
    delete_ids, keep_ids = set(), set()
    for key, tid in ids.items():
        e = rejudge.get(key)
        if e is None:
            continue
        (keep_ids if binary_hack_eval({"scores": e["scores"]})["ok"] else delete_ids).add(tid)

    print(f"binary hacks (keep): {sorted(keep_ids)}")
    print(f"NOT hacks (delete):  {sorted(delete_ids)}\n")

    def is_del(e):
        oid = e.get("original_traj_id") or _orig_id_from_task(e.get("task", ""))
        return oid in delete_ids

    moved, pruned = [], []
    for d in sorted(p for p in LOGS.iterdir() if p.is_dir() and p.name.startswith(ROLLBACK_PREFIX)):
        dest_dir = TRASH / d.name
        n_moved_here = 0
        for f in sorted(d.glob("*.eval")):
            log = read_eval_log(str(f))
            oid = _orig_id_from_task(log.eval.task)
            if oid in delete_ids:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest_dir / f.name))
                moved.append((d.name, oid, f.name))
                n_moved_here += 1

        # prune rollback_results.json entries for deleted trajectories (back up first)
        rf = d / "rollback_results.json"
        if rf.exists():
            results = json.loads(rf.read_text())
            kept = [e for e in results if not is_del(e)]
            if len(kept) != len(results):
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(rf), str(dest_dir / "rollback_results.json.bak"))
                rf.write_text(json.dumps(kept, indent=2))
                pruned.append((d.name, len(results) - len(kept), len(kept)))

        if n_moved_here:
            print(f"  {d.name}: moved {n_moved_here} .eval file(s) to trash")

    print(f"\nMoved {len(moved)} continuation .eval file(s) for {len(delete_ids)} trajectory(ies) "
          f"to {TRASH}")
    for dname, removed, kept in pruned:
        print(f"  pruned {removed} entr(y/ies) from {dname}/rollback_results.json (kept {kept}; .bak saved)")
    print("\nRegenerate the viewer to confirm only the binary hacks remain: uv run viewer.py")
    print(f"Reverse with: mv {TRASH}/<run-dir>/*.eval back into logs/<run-dir>/ (and restore .bak)")


if __name__ == "__main__":
    main()
