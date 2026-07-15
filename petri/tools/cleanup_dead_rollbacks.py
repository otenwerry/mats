"""FREE, REVERSIBLE. Move rollback continuation .eval files whose samples ALL died
(errored / cancelled / produced 0 target output -- e.g. a run that crashed or was
cancelled mid-flight) to a trash folder, and prune their (garbage) entries from
rejudge_scores.json.

A file with a MIX of dead and live samples is left untouched and reported, so good
data is never dropped. Reversible: move the .eval back into logs/<dir>/ and re-judge.

  uv run tools/cleanup_dead_rollbacks.py --dry-run   # preview only, no changes
  uv run tools/cleanup_dead_rollbacks.py             # move all-dead files + prune rejudge
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

from inspect_ai.log import list_eval_logs, read_eval_log
from viewer import DATA, LOGS, ROLLBACK_PREFIX, REJUDGE_FILE, traj_key

TRASH = DATA / "_trash_dead_rollbacks"


def target_out(log, s) -> int:
    roles = log.eval.model_roles or {}
    tm = getattr(roles.get("target"), "model", None) or str(roles.get("target", ""))
    short = tm.split("/")[-1]
    return sum((u.output_tokens or 0) for m, u in (s.model_usage or {}).items()
               if m.split("/")[-1] == short)


def main() -> None:
    DRY = "--dry-run" in sys.argv
    rj = json.loads(REJUDGE_FILE.read_text()) if REJUDGE_FILE.exists() else {}
    moved = pruned = 0
    warned = []
    print(f"{'DRY-RUN: ' if DRY else ''}scanning rollback dirs for all-dead continuation files ...\n")
    for d in sorted(p for p in LOGS.iterdir() if p.is_dir() and p.name.startswith(ROLLBACK_PREFIX)):
        for f in sorted(d.glob("*.eval")):
            log = read_eval_log(str(f))
            samples = log.samples or []
            if not samples:
                continue
            dead = [s for s in samples
                    if getattr(s, "error", None) is not None or target_out(log, s) == 0]
            if not dead:
                continue
            if len(dead) < len(samples):
                warned.append((d.name, f.name, len(dead), len(samples)))
                continue
            keys = [traj_key({"mode": d.name, "task": log.eval.task,
                              "seed": str(s.id), "epoch": s.epoch}) for s in samples]
            in_rj = [k for k in keys if k in rj]
            print(f"  {'WOULD MOVE' if DRY else 'MOVED'} {d.name}/{f.name}  "
                  f"({len(samples)} dead samples, {len(in_rj)} rejudge entries)")
            if not DRY:
                dest = TRASH / d.name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(dest / f.name))
                for k in in_rj:
                    del rj[k]
            moved += 1
            pruned += len(in_rj)

    if not DRY and pruned:
        REJUDGE_FILE.write_text(json.dumps(rj, indent=2, sort_keys=True))

    print(f"\n{'Would move' if DRY else 'Moved'} {moved} all-dead file(s); "
          f"{'would prune' if DRY else 'pruned'} {pruned} rejudge entr(y/ies).")
    for dn, fn, nd, nt in warned:
        print(f"  WARNING (left in place): {dn}/{fn} has {nd}/{nt} dead samples (mixed) -- review by hand.")
    if DRY:
        print("\nDry run only -- re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
