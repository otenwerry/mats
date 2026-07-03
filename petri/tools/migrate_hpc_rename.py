"""One-time, FREE migration: rename the honest_path_clarity dimension -> honest_path_ambiguity.

The dim was renamed on 2026-06-28 so the score direction is self-evident (high = ambiguous /
ill-posed = bad; low = clear honest path = good). The rubric file, viewer, and judge config
were updated to the new key; this brings the stored scores in line. All existing scores live
in rejudge_scores.json (verified earlier: nothing else carries the dim), so this touches only
that file. It:
  1. backs the file up to rejudge_scores.json.bak-hpc-rename;
  2. renames every scores["honest_path_clarity"] -> scores["honest_path_ambiguity"]
     (preserving the value, which was already inverted by migrate_hpc_reframe.py);
  3. renames the dim NAME where it's cited in prose (summary/justification/highlights):
     "honest_path_clarity" -> "honest_path_ambiguity" and the display form
     "Honest Path Clarity" -> "Honest Path Ambiguity";
  4. stamps mats-local/petri/migrations_applied.json so it can't run twice.

FREE: pure local data transform, no API calls. Run once (AFTER migrate_hpc_reframe.py):
  uv run tools/migrate_hpc_rename.py
"""

import json
import pathlib
import sys

_petri = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))

from petri_paths import DATA  # noqa: E402

REJUDGE = DATA / "rejudge_scores.json"
BACKUP = DATA / "rejudge_scores.json.bak-hpc-rename"
MARKERS = DATA / "migrations_applied.json"
MARKER_ID = "hpc_rename_to_ambiguity"
OLD, NEW = "honest_path_clarity", "honest_path_ambiguity"
OLD_DISP, NEW_DISP = "Honest Path Clarity", "Honest Path Ambiguity"
FORCE = "--force" in sys.argv


def main() -> None:
    if not REJUDGE.exists():
        raise SystemExit(f"{REJUDGE} not found.")
    applied = json.loads(MARKERS.read_text()) if MARKERS.exists() else []
    if MARKER_ID in applied and not FORCE:
        raise SystemExit(f"Migration {MARKER_ID!r} already applied (see {MARKERS}). Refusing.")

    data = json.loads(REJUDGE.read_text())
    BACKUP.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"backed up {REJUDGE.name} -> {BACKUP.name}")

    n_key, n_text = 0, 0
    for v in data.values():
        if not isinstance(v, dict):
            continue
        scores = v.get("scores", {})
        if OLD in scores:
            # keep the value; if NEW somehow already exists, don't clobber it
            scores.setdefault(NEW, scores[OLD])
            del scores[OLD]
            n_key += 1
        for f in ("summary", "justification", "highlights"):
            t = v.get(f)
            if isinstance(t, str) and (OLD in t or OLD_DISP in t):
                v[f] = t.replace(OLD, NEW).replace(OLD_DISP, NEW_DISP)
                n_text += 1

    REJUDGE.write_text(json.dumps(data, indent=2, sort_keys=True))
    MARKERS.write_text(json.dumps(applied + [MARKER_ID], indent=2))
    print(f"renamed dim key on {n_key} entrie(s); renamed prose references in {n_text} entrie(s)")
    print(f"stamped {MARKER_ID!r} in {MARKERS.name}")


if __name__ == "__main__":
    main()
