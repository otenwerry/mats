"""Record which flagged runs exceeded the re-judge's context cap. (free)

The Opus re-judge reads the trace via locate.condense_trace(MAX_CHARS=500k);
over-long traces get cut with an in-prompt "… trace truncated at event N of
M …" note — visible to the judge, but until now not recorded anywhere. This
script recomputes the cut deterministically and writes

    "judge_input_truncated": {"seen_events": N, "total_events": M}

into highlights/{run_id}.json for affected runs (and removes the field if a
re-run no longer truncates, e.g. after raising MAX_CHARS). The viewer shows a
caveat in the run banner. Re-run this after any re-judge with changed limits.

Usage: uv run python posttrainbench/flag_truncated_judge_input.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)

import json
import re

import locate
import paths
from exp_localize_hacks import load_events
from exp_rejudge_hacks import MAX_CHARS, OUT_CLIP, TEXT_CLIP

_NOTE = re.compile(r"truncated at event (\d+) of (\d+)")


def main():
    n = 0
    for p in sorted(paths.HIGHLIGHTS.glob("*.json")):
        if p.name.startswith("_debug") or p.name == "categories.json":
            continue
        h = json.loads(p.read_text())
        if not h.get("run_id") or not h.get("reassessment"):
            continue
        rows = locate.condense_trace(load_events(h["run_id"]), MAX_CHARS, OUT_CLIP, TEXT_CLIP)
        m = _NOTE.search(rows[-1].get("text", "")) if rows and rows[-1].get("kind") == "note" else None
        field = {"seen_events": int(m.group(1)), "total_events": int(m.group(2))} if m else None
        if field and h.get("judge_input_truncated") != field:
            h["judge_input_truncated"] = field
            p.write_text(json.dumps(h, indent=1))
            n += 1
            print(f"  ✂ {h['run_id'][:62]}  judge saw {field['seen_events']}/{field['total_events']} events")
        elif field:
            n += 1
        elif "judge_input_truncated" in h:
            del h["judge_input_truncated"]
            p.write_text(json.dumps(h, indent=1))
    print(f"Done: {n} runs flagged judge_input_truncated.")


if __name__ == "__main__":
    main()
