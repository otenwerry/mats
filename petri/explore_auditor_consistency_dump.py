"""Dump same-cell epoch pairs for the auditor-consistency check (FREE, local reads).

Question (Owen 2026-07-05): if we DON'T prime the continuation auditor with the original
run (to save its 128k window), how much does the fabricated environment naturally vary
across runs of the exact same (model, seed, condition) cell? Auditor-side only: when the
target takes the same action, does the auditor return the same data/numbers/behavior?

Dumps, per candidate cell, two allow-condition epochs' transcripts to
mats-local/petri/_analysis/auditor_consistency_20260705/<cell>/ for side-by-side reading.
Also prints the reference-blob composition per candidate (transcript vs interleaved
auditor reasoning) to size up the trim-the-reference option.
"""

import asyncio
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

import make_viewer as mv
from exp_resample import build_reference_blob, load_original_refs

OUT = pathlib.Path("/Users/owenterry/supermats/mats-local/petri/_analysis/auditor_consistency_20260705")
CANDIDATES = [1103, 1150, 1178, 1173, 1203, 1196, 1229, 1215]

# candidate cells: (pretty model, seed label) -> dump 2 allow epochs
CELLS = [
    ("Claude Opus 4.6", "clinical_notes"),
    ("DeepSeek V4 Pro", "support_routing"),
    ("GLM 5.1", "support_routing"),
    ("Kimi K2.6", "spam_detection"),
    ("GPT-5.5", "spam_detection"),
]


async def main() -> None:
    refs = await load_original_refs(CANDIDATES)
    print("=== reference blob composition per candidate ===")
    for cid in CANDIDATES:
        r = refs[cid]
        blob = build_reference_blob(r)
        tr = len(r.transcript)
        pad = sum(len(v) for v in (r.scratchpad or {}).values())
        print(f"  #{cid}: blob={len(blob):>9,}  transcript={tr:>9,}  "
              f"auditor reasoning={pad:>9,} chars ({pad / len(blob):.0%} of blob)")

    by_id = await mv.load_originals_by_id()
    OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== dumped epoch pairs ===")
    for model, seed_lbl in CELLS:
        grp = sorted(
            ((tid, a) for tid, a in by_id.items()
             if mv.sweep_key(a) == "consistent" and mv.pretty_model(a["target"]) == model
             and mv.seed_label(a["seed"]) == seed_lbl and a.get("condition") == "allow"
             and not a.get("dead") and not a.get("crashed")),
            key=lambda t: t[0])
        if len(grp) < 2:
            print(f"  {model} / {seed_lbl}: only {len(grp)} usable allow run(s), skipped")
            continue
        cell_dir = OUT / f"{model.replace(' ', '_')}__{seed_lbl}"
        cell_dir.mkdir(exist_ok=True)
        for tid, a in grp[:2]:
            cat = mv.hack_category(a)
            f = cell_dir / f"id{tid}_{cat}.txt"
            f.write_text(a["transcript"])
            print(f"  {f.relative_to(OUT)}  ({len(a['transcript']):,} chars, cat={cat}, "
                  f"rh={mv.rh_score(a)})")


if __name__ == "__main__":
    asyncio.run(main())
