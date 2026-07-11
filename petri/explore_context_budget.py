"""Context-budget check for the planned split continuation experiment (FREE, local reads).

The 2026-07-05 baseline screening run died with 'Compaction insufficient': the auditor
inherited from the sweep-5 originals is DeepSeek V4 Pro (128k window; compaction threshold
115,200 = 90%), and the auditor's UN-COMPACTABLE prefix (system prompt + seed + the full
original-run reference blob, sent uncut by design) exceeded it outright for the large
candidates. This script prints the sizes that matter for every planned id:

  - candidates: reference-blob chars (what the auditor carries all run long) + whether the
    ORIGINAL run itself compacted (a sign it already neared the auditor window);
  - prefixes: reconstructed prefix-transcript chars (what the TARGET would carry in the
    prefixed conditions -- the second half has the same problem on the target side for
    128k-window targets).

Char->token ratio for this content is bad (~2 chars/token, calibrated from the observed
failure: ref 216,184 chars -> prefix 131,686 tokens incl. system+seed).
"""

import asyncio
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))

import make_viewer as mv
import exp_continuation as C
from exp_resample import build_reference_blob, load_original_refs

PREFIX_IDS = {
    "full_hack_prefix": [1240, 1136, 1159, 1210, 1218],
    "corrected_hack_prefix": [973, 996, 1018, 733, 1077],
    "clean_prefix": [1114, 998, 1158, 1208, 1219],
}
CANDIDATES = [1103, 1150, 1178, 1173, 1203, 1196, 1229, 1215]

RATIO = 2.0          # chars/token, calibrated above -- an ESTIMATE, labeled as such
AUD_WINDOW = 128_000  # deepseek v4 pro (from the observed threshold 115,200 = 90%)


def _msgs_chars(msgs) -> int:
    return sum(len(C._message_text_blob(m)) for m in msgs)


async def main() -> None:
    by_id = await mv.load_originals_by_id()
    all_ids = sorted({i for ids in PREFIX_IDS.values() for i in ids} | set(CANDIDATES))
    refs = await load_original_refs(all_ids)

    print("=== candidates (auditor side: sys+seed+REFERENCE is un-compactable prefix) ===")
    print(f"    est tokens = chars/{RATIO}; auditor threshold 115,200 (window {AUD_WINDOW:,})")
    for cid in CANDIDATES:
        r = refs[cid]
        a = by_id.get(cid, {})
        blob = build_reference_blob(r)
        est = int(len(blob) / RATIO)
        comp = "  ORIGINAL COMPACTED" if a.get("compactions") else ""
        print(f"  #{cid}: {r.target_name:16s} {mv.seed_label(r.seed):18s} "
              f"ref={len(blob):>9,} chars (~{est:>7,} tok)  auditor={r.auditor_model}"
              f"{comp}")

    print("\n=== prefixes (target side, 2nd half: prefix transcript sits in the TARGET's context) ===")
    # build_plans is now per-treatment; this diagnostic just wants every prefix's reconstructed
    # transcript, so pass all ids under one dummy treatment and label each by its PREFIX_IDS flavor.
    flavor = {i: flav for flav, ids in PREFIX_IDS.items() for i in ids}
    plans = await C.build_plans("full-hack", sorted(flavor), CANDIDATES)
    for model, plan in sorted(plans.items()):
        for p in plan.prefixes:
            a = by_id.get(p.ref.traj_id, {})
            chars = _msgs_chars(p.messages)
            comp = "  ORIGINAL COMPACTED" if a.get("compactions") else ""
            print(f"  #{p.ref.traj_id}: {model:16s} {flavor.get(p.ref.traj_id, '?'):22s} "
                  f"{mv.seed_label(p.ref.seed):18s} prefix={chars:>9,} chars "
                  f"(~{int(chars / RATIO):>7,} tok){comp}")


if __name__ == "__main__":
    asyncio.run(main())
