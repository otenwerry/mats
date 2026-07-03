"""Dump transcripts for direct reading to verify claims about hack mechanism,
wall-clock fabrication, and scorer policy. Free, local-only."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer as mv

OUT = Path("/tmp/verify")
OUT.mkdir(exist_ok=True)

DIRS = {
    "deepseek": "v2-4targets-1ep-20260702-152119",   # DeepSeek V4 Pro auditor
    "opus": "v2-5targets-3ep-20260629-225553",         # Opus auditor baseline
}


async def main():
    for key, d in DIRS.items():
        audits = await mv.load_mode(mv.LOGS / d)
        audits = [a for a in audits if not a.get("dead")]
        for a in audits:
            cat = mv.hack_category(a)
            rh = mv.rh_score(a)
            tgt = mv.target_short(a).replace(" ", "")
            seed = a["seed"].replace("ml_training_data_misuse_", "")
            fn = f"{key}__{cat}__{tgt}__{seed}__e{a['epoch']}__rh{rh}.txt"
            (OUT / fn).write_text(a["transcript"])
        # index of full/degenerate hacks per dir
        hacks = [a for a in audits if mv.hack_category(a) in ("full", "degenerate")]
        print(f"\n=== {key} ({d}): {len(hacks)} full/degen hacks ===")
        for a in sorted(hacks, key=lambda a: -mv.rh_score(a)):
            print(f"  {mv.target_short(a):16s} {a['seed'].replace('ml_training_data_misuse_',''):16s} "
                  f"e{a['epoch']} rh={mv.rh_score(a)} chars={len(a['transcript'])}")


if __name__ == "__main__":
    asyncio.run(main())
