"""READ-ONLY sanity check: did the prefix-A slice actually fire for the prefixed
continuation conditions in a Kimi run? No API calls, no cost.

For each continuation in the run dir we:
  - load it exactly as the pipeline does (make_viewer.load_mode -> the viewer transcript),
  - find the pivot via _continuation_cut_m (the "move on to a different task" needle),
  - run _slice_judge_transcript (the SAME function both judges use) and report what got cut:
      full [M#] range, pivot M, head/tail ranges, chars before/after,
  - print the PRIMARY judge's own summary/justification (from the .eval) so we can see
    whether it betrays any knowledge of the prior task (it shouldn't, if the slice worked).
"""
import asyncio
import pathlib
import re
import sys

sys.path.insert(0, "lib")
sys.path.insert(0, ".")

import make_viewer
from make_viewer import load_mode, page_name
import exp_continuation as C
from exp_continuation import _slice_judge_transcript, parse_continuation_task
from petri_paths import LOGS

RUN = sys.argv[1] if len(sys.argv) > 1 else "continuation-1x-20260630-145212"


def m_range(s: str):
    nums = [int(m.group(1)) for m in make_viewer.MSG_HEAD.finditer(s)]
    return (min(nums), max(nums), len(nums)) if nums else (None, None, 0)


async def main():
    run_dir = LOGS / RUN
    print(f"RUN DIR: {run_dir}\n")
    audits = await load_mode(run_dir)
    audits.sort(key=lambda a: a["task"])
    for a in audits:
        parsed = parse_continuation_task(a["task"])
        cond = parsed[0] if parsed else "?"
        page = page_name(a["mode"], a["task"], a["seed"], a["epoch"])
        tr = a["transcript"] or ""
        lo, hi, n = m_range(tr)
        pivot = make_viewer._continuation_cut_m(tr)
        print("=" * 90)
        print(f"TASK {a['task']}  (condition={cond})")
        print(f"  full transcript: [M{lo}..M{hi}] ({n} msgs, {len(tr):,} chars)")
        print(f"  pivot M (_continuation_cut_m): {pivot}")

        sliced = _slice_judge_transcript(tr)
        if sliced is None:
            print("  _slice_judge_transcript -> None  (NO SLICE: judge sees whole transcript)")
        else:
            slo, shi, sn = m_range(sliced)
            # what M-range survived after the head (system) block?
            note_idx = sliced.find("prior, unrelated task ran here")
            head = sliced[:note_idx]
            hlo, hhi, hn = m_range(head)
            print(f"  SLICED transcript: {len(sliced):,} chars ({len(tr) - len(sliced):,} cut)")
            print(f"    head (kept, pre-note) M-range: [M{hlo}..M{hhi}] ({hn} msgs)")
            print(f"    placeholder note present: {'yes' if note_idx >= 0 else 'NO'}")
            print(f"    tail M-range (kept): from pivot onward -> overall sliced [M{slo}..M{shi}] ({sn} msgs)")
            # sanity: is the body between head and pivot actually gone?
            body_present = any(lo2 < (pivot or 0) and lo2 > (hhi or 0)
                               for lo2 in [int(m.group(1)) for m in make_viewer.MSG_HEAD.finditer(sliced)])
            print(f"    prefix body (M{(hhi or 0)+1}..M{(pivot or 1)-1}) still present in sliced? "
                  f"{'YES -- BUG' if body_present else 'no (good)'}")
            print(f"    --- first 240 chars of sliced ---\n{sliced[:240]}")
            print(f"    --- tail start (around pivot) ---")
            tailstart = sliced.find(f"[M{pivot}]")
            print(sliced[tailstart:tailstart + 240] if tailstart >= 0 else "(could not locate pivot in sliced!)")

        # primary judge's own words (does it know about the prior task?)
        summ = (a.get("summary") or "")[:500]
        print(f"  --- PRIMARY judge summary (first 500 chars) ---\n  {summ}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
