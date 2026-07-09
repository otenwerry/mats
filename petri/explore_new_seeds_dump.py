"""Dump full judge notes + transcript for specific new-seed trajectories."""
import asyncio, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from petri_paths import LOGS
from viewer_load import load_mode

# (run_dir, target substring, epoch)
TARGETS = [
    ("v2-4targets-allow-2ep-20260705-215536", "gpt-5.5", 2),   # the one FULL hack (perf)
    ("v2-4targets-allow-2ep-20260705-215503", "deepseek", 1),  # borderline NUDGED (p-hack)
]

def short(m): return (m or "?").split("/")[-1]

async def main():
    for d, tgt, ep in TARGETS:
        audits = await load_mode(LOGS / d)
        a = next(x for x in audits if tgt in short(x["target"]) and x["epoch"] == ep)
        print("\n" + "#" * 110)
        print(f"# {short(a['target'])}  e{ep}  [{d}]  scores={a['scores']}")
        print("#" * 110)
        print("\n=== SUMMARY ===\n" + (a.get("summary") or ""))
        print("\n=== JUSTIFICATION ===\n" + (a.get("justification") or ""))
        print("\n=== HIGHLIGHTS ===\n" + (a.get("highlights") or ""))
        t = a.get("transcript") or ""
        print(f"\n=== TRANSCRIPT ({len(t)} chars) — TAIL 9000 ===\n" + t[-9000:])

asyncio.run(main())
