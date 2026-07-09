"""Did the perf auditor deliver the scripted deadline notices? Read-only."""
import asyncio, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from petri_paths import LOGS
from viewer_load import load_mode

PERF = "v2-4targets-allow-2ep-20260705-215536"
# The two scripted notices from core.md (match on distinctive fragments)
PHRASES = ["compute reservation ends", "Final notice", "reservation ends in a few minutes",
           "Heads up", "your reservation ends", "almost up", "time is almost", "running out"]

def short(m): return (m or "?").split("/")[-1]

async def main():
    audits = await load_mode(LOGS / PERF)
    audits.sort(key=lambda a: (short(a["target"]), a["epoch"]))
    for a in audits:
        t = a.get("transcript") or ""
        hits = []
        for ph in PHRASES:
            for m in re.finditer(re.escape(ph), t):
                s = max(0, m.start() - 90); e = min(len(t), m.end() + 160)
                hits.append((ph, t[s:e].replace("\n", " ")))
        cap = "HIT-CAP" if not a.get("ended_via_end_conv") else "ended"
        print(f"\n=== {short(a['target'])} e{a['epoch']}  [{cap}, incompleteness={a['scores'].get('incompleteness')}] ===")
        if not hits:
            print("   (no deadline-notice phrases found)")
        for ph, ctx in hits:
            print(f"   <{ph}>  ...{ctx}...")

asyncio.run(main())
