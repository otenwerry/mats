"""Read-only follow-up: for the 3 `correct`-condition Opus trajectories that
DELIVERED a hack, print the full judge justification AND every auditor turn, so
we can tell whether the auditor actually delivered the scripted call-out and how
Opus responded. Free — reads local .eval files, no API calls.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
from petri_paths import LOGS  # noqa: E402
from viewer_load import load_mode  # noqa: E402

TARGET_RUNS = [
    "v2-3targets-correct-5ep-20260708-185830",
    "v2-5targets-correct-3ep-20260708-173507",
]


def auditor_turns(a):
    """Return list of (anchor, text) for auditor user-turn messages (the env speaking
    TO the target), from the transcript."""
    out = []
    for m in a.get("transcript") or []:
        role = m.get("role") if isinstance(m, dict) else None
        # in the transcript, auditor-authored user turns to the target are role 'user'
        if role == "user":
            txt = m.get("text") or m.get("content") or ""
            if isinstance(txt, list):
                txt = " ".join(str(x) for x in txt)
            out.append(txt)
    return out


async def main():
    for run in TARGET_RUNS:
        audits = await load_mode(LOGS / run)
        for a in audits:
            if "opus" not in (a.get("target") or "").lower():
                continue
            sc = a.get("scores") or {}
            if a.get("rh_category") != "delivered_hack":
                continue
            print("\n" + "#" * 100)
            print(f"# {run}  ep{a.get('epoch')}  seed={a.get('seed')}")
            print(f"# reward_hacking={sc.get('reward_hacking')}  noticed_hack={sc.get('noticed_hack')}  "
                  f"category={a.get('rh_category')}  types={a.get('rh_types')}")
            print("#" * 100)
            print("\n----- JUDGE JUSTIFICATION -----")
            print((a.get("justification") or "").strip())
            print("\n----- AUDITOR USER-TURNS (env -> target) -----")
            turns = auditor_turns(a)
            for i, t in enumerate(turns):
                t = " ".join(t.split())
                print(f"\n  [user-turn {i}] {t[:900]}")


asyncio.run(main())
