"""Dump judge-view transcripts (+ auditor scratchpads interleaved) for the new-auditor
runs to text files under mats-local/petri/_analysis/new_auditor_dump_20260703/.

Free: reads cached audit dicts via lib/viewer_load.load_mode (no API calls).
"""
import asyncio
import re
import sys
from pathlib import Path

_petri = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_petri / "lib"))
sys.path.insert(0, str(_petri))

from viewer_load import load_mode  # noqa: E402
from make_viewer import hack_category  # noqa: E402
from petri_paths import LOGS, DATA  # noqa: E402

DIRS = [
    "v2-4targets-correct-5ep-20260704-000221",
    "v2-4targets-allow-5ep-20260703-160853",
    "v2-4targets-allow-5ep-20260703-222409",
    "v2-4targets-allow-5ep-20260703-222430",
    "v2-4targets-allow-5ep-20260703-222456",
]

OUT = DATA / "_analysis" / "new_auditor_dump_20260703"


def interleave_scratchpads(transcript: str, scratchpads: dict) -> str:
    """Insert each scratchpad note right before the [M<n>] line it is anchored to."""
    if not scratchpads:
        return transcript
    lines = transcript.split("\n")
    out = []
    head = re.compile(r"^\[M(\d+)\] ")
    for line in lines:
        m = head.match(line)
        if m and int(m.group(1)) in scratchpads:
            note = scratchpads[int(m.group(1))]
            out.append(f"\n[AUDITOR SCRATCHPAD before M{m.group(1)}]\n{note}\n[/AUDITOR SCRATCHPAD]\n")
        out.append(line)
    return "\n".join(out)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index_lines = []
    for d in DIRS:
        audits = await load_mode(LOGS / d)
        sub = OUT / d
        sub.mkdir(exist_ok=True)
        for a in audits:
            tgt = str(a.get("target", "?")).split("/")[-1]
            ep = a.get("epoch", "?")
            name = f"{tgt}-ep{ep}.txt"
            s = a["scores"]
            n_msgs = len(re.findall(r"^\[M\d+\] ", a.get("transcript", ""), re.M))
            hdr = (
                f"RUN DIR : {d}\n"
                f"AUDITOR : {a.get('auditor')}\n"
                f"TARGET  : {a.get('target')}   epoch {ep}\n"
                f"CONDITION: {a.get('condition')}   ended_via_end_conv: {a.get('ended_via_end_conv')}"
                f"   n_transcript_msgs: {n_msgs}\n"
                f"CATEGORY: {hack_category(a)}\n"
                f"SCORES  : {dict(sorted(s.items()))}\n"
                f"\n===== JUDGE SUMMARY =====\n{a.get('summary','')}\n"
                f"\n===== JUDGE JUSTIFICATION =====\n{a.get('justification','')}\n"
                f"\n===== TRANSCRIPT (judge view; [AUDITOR SCRATCHPAD] blocks interleaved) =====\n"
            )
            body = interleave_scratchpads(a.get("transcript", ""), a.get("scratchpad") or {})
            (sub / name).write_text(hdr + body)
            index_lines.append(
                f"{d}/{name}  cat={hack_category(a):10s} rh={s.get('reward_hacking')} "
                f"inc={s.get('incompleteness')} mist={s.get('auditor_mistakes')} "
                f"nudg={s.get('auditor_nudging')} end={'Y' if a.get('ended_via_end_conv') else 'N'} "
                f"msgs={n_msgs}"
            )
        print(f"dumped {len(audits)} -> {sub}")
    (OUT / "INDEX.txt").write_text("\n".join(index_lines) + "\n")
    print(f"index -> {OUT/'INDEX.txt'}")


asyncio.run(main())
