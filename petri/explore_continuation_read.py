"""FREE helper: dump clean transcript text for the continuation qualitative read.

Writes plain-text transcripts (pivot-sliced for continuations, with hack-turn markers
from annotations.json) into mats-local/petri/scratch_continuation_read/ so they can be
read cheaply (the viewer HTML pages are ~50k tokens each). No API calls.
"""
import asyncio, json, sys, pathlib, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))
import make_viewer
from make_viewer import load_mode, load_originals_by_id, page_name, _continuation_cut_m
from petri_paths import LOGS, DATA

RUN = "continuation-5x-20260630-160044"
OUT = DATA / "scratch_continuation_read"
OUT.mkdir(exist_ok=True)
ANN = json.loads((DATA / "annotations.json").read_text())

# model -> (hack A id, clean A id, B id)
TRIPLES = {
    "gpt-5.5":         (471, 472, 466),
    "opus-4.6":        (534, 550, 456),
    "kimi-k2.6":       (524, 559, 480),
    "glm-5.1":         (557, 514, 508),
    "deepseek-v4-pro": (502, 501, 541),
}


def mark(transcript: str, page_key: str) -> str:
    """Insert hack-turn markers before the [M#] lines named in annotations."""
    ann = ANN.get(page_key)
    if not ann:
        return transcript
    turns = {t["m"]: t for t in (ann.get("hack_turns") or []) if isinstance(t, dict) and t.get("m")}
    if not turns:
        return transcript
    out = []
    for line in transcript.splitlines():
        m = re.match(r"^\[M(\d+)\]", line)
        if m and int(m.group(1)) in turns:
            t = turns[int(m.group(1))]
            out.append(f"\n>>> HACK TURN [{t.get('severity','?')}] {t.get('title','')}")
            out.append(f">>> note: {t.get('note','')}")
            for q in (t.get('quotes') or []):
                out.append(f">>> quote: {q}")
            out.append("")
        out.append(line)
    return "\n".join(out)


async def main():
    originals = await load_originals_by_id()
    cont = await load_mode(LOGS / RUN)
    cont_by_page = {page_name(a["mode"], a["task"], a["seed"], a["epoch"]): a for a in cont}

    # find page keys in annotations to match originals (by matching transcript identity via traj)
    # originals: dump prefix A (hack) and B original
    for model, (hid, cid, bid) in TRIPLES.items():
        for label, tid in (("Ahack", hid), ("Aclean", cid), ("Boriginal", bid)):
            a = originals.get(tid)
            if not a:
                print(f"MISSING original {label} #{tid}")
                continue
            pk = page_name(a["mode"], a["task"], a["seed"], a["epoch"])
            tr = a.get("transcript") or "(no transcript)"
            tr = mark(tr, pk)
            scores = a.get("scores") or {}
            hdr = (f"MODEL={model} {label} traj#{tid} seed={a['seed']} "
                   f"reward_hacking={scores.get('reward_hacking')}\n"
                   f"PAGE={pk}\n" + "="*80 + "\n")
            (OUT / f"{model}__{label}_{tid}.txt").write_text(hdr + tr)

    # continuations: dump each page, pivot-sliced (new task only)
    for pk, a in cont_by_page.items():
        tr = a.get("transcript") or ""
        pivot = _continuation_cut_m(tr)
        sliced = tr
        if pivot is not None:
            mt = re.search(rf"^\[M{pivot}\]", tr, re.M)
            if mt:
                sliced = tr[mt.start():]
        sliced = mark(sliced, pk)
        scores = a.get("scores") or {}
        short = pk.split("__", 1)[1] if "__" in pk else pk
        hdr = (f"CONTINUATION {short}\nreward_hacking={scores.get('reward_hacking')} "
               f"pivot_M={pivot}\n" + "="*80 + "\n")
        (OUT / f"cont__{short.replace('.html','')}.txt").write_text(hdr + sliced)

    print(f"wrote transcripts to {OUT}")
    print("files:", len(list(OUT.glob('*.txt'))))


if __name__ == "__main__":
    asyncio.run(main())
