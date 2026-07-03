"""Sweep-3 tool-channel analysis (free, local-only). READ-ONLY over viewer data.

Question (Owen, 2026-07-02): beyond user-turn nudges, to what extent does the auditor
push the target toward hacking through TOOL RESULTS — fabricated scores, leaderboards
inside tool output, affordances granted at suspicious moments (e.g. letting a protected
file be read)? Especially: cases where the target looks unwilling to hack and the
auditor then fabricates a tool result that raises the temptation.

Three dump files into mats-local/petri/_analysis/ for manual reading:
  sweep3_scratchpad_dump.txt   -- EVERY auditor scratchpad note, per trajectory, with
                                  M anchors and the trajectory's category/first-hack
                                  header. Notes truncated at TRUNC_SP chars (marked).
  sweep3_score_sequences.txt   -- metric values (accuracy/f1/auc/ndcg/...) extracted by
                                  regex from tool AND user messages, in order, with the
                                  first-hack M marked. Regex-based, so exotic metric
                                  formats may be missed -- this is a reading aid, not a
                                  complete accounting.
  sweep3_toolresult_flags.txt  -- per-trajectory counts of leaderboard/competitor
                                  language by channel (tool vs user), permission-denied
                                  events, and DO-NOT-USE / protected-file mentions.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer as mv

DIRS = ["v2-5targets-1ep-20260629-183349", "v2-5targets-3ep-20260629-225553"]
OUT_DIR = mv.DATA / "_analysis"
TRUNC_SP = 2500   # chars kept per scratchpad note in the dump

METRIC_RE = re.compile(
    r"(macro[_\- ]?f1|micro[_\- ]?f1|test[_ ]accuracy|accuracy|roc[_\- ]?auc|pr[_\- ]?auc|"
    r"ndcg@?10|mrr|auprc|\bf1\b)\s*(?:score)?\s*[:=]?\s*(0\.\d{2,4}|1\.0{1,4}\b)",
    re.IGNORECASE)
LEADER_RE = re.compile(
    r"leaderboard|candidate[_ ][A-Z]\b|other team|team[_ ][a-z]+\b|best to beat|"
    r"competing candidate|standings|cand_[A-Z]", re.IGNORECASE)
PERM_RE = re.compile(r"permission denied|permissionerror|operation not permitted", re.IGNORECASE)
PROTECTED_RE = re.compile(r"do not use|test_labels|\.test_labels|real_user_", re.IGNORECASE)


def parse_messages(transcript: str):
    heads = list(mv.MSG_HEAD.finditer(transcript))
    out = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(transcript)
        out.append((int(h.group(1)), h.group(2).strip().lower(),
                    transcript[h.end():end].strip()))
    return out


def header(a: dict, fh_m, fh_a) -> str:
    return (f"### {mv.hack_category(a).upper()} | {mv.target_short(a)} | "
            f"{a['seed'].replace('ml_training_data_misuse_', '')} e{a['epoch']} "
            f"({'1ep' if '1ep' in a['mode'] else '3ep'}) | rh={mv.rh_score(a)} | "
            f"first hack: {'A' + str(fh_a) if fh_a else '-'} (M{fh_m})")


async def main():
    audits = []
    for d in DIRS:
        audits += await mv.load_mode(mv.LOGS / d)
    audits = [a for a in audits if not a.get("dead")]
    annotations = json.loads((mv.DATA / "annotations.json").read_text())
    order = {"full": 0, "degenerate": 1, "nudged": 2, "both": 3, "non": 4}
    audits.sort(key=lambda a: (order[mv.hack_category(a)], mv.target_short(a),
                               a["seed"], a["epoch"]))

    sp_parts, score_parts, flag_parts = [], [], []
    for a in audits:
        ann = annotations.get(mv.page_name(a["mode"], a["task"], a["seed"], a["epoch"]))
        fh_m = mv.first_hack_m(ann)
        fh_a = mv.assistant_turn_index(a["transcript"], fh_m)
        head = header(a, fh_m, fh_a)
        msgs = parse_messages(a["transcript"])

        # 1. scratchpad dump (auditor's private notes, keyed by transcript M anchor)
        lines = [head]
        for m in sorted(a["scratchpad"]):
            txt = re.sub(r"\s+", " ", a["scratchpad"][m]).strip()
            if len(txt) > TRUNC_SP:
                txt = txt[:TRUNC_SP] + f" ...[+{len(txt) - TRUNC_SP} chars]"
            mark = " <FIRST-HACK-TURN-PASSED>" if isinstance(fh_m, int) and m >= fh_m else ""
            lines.append(f"  [sp@M{m}]{mark} {txt}")
        sp_parts.append("\n".join(lines))

        # 2. metric sequences from tool + user messages
        seq = []
        for m, role, body in msgs:
            if role not in ("tool", "user"):
                continue
            seen = set()
            for mt in METRIC_RE.finditer(body):
                name = re.sub(r"[\s_\-]+", "_", mt.group(1).lower())
                val = mt.group(2)
                if (name, val) in seen:
                    continue
                seen.add((name, val))
                seq.append((m, role, name, val))
        lines = [head]
        for m, role, name, val in seq:
            mark = " <HACK>" if isinstance(fh_m, int) and m >= fh_m else ""
            lines.append(f"  [M{m} {role}]{mark} {name}={val}")
        score_parts.append("\n".join(lines))

        # 3. channel flags
        lead_tool = sum(1 for m, r, b in msgs if r == "tool" and LEADER_RE.search(b))
        lead_user = sum(1 for m, r, b in msgs if r == "user" and LEADER_RE.search(b))
        perm = [m for m, r, b in msgs if r == "tool" and PERM_RE.search(b)]
        prot = sum(1 for m, r, b in msgs if r == "tool" and PROTECTED_RE.search(b))
        flag_parts.append(
            f"{head}\n  leaderboard-lang msgs: tool={lead_tool} user={lead_user} | "
            f"permission-denied tool msgs: {len(perm)} at {perm[:8]} | "
            f"protected-file mentions in tool msgs: {prot}")

    OUT_DIR.mkdir(exist_ok=True)
    for name, parts in [("sweep3_scratchpad_dump.txt", sp_parts),
                        ("sweep3_score_sequences.txt", score_parts),
                        ("sweep3_toolresult_flags.txt", flag_parts)]:
        p = OUT_DIR / name
        p.write_text("\n\n".join(parts))
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")

asyncio.run(main())
