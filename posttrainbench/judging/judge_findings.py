"""Parse the post-hoc judge agent's output for a flagged task.

After each task a *judge agent* (Codex-format) investigated the workspace and
emitted a structured markdown verdict. It is stored as `judge_output.json` in
the raw task dir — JSONL (one stray non-JSON warning line at the top), events of
type `item.completed` whose `item.type` is `reasoning` / `command_execution` /
`agent_message`. The LAST `agent_message` is the verdict report.

That report cites **workspace files with line ranges** (e.g.
`training_data_v5.jsonl:600`, `train_v7.py:19-46`) and includes verbatim quoted
snippets. This module extracts those deterministically; mapping the cited files
onto trace turns happens in locate.py.

NOTE: `judgement.log` is NOT judge reasoning — it is byte-identical to
`error.log` (the eval process's vLLM/CUDA stdout). Ignore it for localization.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "utilities"))  # shared modules (paths, locate)

import json
import re
from dataclasses import dataclass, field

import paths


@dataclass
class Findings:
    run_id: str
    has_judge: bool                       # judge_output.json present + parsed
    report_md: str = ""                   # the judge's final markdown verdict
    cited_files: list[str] = field(default_factory=list)   # unique, in order
    citations: list[dict] = field(default_factory=list)    # {path, ranges:[(a,b)]}
    quotes: list[str] = field(default_factory=list)        # verbatim snippets


# A path citation: a filename with an extension, optional :line(-line)(,line-line)
# Examples: train_v7.py:19-46 | data/x.jsonl:1-5,600 | scripts/gen.py | final_model/config.json:1-34
_CITE = re.compile(
    r"(?P<path>[\w./\-]+\.[A-Za-z0-9]+)"
    r"(?::(?P<lines>\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*))?"
)
# Snippets the judge quoted verbatim. The judge uses curly quotes most often.
_QUOTE = re.compile(r"[“\"]([^”\"]{4,200}?)[”\"]")


def _parse_ranges(lines: str | None) -> list[tuple[int, int]]:
    if not lines:
        return []
    out = []
    for part in lines.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(part), int(part)))
    return out


def _final_agent_message(judge_path) -> str:
    msgs = []
    with open(judge_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (obj.get("type") == "item.completed"
                    and obj.get("item", {}).get("type") == "agent_message"):
                msgs.append(obj["item"].get("text", ""))
    return msgs[-1] if msgs else ""


def parse(run_id: str) -> Findings:
    judge_path = paths.raw_dir(run_id) / "judge_output.json"
    if not judge_path.exists() or judge_path.stat().st_size == 0:
        return Findings(run_id=run_id, has_judge=False)

    report = _final_agent_message(judge_path)
    if not report:
        return Findings(run_id=run_id, has_judge=False)

    # Only mine citations from inside backticks — that's where the judge puts
    # paths; bare prose like "the train.py script" would create false hits.
    citations: list[dict] = []
    seen: set[str] = set()
    for token in re.findall(r"`([^`]+)`", report):
        for m in _CITE.finditer(token):
            path = m.group("path")
            # skip the verdict txt files themselves and bare model names
            if path in ("contamination_judgement.txt", "disallowed_model_judgement.txt"):
                continue
            ranges = _parse_ranges(m.group("lines"))
            if path not in seen:
                seen.add(path)
                citations.append({"path": path, "ranges": ranges})
            elif ranges:
                # merge ranges into the existing entry
                ex = next(c for c in citations if c["path"] == path)
                ex["ranges"] = ex["ranges"] + ranges

    quotes = []
    for q in _QUOTE.findall(report):
        q = q.strip()
        if q and q not in quotes:
            quotes.append(q)

    return Findings(
        run_id=run_id, has_judge=True, report_md=report,
        cited_files=[c["path"] for c in citations],
        citations=citations, quotes=quotes,
    )


if __name__ == "__main__":  # quick manual check
    import sys
    f = parse(sys.argv[1])
    print("has_judge:", f.has_judge)
    print("cited_files:", f.cited_files)
    print("quotes:", f.quotes[:5])
    print("--- report ---")
    print(f.report_md[:1500])
