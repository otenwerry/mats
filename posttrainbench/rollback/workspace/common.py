"""Shared helpers for both workspace rebuild strategies."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from .. import ptbio


# Files that are part of the provided scaffold or our own bookkeeping — never
# treated as agent-created, never deleted by the backward roller.
SCAFFOLD_PROTECTED = {
    "evaluate.py", "timer.sh", "templates", "evaluation_code",
    "benchmark.txt", ".rollback_prompt.txt", "opencode.json",
}


def path_set(root: Path) -> set[str]:
    """Relative paths of all files (not dirs) under root."""
    out = set()
    for dp, _dirs, files in os.walk(root):
        for f in files:
            out.add(str(Path(dp, f).relative_to(root)))
    return out


def file_sha(p: Path, clip: int | None = None) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        data = f.read() if clip is None else f.read(clip)
        h.update(data)
    return h.hexdigest()[:16]


# Filenames that embed their own creation time, e.g. inspect-ai eval logs:
# logs/2026-02-13T16-54-57+01-00_gpqa-main_<id>.json. These are bash side-
# effects (eval runs) whose names never appear in any command, so creation_index
# can't date them — but the embedded timestamp can.
_NAME_TS = re.compile(
    r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})(?:([+-])(\d{2})-(\d{2}))?")


def name_embedded_epoch(rel_path: str) -> float | None:
    """Unix epoch parsed from a timestamp embedded in the filename, if any."""
    m = _NAME_TS.search(Path(rel_path).name)
    if not m:
        return None
    date, hh, mm, ss, sign, ohh, omm = m.groups()
    iso = f"{date}T{hh}:{mm}:{ss}"
    iso += f"{sign}{ohh}:{omm}" if sign else "+00:00"
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


def creation_index(events: list[dict], rel_path: str) -> int | None:
    """Best-effort earliest event index at which the agent created `rel_path`.

    1) a Write/create tool_use targeting it (exact, from first_write_index), else
    2) the first bash command that references its basename (creation proxy for
       files emitted by a script the agent ran).
    Returns None if never referenced (treat as pre-existing scaffold)."""
    fw = ptbio.first_write_index(events)
    base = Path(rel_path).name
    if rel_path in fw:
        return fw[rel_path]
    if base in fw:
        return fw[base]
    # bash creation proxy: first command mentioning the basename
    if len(base) >= 4:
        for tc in ptbio.iter_tool_calls(events):
            if tc.name.lower() == "bash":
                cmd = tc.input.get("command")
                if isinstance(cmd, str) and base in cmd:
                    return tc.idx
    return None
