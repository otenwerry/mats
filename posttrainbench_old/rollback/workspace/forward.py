"""Forward rebuild: replay the agent's tool calls from t=0 up to the cut.

Starts from scaffold.assemble(), then applies each persistent tool op in order:
  - Write/create : write file contents verbatim from the trace.
  - Edit         : apply old->new string replacement (best-effort).
  - Bash         : execute the command in the task dir.

Bash is where non-determinism and environment dependence live (HF downloads,
pip, training, eval). `bash_mode` controls it:
  - "execute" : run every bash command (real rebuild; do this on the Linux/GPU
                box inside the container).
  - "safe"    : run only commands that don't match heavy/external patterns
                (datasets/pip/train/eval/curl) — enough to materialize small
                local files on a laptop.
  - "skip"    : don't run bash at all (plan-only; for cheap Mac validation —
                valid here because nothing before the default cut writes
                workspace files via bash, only via downloads to the HF cache).

Every executed op is journaled to dest/../forward_journal.jsonl so a Ctrl-C'd
run resumes without re-doing completed ops (see `resume`).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .. import config, ptbio, scaffold
from . import common

_HEAVY = re.compile(
    r"load_dataset|datasets|huggingface|hf_hub|snapshot_download|pip install|"
    r"uv pip|accelerate|torchrun|train|finetune|evaluate\.py|inspect|curl|wget|"
    r"vllm|model\.from_pretrained|AutoModel", re.I)


def _apply_write(dest: Path, op: ptbio.FileOp) -> dict:
    rel = op.path
    content = op.payload.get("content") or op.payload.get("file_text") or ""
    if not rel:
        return {"idx": op.idx, "kind": "write", "status": "no_path"}
    target = dest / Path(rel).relative_to("/") if rel.startswith("/") else dest / rel
    # the agent works in /home/ben/task; map that prefix onto dest
    target = _map_into_dest(dest, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"idx": op.idx, "kind": "write", "path": str(target), "bytes": len(content)}


def _apply_edit(dest: Path, op: ptbio.FileOp) -> dict:
    rel = op.path
    target = _map_into_dest(dest, rel) if rel else None
    edits = op.payload.get("edits")
    pairs = edits if isinstance(edits, list) else [op.payload]
    if not target or not target.exists():
        return {"idx": op.idx, "kind": "edit", "status": "target_missing", "path": rel}
    text = target.read_text()
    applied = 0
    for e in pairs:
        old = e.get("old_string") if e.get("old_string") is not None else e.get("oldString")
        new = e.get("new_string") if e.get("new_string") is not None else e.get("newString")
        if old is not None and old in text:
            text = text.replace(old, new if new is not None else "", 1)
            applied += 1
    target.write_text(text)
    return {"idx": op.idx, "kind": "edit", "path": str(target),
            "applied": applied, "of": len(pairs)}


def _apply_bash(dest: Path, op: ptbio.FileOp, bash_mode: str) -> dict:
    cmd = op.payload.get("command")
    if not isinstance(cmd, str):
        return {"idx": op.idx, "kind": "bash", "status": "no_command"}
    heavy = bool(_HEAVY.search(cmd))
    if bash_mode == "skip" or (bash_mode == "safe" and heavy):
        return {"idx": op.idx, "kind": "bash", "status": "skipped",
                "heavy": heavy, "cmd": cmd[:120]}
    try:
        r = subprocess.run(cmd, shell=True, cwd=dest, capture_output=True,
                           text=True, timeout=op.payload.get("timeout", 600) / 1000
                           if op.payload.get("timeout", 0) > 1000 else 600)
        return {"idx": op.idx, "kind": "bash", "status": "ran",
                "returncode": r.returncode, "cmd": cmd[:120]}
    except Exception as e:
        return {"idx": op.idx, "kind": "bash", "status": "error",
                "error": str(e)[:200], "cmd": cmd[:120]}


def _map_into_dest(dest: Path, rel: str) -> Path:
    """Map an absolute /home/ben/task/... path (or relative) into dest."""
    for prefix in ("/home/ben/task/", "/home/ben/task", "task/"):
        if rel.startswith(prefix):
            return dest / rel[len(prefix):].lstrip("/")
    if rel.startswith("/"):
        return dest / rel.lstrip("/")
    return dest / rel


def build(spec: config.ExperimentSpec, *, creation_epoch: int,
          bash_mode: str = "skip") -> dict:
    traj = spec.trajectory
    cut = spec.cut_before_event
    dest = spec.build_dir / "task"
    journal = spec.build_dir / "forward_journal.jsonl"
    if dest.exists():
        shutil.rmtree(dest)
    spec.build_dir.mkdir(parents=True, exist_ok=True)

    scaffold_manifest = scaffold.assemble(traj, dest, creation_epoch=creation_epoch)

    events = ptbio.load_events(traj)
    ops = ptbio.file_ops(events, cut)

    done = _resume_set(journal)
    results = []
    with open(journal, "a") as jf:
        for op in ops:
            key = f"{op.idx}:{op.kind}"
            if key in done:
                continue
            if op.kind == "write":
                res = _apply_write(dest, op)
            elif op.kind == "edit":
                res = _apply_edit(dest, op)
            else:
                res = _apply_bash(dest, op, bash_mode)
            jf.write(json.dumps(res) + "\n")
            jf.flush()
            results.append(res)

    return {
        "strategy": "forward",
        "cut_before_event": cut,
        "bash_mode": bash_mode,
        "scaffold": scaffold_manifest,
        "ops_total": len(ops),
        "ops_by_kind": {k: sum(1 for o in ops if o.kind == k)
                        for k in ("write", "edit", "bash")},
        "results_tail": results[-12:],
        "result_file_count": len(common.path_set(dest)),
        "dest": str(dest),
        "journal": str(journal),
    }


def _resume_set(journal: Path) -> set[str]:
    """Ops already completed (from a prior interrupted run)."""
    done = set()
    if journal.exists():
        for line in journal.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add(f"{r['idx']}:{r['kind']}")
            except Exception:
                pass
    return done
