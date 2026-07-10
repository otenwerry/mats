"""Snapshot each rollback run's final task/ workspace into viewer_data (FREE — no API).

The original PTB runs ship a <run_id>.workspace.json next to their trajectory
(file list + small files inlined); rollback continuations never got one — the
agent's work products (training data, final scripts, adapters) lived only in
the pulled result archives. This script closes that gap so the archives can
move to cold storage (S3) without losing viewer access to the file-level view.

For every rollback_*.json in config.ROLLBACK_VIEWER_DATA, it follows
meta.result_dir to config.ROLLBACK_RESULTS/<dir>/task and writes
<run_id>.workspace.json in the SAME schema as the originals:

    {root, files: [{path, size, mtime, inlined, text}], inlined_bytes}

Inline policy (recorded in the output as `inline_policy`):
  - a file is inlined iff it is valid UTF-8 and <= 100 KB;
  - total inlined text is capped at 8 MB per run — if the cap is hit, remaining
    files are listed but not inlined and `inline_cap_hit: true` is stored
    (none of the current runs come close; the flag exists so a silent
    truncation can never masquerade as a complete snapshot).
Everything under task/ is LISTED regardless (path/size/mtime), so the full
file inventory survives even where contents don't.

MUST run while the result archives are still local — i.e. before exp_s3_archive
pushes/deletes them. Idempotent; --force rebuilds existing snapshots.

    uv run python -m rollback.build_rollback_workspaces [--force]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config

VIEWER_DATA = config.ROLLBACK_VIEWER_DATA
RESULTS = config.ROLLBACK_RESULTS

INLINE_MAX_FILE_BYTES = 100_000
INLINE_TOTAL_CAP_BYTES = 8_000_000


def snapshot_task_dir(task_dir: Path) -> dict:
    files, inlined_bytes, cap_hit = [], 0, False
    for p in sorted(task_dir.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        st = p.stat()
        entry = {"path": str(p.relative_to(task_dir)), "size": st.st_size,
                 "mtime": st.st_mtime, "inlined": False, "text": None}
        if st.st_size <= INLINE_MAX_FILE_BYTES:
            if inlined_bytes + st.st_size > INLINE_TOTAL_CAP_BYTES:
                cap_hit = True
            else:
                try:
                    entry["text"] = p.read_text(encoding="utf-8")
                    entry["inlined"] = True
                    inlined_bytes += st.st_size
                except (UnicodeDecodeError, OSError):
                    pass  # binary or unreadable: listed, not inlined
        files.append(entry)
    out = {"root": str(task_dir), "files": files, "inlined_bytes": inlined_bytes,
           "inline_policy": {"max_file_bytes": INLINE_MAX_FILE_BYTES,
                             "total_cap_bytes": INLINE_TOTAL_CAP_BYTES},
           "inline_cap_hit": cap_hit}
    return out


def build(force: bool = False) -> None:
    built = skipped = missing = 0
    for vf in sorted(VIEWER_DATA.glob("rollback_*.json")):
        out_path = VIEWER_DATA / f"{vf.stem}.workspace.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        try:
            meta = json.loads(vf.read_text()).get("meta", {})
        except (json.JSONDecodeError, OSError):
            print(f"! {vf.name}: unreadable viewer file, skipping")
            continue
        rdir = meta.get("result_dir")
        task_dir = RESULTS / rdir / "task" if rdir else None
        if not rdir or not task_dir.is_dir():
            # archive already gone (or never had a task dir) — nothing to snapshot
            print(f"! {vf.stem}: result archive not local "
                  f"({rdir or 'no result_dir in meta'}), skipping")
            missing += 1
            continue
        ws = snapshot_task_dir(task_dir)
        ws["source_result_dir"] = rdir
        out_path.write_text(json.dumps(ws) + "\n")
        n_inl = sum(1 for f in ws["files"] if f["inlined"])
        print(f"+ {vf.stem}: {len(ws['files'])} files, {n_inl} inlined "
              f"({ws['inlined_bytes'] / 1e3:.0f} KB)"
              + ("  CAP HIT — some small files not inlined" if ws["inline_cap_hit"] else ""))
        built += 1
    print(f"\nworkspaces: {built} built, {skipped} already present, {missing} missing archive.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild existing snapshots")
    build(ap.parse_args().force)
