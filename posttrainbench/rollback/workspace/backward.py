"""Backward rebuild: archived FINAL workspace -> rolled back to the cut point.

Rationale: the final workspace holds file CONTENTS that forward replay can't
reproduce (non-deterministically generated data). We recover the cut-point
state by removing everything the agent produced at/after the cut.

Steps:
  1. Copy the archived final task/ into dest.
  2. Overlay the repo's static scaffold to RESTORE files the dataset archive
     stripped (notably evaluation_code/data/healthbench.jsonl, the test data) —
     the agent had these at runtime; the archive does not.
  3. Delete every agent-added file whose creation index is >= cut. Files created
     before the cut are kept (caveat: if such a file was also MODIFIED after the
     cut, the kept copy is the post-cut version — flagged, not auto-resolved;
     does not occur at the default cut where nothing is written pre-cut).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .. import config, ptbio, scaffold, timing
from . import common


def build(spec: config.ExperimentSpec, *, creation_epoch: int) -> dict:
    traj = spec.trajectory
    cut = spec.cut_before_event
    dest = spec.build_dir / "task"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1. copy final workspace
    if not traj.final_task_dir.exists():
        raise FileNotFoundError(f"final workspace missing: {traj.final_task_dir}")
    shutil.copytree(traj.final_task_dir, dest)

    # 2. determine the static scaffold file set (assemble into a temp dir) and
    #    overlay it onto dest to RESTORE files the archive stripped.
    scaffold_files = _scaffold_file_set(traj, creation_epoch)
    repo_manifest = scaffold.assemble(traj, dest, creation_epoch=creation_epoch)

    events = ptbio.load_events(traj)
    final_paths = common.path_set(dest)

    # 3. roll back. KEEP a file iff it is part of the scaffold OR was created
    #    strictly before the cut — dated either by the trace (creation_index)
    #    or, for bash side-effect files whose names never appear in a command
    #    (inspect-ai eval logs), by the timestamp embedded in the filename
    #    compared against the cut's wall-clock. Everything else in the final
    #    workspace is a post-cut artifact (later data/script versions, training
    #    outputs, post-hoc judge verdicts) and is removed.
    cut_wall = timing.cut_epoch(events, cut)
    removed, kept_pre_cut, kept_by_name_ts, flagged = [], [], [], []
    for rel in sorted(final_paths):
        if rel in scaffold_files:
            continue
        ci = common.creation_index(events, rel)
        if ci is not None and ci < cut:
            kept_pre_cut.append(rel)
            if _modified_after(events, rel, cut):
                flagged.append(rel)
            continue
        ne = common.name_embedded_epoch(rel)
        if ci is None and ne is not None and cut_wall is not None and ne < cut_wall:
            kept_pre_cut.append(rel)
            kept_by_name_ts.append(rel)
            continue
        (dest / rel).unlink(missing_ok=True)
        removed.append(rel)

    _prune_empty_dirs(dest, keep_top=common.SCAFFOLD_PROTECTED)

    return {
        "strategy": "backward",
        "cut_before_event": cut,
        "final_file_count": len(final_paths),
        "removed_after_cut": removed,
        "kept_pre_cut": kept_pre_cut,
        "kept_by_name_timestamp": kept_by_name_ts,
        "flagged_modified_after_cut": flagged,
        "scaffold_overlay": repo_manifest,
        "result_file_count": len(common.path_set(dest)),
        "dest": str(dest),
    }


def _scaffold_file_set(traj: config.Trajectory, creation_epoch: int) -> set[str]:
    """Recursive set of relative paths in a freshly-assembled scaffold."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp) / "task"
        scaffold.assemble(traj, td, creation_epoch=creation_epoch,
                          write_prompt_file=True)
        return common.path_set(td)


def _modified_after(events: list[dict], rel: str, cut: int) -> bool:
    base = Path(rel).name
    hits = [op.idx for op in ptbio.file_ops(events)
            if op.kind in ("write", "edit") and op.path
            and Path(op.path).name == base]
    return any(i >= cut for i in hits) and any(i < cut for i in hits)


def _prune_empty_dirs(root: Path, keep_top: set[str]) -> None:
    for dp, dirs, files in os.walk(root, topdown=False):
        p = Path(dp)
        if p == root:
            continue
        if p.relative_to(root).parts[0] in keep_top:
            continue
        if not any(p.iterdir()):
            p.rmdir()
