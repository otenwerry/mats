"""Shared manifest for runs and trajectories shown and rejudged as Old."""

from __future__ import annotations

import json
from pathlib import Path

from project_paths import ENVIRONMENTS_ROOT


OLD_RUNS_FILE = ENVIRONMENTS_ROOT / "viewer_old_runs.json"
OLD_RUNS_FORMAT = "environments-viewer-old-runs-v1"


def _old_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("format") != OLD_RUNS_FORMAT:
        raise ValueError(f"unsupported old-run manifest format: {path}")
    return payload


def _string_set(payload: dict, field: str, path: Path) -> frozenset[str]:
    values = payload.get(field, [])
    if (
        not isinstance(values, list)
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError(f"invalid {field} in {path}")
    return frozenset(values)


def old_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    payload = _old_manifest(path)
    names = payload.get("run_directories")
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError(f"invalid run_directories in {path}")
    return frozenset(names)


def old_trajectory_keys(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    return _string_set(_old_manifest(path), "trajectory_keys", path)


def old_prefix_files(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Purpose-built prefix payload basenames shown in the Old viewer window."""

    return _string_set(_old_manifest(path), "prefix_files", path)


def promoted_rejudge_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Rejudge runs explicitly approved to replace their source judgments."""

    return _string_set(
        _old_manifest(path), "promoted_rejudge_run_directories", path
    )


def old_source_run_names(path: Path = OLD_RUNS_FILE) -> frozenset[str]:
    """Original real-v* directories needed by retrospective rejudging."""

    keys = old_trajectory_keys(path)
    malformed = sorted(key for key in keys if "__" not in key)
    if malformed:
        raise ValueError(f"invalid trajectory_keys in {path}: {malformed[:3]}")
    source_names = old_run_names(path) | frozenset(
        key.split("__", 1)[0] for key in keys
    )
    return frozenset(name for name in source_names if name.startswith("real-v"))
