"""Archive a fixed cohort's judgments and restore judgment-free source logs.

This is a free, local maintenance command. It never calls a model. The selected
original run directories are moved into an exact archive, then copied back with every
stored gate/final judgment removed. Existing retrospective rejudgments and generated
viewer caches can be archived in the same operation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspect_ai.log import (  # noqa: E402
    read_eval_log,
    resolve_sample_attachments,
    write_eval_log,
)
from inspect_ai.model import ModelUsage  # noqa: E402

from lib.project_paths import DATA_ROOT, LOGS_ROOT, VIEWER_ROOT  # noqa: E402


SCORE_KEY = "environment_judge"
JUDGMENT_ROLES = {"gate", "judge"}
JUDGMENT_SPAN_NAMES = {"scorers", SCORE_KEY, "submit_judgment"}
JUDGE_METADATA_KEYS = {
    "judge",
    "gate_model",
    "judge_schema_version",
    "judge_dimensions",
    "judge_dimension_files",
    "judge_evidence_version",
}


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump(mode="json"))
    return str(value)


def sanitize_real_env(raw: dict) -> dict:
    """Keep trajectory facts and remove only stored model judgments."""

    real_env = deepcopy(raw)
    real_env.pop("final_judgment", None)
    protocol = real_env.get("protocol")
    if isinstance(protocol, dict):
        protocol.pop("first_gate_verdict", None)
    gates = real_env.get("gates")
    if isinstance(gates, list):
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            for key in ("verdict", "judgment", "judge_metadata", "rule"):
                gate.pop(key, None)
    return real_env


def _judgment_span_ids(events: Iterable[Any]) -> set[str]:
    begins = [
        event for event in events
        if getattr(event, "event", None) == "span_begin"
    ]
    removed = {
        str(getattr(event, "id", None) or getattr(event, "span_id", ""))
        for event in begins
        if getattr(event, "name", None) in JUDGMENT_SPAN_NAMES
    }
    removed.discard("")
    changed = True
    while changed:
        changed = False
        for event in begins:
            parent = str(getattr(event, "parent_id", None) or "")
            span_id = str(
                getattr(event, "id", None) or getattr(event, "span_id", "")
            )
            if parent in removed and span_id and span_id not in removed:
                removed.add(span_id)
                changed = True
    return removed


def _is_judgment_event(event: Any, removed_spans: set[str]) -> bool:
    event_type = getattr(event, "event", None)
    span_id = str(getattr(event, "span_id", None) or "")
    event_id = str(getattr(event, "id", None) or "")
    if span_id in removed_spans or event_id in removed_spans:
        return True
    if getattr(event, "role", None) in JUDGMENT_ROLES:
        return True
    if event_type == "tool" and getattr(event, "function", None) == "submit_judgment":
        return True
    return event_type == "score" and getattr(event, "scorer", None) == SCORE_KEY


def _sanitize_retained_event(event: Any) -> Any:
    if getattr(event, "event", None) != "state":
        return event
    changes = []
    for change in getattr(event, "changes", None) or []:
        if (
            getattr(change, "path", None) == "/metadata/real_env"
            and isinstance(getattr(change, "value", None), dict)
        ):
            change = change.model_copy(update={
                "value": sanitize_real_env(change.value),
            })
        changes.append(change)
    return event.model_copy(update={"changes": changes})


def _add_usage(current: ModelUsage | None, incoming: ModelUsage) -> ModelUsage:
    return incoming if current is None else current + incoming


def _retained_model_usage(
    role_usage: dict[str, ModelUsage], role_models: dict[str, str]
) -> dict[str, ModelUsage]:
    model_usage: dict[str, ModelUsage] = {}
    for role, usage in role_usage.items():
        model = role_models.get(role)
        if model:
            model_usage[model] = _add_usage(model_usage.get(model), usage)
    return model_usage


def _sanitize_metadata(raw: dict | None) -> dict:
    return {
        key: value for key, value in (raw or {}).items()
        if key not in JUDGE_METADATA_KEYS
    }


def sanitize_log(log):
    roles = getattr(log.eval, "model_roles", None) or {}
    role_models = {
        str(role): str(getattr(config, "model", None) or config)
        for role, config in roles.items()
    }
    retained_roles = {
        role: config for role, config in roles.items()
        if str(role) not in JUDGMENT_ROLES
    }

    samples = []
    for raw_sample in log.samples or []:
        # Resolve every attachment before removing judge events. The writer may create
        # fresh attachments for retained large fields, but no detached old blob survives.
        sample = resolve_sample_attachments(raw_sample, "full")
        events = list(sample.events or [])
        removed_spans = _judgment_span_ids(events)
        retained_events = [
            _sanitize_retained_event(event) for event in events
            if not _is_judgment_event(event, removed_spans)
        ]
        metadata = deepcopy(sample.metadata or {})
        real_env = metadata.get("real_env")
        if isinstance(real_env, dict):
            metadata["real_env"] = sanitize_real_env(real_env)
        role_usage = {
            str(role): usage
            for role, usage in (sample.role_usage or {}).items()
            if str(role) not in JUDGMENT_ROLES
        }
        samples.append(sample.model_copy(update={
            "scores": {},
            "events": retained_events,
            "metadata": metadata,
            "role_usage": role_usage,
            "model_usage": _retained_model_usage(role_usage, role_models),
            "attachments": {},
        }))

    aggregate_roles: dict[str, ModelUsage] = {}
    aggregate_models: dict[str, ModelUsage] = {}
    for sample in samples:
        for role, usage in (sample.role_usage or {}).items():
            aggregate_roles[str(role)] = _add_usage(
                aggregate_roles.get(str(role)), usage
            )
        for model, usage in (sample.model_usage or {}).items():
            aggregate_models[str(model)] = _add_usage(
                aggregate_models.get(str(model)), usage
            )
    stats = log.stats
    if stats is not None:
        stats = stats.model_copy(update={
            "role_usage": aggregate_roles,
            "model_usage": aggregate_models,
        })
    scorers = [
        scorer for scorer in (log.eval.scorers or [])
        if getattr(scorer, "name", None) != SCORE_KEY
    ]
    eval_record = log.eval.model_copy(update={
        "model_roles": retained_roles,
        "scorers": scorers,
        "metadata": _sanitize_metadata(log.eval.metadata),
    })
    return log.model_copy(update={
        "eval": eval_record,
        "metadata": _sanitize_metadata(log.metadata),
        "samples": samples,
        "stats": stats,
        "results": None,
        "reductions": [],
    })


def _assert_judgment_free(log) -> int:
    roles = getattr(log.eval, "model_roles", None) or {}
    if JUDGMENT_ROLES.intersection(str(role) for role in roles):
        raise ValueError("sanitized log still has a gate or judge model role")
    if any(getattr(scorer, "name", None) == SCORE_KEY for scorer in log.eval.scorers or []):
        raise ValueError("sanitized log still declares environment_judge")
    if log.results is not None or log.reductions:
        raise ValueError("sanitized log still has aggregate judgment results")
    for sample in log.samples or []:
        if sample.scores:
            raise ValueError(f"sample {sample.id} still has scores")
        real_env = (sample.metadata or {}).get("real_env") or {}
        if "final_judgment" in real_env:
            raise ValueError(f"sample {sample.id} still has final_judgment")
        if "first_gate_verdict" in (real_env.get("protocol") or {}):
            raise ValueError(f"sample {sample.id} still has first_gate_verdict")
        for gate in real_env.get("gates") or []:
            if {"verdict", "judgment", "judge_metadata", "rule"}.intersection(gate):
                raise ValueError(f"sample {sample.id} still has a stored gate judgment")
        removed_spans = _judgment_span_ids(sample.events or [])
        if removed_spans or any(
            _is_judgment_event(event, removed_spans) for event in sample.events or []
        ):
            raise ValueError(f"sample {sample.id} still has judgment events")
        for event in sample.events or []:
            if getattr(event, "event", None) != "state":
                continue
            for change in getattr(event, "changes", None) or []:
                value = getattr(change, "value", None)
                if (
                    getattr(change, "path", None) == "/metadata/real_env"
                    and isinstance(value, dict)
                    and sanitize_real_env(value) != value
                ):
                    raise ValueError(
                        f"sample {sample.id} still has a judgment-bearing state event"
                    )
        if JUDGMENT_ROLES.intersection(str(role) for role in sample.role_usage or {}):
            raise ValueError(f"sample {sample.id} still has judge usage")
    return len(log.samples or [])


def sanitize_eval_file(path: Path) -> int:
    log = read_eval_log(str(path))
    sanitized = sanitize_log(log)
    temporary = path.with_name(path.stem + ".sanitizing.eval")
    write_eval_log(sanitized, temporary, format="eval")
    try:
        verified = read_eval_log(str(temporary))
        count = _assert_judgment_free(verified)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count


def _copy_without_judgment_carriers(source: Path, destination: Path) -> None:
    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {"logs.json"}.intersection(names)
        if Path(directory) == source and "remote_cells" in names:
            ignored.add("remote_cells")
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def _sanitize_real_env_sidecars(run_dir: Path) -> int:
    count = 0
    for path in run_dir.rglob("_real_env.json"):
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"real-environment sidecar is not an object: {path}")
        path.write_text(json.dumps(sanitize_real_env(value), indent=2, sort_keys=True) + "\n")
        count += 1
    return count


def _move(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def reset(
    *,
    run_names: list[str],
    archive_root: Path,
    archive_rejudges: bool,
    archive_viewer: bool,
) -> dict:
    logs_root = LOGS_ROOT.resolve()
    archive_root = archive_root.resolve()
    if archive_root.exists():
        raise FileExistsError(f"archive destination already exists: {archive_root}")
    if not run_names or len(run_names) != len(set(run_names)):
        raise ValueError("provide one or more unique --source-run names")
    run_dirs = []
    for name in run_names:
        if not name.startswith("real-v") or Path(name).name != name:
            raise ValueError(f"unsafe source run name: {name!r}")
        path = (logs_root / name).resolve()
        if path.parent != logs_root or not path.is_dir():
            raise FileNotFoundError(f"source run not found: {path}")
        run_dirs.append(path)

    rejudge_dirs = (
        sorted(path for path in logs_root.glob("rejudge-current-*") if path.is_dir())
        if archive_rejudges else []
    )
    archive_root.mkdir(parents=True)
    archive_runs = archive_root / "original_runs"
    archive_rejudge_root = archive_root / "old_rejudgments"
    sanitized_sample_count = 0
    sidecar_count = 0

    for run_dir in run_dirs:
        archived = archive_runs / run_dir.name
        _move(run_dir, archived)
        staging = logs_root / f".{run_dir.name}.judgment-reset"
        _copy_without_judgment_carriers(archived, staging)
        for log_path in sorted(staging.glob("*.eval")):
            sanitized_sample_count += sanitize_eval_file(log_path)
        sidecar_count += _sanitize_real_env_sidecars(staging)
        staging.replace(run_dir)

    for rejudge_dir in rejudge_dirs:
        _move(rejudge_dir, archive_rejudge_root / rejudge_dir.name)

    archived_generated: list[str] = []
    if archive_viewer:
        generated = [
            VIEWER_ROOT,
            DATA_ROOT / "viewer_cache",
            DATA_ROOT / ".viewer_cache",
        ]
        for path in generated:
            if path.exists():
                destination = archive_root / "generated_viewer_state" / path.name
                _move(path, destination)
                archived_generated.append(path.name)

    manifest = {
        "version": 1,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "source_runs": run_names,
        "sanitized_sample_count": sanitized_sample_count,
        "sanitized_real_env_sidecars": sidecar_count,
        "archived_rejudge_runs": [path.name for path in rejudge_dirs],
        "archived_generated_viewer_state": archived_generated,
        "live_logs_root": str(logs_root),
    }
    (archive_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", action="append", required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive-rejudges", action="store_true")
    parser.add_argument("--archive-viewer", action="store_true")
    args = parser.parse_args()
    manifest = reset(
        run_names=args.source_run,
        archive_root=args.archive_root,
        archive_rejudges=args.archive_rejudges,
        archive_viewer=args.archive_viewer,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
