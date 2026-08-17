"""Recoverably quarantine campaigns whose every stored sample has no agent output.

The default is a read-only plan. ``--apply`` moves both the imported log directory and
the controller state into a timestamped directory under
``mats-local/environments/trash/fully_failed_campaigns`` and writes a manifest.  It
never deletes evidence or makes AWS/model calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from inspect_ai.log import read_eval_log


ENVIRONMENTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ENVIRONMENTS_ROOT / "lib"))

from project_paths import DATA_ROOT, LOGS_ROOT  # noqa: E402
from real_integrity import sample_has_target_output  # noqa: E402


CAMPAIGNS_ROOT = DATA_ROOT / "remote_campaigns"
TRASH_ROOT = DATA_ROOT / "trash" / "fully_failed_campaigns"
SAFE_CAMPAIGN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ACTIVE_CELL_STATES = frozenset({"launching", "pending", "running", "shutting-down"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_tokens(sample) -> int:
    usage = (getattr(sample, "role_usage", None) or {}).get("target")
    return int(getattr(usage, "output_tokens", 0) or 0)


def inspect_campaign(campaign_id: str) -> dict:
    if not SAFE_CAMPAIGN.fullmatch(campaign_id) or Path(campaign_id).name != campaign_id:
        raise ValueError(f"unsafe campaign id: {campaign_id!r}")
    state_path = CAMPAIGNS_ROOT / f"{campaign_id}.json"
    log_dir = LOGS_ROOT / campaign_id
    if not state_path.is_file():
        raise FileNotFoundError(f"campaign state not found: {state_path}")
    if not log_dir.is_dir():
        raise FileNotFoundError(f"campaign log directory not found: {log_dir}")
    state = json.loads(state_path.read_text())
    if state.get("campaign_id") != campaign_id:
        raise ValueError(f"campaign id mismatch in {state_path}")
    cells = state.get("cells") or []
    if not cells:
        raise ValueError(f"campaign has no cells: {campaign_id}")
    status_counts = Counter(str(cell.get("status") or "missing") for cell in cells)
    active = sorted(set(status_counts).intersection(ACTIVE_CELL_STATES))
    if active:
        raise ValueError(f"campaign still has active cells {active}: {campaign_id}")
    cleanup = state.get("s3_cleanup") or {}
    if cleanup.get("status") != "deleted":
        raise ValueError(
            f"campaign S3 cleanup is not confirmed deleted: {campaign_id}"
        )

    eval_paths = sorted(log_dir.glob("*.eval"))
    if not eval_paths:
        raise ValueError(f"campaign has no imported eval logs: {campaign_id}")
    sample_count = 0
    output_sample_count = 0
    for eval_path in eval_paths:
        log = read_eval_log(str(eval_path))
        samples = log.samples or []
        if not samples:
            raise ValueError(f"eval log has no samples: {eval_path}")
        for sample in samples:
            sample_count += 1
            if sample_has_target_output(sample) or _output_tokens(sample) > 0:
                output_sample_count += 1
    if sample_count != len(cells):
        raise ValueError(
            f"stored sample/cell count mismatch for {campaign_id}: "
            f"samples={sample_count} cells={len(cells)}"
        )
    if output_sample_count:
        raise ValueError(
            f"refusing to quarantine {campaign_id}: {output_sample_count}/"
            f"{sample_count} samples contain agent output"
        )

    exit_counts = Counter(
        str((cell.get("terminal") or {}).get("pipeline_exit_code", "missing"))
        for cell in cells
    )
    return {
        "campaign_id": campaign_id,
        "log_directory": str(log_dir),
        "campaign_state": str(state_path),
        "campaign_state_sha256": _sha256(state_path),
        "eval_files": len(eval_paths),
        "samples": sample_count,
        "samples_with_agent_output": output_sample_count,
        "cell_status_counts": dict(sorted(status_counts.items())),
        "pipeline_exit_counts": dict(sorted(exit_counts.items())),
        "s3_cleanup": cleanup,
    }


def quarantine(campaigns: list[dict], *, reason: str) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = TRASH_ROOT / stamp
    if destination.exists():
        raise FileExistsError(f"quarantine destination already exists: {destination}")
    logs_destination = destination / "logs"
    states_destination = destination / "remote_campaigns"
    moved: list[tuple[Path, Path]] = []
    try:
        logs_destination.mkdir(parents=True)
        states_destination.mkdir()
        for record in campaigns:
            source_log = Path(record["log_directory"])
            source_state = Path(record["campaign_state"])
            target_log = logs_destination / source_log.name
            target_state = states_destination / source_state.name
            shutil.move(str(source_log), str(target_log))
            moved.append((target_log, source_log))
            shutil.move(str(source_state), str(target_state))
            moved.append((target_state, source_state))
            record["quarantined_log_directory"] = str(target_log)
            record["quarantined_campaign_state"] = str(target_state)
        manifest = {
            "format": "environments-fully-failed-quarantine-v1",
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "definition": (
                "every stored sample lacked assistant-visible output, tool calls, "
                "and target output tokens"
            ),
            "recoverable": True,
            "campaigns": campaigns,
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest["manifest"] = str(manifest_path)
        return manifest
    except BaseException:
        for source, target in reversed(moved):
            if source.exists() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="append", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if len(args.campaign) != len(set(args.campaign)):
        raise SystemExit("campaign ids must be unique")
    records = [inspect_campaign(campaign_id) for campaign_id in args.campaign]
    if args.apply:
        result = quarantine(records, reason=args.reason)
    else:
        result = {
            "mode": "plan",
            "reason": args.reason,
            "campaigns": records,
            "filesystem_writes": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
