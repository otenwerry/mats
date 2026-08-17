#!/usr/bin/env python3
"""Resume/import the desired campaigns orphaned by the 2026-08-15 auth failure.

This is deliberately resume-only. It never creates a campaign, launches a VM, retries
a cell, or touches the accidentally cancelled multi-agent ML campaigns. After this
salvage pass, the terminal campaign states can be used to construct an exact retry of
only not-launched or infrastructure-failure cells.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys


ENVIRONMENTS = Path(__file__).resolve().parent
DATA_ROOT = (ENVIRONMENTS / "../../mats-local/environments").resolve()
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"
LOCK_DIR = RUN_ROOT / ".continuation-push-20260815-salvage.lock"
AWS_REGION = "us-west-2"


@dataclass(frozen=True)
class Campaign:
    label: str
    campaign_id: str
    harness: str
    cells: int
    launched: int


CAMPAIGNS = (
    Campaign(
        "wiki-ml-fill",
        "continuation-aws-wikipedia-summaries-40ep-20260815-015949-82cceb1f-"
        "retry-20260815-123006-c16c95f5-retry-20260815-185954-93b0ed3c",
        "production", 46, 18,
    ),
    Campaign(
        "phack-ml-hack1-production-fill",
        "continuation-aws-hack-in-one-turn-40ep-20260815-015952-72c5744f-"
        "retry-20260815-123106-3ed4f92a-retry-20260815-190015-92b0795c",
        "production", 5, 5,
    ),
    Campaign(
        "phack-ml-hack2-production-fill",
        "continuation-aws-hack-in-two-turns-40ep-20260815-015952-d018a77c-"
        "retry-20260815-123136-48056a34-retry-20260815-190035-2a270710",
        "production", 37, 37,
    ),
    Campaign(
        "phack-ml-hack2-gpt-fill",
        "continuation-aws-hack-in-two-turns-40ep-20260815-015951-96acfbd5-"
        "retry-20260815-123206-2683a12f-retry-20260815-190055-b2cf0bb9",
        "subscription", 1, 1,
    ),
    Campaign(
        "phack-ml-nohack-production-fill",
        "continuation-aws-no-hack-40ep-20260815-015952-474ed33c-"
        "retry-20260815-123237-aa4cc4db-retry-20260815-190115-5e4bd5ed",
        "production", 6, 6,
    ),
    Campaign(
        "phack-ml-nohack-gpt-fill",
        "continuation-aws-no-hack-40ep-20260815-015951-c7e12a59-"
        "retry-20260815-123306-d7a29fb4-retry-20260815-190135-05c4d71b",
        "subscription", 1, 1,
    ),
    Campaign(
        "inline-phack-hack1-production",
        "continuation-aws-hack-in-one-turn-40ep-20260815-190158-5d8b36c8",
        "production", 120, 20,
    ),
    Campaign(
        "inline-phack-hack2-production",
        "continuation-aws-hack-in-two-turns-40ep-20260815-190218-83b32284",
        "production", 120, 20,
    ),
    Campaign(
        "inline-phack-hack2-gpt",
        "continuation-aws-hack-in-two-turns-40ep-20260815-190238-662ce43b",
        "subscription", 40, 3,
    ),
    Campaign(
        "inline-phack-nohack-production",
        "continuation-aws-no-hack-40ep-20260815-190258-0ce87105",
        "production", 80, 14,
    ),
    Campaign(
        "inline-phack-nohack-gpt",
        "continuation-aws-no-hack-40ep-20260815-190318-3cddd339",
        "subscription", 40, 3,
    ),
)


def state_path(campaign: Campaign) -> Path:
    return CAMPAIGN_ROOT / f"{campaign.campaign_id}.json"


def read_state(campaign: Campaign) -> dict:
    path = state_path(campaign)
    if not path.is_file():
        raise RuntimeError(f"missing campaign state for {campaign.label}: {path}")
    return json.loads(path.read_text())


def validate(campaign: Campaign) -> dict:
    state = read_state(campaign)
    cells = state.get("cells") or []
    config = state.get("pipeline_config") or {}
    errors: list[str] = []
    if state.get("campaign_id") != campaign.campaign_id:
        errors.append("campaign ID mismatch")
    if config.get("harness") != campaign.harness:
        errors.append("harness mismatch")
    if len(cells) != campaign.cells:
        errors.append(f"expected {campaign.cells} cells, found {len(cells)}")
    launched = sum(bool(cell.get("instance_id")) for cell in cells)
    if launched != campaign.launched:
        errors.append(f"expected {campaign.launched} launched cells, found {launched}")
    if config.get("multi_agent"):
        errors.append("multi-agent campaign is forbidden in this salvage")
    if any("opus" in str(cell.get("target", "")).lower() for cell in cells):
        errors.append("Opus target is forbidden in this salvage")
    if errors:
        raise RuntimeError(f"{campaign.label}: " + "; ".join(errors))
    return state


def counts(state: dict) -> dict[str, int]:
    result: dict[str, int] = {}
    for cell in state.get("cells") or []:
        status = str(cell.get("status") or "unknown")
        result[status] = result.get(status, 0) + 1
    return dict(sorted(result.items()))


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


class Logger:
    def __init__(self, path: Path):
        self.stream = path.open("a", buffering=1)

    def emit(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def child(self, label: str, message: str) -> str:
        line = f"[{label}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")
        return line


async def stream_resume(
    campaign: Campaign, *, run_dir: Path, logger: Logger, environment: dict[str, str]
) -> int:
    command = [
        "uv", "run", "exp_continuation_pipeline.py",
        f"--resume-campaign={campaign.campaign_id}", "--skip-viewer",
    ]
    log_path = run_dir / f"{campaign.label}.log"
    logger.emit(
        f"Starting resume-only {campaign.label} — existing_launched="
        f"{campaign.launched}/{campaign.cells}; log={log_path}"
    )
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=ENVIRONMENTS,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    with log_path.open("a", buffering=1) as child_log:
        assert process.stdout is not None
        while raw := await process.stdout.readline():
            line = logger.child(
                campaign.label, raw.decode(errors="replace").rstrip("\n")
            )
            child_log.write(line + "\n")
    status = await process.wait()
    logger.emit(f"Finished resume-only {campaign.label} — exit={status}")
    return status


async def run_salvage(run_dir: Path, logger: Logger) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"

    identity = await asyncio.create_subprocess_exec(
        "aws", "sts", "get-caller-identity",
        "--profile", "mats-run", "--region", AWS_REGION,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=environment,
    )
    if await identity.wait() != 0:
        raise RuntimeError(
            "mats-run authentication failed; run `aws login --profile mats-login "
            "--region us-west-2` and retry this salvage command"
        )

    logger.emit(
        f"Starting {len(CAMPAIGNS)} resume-only controllers for "
        f"{sum(item.launched for item in CAMPAIGNS)} already-launched cells."
    )
    logger.emit("No campaign, VM, retry, or multi-agent ML work will be launched.")
    tasks: dict[str, asyncio.Task[int]] = {}
    for campaign in CAMPAIGNS:
        tasks[campaign.label] = asyncio.create_task(
            stream_resume(
                campaign, run_dir=run_dir, logger=logger, environment=environment
            )
        )
        await asyncio.sleep(2)

    statuses = {label: await task for label, task in tasks.items()}
    logger.emit("All resume-only controllers exited. Building the viewer once.")
    viewer = await asyncio.create_subprocess_exec(
        "uv", "run", "viewer.py",
        cwd=ENVIRONMENTS,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert viewer.stdout is not None
    with (run_dir / "viewer.log").open("a", buffering=1) as viewer_log:
        while raw := await viewer.stdout.readline():
            line = logger.child("viewer", raw.decode(errors="replace").rstrip("\n"))
            viewer_log.write(line + "\n")
    viewer_status = await viewer.wait()

    rows = []
    active = 0
    retryable = 0
    completed = 0
    for campaign in CAMPAIGNS:
        state = read_state(campaign)
        status_counts = counts(state)
        active += sum(status_counts.get(key, 0) for key in ("launching", "running", "finishing"))
        retryable += status_counts.get("not_launched", 0) + status_counts.get(
            "infrastructure_failure", 0
        )
        completed += status_counts.get("completed", 0)
        rows.append({
            "label": campaign.label,
            "campaign_id": campaign.campaign_id,
            "controller_exit_status": statuses[campaign.label],
            "counts": status_counts,
        })
        logger.emit(f"Salvaged {campaign.label} — {status_counts}")

    final = {
        "format": "continuation-push-20260815-salvage-v1",
        "run_directory": str(run_dir),
        "resume_only": True,
        "multi_agent_ml_included": False,
        "already_launched_cells": sum(item.launched for item in CAMPAIGNS),
        "completed": completed,
        "retryable_without_pipeline_failures": retryable,
        "active": active,
        "viewer_exit_status": viewer_status,
        "jobs": rows,
    }
    (run_dir / "final_status.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n"
    )
    logger.emit(f"Final salvage status: {run_dir / 'final_status.json'}")
    if active:
        logger.emit(
            "SALVAGE INCOMPLETE: active cells remain; refresh AWS login and rerun "
            "this same resume-only command."
        )
        return 1
    logger.emit(
        f"SALVAGE COMPLETE: completed={completed}; exact infrastructure/not-launched "
        f"retry candidates={retryable}. No retry was launched."
    )
    return 0 if viewer_status == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan", action="store_true",
        help="validate and print the pinned resume-only slate without writes or AWS calls",
    )
    args = parser.parse_args()
    for campaign in CAMPAIGNS:
        state = validate(campaign)
        print(
            f"[{campaign.label}] campaign={campaign.campaign_id} "
            f"cells={campaign.cells} existing_launched={campaign.launched} "
            f"current={counts(state)}"
        )
    print(
        f"TOTAL resume_only_campaigns={len(CAMPAIGNS)} "
        f"already_launched_cells={sum(item.launched for item in CAMPAIGNS)}"
    )
    print("EXCLUDED: all multi-agent ML campaigns and all new/retry launches")
    if args.plan:
        return 0

    for command in ("aws", "uv"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required command is unavailable: {command}")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"another salvage wrapper may be active: {LOCK_DIR}") from error
    try:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = RUN_ROOT / f"continuation-push-20260815-salvage-{stamp}"
        run_dir.mkdir()
        logger = Logger(run_dir / "orchestrator.log")
        return asyncio.run(run_salvage(run_dir, logger))
    finally:
        LOCK_DIR.rmdir()


def caffeinated_main() -> int:
    if (
        not os.environ.get("MATS_CONTINUATION_SALVAGE_CAFFEINATED")
        and shutil.which("caffeinate")
    ):
        environment = dict(os.environ)
        environment["MATS_CONTINUATION_SALVAGE_CAFFEINATED"] = "1"
        os.execvpe(
            "caffeinate",
            ["caffeinate", "-i", sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            environment,
        )
    return main()


if __name__ == "__main__":
    raise SystemExit(caffeinated_main())
