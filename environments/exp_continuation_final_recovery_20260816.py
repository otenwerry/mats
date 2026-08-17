#!/usr/bin/env python3
"""Retry the exact 46 incomplete cells from the 2026-08-15 remaining-work run.

This paid one-off wrapper selects only:

* 40 Kimi one-turn p-hacking continuations that failed before OpenCode started;
* one GLM two-turn continuation that reached its wall-clock limit; and
* five production demand-forecasting prefixes that did not finish pass two.

Usage from ``mats/environments``::

    uv run exp_continuation_final_recovery_20260816.py --plan
    uv run exp_continuation_final_recovery_20260816.py

The plan mode is local and read-only. Paid mode performs AWS setup once, streams
labeled controller/VM output, retains logs and final state, and rebuilds the viewer
once. It never retries a child automatically.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys


ENVIRONMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from exp_continuation_push_20260815 import (  # noqa: E402
    Job,
    RunLogger,
    _atomic_json,
    _campaign_ids,
    source_fingerprint,
    stream_command,
)
from exp_opencode_transport import (  # noqa: E402
    _opencode_positional_message,
    rewrite_opencode_exec_remote,
)
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions  # noqa: E402


DATA_ROOT = (ENVIRONMENTS / "../../mats-local/environments").resolve()
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"
AWS_REGION = "us-west-2"
AWS_INSTANCE_TYPE = "c7a.xlarge"
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180
EXPECTED_TRAJECTORIES = 46
EXPECTED_VM_CAP = 24
RUN_BASENAME = "continuation-final-recovery-20260816"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class RetryParent:
    label: str
    campaign_id: str
    kind: str
    expected_target_epochs: tuple[tuple[str, tuple[int, ...]], ...]
    vm_cap: int

    @property
    def cells(self) -> int:
        return sum(len(epochs) for _, epochs in self.expected_target_epochs)

    @property
    def state_path(self) -> Path:
        return CAMPAIGN_ROOT / f"{self.campaign_id}.json"


PARENTS = (
    RetryParent(
        "kimi-hack1-opencode-retry",
        "continuation-aws-hack-in-one-turn-40ep-20260815-190158-5d8b36c8-"
        "retry-20260815-233432-4be1fa82",
        "continuation",
        (("kimi-k2.6", tuple(range(1, 41))),),
        20,
    ),
    RetryParent(
        "glm-hack2-wallclock-retry",
        "continuation-aws-hack-in-two-turns-40ep-20260815-190218-83b32284-"
        "retry-20260815-233452-e622287e",
        "continuation",
        (("glm-5.1", (19,)),),
        1,
    ),
    RetryParent(
        "ml-demand-second-pass-retry",
        "ml-prefix-aws-continuation-push-20260815-20260815185916-ml-dem-10ep-"
        "20260815-233615-406c3f5c",
        "prefix",
        (("glm-5.1", (1, 3, 10)), ("kimi-k2.6", (2, 7))),
        3,
    ),
)


def _retryable(cell: dict, *, prefix_only: bool) -> bool:
    if cell.get("status") in {"not_launched", "infrastructure_failure"}:
        return True
    if cell.get("status") != "completed":
        return False
    if not prefix_only and cell.get("terminal") is None:
        return True
    try:
        return int((cell.get("terminal") or {}).get("pipeline_exit_code")) != 0
    except (TypeError, ValueError):
        return True


def _job(parent: RetryParent) -> Job:
    endpoint = (
        "exp_continuation_pipeline.py"
        if parent.kind == "continuation"
        else "prefixes/exp_ml_prefix.py"
    )
    args = [
        f"--retry-failed={parent.campaign_id}",
        "--harness=production",
        f"--vm-concurrency={parent.vm_cap}",
        "--skip-viewer",
    ]
    if parent.kind == "continuation":
        args.insert(1, "--retry-pipeline-failures")
    return Job(
        label=parent.label,
        group=(
            "exact continuation recovery"
            if parent.kind == "continuation"
            else "exact prefix recovery"
        ),
        endpoint=endpoint,
        harness="production",
        cells=parent.cells,
        vm_cap=parent.vm_cap,
        args=tuple(args),
        retry_parent=parent.campaign_id,
    )


def build_jobs() -> list[Job]:
    return [_job(parent) for parent in PARENTS]


def _validate_parent(parent: RetryParent) -> None:
    if not parent.state_path.is_file():
        raise RuntimeError(f"missing retry parent: {parent.state_path}")
    state = json.loads(parent.state_path.read_text())
    config = state.get("pipeline_config") or {}
    if state.get("campaign_id") != parent.campaign_id:
        raise RuntimeError(f"{parent.label}: campaign identity changed")
    if config.get("harness") != "production":
        raise RuntimeError(f"{parent.label}: parent harness changed")
    if parent.kind == "continuation" and not config.get("continuation"):
        raise RuntimeError(f"{parent.label}: parent is not a continuation campaign")
    if parent.kind == "prefix" and config.get("prefix_only") is not True:
        raise RuntimeError(f"{parent.label}: parent is not a prefix-only campaign")
    if config.get("multi_agent") or any(
        "opus" in str(target).lower() for target in config.get("targets") or []
    ):
        raise RuntimeError(f"{parent.label}: forbidden campaign type or target")
    if (state.get("s3_cleanup") or {}).get("status") != "deleted":
        raise RuntimeError(f"{parent.label}: parent S3 cleanup is incomplete")
    if any(
        cell.get("status") in {"planned", "launching", "running", "finishing"}
        for cell in state.get("cells") or []
    ):
        raise RuntimeError(f"{parent.label}: parent campaign is not terminal")
    if any(
        cell.get("instance_state") != "terminated"
        for cell in state.get("cells") or []
    ):
        raise RuntimeError(f"{parent.label}: a parent instance is not terminated")

    prefix_only = parent.kind == "prefix"
    selected = [
        cell for cell in state.get("cells") or []
        if _retryable(cell, prefix_only=prefix_only)
    ]
    actual_counts = Counter(str(cell.get("target")) for cell in selected)
    expected_counts = Counter({
        target: len(epochs) for target, epochs in parent.expected_target_epochs
    })
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"{parent.label}: retry targets drifted; expected {dict(expected_counts)}, "
            f"found {dict(actual_counts)}"
        )
    for target, expected_epochs in parent.expected_target_epochs:
        actual_epochs = sorted(
            int(cell.get("original_epoch"))
            for cell in selected
            if cell.get("target") == target
        )
        if actual_epochs != sorted(expected_epochs):
            raise RuntimeError(
                f"{parent.label}: {target} epochs drifted; "
                f"expected {sorted(expected_epochs)}, found {actual_epochs}"
            )
    local_log_dir = Path(str(state.get("local_log_dir") or ""))
    if not local_log_dir.is_dir():
        raise RuntimeError(f"{parent.label}: imported evidence directory is missing")


def _validate_opencode_fix() -> None:
    prompt = ('long activity log says "value"\n' * 20_000) + "finish"
    command = [
        "bash", "-c", 'exec 0</dev/null; "$@"', "bash",
        "/opt/inspect-swe/opencode/node_modules/.bin/opencode", "run",
        "--model", "openrouter/moonshotai/kimi-k2.6", "--format", "json",
        "--dangerously-skip-permissions", "--continue", prompt,
    ]
    rewritten, options, matched = rewrite_opencode_exec_remote(
        command,
        ExecRemoteAwaitableOptions(cwd="/workspace", concurrency=False),
    )
    if (
        not matched
        or prompt in rewritten
        or rewritten[2] != 'exec "$@"'
        or options.input != _opencode_positional_message(prompt)
        or sum(len(item.encode()) + 1 for item in rewritten) >= 1024
    ):
        raise RuntimeError("OpenCode long-prompt stdin transport is not active")


def validate_plan(jobs: list[Job]) -> str:
    if len(jobs) != 3 or len({job.label for job in jobs}) != 3:
        raise RuntimeError("final recovery controller matrix drifted")
    if sum(job.cells for job in jobs) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("final recovery trajectory count drifted")
    if sum(job.vm_cap for job in jobs) != EXPECTED_VM_CAP:
        raise RuntimeError("final recovery VM cap drifted")
    if any(job.harness != "production" for job in jobs):
        raise RuntimeError("non-production harness entered final recovery")
    joined = " ".join(item for job in jobs for item in job.command()).lower()
    if any(forbidden in joined for forbidden in ("opus", "gpt", "multi-agent")):
        raise RuntimeError("forbidden target or condition entered final recovery")
    for parent in PARENTS:
        _validate_parent(parent)
    _validate_opencode_fix()
    return source_fingerprint()


def print_plan(jobs: list[Job], fingerprint: str) -> None:
    print("2026-08-16 continuation push — exact final recovery plan")
    print(f"source fingerprint: {fingerprint}\n")
    group = None
    for job in jobs:
        if job.group != group:
            group = job.group
            print(f"{group}:")
        print(
            f"  [{job.label}] cells={job.cells} harness={job.harness} "
            f"vm_cap={job.vm_cap}"
        )
        print("    " + " ".join(job.command()))
    print(
        f"\nTOTAL controllers=3 trajectories={EXPECTED_TRAJECTORIES} "
        f"vm_caps={EXPECTED_VM_CAP} subscription_caps=0"
    )
    print("Breakdown: 40 Kimi OpenCode retries + 1 GLM timeout + 5 ML prefixes.")
    print("Excluded: every successful cell, GPT, Opus, and all multi-agent work.")
    print("OpenCode stdin transport validated with a >500 KB resumed prompt.")
    print("No AWS, VM, model, judge, credential, or filesystem write was performed.")


async def run_paid_batch(
    jobs: list[Job], *, run_dir: Path, fingerprint: str, logger: RunLogger
) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"
    setup_commands = (
        ("aws-login", [
            "aws", "login", "--profile", "mats-login", "--region", AWS_REGION,
        ]),
        ("aws-identity", [
            "aws", "sts", "get-caller-identity", "--profile", "mats-run",
            "--region", AWS_REGION,
        ]),
        ("aws-setup", [
            "uv", "run", "exp_real_audit_pipeline.py", "--aws-setup",
            "--confirm-personal-account", "--harness=production",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
        ]),
    )
    for label, command in setup_commands:
        status = await stream_command(
            label=label,
            command=command,
            log_path=run_dir / f"{label}.log",
            logger=logger,
            environment=environment,
            quiet_success=label == "aws-identity",
        )
        if status != 0:
            raise RuntimeError(f"{label} failed; no recovery controller was started")

    STARTED_MARKER.write_text(str(run_dir) + "\n")
    existing_campaigns = _campaign_ids()
    logger.log(
        f"Starting 3 controllers for {EXPECTED_TRAJECTORIES} exact retries; "
        f"global VM cap={EXPECTED_VM_CAP}"
    )
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        if source_fingerprint() != fingerprint:
            launch_failure = f"source changed before {job.label}; stopped later submissions"
            logger.log("STOPPED LAUNCHING: " + launch_failure)
            break
        logger.log(
            f"Submitting {job.label} — cells={job.cells} vm_cap={job.vm_cap}"
        )
        tasks[job.label] = asyncio.create_task(stream_command(
            label=job.label,
            command=job.command(),
            log_path=run_dir / f"{job.label}.log",
            logger=logger,
            environment=environment,
        ))
        if index + 1 < len(jobs):
            await asyncio.sleep(LAUNCH_STAGGER_SECONDS)

    async def heartbeat() -> None:
        while any(not task.done() for task in tasks.values()):
            await asyncio.sleep(HEARTBEAT_SECONDS)
            running = sum(not task.done() for task in tasks.values())
            if running:
                logger.log(f"Recovery controllers still running: {running}/{len(tasks)}")

    heartbeat_task = asyncio.create_task(heartbeat())
    statuses: dict[str, int | None] = {job.label: None for job in jobs}
    for label, task in tasks.items():
        try:
            statuses[label] = await task
        except Exception as error:
            logger.log(f"{label} controller task crashed locally: {error!r}")
            statuses[label] = 1
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    logger.log("All submitted controllers exited. Building the viewer once.")
    viewer_status = await stream_command(
        label="viewer",
        command=["uv", "run", "viewer.py"],
        log_path=run_dir / "viewer.log",
        logger=logger,
        environment=environment,
    )
    rows = []
    for job in jobs:
        row = asdict(job)
        row.update({
            "args": list(job.args),
            "command": job.command(),
            "exit_status": statuses[job.label],
            "log": str(run_dir / f"{job.label}.log"),
        })
        rows.append(row)
    success = (
        launch_failure is None
        and all(status == 0 for status in statuses.values())
        and viewer_status == 0
    )
    final = {
        "format": "continuation-final-recovery-20260816-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": fingerprint,
        "trajectories": EXPECTED_TRAJECTORIES,
        "maximum_vms": EXPECTED_VM_CAP,
        "subscription_maximum_vms": 0,
        "automatic_campaign_retries": False,
        "launch_failure": launch_failure,
        "viewer_exit_status": viewer_status,
        "new_campaign_ids": sorted(_campaign_ids() - existing_campaigns),
        "jobs": rows,
        "success": success,
    }
    _atomic_json(run_dir / "final_status.json", final)
    logger.log(f"Final status: {run_dir / 'final_status.json'}")
    logger.log(
        "DONE." if success else
        "DONE WITH FAILURES. Preserve evidence and inspect exact cells before retrying."
    )
    return 0 if success else 1


def maybe_caffeinate(args: list[str]) -> None:
    variable = "MATS_CONTINUATION_FINAL_RECOVERY_CAFFEINATED"
    if os.environ.get(variable) or not shutil.which("caffeinate"):
        return
    environment = dict(os.environ)
    environment[variable] = "1"
    os.execvpe(
        "caffeinate",
        ["caffeinate", "-i", sys.executable, str(Path(__file__).resolve()), *args],
        environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    jobs = build_jobs()
    fingerprint = validate_plan(jobs)
    if args.plan:
        print_plan(jobs, fingerprint)
        return 0

    maybe_caffeinate(sys.argv[1:])
    for command in ("aws", "git", "uv"):
        if shutil.which(command) is None:
            raise SystemExit(f"required command is unavailable: {command}")
    if STARTED_MARKER.exists():
        raise SystemExit(
            "this exact paid recovery was already started; inspect: "
            + STARTED_MARKER.read_text().strip()
        )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError as error:
        raise SystemExit(f"another final recovery may be active: {LOCK_DIR}") from error

    run_dir = RUN_ROOT / f"{RUN_BASENAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        run_dir.mkdir()
        logger = RunLogger(run_dir / "orchestrator.log")
        try:
            plan = {
                "format": "continuation-final-recovery-20260816-plan-v1",
                "created_at": datetime.now().astimezone().isoformat(),
                "run_directory": str(run_dir),
                "source_fingerprint": fingerprint,
                "trajectories": EXPECTED_TRAJECTORIES,
                "maximum_vms": EXPECTED_VM_CAP,
                "subscription_maximum_vms": 0,
                "automatic_campaign_retries": False,
                "jobs": [
                    {
                        **asdict(job),
                        "args": list(job.args),
                        "command": job.command(),
                        "log": str(run_dir / f"{job.label}.log"),
                    }
                    for job in jobs
                ],
            }
            _atomic_json(run_dir / "plan.json", plan)
            logger.log(f"Run directory: {run_dir}")
            logger.log(f"Frozen plan: {run_dir / 'plan.json'}")
            logger.log(f"Source fingerprint: {fingerprint}")
            logger.log(
                f"Plan: {EXPECTED_TRAJECTORIES} exact retries; "
                f"{EXPECTED_VM_CAP} maximum VMs"
            )
            return asyncio.run(run_paid_batch(
                jobs, run_dir=run_dir, fingerprint=fingerprint, logger=logger
            ))
        finally:
            logger.close()
    finally:
        try:
            LOCK_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
