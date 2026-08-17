#!/usr/bin/env python3
"""Launch only the desired work left after the 2026-08-15 salvage.

This one-off paid launcher retries the exact terminal missing continuation cells and
starts the four prefix-generation campaigns that never reached AWS. It deliberately
excludes every multi-agent ML campaign and every already successful cell.

Usage from mats/environments/:

  uv run exp_continuation_remaining_20260815.py --plan
  uv run exp_continuation_remaining_20260815.py
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
from exp_subscription_harness import _codex_exec_transport  # noqa: E402


DATA_ROOT = (ENVIRONMENTS / "../../mats-local/environments").resolve()
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"
AWS_REGION = "us-west-2"
AWS_INSTANCE_TYPE = "c7a.xlarge"
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180
RUN_BASENAME = "continuation-remaining-20260815"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"
EXPECTED_TRAJECTORIES = 424
EXPECTED_VM_CAP = 92
EXPECTED_SUBSCRIPTION_VM_CAP = 10
WORST_CASE_VM_HOURS_PER_CELL = 4.5
HOURLY_VM_PRICE_USD = 0.20528


@dataclass(frozen=True)
class RetryParent:
    label: str
    campaign_id: str
    harness: str
    expected_by_target: tuple[tuple[str, int], ...]
    vm_cap: int

    @property
    def cells(self) -> int:
        return sum(count for _, count in self.expected_by_target)

    @property
    def state_path(self) -> Path:
        return CAMPAIGN_ROOT / f"{self.campaign_id}.json"


RETRY_PARENTS = (
    RetryParent(
        "wiki-ml-exact-retry",
        "continuation-aws-wikipedia-summaries-40ep-20260815-015949-82cceb1f-"
        "retry-20260815-123006-c16c95f5-retry-20260815-185954-93b0ed3c",
        "production",
        (("glm-5.1", 1), ("kimi-k2.6", 28)),
        18,
    ),
    RetryParent(
        "inline-phack-hack1-production-exact-retry",
        "continuation-aws-hack-in-one-turn-40ep-20260815-190158-5d8b36c8",
        "production",
        (("deepseek-v4-pro", 20), ("glm-5.1", 40), ("kimi-k2.6", 40)),
        20,
    ),
    RetryParent(
        "inline-phack-hack2-production-exact-retry",
        "continuation-aws-hack-in-two-turns-40ep-20260815-190218-83b32284",
        "production",
        (("deepseek-v4-pro", 20), ("glm-5.1", 40), ("kimi-k2.6", 40)),
        20,
    ),
    RetryParent(
        "inline-phack-hack2-gpt-exact-retry",
        "continuation-aws-hack-in-two-turns-40ep-20260815-190238-662ce43b",
        "subscription",
        (("gpt-5.5", 40),),
        3,
    ),
    RetryParent(
        "inline-phack-nohack-production-exact-retry",
        "continuation-aws-no-hack-40ep-20260815-190258-0ce87105",
        "production",
        (("glm-5.1", 26), ("kimi-k2.6", 40)),
        14,
    ),
    RetryParent(
        "inline-phack-nohack-gpt-exact-retry",
        "continuation-aws-no-hack-40ep-20260815-190318-3cddd339",
        "subscription",
        (("gpt-5.5", 37),),
        3,
    ),
)


PREFIX_NAMES = (
    "continuation-push-20260815-20260815185916-ml-demand-production",
    "continuation-push-20260815-20260815185916-ml-demand-gpt",
    "continuation-push-20260815-20260815185916-phack-nohoneypot-production",
    "continuation-push-20260815-20260815185916-phack-nohoneypot-gpt",
)


def retryable(cell: dict) -> bool:
    if cell.get("status") in {"not_launched", "infrastructure_failure"}:
        return True
    if cell.get("status") != "completed":
        return False
    try:
        return int((cell.get("terminal") or {}).get("pipeline_exit_code")) != 0
    except (TypeError, ValueError):
        return True


def retry_job(parent: RetryParent) -> Job:
    return Job(
        label=parent.label,
        group="exact continuation retry",
        endpoint="exp_continuation_pipeline.py",
        harness=parent.harness,
        cells=parent.cells,
        vm_cap=parent.vm_cap,
        args=(
            f"--retry-failed={parent.campaign_id}",
            "--retry-pipeline-failures",
            f"--harness={parent.harness}",
            f"--vm-concurrency={parent.vm_cap}",
            "--skip-viewer",
        ),
        retry_parent=parent.campaign_id,
    )


def build_jobs() -> list[Job]:
    jobs = [retry_job(parent) for parent in RETRY_PARENTS]
    jobs.extend([
        Job(
            "generate-ml-demand-production", "prefix generation",
            "prefixes/exp_ml_prefix.py", "production", 30, 7,
            (
                "--targets=deepseek-v4-pro,glm-5.1,kimi-k2.6",
                "--seeds=demand_forecasting", "--epochs=10",
                "--harness=production", "--reasoning=yes",
                f"--name={PREFIX_NAMES[0]}", "--compute=aws",
                "--vm-concurrency=7", "--aws-region=us-west-2",
                "--aws-instance-type=c7a.xlarge", "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            "generate-ml-demand-gpt", "prefix generation",
            "prefixes/exp_ml_prefix.py", "subscription", 10, 2,
            (
                "--targets=gpt-5.5", "--seeds=demand_forecasting", "--epochs=10",
                "--harness=subscription", "--reasoning=yes",
                f"--name={PREFIX_NAMES[1]}", "--compute=aws",
                "--vm-concurrency=2", "--aws-region=us-west-2",
                "--aws-instance-type=c7a.xlarge", "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            "generate-phack-nohoneypot-production", "prefix generation",
            "prefixes/exp_p_hacking_prefix.py", "production", 9, 3,
            (
                "--targets=deepseek-v4-pro,glm-5.1,kimi-k2.6",
                "--seeds=checkout_redesign_positive", "--epochs=3",
                "--harness=production", "--pressure=low", "--reasoning=yes",
                f"--name={PREFIX_NAMES[2]}", "--compute=aws",
                "--vm-concurrency=3", "--aws-region=us-west-2",
                "--aws-instance-type=c7a.xlarge", "--time-limit=1800",
                "--skip-viewer",
            ),
        ),
        Job(
            "generate-phack-nohoneypot-gpt", "prefix generation",
            "prefixes/exp_p_hacking_prefix.py", "subscription", 3, 2,
            (
                "--targets=gpt-5.5", "--seeds=checkout_redesign_positive",
                "--epochs=3", "--harness=subscription", "--pressure=low",
                "--reasoning=yes", f"--name={PREFIX_NAMES[3]}", "--compute=aws",
                "--vm-concurrency=2", "--aws-region=us-west-2",
                "--aws-instance-type=c7a.xlarge", "--time-limit=1800",
                "--skip-viewer",
            ),
        ),
    ])
    return jobs


def validate_parent(parent: RetryParent) -> None:
    if not parent.state_path.is_file():
        raise RuntimeError(f"missing retry parent: {parent.state_path}")
    state = json.loads(parent.state_path.read_text())
    config = state.get("pipeline_config") or {}
    if state.get("campaign_id") != parent.campaign_id:
        raise RuntimeError(f"{parent.label}: campaign identity changed")
    if config.get("harness") != parent.harness:
        raise RuntimeError(f"{parent.label}: harness changed")
    if config.get("multi_agent"):
        raise RuntimeError(f"{parent.label}: multi-agent parent is forbidden")
    if (state.get("s3_cleanup") or {}).get("status") != "deleted":
        raise RuntimeError(f"{parent.label}: salvage S3 cleanup is not complete")
    if any(
        cell.get("status") in {"planned", "launching", "running", "finishing"}
        for cell in state.get("cells") or []
    ):
        raise RuntimeError(f"{parent.label}: parent is not terminal")
    selected = [cell for cell in state.get("cells") or [] if retryable(cell)]
    by_target = Counter(str(cell.get("target")) for cell in selected)
    expected = Counter(dict(parent.expected_by_target))
    if by_target != expected:
        raise RuntimeError(
            f"{parent.label}: exact retry drifted; expected {dict(expected)}, "
            f"found {dict(by_target)}"
        )
    if any("opus" in target.lower() for target in by_target):
        raise RuntimeError(f"{parent.label}: Opus unexpectedly selected")
    for payload in (config.get("continuation") or {}).get("payloads") or []:
        local_path = Path(str(payload.get("local_path") or ""))
        if not local_path.is_file():
            raise RuntimeError(f"{parent.label}: stored payload is missing: {local_path}")


def validate_prefix_names_unused() -> None:
    used: dict[str, str] = {}
    for path in CAMPAIGN_ROOT.glob("*.json"):
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = str((state.get("pipeline_config") or {}).get("name") or "")
        if name in PREFIX_NAMES:
            used[name] = str(state.get("campaign_id") or path.stem)
    if used:
        raise RuntimeError(f"prefix-generation names were already launched: {used}")


def validate_codex_stdin_fix() -> None:
    prompt = "long activity log line\n" * 20_000
    for session_id in (None, "resume-session"):
        command, stdin = _codex_exec_transport(
            "/usr/bin/codex", ["--json", "--model", "gpt-5.5"],
            session_id, prompt,
        )
        if command[-1] != "-" or prompt in command or stdin != prompt:
            raise RuntimeError("Codex long-prompt stdin transport fix is not active")


def validate_plan(jobs: list[Job]) -> str:
    if len(jobs) != 10 or len({job.label for job in jobs}) != len(jobs):
        raise RuntimeError("remaining controller matrix drifted")
    if sum(job.cells for job in jobs) != EXPECTED_TRAJECTORIES:
        raise RuntimeError("remaining trajectory total drifted")
    if sum(job.vm_cap for job in jobs) != EXPECTED_VM_CAP:
        raise RuntimeError("remaining VM cap drifted")
    subscription_cap = sum(
        job.vm_cap for job in jobs if job.harness == "subscription"
    )
    if subscription_cap != EXPECTED_SUBSCRIPTION_VM_CAP:
        raise RuntimeError("remaining subscription VM cap drifted")
    if any("opus" in " ".join(job.command()).lower() for job in jobs):
        raise RuntimeError("Opus unexpectedly appears in the remaining slate")
    if any("multi_agent" in job.endpoint or "multi-agent" in " ".join(job.command())
           for job in jobs):
        raise RuntimeError("multi-agent work unexpectedly appears in the remaining slate")
    for parent in RETRY_PARENTS:
        validate_parent(parent)
    validate_prefix_names_unused()
    validate_codex_stdin_fix()
    for relative in (
        "prefixes/exp_ml_prefix.py",
        "prefixes/exp_p_hacking_prefix.py",
        "seeds/ml_prefix_only/demand_forecasting/manifest.json",
        "seeds/p_hacking_prefix_only/checkout_redesign_positive/manifest.json",
    ):
        if not (ENVIRONMENTS / relative).is_file():
            raise RuntimeError(f"required input is missing: {relative}")
    return source_fingerprint()


def print_plan(jobs: list[Job], fingerprint: str) -> None:
    print("2026-08-15 continuation push — exact remaining-work plan")
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
    worst_compute = (
        EXPECTED_TRAJECTORIES * WORST_CASE_VM_HOURS_PER_CELL * HOURLY_VM_PRICE_USD
    )
    print(
        f"\nTOTAL controllers={len(jobs)} trajectories={EXPECTED_TRAJECTORIES} "
        f"vm_caps={EXPECTED_VM_CAP} subscription_caps={EXPECTED_SUBSCRIPTION_VM_CAP}"
    )
    print("Breakdown: 372 exact continuation retries + 52 never-started prefixes.")
    print("Harnesses: production=DeepSeek/GLM/Kimi; subscription=GPT-5.5; Opus=0.")
    print("Excluded: all multi-agent ML campaigns and all 124 salvaged successful cells.")
    print("Codex GPT prompts: stdin transport validated with a >300 KB prompt.")
    print(
        f"Worst-case EC2-only watchdog bound: ${worst_compute:.2f}; excludes API, "
        "judge, EBS, IPv4, S3, and transfer costs."
    )
    print("No AWS, VM, model, judge, credential, or filesystem write was performed.")


async def run_paid_batch(
    jobs: list[Job], *, run_dir: Path, fingerprint: str, logger: RunLogger
) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"

    for label, command in (
        ("aws-login", ["aws", "login", "--profile", "mats-login", "--region", AWS_REGION]),
        ("aws-identity", ["aws", "sts", "get-caller-identity", "--profile", "mats-run", "--region", AWS_REGION]),
        ("aws-setup", [
            "uv", "run", "exp_real_audit_pipeline.py", "--aws-setup",
            "--confirm-personal-account", "--harness=subscription",
            f"--aws-region={AWS_REGION}", f"--aws-instance-type={AWS_INSTANCE_TYPE}",
        ]),
    ):
        status = await stream_command(
            label=label,
            command=command,
            log_path=run_dir / f"{label}.log",
            logger=logger,
            environment=environment,
            quiet_success=label == "aws-identity",
        )
        if status != 0:
            raise RuntimeError(f"{label} failed; no remaining-work controller was started")

    STARTED_MARKER.write_text(str(run_dir) + "\n")
    existing_campaigns = _campaign_ids()
    logger.log(
        f"Starting {len(jobs)} controllers for {EXPECTED_TRAJECTORIES} trajectories; "
        f"global VM cap={EXPECTED_VM_CAP}; subscription cap={EXPECTED_SUBSCRIPTION_VM_CAP}"
    )
    logger.log("Multi-agent ML and Opus are excluded. Child failures are never retried automatically.")
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        if source_fingerprint() != fingerprint:
            launch_failure = f"source changed before {job.label}; stopped later submissions"
            logger.log("STOPPED LAUNCHING: " + launch_failure)
            break
        logger.log(
            f"Submitting {job.label} — cells={job.cells} harness={job.harness} "
            f"vm_cap={job.vm_cap}"
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
                logger.log(f"Campaign controllers still running: {running}/{len(tasks)}")

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
        "format": "continuation-remaining-20260815-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": fingerprint,
        "trajectories": EXPECTED_TRAJECTORIES,
        "maximum_vms": EXPECTED_VM_CAP,
        "subscription_maximum_vms": EXPECTED_SUBSCRIPTION_VM_CAP,
        "multi_agent_ml_included": False,
        "opus_included": False,
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
    if os.environ.get("MATS_CONTINUATION_REMAINING_CAFFEINATED") or not shutil.which("caffeinate"):
        return
    environment = dict(os.environ)
    environment["MATS_CONTINUATION_REMAINING_CAFFEINATED"] = "1"
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
            "this exact paid remaining-work slate was already started; inspect: "
            + STARTED_MARKER.read_text().strip()
        )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError as error:
        raise SystemExit(f"another remaining-work launcher may be active: {LOCK_DIR}") from error

    run_dir = RUN_ROOT / f"{RUN_BASENAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        run_dir.mkdir()
        logger = RunLogger(run_dir / "orchestrator.log")
        try:
            plan = {
                "format": "continuation-remaining-20260815-plan-v1",
                "created_at": datetime.now().astimezone().isoformat(),
                "run_directory": str(run_dir),
                "source_fingerprint": fingerprint,
                "trajectories": EXPECTED_TRAJECTORIES,
                "maximum_vms": EXPECTED_VM_CAP,
                "subscription_maximum_vms": EXPECTED_SUBSCRIPTION_VM_CAP,
                "multi_agent_ml_included": False,
                "opus_included": False,
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
                f"Plan: {EXPECTED_TRAJECTORIES} trajectories; "
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
