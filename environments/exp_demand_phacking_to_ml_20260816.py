#!/usr/bin/env python3
"""Generate demand prefixes and run no-honeypot p-hacking -> fraud together.

This paid, one-off wrapper starts two independent experiments in one concurrent slate:

1. Generate eight demand-forecasting prefixes for each available agent. The generated
   prefixes are left untouched for manual inspection and possible continuation later.
2. Continue one frozen positive checkout no-honeypot p-hacking prefix per agent into
   40 fraud-detection trajectories.

The four-agent cohort matches the successful 2026-08-15 prefix push: DeepSeek V4 Pro,
GLM-5.1, and Kimi K2.6 through production, plus GPT-5.5 through subscription. Opus is
excluded because its recent real-environment attempts produced no agent-visible output
and were quarantined.

The slate uses as much useful concurrency as its 192 cells allow (162 simultaneous VM
slots, below the 250-VM account allowance) and caps simultaneous GPT subscription
workers at 18. Child controllers stream labeled output, write separate logs, never
retry automatically, and skip the viewer build because it is managed separately.

Usage (from ``mats/environments``)::

    uv run exp_demand_phacking_to_ml_20260816.py --plan
    uv run exp_demand_phacking_to_ml_20260816.py

``--plan`` is local and read-only. Paid mode is protected by a durable one-use marker;
recovery must use the recorded campaign IDs and exact-cell retry semantics.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ENVIRONMENTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENVIRONMENTS_ROOT.parent
DATA_ROOT = REPO_ROOT.parent / "mats-local" / "environments"
PREFIX_ROOT = DATA_ROOT / "continuation_prefixes"
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"

AWS_REGION = "us-west-2"
AWS_INSTANCE_TYPE = "c7a.xlarge"
JUDGE = "gpt-5.6-luna"
PRODUCTION_TARGETS = ("deepseek-v4-pro", "glm-5.1", "kimi-k2.6")
SUBSCRIPTION_TARGETS = ("gpt-5.5",)
ALL_TARGETS = (*PRODUCTION_TARGETS, *SUBSCRIPTION_TARGETS)

PREFIX_EPOCHS = 8
CONTINUATION_EPOCHS = 40
GLOBAL_VM_LIMIT = 250
SUBSCRIPTION_VM_LIMIT = 18
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180

PLANNED_VM_CAP = 162
PLANNED_PREFIX_GENERATION_CELLS = 32
PLANNED_PHACK_CONTINUATION_CELLS = 160
PLANNED_CELLS = 192
PLANNED_CONTROLLERS = 4

RUN_BASENAME = "demand-phacking-to-ml-20260816"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class FrozenPrefix:
    filename: str
    file_sha256: str
    target: str
    harness: str
    epoch: int

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


# These are the earliest eligible no-honeypot p-hacking payloads produced by the
# 2026-08-15 push: exactly one per model. Freeze their bytes so the paid continuation
# slate cannot silently select a newer or different prefix campaign.
PHACK_PREFIXES = (
    FrozenPrefix(
        "continuation-push-20260815-20260815185916-phack-nohoneypot-production-deepseek-v4-pro-checkout-redesign-positive-e1-d7a09add8fd8.json",
        "a02cf76a7ea59769a441cc5e6e23b77376129c19cb7c8ed0c085353b3120684a",
        "deepseek-v4-pro", "production", 1,
    ),
    FrozenPrefix(
        "continuation-push-20260815-20260815185916-phack-nohoneypot-production-glm-5-1-checkout-redesign-positive-e1-b0d365dcddd7.json",
        "de0e6bd46dfe88d6a881a813532a412b547934649f3890c5a243b2d98870424d",
        "glm-5.1", "production", 1,
    ),
    FrozenPrefix(
        "continuation-push-20260815-20260815185916-phack-nohoneypot-production-kimi-k2-6-checkout-redesign-positive-e1-8e7a35d58ea8.json",
        "34c658e646a10c6c0120c831ead7313b77fd3400f8f6dae45609338d939f5040",
        "kimi-k2.6", "production", 1,
    ),
    FrozenPrefix(
        "continuation-push-20260815-20260815185916-phack-nohoneypot-gpt-gpt-5-5-checkout-redesign-positive-e1-fca38e827408.json",
        "a11d5c23623ebebc0331e5d90d964a8dfa0313aad4e078c36851ad1deb94b18e",
        "gpt-5.5", "subscription", 1,
    ),
)


@dataclass(frozen=True)
class Job:
    label: str
    wave: int
    group: str
    endpoint: str
    harness: str
    cells: int
    vm_cap: int
    args: tuple[str, ...]

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _csv(paths: Iterable[Path]) -> str:
    return ",".join(str(path) for path in paths)


def _prefix_generation_job(
    *, run_slug: str, harness: str, targets: tuple[str, ...], vm_cap: int
) -> Job:
    label = f"generate-demand-{harness}"
    name = f"{run_slug}-demand-{harness}"
    return Job(
        label=label,
        wave=1,
        group="demand prefix generation",
        endpoint="prefixes/exp_ml_prefix.py",
        harness=harness,
        cells=len(targets) * PREFIX_EPOCHS,
        vm_cap=vm_cap,
        args=(
            f"--targets={','.join(targets)}",
            "--seeds=demand_forecasting",
            f"--epochs={PREFIX_EPOCHS}",
            f"--harness={harness}",
            "--reasoning=yes",
            f"--name={name}",
            "--compute=aws",
            f"--vm-concurrency={vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            "--time-limit=4200",
            "--skip-viewer",
        ),
    )


def _continuation_job(
    *, label: str, wave: int, group: str, harness: str,
    prefix_paths: Iterable[Path], vm_cap: int,
) -> Job:
    paths = tuple(prefix_paths)
    return Job(
        label=label,
        wave=wave,
        group=group,
        endpoint="exp_continuation_pipeline.py",
        harness=harness,
        cells=len(paths) * CONTINUATION_EPOCHS,
        vm_cap=vm_cap,
        args=(
            "--treatment=no-honeypot",
            f"--prefix-files={_csv(paths)}",
            "--seed-dir=ml_training_data_misuse",
            "--seeds=fraud_detection",
            f"--epochs={CONTINUATION_EPOCHS}",
            f"--harness={harness}",
            "--condition=allow",
            f"--judge={JUDGE}",
            f"--gate-model={JUDGE}",
            "--compute=aws",
            f"--vm-concurrency={vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            "--time-limit=4200",
            "--skip-viewer",
        ),
    )


def build_jobs(run_slug: str) -> list[Job]:
    production_phack = [
        item.path for item in PHACK_PREFIXES if item.harness == "production"
    ]
    subscription_phack = [
        item.path for item in PHACK_PREFIXES if item.harness == "subscription"
    ]
    return [
        _prefix_generation_job(
            run_slug=run_slug,
            harness="production",
            targets=PRODUCTION_TARGETS,
            vm_cap=24,
        ),
        _prefix_generation_job(
            run_slug=run_slug,
            harness="subscription",
            targets=SUBSCRIPTION_TARGETS,
            vm_cap=8,
        ),
        _continuation_job(
            label="phacking-nohoneypot-to-fraud-production",
            wave=1,
            group="p-hacking no-honeypot -> ML",
            harness="production",
            prefix_paths=production_phack,
            vm_cap=120,
        ),
        _continuation_job(
            label="phacking-nohoneypot-to-fraud-subscription",
            wave=1,
            group="p-hacking no-honeypot -> ML",
            harness="subscription",
            prefix_paths=subscription_phack,
            vm_cap=10,
        ),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prefix_specs(paths: list[Path], *, harness: str) -> None:
    lib_path = str(ENVIRONMENTS_ROOT / "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    from exp_real_continuation import load_prefix_specs  # noqa: PLC0415

    specs = load_prefix_specs([], [str(path) for path in paths], harness=harness)
    if len(specs) != len(paths):
        raise RuntimeError(
            f"{harness}: expected {len(paths)} prefix specs, loaded {len(specs)}"
        )


def validate_frozen_phack_prefixes() -> None:
    expected = {(target, 1) for target in ALL_TARGETS}
    observed = {(item.target, item.epoch) for item in PHACK_PREFIXES}
    if observed != expected or len(PHACK_PREFIXES) != len(expected):
        raise RuntimeError(
            "frozen p-hacking prefix matrix is not one epoch-1 prefix per agent"
        )

    for item in PHACK_PREFIXES:
        if not item.path.is_file():
            raise RuntimeError(f"missing frozen p-hacking prefix: {item.path}")
        if _sha256_file(item.path) != item.file_sha256:
            raise RuntimeError(f"frozen p-hacking prefix bytes drifted: {item.path}")
        payload = json.loads(item.path.read_text())
        source = payload.get("source") or {}
        observed_identity = (
            payload.get("format"), payload.get("target"), payload.get("reasoning"),
            source.get("generator"), source.get("prefix_type"), source.get("seed"),
            source.get("comparison_source_seed"), source.get("harness"),
            source.get("epoch"), source.get("analysis_honeypot"),
            (source.get("continuation_eligibility") or {}).get("eligible_by_default"),
        )
        expected_identity = (
            "environments-continuation-prefix-v1", item.target, True,
            "exp_p_hacking_prefix.py", "p_hacking_no_honeypot",
            "checkout_redesign_positive", "checkout_redesign", item.harness,
            item.epoch, False, True,
        )
        if observed_identity != expected_identity:
            raise RuntimeError(
                f"frozen p-hacking prefix identity drifted for {item.filename}: "
                f"expected {expected_identity!r}, got {observed_identity!r}"
            )

    for harness in ("production", "subscription"):
        paths = [item.path for item in PHACK_PREFIXES if item.harness == harness]
        _load_prefix_specs(paths, harness=harness)


def _validate_jobs(jobs: list[Job]) -> None:
    if not jobs or any(job.wave != 1 for job in jobs):
        raise RuntimeError("job membership drifted")
    labels = [job.label for job in jobs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate job labels")
    vm_cap = sum(job.vm_cap for job in jobs)
    if vm_cap != PLANNED_VM_CAP or vm_cap > GLOBAL_VM_LIMIT:
        raise RuntimeError(
            f"VM cap must be {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}, got {vm_cap}"
        )
    subscription_cap = sum(
        job.vm_cap for job in jobs if job.harness == "subscription"
    )
    if subscription_cap != SUBSCRIPTION_VM_LIMIT:
        raise RuntimeError(
            f"subscription cap must be {SUBSCRIPTION_VM_LIMIT}, "
            f"got {subscription_cap}"
        )
    for job in jobs:
        command = " ".join(job.command()).lower()
        if "opus" in command:
            raise RuntimeError(f"{job.label}: Opus unexpectedly appears in command")
        if "--skip-viewer" not in job.args:
            raise RuntimeError(f"{job.label}: child must defer the viewer build")


def validate_plan(run_slug: str) -> str:
    required = (
        ENVIRONMENTS_ROOT / "prefixes" / "exp_ml_prefix.py",
        ENVIRONMENTS_ROOT / "exp_continuation_pipeline.py",
        ENVIRONMENTS_ROOT / "seeds" / "ml_prefix_only" / "demand_forecasting" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "ml_training_data_misuse" / "fraud_detection" / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required experiment inputs are missing: {missing}")
    validate_frozen_phack_prefixes()
    jobs = build_jobs(run_slug)
    _validate_jobs(jobs)
    if sum(job.cells for job in jobs) != (
        PLANNED_PREFIX_GENERATION_CELLS + PLANNED_PHACK_CONTINUATION_CELLS
    ):
        raise RuntimeError("cell total drifted")
    return source_fingerprint()


def source_fingerprint() -> str:
    digest = hashlib.sha256()

    def git_bytes(*args: str) -> bytes:
        result = subprocess.run(
            ["git", *args], cwd=ENVIRONMENTS_ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout

    digest.update(git_bytes("rev-parse", "HEAD"))
    digest.update(git_bytes("diff", "HEAD", "--binary", "--", "."))
    untracked = git_bytes("ls-files", "--others", "--exclude-standard", "--", ".")
    for raw_name in sorted(line for line in untracked.splitlines() if line):
        relative = raw_name.decode()
        path = ENVIRONMENTS_ROOT / relative
        if not path.is_file():
            continue
        digest.update(raw_name + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def print_plan(run_slug: str, fingerprint: str) -> None:
    jobs = build_jobs(run_slug)
    print("2026-08-16 demand prefixes + p-hacking -> ML batch — exact free plan")
    print(f"source fingerprint: {fingerprint}")
    print()
    for job in jobs:
        print(
            f"[{job.label}] cells={job.cells} harness={job.harness} "
            f"vm_cap={job.vm_cap}"
        )
        print("  " + " ".join(job.command()))
    print()
    print(
        f"TOTAL controllers={PLANNED_CONTROLLERS} paid_cells={PLANNED_CELLS} "
        f"(prefix_generation={PLANNED_PREFIX_GENERATION_CELLS}, "
        f"p_hacking_to_ml={PLANNED_PHACK_CONTINUATION_CELLS})"
    )
    print(
        f"Concurrent VM caps: {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}; "
        f"subscription cap={SUBSCRIPTION_VM_LIMIT}."
    )
    print(
        "Agents: DeepSeek V4 Pro, GLM-5.1, Kimi K2.6, GPT-5.5. "
        "Opus is deliberately excluded after its confirmed no-response failures."
    )
    print(
        "Exactly one frozen positive checkout prefix is used per agent (the earliest "
        "eligible prefix, epoch 1); each gets 40 fraud-detection continuations."
    )
    print(
        "Demand prefix generation is independent: its 8 attempts per agent are saved "
        "for manual inspection and are not continued by this wrapper."
    )
    print("The viewer is not rebuilt or reloaded. No automatic retries are attempted.")
    print("No AWS, VM, model, judge, credential, or filesystem write was performed.")


class RunLogger:
    def __init__(self, path: Path):
        self._handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def log(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()

    def child(self, label: str, message: str) -> str:
        line = f"[{label}] {message}"
        print(line, flush=True)
        self._handle.write(line + "\n")
        self._handle.flush()
        return line


async def stream_command(
    *, label: str, command: list[str], log_path: Path, logger: RunLogger,
    environment: dict[str, str], quiet_success: bool = False,
) -> int:
    logger.log(f"Starting {label} — log={log_path}")
    with log_path.open("w", encoding="utf-8") as job_log:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=ENVIRONMENTS_ROOT,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as error:
            line = logger.child(label, f"could not start: {error}")
            job_log.write(line + "\n")
            return 127
        assert process.stdout is not None
        while True:
            raw_line = await process.stdout.readline()
            if not raw_line:
                break
            line = logger.child(label, raw_line.decode(errors="replace").rstrip("\n"))
            job_log.write(line + "\n")
            job_log.flush()
        status = await process.wait()
        if not quiet_success or status != 0:
            line = logger.child(label, f"process exited — status={status}")
            job_log.write(line + "\n")
    logger.log(f"Finished {label} — exit={status}")
    return status


async def run_wave(
    jobs: list[Job], *, run_dir: Path, logger: RunLogger,
    environment: dict[str, str], fingerprint: str,
) -> tuple[dict[str, int | None], str | None]:
    logger.log(
        f"Starting concurrent slate: controllers={len(jobs)} "
        f"cells={sum(job.cells for job in jobs)} vm_cap={sum(job.vm_cap for job in jobs)}"
    )
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        if source_fingerprint() != fingerprint:
            launch_failure = (
                f"source changed before {job.label}; stopped submitting later controllers"
            )
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
                logger.log(
                    "Slate controllers still running: "
                    f"{running}/{len(tasks)}"
                )

    heartbeat_task = asyncio.create_task(heartbeat())
    statuses: dict[str, int | None] = {job.label: None for job in jobs}
    for label, task in tasks.items():
        try:
            statuses[label] = await task
        except Exception as error:  # retain other controllers and their evidence
            logger.log(f"{label} controller task crashed locally: {error!r}")
            statuses[label] = 1
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass
    return statuses, launch_failure


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _campaign_ids() -> set[str]:
    if not CAMPAIGN_ROOT.is_dir():
        return set()
    return {path.stem for path in CAMPAIGN_ROOT.glob("*.json")}


def _serialized_job(job: Job, status: int | None, run_dir: Path) -> dict:
    row = asdict(job)
    row["args"] = list(job.args)
    row["command"] = job.command()
    row["exit_status"] = status
    row["log"] = str(run_dir / f"{job.label}.log")
    return row


async def run_paid_batch(
    *, run_slug: str, run_dir: Path, fingerprint: str, logger: RunLogger,
) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"

    setup_commands = (
        (
            "aws-login",
            ["aws", "login", "--profile", "mats-login", "--region", AWS_REGION],
        ),
        (
            "aws-identity",
            [
                "aws", "sts", "get-caller-identity", "--profile", "mats-run",
                "--region", AWS_REGION,
            ],
        ),
        (
            "aws-setup",
            [
                "uv", "run", "exp_real_audit_pipeline.py", "--aws-setup",
                "--confirm-personal-account", "--harness=subscription",
                f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            ],
        ),
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
            raise RuntimeError(f"{label} failed; no experiment controller was started")

    STARTED_MARKER.write_text(str(run_dir) + "\n")
    existing_campaigns = _campaign_ids()
    jobs = build_jobs(run_slug)
    statuses, launch_failure = await run_wave(
        jobs,
        run_dir=run_dir,
        logger=logger,
        environment=environment,
        fingerprint=fingerprint,
    )
    logger.log(
        "All submitted controllers exited. Generated demand prefixes remain "
        "uncontinued for manual inspection; viewer rebuild is intentionally skipped."
    )
    new_campaigns = sorted(_campaign_ids() - existing_campaigns)
    success = (
        launch_failure is None
        and all(statuses.get(job.label) == 0 for job in jobs)
    )
    final_status = {
        "format": "environments-demand-phacking-to-ml-20260816-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": fingerprint,
        "planned_cells": PLANNED_CELLS,
        "planned_controllers": PLANNED_CONTROLLERS,
        "maximum_vms": GLOBAL_VM_LIMIT,
        "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
        "automatic_campaign_retries": False,
        "opus_included": False,
        "launch_failure": launch_failure,
        "demand_prefixes_continued": False,
        "viewer_rebuilt": False,
        "new_campaign_ids": new_campaigns,
        "frozen_p_hacking_prefixes": [
            {
                **asdict(item),
                "path": str(item.path),
            }
            for item in PHACK_PREFIXES
        ],
        "jobs": [
            _serialized_job(job, statuses.get(job.label), run_dir)
            for job in jobs
        ],
        "success": success,
    }
    _atomic_json(run_dir / "final_status.json", final_status)
    logger.log(f"Final status: {run_dir / 'final_status.json'}")
    if success:
        logger.log("DONE. Results were retained without rebuilding the viewer.")
        return 0
    logger.log(
        "DONE WITH FAILURES. Evidence was retained; inspect labeled logs and "
        "campaign states before any exact-cell recovery."
    )
    return 1


def _maybe_caffeinate(args: list[str]) -> None:
    variable = "MATS_DEMAND_PHACKING_TO_ML_CAFFEINATED"
    if os.environ.get(variable) or not shutil.which("caffeinate"):
        return
    environment = dict(os.environ)
    environment[variable] = "1"
    os.execvpe(
        "caffeinate",
        ["caffeinate", "-i", sys.executable, str(Path(__file__).resolve()), *args],
        environment,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_slug = f"{RUN_BASENAME}-plan"
    fingerprint = validate_plan(plan_slug)
    if args.plan:
        print_plan(plan_slug, fingerprint)
        return 0

    _maybe_caffeinate(sys.argv[1:])
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if STARTED_MARKER.exists():
        raise SystemExit(
            f"this exact paid slate already started; inspect {STARTED_MARKER} and "
            "recover from recorded campaign IDs instead of repeating it"
        )
    try:
        LOCK_DIR.mkdir()
    except FileExistsError as error:
        raise SystemExit(f"another {RUN_BASENAME} wrapper may be active: {LOCK_DIR}") from error

    run_slug = f"{RUN_BASENAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir = RUN_ROOT / run_slug
    run_dir.mkdir()
    logger = RunLogger(run_dir / "orchestrator.log")
    try:
        plan_record = {
            "format": "environments-demand-phacking-to-ml-20260816-plan-v1",
            "run_slug": run_slug,
            "source_fingerprint": fingerprint,
            "planned_cells": PLANNED_CELLS,
            "planned_controllers": PLANNED_CONTROLLERS,
            "demand_prefixes_continued": False,
            "viewer_rebuilt": False,
            "jobs": [asdict(job) for job in build_jobs(run_slug)],
        }
        _atomic_json(run_dir / "plan.json", plan_record)
        logger.log(f"Run directory: {run_dir}")
        logger.log(f"Source fingerprint: {fingerprint}")
        return asyncio.run(run_paid_batch(
            run_slug=run_slug,
            run_dir=run_dir,
            fingerprint=fingerprint,
            logger=logger,
        ))
    finally:
        logger.close()
        try:
            LOCK_DIR.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
