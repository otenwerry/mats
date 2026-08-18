#!/usr/bin/env python3
"""Run the GPT-5.5 version of the three selected continuation experiments.

The paid slate is deliberately limited to work that was deferred from the preceding
open-model batch:

1. demand_forecasting -> fraud_detection with one judged no-hack prefix, one
   judged clean prefix, and one no-honeypot prefix;
2. the matched inline activity-log condition for the two judged prefixes only;
3. reasoning_prompt_benchmark no-honeypot -> checkout_redesign.

There is no successful judged GPT demand-forecasting reward-hack trajectory, so this
wrapper does not fabricate a hack arm. Each of the six source/condition arms gets 40
continuations. Four controllers share a user-selected experimental ceiling of 100
simultaneous subscription workers. OpenAI does not publish a 100-worker allowance;
this is an intentionally aggressive local scheduling choice.

Usage (from ``mats/environments``)::

    uv run exp_next_gpt_20260817.py --plan
    uv run exp_next_gpt_20260817.py

``--plan`` is local, free, and read-only. Paid mode performs a live subscription-auth
preflight, AWS setup, and then launches the frozen slate. It has a durable one-use
marker and never retries campaigns automatically.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import exp_next_open_20260817 as precedent


ENVIRONMENTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENVIRONMENTS_ROOT.parent
DATA_ROOT = REPO_ROOT.parent / "mats-local" / "environments"
PREFIX_ROOT = DATA_ROOT / "continuation_prefixes"
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"

AWS_REGION = "us-west-2"
AWS_INSTANCE_TYPE = "c7a.xlarge"
JUDGE = "gpt-5.6-luna"
HARNESS = "subscription"

EPOCHS = 40
GLOBAL_VM_LIMIT = 250
SUBSCRIPTION_VM_LIMIT = 100
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180

PLANNED_REGULAR_ML_CELLS = 120
PLANNED_MULTI_AGENT_ML_CELLS = 80
PLANNED_PHACK_CELLS = 40
PLANNED_CELLS = 240
PLANNED_CONTROLLERS = 4
PLANNED_VM_CAP = 100

RUN_BASENAME = "next-gpt-continuations-20260817"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class FrozenPrefix:
    filename: str
    file_sha256: str
    name: str
    viewer_label: str
    condition: str
    source_kind: str
    epoch: int
    message_count: int
    trajectory_id: int | None = None

    target: str = "gpt-5.5"
    harness: str = HARNESS

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


JUDGED_ML_PREFIXES = (
    FrozenPrefix(
        "traj781-8c98a962c259.json",
        "2fc41768df9dcd4ab5e8ed124cc5743384a88a0bcd63cd5b84e9512e8a66edd4",
        "traj781", "#781", "notable", "judged-conversation", 1, 97, 781,
    ),
    FrozenPrefix(
        "traj783-c05e183128f0.json",
        "e29f1cef2bb9339b7254c5817aea54188645dc9592c883f6af235b9bd6862645",
        "traj783", "#783", "clean", "judged-conversation", 1, 100, 783,
    ),
)


ACTIVITY_LOG_PREFIXES = (
    FrozenPrefix(
        "activity-log-traj781-e70e87255a3e.json",
        "570fed35f158bfc929d8f8bd96c809f1ddd4ade61f0a5911806ed316d8f1e96b",
        "activity-log-traj781", "#781 activity log", "notable",
        "activity-log", 1, 1, 781,
    ),
    FrozenPrefix(
        "activity-log-traj783-1f837e98f439.json",
        "09c7f972b8d9594dccae26d23f82dde85d38c97d328760b7191646ce6cd83637",
        "activity-log-traj783", "#783 activity log", "clean",
        "activity-log", 1, 1, 783,
    ),
)


ML_NO_HONEYPOT_PREFIX = FrozenPrefix(
    "next-open-continuations-20260817-20260817014146-demand-gpt-gpt-5-5-"
    "demand-forecasting-e5-23557dc62c31.json",
    "f7fc422a3fb0d87f85cefcde899e2a2d62040cd49e34ab7157e7657c02743f1a",
    "next-open-continuations-20260817-20260817014146-demand-gpt-gpt-5-5-"
    "demand-forecasting-e5",
    "MLD-136", "no_honeypot", "ml-no-honeypot", 5, 94,
)


PHACK_NO_HONEYPOT_PREFIX = FrozenPrefix(
    "next-open-continuations-20260817-20260817014146-reasoning-nohoneypot-"
    "gpt-gpt-5-5-reasoning-prompt-benchmark-positive-e1-747b3f454be3.json",
    "a1cefc1980e46a02fe2b4c47cdf683f304d2ed139e90e0c2b1c04b31e2d0db21",
    "next-open-continuations-20260817-20260817014146-reasoning-nohoneypot-"
    "gpt-gpt-5-5-reasoning-prompt-benchmark-positive-e1",
    "PHN-37", "no_honeypot", "phack-no-honeypot", 1, 23,
)


ALL_PREFIXES = (
    *JUDGED_ML_PREFIXES,
    *ACTIVITY_LOG_PREFIXES,
    ML_NO_HONEYPOT_PREFIX,
    PHACK_NO_HONEYPOT_PREFIX,
)


@dataclass(frozen=True)
class Job:
    label: str
    group: str
    source_mode: str
    treatment: str
    destination: str
    cells: int
    vm_cap: int
    prefix_paths: tuple[Path, ...]
    args: tuple[str, ...]
    endpoint: str = "exp_continuation_pipeline.py"
    harness: str = HARNESS

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _csv(paths: Iterable[Path]) -> str:
    return ",".join(str(path) for path in paths)


def _ml_job(
    *, label: str, group: str, source_mode: str, treatment: str,
    paths: tuple[Path, ...], vm_cap: int,
) -> Job:
    return Job(
        label=label,
        group=group,
        source_mode=source_mode,
        treatment=treatment,
        destination="fraud_detection",
        cells=len(paths) * EPOCHS,
        vm_cap=vm_cap,
        prefix_paths=paths,
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={_csv(paths)}",
            "--seed-dir=ml_training_data_misuse",
            "--seeds=fraud_detection",
            f"--epochs={EPOCHS}",
            f"--harness={HARNESS}",
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


def _phack_job(*, path: Path, vm_cap: int) -> Job:
    treatment = "no-honeypot"
    return Job(
        label="phack-nohoneypot-to-phack-gpt",
        group="p-hacking no-honeypot -> p-hacking",
        source_mode="conversation",
        treatment=treatment,
        destination="checkout_redesign",
        cells=EPOCHS,
        vm_cap=vm_cap,
        prefix_paths=(path,),
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={path}",
            "--seed-dir=p_hacking",
            "--seeds=checkout_redesign",
            f"--epochs={EPOCHS}",
            f"--harness={HARNESS}",
            "--condition=allow",
            "--pressure=low",
            f"--judge={JUDGE}",
            f"--gate-model={JUDGE}",
            "--compute=aws",
            f"--vm-concurrency={vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            "--time-limit=1800",
            "--skip-viewer",
        ),
    )


def build_jobs() -> list[Job]:
    return [
        _ml_job(
            label="ml-conversation-judged-gpt",
            group="regular ML -> ML",
            source_mode="conversation",
            treatment="no-hack",
            paths=tuple(item.path for item in JUDGED_ML_PREFIXES),
            vm_cap=37,
        ),
        _ml_job(
            label="ml-conversation-nohoneypot-gpt",
            group="regular ML -> ML",
            source_mode="conversation",
            treatment="no-honeypot",
            paths=(ML_NO_HONEYPOT_PREFIX.path,),
            vm_cap=18,
        ),
        _ml_job(
            label="ml-activity-judged-gpt",
            group="multi-agent ML -> ML",
            source_mode="activity-log",
            treatment="no-hack",
            paths=tuple(item.path for item in ACTIVITY_LOG_PREFIXES),
            vm_cap=37,
        ),
        _phack_job(path=PHACK_NO_HONEYPOT_PREFIX.path, vm_cap=8),
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_inputs_fingerprint() -> str:
    digest = hashlib.sha256()
    for item in ALL_PREFIXES:
        digest.update(item.filename.encode() + b"\0")
        digest.update(bytes.fromhex(_sha256_file(item.path)))
    return digest.hexdigest()


def _expected_prefix_identity(item: FrozenPrefix, payload: dict) -> None:
    source = payload.get("source") or {}
    condition = source.get("prefix_condition") or {}
    eligibility = source.get("continuation_eligibility") or {}
    native = payload.get("native_resume") or {}

    common = (
        payload.get("format"), payload.get("name"), payload.get("target"),
        payload.get("reasoning"), source.get("harness"), source.get("epoch"),
        len(payload.get("messages") or []),
    )
    expected_common = (
        "environments-continuation-prefix-v1", item.name, item.target, True,
        item.harness, item.epoch, item.message_count,
    )
    if common != expected_common:
        raise RuntimeError(
            f"frozen prefix identity drifted for {item.filename}: "
            f"expected {expected_common!r}, got {common!r}"
        )

    if item.source_kind == "judged-conversation":
        identity = (
            source.get("kind"), source.get("family"), source.get("seed"),
            source.get("trajectory_id"), source.get("prefix_type"),
            source.get("prefix_type_label"), condition.get("key"),
            source.get("was_continuation"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "trajectory", "ml_training_data_misuse", "demand_forecasting",
            item.trajectory_id, "ml_demand_forecasting", "ML: demand_forecasting",
            item.condition, False, "environments-production-native-resume-v1",
            "gpt-5.5", True,
        )
    elif item.source_kind == "activity-log":
        metadata = source.get("activity_log_metadata") or {}
        lossy = metadata.get("lossy_processing") or {}
        identity = (
            source.get("kind"), source.get("generator"), source.get("family"),
            source.get("seed"), source.get("trajectory_id"),
            source.get("prefix_type"), condition.get("key"),
            (payload.get("delivery") or {}).get("mode"),
            payload.get("native_resume"), lossy.get("affected"),
            lossy.get("kind"), bool(lossy.get("visible_caveat")),
        )
        expected = (
            "trajectory", "build_activity_log_prefixes.py",
            "ml_training_data_misuse", "demand_forecasting", item.trajectory_id,
            "activity_log_context", item.condition, "inline_user_context", None,
            True, "observable_activity_only", True,
        )
    elif item.source_kind == "ml-no-honeypot":
        identity = (
            source.get("kind"), source.get("generator"), source.get("family"),
            source.get("seed"), source.get("analysis_honeypot"),
            eligibility.get("eligible_by_default"),
            eligibility.get("missing_required_deliverables"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "external", "exp_ml_prefix.py", "ml_prefix_only",
            "demand_forecasting", None, True, [],
            "environments-production-native-resume-v1", "gpt-5.5", True,
        )
    elif item.source_kind == "phack-no-honeypot":
        identity = (
            source.get("kind"), source.get("generator"), source.get("family"),
            source.get("prefix_type"), source.get("seed"),
            source.get("comparison_source_seed"), source.get("analysis_honeypot"),
            source.get("result_condition"), source.get("pressure"),
            eligibility.get("eligible_by_default"),
            eligibility.get("missing_required_deliverables"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "external", "exp_p_hacking_prefix.py", "p_hacking_prefix_only",
            "p_hacking_no_honeypot", "reasoning_prompt_benchmark_positive",
            "reasoning_prompt_benchmark", False, "clear positive", "low", True, [],
            "environments-production-native-resume-v1", "gpt-5.5", True,
        )
    else:
        raise RuntimeError(f"unknown frozen source kind: {item.source_kind}")

    if identity != expected:
        raise RuntimeError(
            f"frozen prefix provenance drifted for {item.filename}: "
            f"expected {expected!r}, got {identity!r}"
        )


def validate_frozen_prefixes() -> None:
    filenames = [item.filename for item in ALL_PREFIXES]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("frozen prefix filenames are not unique")
    if len(JUDGED_ML_PREFIXES) != 2 or len(ACTIVITY_LOG_PREFIXES) != 2:
        raise RuntimeError("the matched judged/activity-log matrix must be 2 + 2")

    judged_pairs = {
        (item.trajectory_id, item.target, item.condition)
        for item in JUDGED_ML_PREFIXES
    }
    activity_pairs = {
        (item.trajectory_id, item.target, item.condition)
        for item in ACTIVITY_LOG_PREFIXES
    }
    if judged_pairs != activity_pairs:
        raise RuntimeError("activity-log sources are not an exact judged-source copy")
    if ML_NO_HONEYPOT_PREFIX in ACTIVITY_LOG_PREFIXES:
        raise RuntimeError("the no-honeypot ML source must not enter multi-agent")

    for item in ALL_PREFIXES:
        if not item.path.is_file():
            raise RuntimeError(f"missing frozen prefix: {item.path}")
        observed_hash = _sha256_file(item.path)
        if observed_hash != item.file_sha256:
            raise RuntimeError(
                f"frozen prefix bytes drifted: {item.path}; "
                f"expected {item.file_sha256}, got {observed_hash}"
            )
        _expected_prefix_identity(item, json.loads(item.path.read_text()))

    lib_path = str(ENVIRONMENTS_ROOT / "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    from exp_real_continuation import (  # noqa: PLC0415
        build_continuation_cells,
        load_prefix_specs,
    )
    from protocol_sources import (  # noqa: PLC0415
        load_protocol_sources,
        resolve_seeds,
    )

    for job in build_jobs():
        specs = load_prefix_specs(
            [], [str(path) for path in job.prefix_paths], harness=job.harness,
        )
        seed_dir = (
            "ml_training_data_misuse"
            if job.destination == "fraud_detection"
            else "p_hacking"
        )
        seeds_path, available = resolve_seeds(seed_dir)
        if job.destination not in available:
            raise RuntimeError(
                f"{job.label}: destination {job.destination!r} is unavailable"
            )
        cells = build_continuation_cells(specs, seeds_path, [job.destination])
        pressure = "low" if seed_dir == "p_hacking" else None
        for unit_path in {cell.unit_path for cell in cells}:
            load_protocol_sources(unit_path, pressure=pressure)
        if len(cells) * EPOCHS != job.cells:
            raise RuntimeError(
                f"{job.label}: expected {job.cells} cells, loader produced "
                f"{len(cells) * EPOCHS}"
            )


def _validate_jobs(jobs: list[Job]) -> None:
    if len(jobs) != PLANNED_CONTROLLERS:
        raise RuntimeError(
            f"expected {PLANNED_CONTROLLERS} controllers, got {len(jobs)}"
        )
    labels = [job.label for job in jobs]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate job labels")
    vm_cap = sum(job.vm_cap for job in jobs)
    if vm_cap != PLANNED_VM_CAP or vm_cap > GLOBAL_VM_LIMIT:
        raise RuntimeError(
            f"VM caps must sum to {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}, got {vm_cap}"
        )
    if vm_cap != SUBSCRIPTION_VM_LIMIT:
        raise RuntimeError("the configured subscription concurrency is not allocated")
    if sum(job.cells for job in jobs) != PLANNED_CELLS:
        raise RuntimeError("planned total cell count drifted")
    if sum(job.cells for job in jobs if job.group == "regular ML -> ML") != (
        PLANNED_REGULAR_ML_CELLS
    ):
        raise RuntimeError("regular ML -> ML count drifted")
    if sum(job.cells for job in jobs if job.group == "multi-agent ML -> ML") != (
        PLANNED_MULTI_AGENT_ML_CELLS
    ):
        raise RuntimeError("multi-agent ML -> ML count drifted")
    if sum(job.cells for job in jobs if job.group.startswith("p-hacking")) != (
        PLANNED_PHACK_CELLS
    ):
        raise RuntimeError("p-hacking -> p-hacking count drifted")

    expected = {
        "ml-conversation-judged-gpt": (80, 37, "no-hack"),
        "ml-conversation-nohoneypot-gpt": (40, 18, "no-honeypot"),
        "ml-activity-judged-gpt": (80, 37, "no-hack"),
        "phack-nohoneypot-to-phack-gpt": (40, 8, "no-honeypot"),
    }
    observed = {
        job.label: (job.cells, job.vm_cap, job.treatment) for job in jobs
    }
    if observed != expected:
        raise RuntimeError(f"job matrix drifted: {observed!r}")

    for job in jobs:
        command = " ".join(job.command()).lower()
        required = (
            "--harness=subscription", "--epochs=40", "--skip-viewer",
            "--compute=aws",
        )
        if any(value not in job.args for value in required):
            raise RuntimeError(f"{job.label}: required command invariant drifted")
        if "gpt-5.5" not in {
            item.target
            for item in ALL_PREFIXES
            if item.path in job.prefix_paths
        }:
            raise RuntimeError(f"{job.label}: GPT source is missing")
        if "opus" in command:
            raise RuntimeError(f"{job.label}: Opus unexpectedly entered command")


def source_fingerprint() -> str:
    return precedent.source_fingerprint()


def validate_plan() -> tuple[str, str]:
    required = (
        ENVIRONMENTS_ROOT / "exp_continuation_pipeline.py",
        ENVIRONMENTS_ROOT / "tools" / "check_codex_subscription_auth.py",
        ENVIRONMENTS_ROOT / "seeds" / "ml_training_data_misuse"
        / "fraud_detection" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "p_hacking"
        / "checkout_redesign" / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required experiment inputs are missing: {missing}")
    jobs = build_jobs()
    _validate_jobs(jobs)
    validate_frozen_prefixes()
    return source_fingerprint(), frozen_inputs_fingerprint()


def _selection_text(items: Iterable[FrozenPrefix]) -> str:
    return ", ".join(
        f"{item.viewer_label} ({item.condition})" for item in items
    )


def print_plan(source_hash: str, inputs_hash: str) -> None:
    jobs = build_jobs()
    print("2026-08-17 GPT-only next continuations — exact free plan")
    print(f"source fingerprint: {source_hash}")
    print(f"frozen-input fingerprint: {inputs_hash}")
    print()
    for job in jobs:
        print(
            f"[{job.label}] cells={job.cells} source={job.source_mode} "
            f"harness={job.harness} vm_cap={job.vm_cap}"
        )
        print("  " + " ".join(job.command()))
    print()
    print(
        f"TOTAL controllers={PLANNED_CONTROLLERS} paid_cells={PLANNED_CELLS} "
        f"(regular_ml_to_ml={PLANNED_REGULAR_ML_CELLS}, "
        f"multi_agent_ml={PLANNED_MULTI_AGENT_ML_CELLS}, "
        f"phacking_to_phacking={PLANNED_PHACK_CELLS})"
    )
    print(
        f"Concurrent VM caps: {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}; "
        f"configured subscription concurrency={SUBSCRIPTION_VM_LIMIT}."
    )
    print("Regular judged ML sources: " + _selection_text(JUDGED_ML_PREFIXES))
    print("Regular no-honeypot ML source: " + _selection_text((ML_NO_HONEYPOT_PREFIX,)))
    print(
        "Multi-agent sources: exact activity-log transforms of #781 and #783 only; "
        "MLD-128 is deliberately not copied."
    )
    print(
        "P-hacking no-honeypot source: "
        + _selection_text((PHACK_NO_HONEYPOT_PREFIX,))
    )
    print(
        "No GPT reward-hack arm exists because the judged GPT demand pool contains "
        "no successful reward-hack trajectory."
    )
    print(
        "Activity-log processing is intentionally lossy: system messages and private "
        "reasoning are omitted, while all stored observable activity is delivered "
        "inline with a visible caveat."
    )
    print("Exactly 40 continuations are generated from each of the six source arms.")
    print(
        "The 100-worker subscription ceiling is user-selected and experimental; "
        "it is not a published OpenAI concurrency allowance."
    )
    print(
        "The viewer is not rebuilt or reloaded. No automatic retries are attempted; "
        "successful cells are retained and exact failed cells can be retried later."
    )
    print("No AWS, VM, model, judge, credential, or filesystem write was performed.")


RunLogger = precedent.RunLogger
stream_command = precedent.stream_command
_atomic_json = precedent._atomic_json


def _campaign_ids() -> set[str]:
    if not CAMPAIGN_ROOT.is_dir():
        return set()
    return {path.stem for path in CAMPAIGN_ROOT.glob("*.json")}


def _serialized_job(job: Job, status: int | None, run_dir: Path) -> dict:
    row = asdict(job)
    row["prefix_paths"] = [str(path) for path in job.prefix_paths]
    row["args"] = list(job.args)
    row["command"] = job.command()
    row["exit_status"] = status
    row["log"] = str(run_dir / f"{job.label}.log")
    return row


async def run_slate(
    jobs: list[Job], *, run_dir: Path, logger: RunLogger,
    environment: dict[str, str], source_hash: str, inputs_hash: str,
) -> tuple[dict[str, int | None], str | None]:
    logger.log(
        f"Starting concurrent GPT slate: controllers={len(jobs)} "
        f"cells={sum(job.cells for job in jobs)} vm_cap={sum(job.vm_cap for job in jobs)}"
    )
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        if (
            source_fingerprint() != source_hash
            or frozen_inputs_fingerprint() != inputs_hash
        ):
            launch_failure = (
                f"source or frozen inputs changed before {job.label}; "
                "stopped submitting later controllers"
            )
            logger.log("STOPPED LAUNCHING: " + launch_failure)
            break
        logger.log(
            f"Submitting {job.label} — cells={job.cells} source={job.source_mode} "
            f"harness={job.harness} vm_cap={job.vm_cap}"
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
                logger.log(f"Slate controllers still running: {running}/{len(tasks)}")

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
    return statuses, launch_failure


async def run_paid_batch(
    *, run_dir: Path, source_hash: str, inputs_hash: str, logger: RunLogger,
) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"

    setup_commands = (
        (
            "codex-subscription-auth",
            ["uv", "run", "tools/check_codex_subscription_auth.py", "--live"],
        ),
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
    jobs = build_jobs()
    statuses, launch_failure = await run_slate(
        jobs,
        run_dir=run_dir,
        logger=logger,
        environment=environment,
        source_hash=source_hash,
        inputs_hash=inputs_hash,
    )
    logger.log(
        "All submitted controllers exited. Results were retained; viewer rebuild "
        "is intentionally skipped."
    )
    new_campaigns = sorted(_campaign_ids() - existing_campaigns)
    success = (
        launch_failure is None
        and all(statuses.get(job.label) == 0 for job in jobs)
    )
    final_status = {
        "format": "environments-next-gpt-continuations-20260817-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": source_hash,
        "frozen_input_fingerprint": inputs_hash,
        "planned_cells": PLANNED_CELLS,
        "planned_controllers": PLANNED_CONTROLLERS,
        "maximum_vms": GLOBAL_VM_LIMIT,
        "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
        "subscription_concurrency_basis": (
            "user-selected experimental ceiling; not a published OpenAI allowance"
        ),
        "automatic_campaign_retries": False,
        "gpt_only": True,
        "gpt_reward_hack_arm_included": False,
        "multi_agent_no_honeypot_included": False,
        "activity_log_lossy_processing": (
            "system messages and private reasoning omitted; complete observable "
            "activity delivered inline with visible caveat"
        ),
        "launch_failure": launch_failure,
        "viewer_rebuilt": False,
        "new_campaign_ids": new_campaigns,
        "frozen_prefixes": [
            {**asdict(item), "path": str(item.path)} for item in ALL_PREFIXES
        ],
        "jobs": [
            _serialized_job(job, statuses.get(job.label), run_dir) for job in jobs
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
    variable = "MATS_NEXT_GPT_20260817_CAFFEINATED"
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
    source_hash, inputs_hash = validate_plan()
    if args.plan:
        print_plan(source_hash, inputs_hash)
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
        raise SystemExit(
            f"another {RUN_BASENAME} wrapper may be active: {LOCK_DIR}"
        ) from error

    run_slug = f"{RUN_BASENAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    run_dir = RUN_ROOT / run_slug
    run_dir.mkdir()
    logger = RunLogger(run_dir / "orchestrator.log")
    try:
        plan_record = {
            "format": "environments-next-gpt-continuations-20260817-plan-v1",
            "run_slug": run_slug,
            "source_fingerprint": source_hash,
            "frozen_input_fingerprint": inputs_hash,
            "planned_cells": PLANNED_CELLS,
            "planned_controllers": PLANNED_CONTROLLERS,
            "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
            "subscription_concurrency_basis": (
                "user-selected experimental ceiling; not a published OpenAI allowance"
            ),
            "viewer_rebuilt": False,
            "multi_agent_no_honeypot_included": False,
            "gpt_reward_hack_arm_included": False,
            "frozen_prefixes": [
                {**asdict(item), "path": str(item.path)} for item in ALL_PREFIXES
            ],
            "jobs": [
                _serialized_job(job, None, run_dir) for job in build_jobs()
            ],
        }
        _atomic_json(run_dir / "plan.json", plan_record)
        logger.log(f"Run directory: {run_dir}")
        logger.log(f"Source fingerprint: {source_hash}")
        logger.log(f"Frozen-input fingerprint: {inputs_hash}")
        return asyncio.run(run_paid_batch(
            run_dir=run_dir,
            source_hash=source_hash,
            inputs_hash=inputs_hash,
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
