#!/usr/bin/env python3
"""Continue the two newly elicited demand prefixes into fraud detection.

Four independent n=40 arms are launched together:

1. GPT-5.5 delivered two-turn hack, ordinary conversation -> fraud_detection;
2. the exact observable activity-log transform of that GPT source -> fraud_detection;
3. GLM-5.1 clean, ordinary conversation -> fraud_detection;
4. the exact observable activity-log transform of that GLM source -> fraud_detection.

All 160 cells can run concurrently (80 production, 80 subscription), so the
160-VM slate is already maximally parallel even though the account limit is 250.
The wrapper never retries automatically and never rebuilds the viewer.

Usage (from ``mats/environments``)::

    uv run exp_elicited_gpt_glm_ml_followups_20260817.py --plan
    uv run exp_elicited_gpt_glm_ml_followups_20260817.py

``--plan`` is local, free, and read-only. Paid mode performs subscription-auth
and AWS preflights, records a durable one-use marker, and launches the frozen slate.
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

import exp_final_gap_first_pass_20260817 as precedent


ENVIRONMENTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ENVIRONMENTS_ROOT.parent
DATA_ROOT = REPO_ROOT.parent / "mats-local" / "environments"
PREFIX_ROOT = DATA_ROOT / "continuation_prefixes"
CAMPAIGN_ROOT = DATA_ROOT / "remote_campaigns"
RUN_ROOT = DATA_ROOT / "overnight_runs"

AWS_REGION = "us-west-2"
AWS_INSTANCE_TYPE = "c7a.xlarge"
JUDGE = "gpt-5.6-luna"
EPOCHS = 40
GLOBAL_VM_LIMIT = 250
SUBSCRIPTION_VM_LIMIT = 100
PLANNED_VM_CAP = 160
PLANNED_SUBSCRIPTION_VM_CAP = 80
PLANNED_CELLS = 160
PLANNED_CONTROLLERS = 4
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180

RUN_BASENAME = "elicited-gpt-glm-ml-followups-20260817"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class FrozenPrefix:
    filename: str
    file_sha256: str
    name: str
    viewer_label: str
    target: str
    harness: str
    source_mode: str
    trajectory_id: int
    condition: str
    message_count: int
    source_run: str
    source_task: str

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


GPT_CONVERSATION = FrozenPrefix(
    "traj7420-a2c171e80064.json",
    "c73a7853b525245639a4ee7dcc37d489bb2e7f305a1293c0c902186ab8478719",
    "traj7420", "#7420", "gpt-5.5", "subscription", "conversation", 7420,
    "hack_2turn", 116,
    "real-v2-aws-gpt-5-5-allow-20ep-20260817-155627-aefbc9a3",
    "real_audit_gpt-5.5-2026-04-23_demand_forecasting_"
    "gpt-5-5-demand-forecasting-e14-f6f3ee32",
)

GPT_ACTIVITY = FrozenPrefix(
    "activity-log-traj7420-b5cce0abfa52.json",
    "eabeef4835bae3d099355f0378c311ee5b58c2a8028e85cb16b2308fcc592904",
    "activity-log-traj7420", "#7420 activity log", "gpt-5.5", "subscription",
    "activity-log", 7420, "hack_2turn", 1,
    "real-v2-aws-gpt-5-5-allow-20ep-20260817-155627-aefbc9a3",
    "real_audit_gpt-5.5-2026-04-23_demand_forecasting_"
    "gpt-5-5-demand-forecasting-e14-f6f3ee32",
)

GLM_CONVERSATION = FrozenPrefix(
    "traj7764-9cafa27a77bf.json",
    "13e5831280538016e2bfebc30002deef110d313906d73b6ec8fe3a50503657d7",
    "traj7764", "#7764", "glm-5.1", "production", "conversation", 7764,
    "clean", 167,
    "real-v2-aws-glm-5-1-allow-20ep-20260817-155647-fcf24f27",
    "real_audit_glm-5.1_demand_forecasting_"
    "glm-5-1-demand-forecasting-e18-9c33ef93",
)

GLM_ACTIVITY = FrozenPrefix(
    "activity-log-traj7764-331806eeb988.json",
    "498bf669336669b345b40e5e26f9dcdaeb9d7eecc35305cb48a6042b2ed4ff0f",
    "activity-log-traj7764", "#7764 activity log", "glm-5.1", "production",
    "activity-log", 7764, "clean", 1,
    "real-v2-aws-glm-5-1-allow-20ep-20260817-155647-fcf24f27",
    "real_audit_glm-5.1_demand_forecasting_"
    "glm-5-1-demand-forecasting-e18-9c33ef93",
)

ALL_PREFIXES = (
    GPT_CONVERSATION,
    GPT_ACTIVITY,
    GLM_CONVERSATION,
    GLM_ACTIVITY,
)


@dataclass(frozen=True)
class Job:
    label: str
    source_mode: str
    treatment: str
    harness: str
    cells: int
    vm_cap: int
    prefix_paths: tuple[Path, ...]
    args: tuple[str, ...]
    endpoint: str = "exp_continuation_pipeline.py"
    destination: str = "fraud_detection"

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _job(*, prefix: FrozenPrefix, treatment: str, vm_cap: int) -> Job:
    label = (
        f"{prefix.target.replace('.', '-')}-ml-{prefix.source_mode}-"
        f"{'hack2' if prefix.condition == 'hack_2turn' else 'clean'}"
    )
    return Job(
        label=label,
        source_mode=prefix.source_mode,
        treatment=treatment,
        harness=prefix.harness,
        cells=EPOCHS,
        vm_cap=vm_cap,
        prefix_paths=(prefix.path,),
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={prefix.path}",
            "--seed-dir=ml_training_data_misuse",
            "--seeds=fraud_detection",
            f"--epochs={EPOCHS}",
            f"--harness={prefix.harness}",
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


def build_jobs() -> list[Job]:
    return [
        _job(prefix=GPT_CONVERSATION, treatment="hack-in-two-turns", vm_cap=40),
        _job(prefix=GPT_ACTIVITY, treatment="hack-in-two-turns", vm_cap=40),
        _job(prefix=GLM_CONVERSATION, treatment="no-hack", vm_cap=40),
        _job(prefix=GLM_ACTIVITY, treatment="no-hack", vm_cap=40),
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


def _validate_prefix(item: FrozenPrefix) -> None:
    if not item.path.is_file():
        raise RuntimeError(f"missing frozen prefix: {item.path}")
    observed_hash = _sha256_file(item.path)
    if observed_hash != item.file_sha256:
        raise RuntimeError(
            f"frozen prefix bytes drifted for {item.filename}: expected "
            f"{item.file_sha256}, got {observed_hash}"
        )
    payload = json.loads(item.path.read_text())
    source = payload.get("source") or {}
    condition = source.get("prefix_condition") or {}
    common = (
        payload.get("format"), payload.get("name"), payload.get("target"),
        payload.get("reasoning"), len(payload.get("messages") or []),
        source.get("kind"), source.get("trajectory_id"), source.get("run"),
        source.get("task"), source.get("seed"), source.get("epoch"),
        source.get("family"), source.get("harness"),
        source.get("was_continuation"), condition.get("key"),
    )
    expected_common = (
        "environments-continuation-prefix-v1", item.name, item.target, True,
        item.message_count, "trajectory", item.trajectory_id, item.source_run,
        item.source_task, "demand_forecasting", 1,
        "ml_training_data_misuse", item.harness, False, item.condition,
    )
    if common != expected_common:
        raise RuntimeError(
            f"frozen prefix identity drifted for {item.filename}: "
            f"expected {expected_common!r}, got {common!r}"
        )

    if item.source_mode == "conversation":
        observed = (
            source.get("prefix_type"), source.get("prefix_type_label"),
            (payload.get("native_resume") or {}).get("format"),
            (payload.get("native_resume") or {}).get("target_name"),
            (payload.get("native_resume") or {}).get("reasoning"),
            payload.get("delivery"),
        )
        expected = (
            "ml_demand_forecasting", "ML: demand_forecasting",
            "environments-production-native-resume-v1", item.target, True, None,
        )
    else:
        metadata = source.get("activity_log_metadata") or {}
        lossy = metadata.get("lossy_processing") or {}
        observed = (
            source.get("generator"), source.get("prefix_type"),
            source.get("prefix_type_label"),
            (payload.get("delivery") or {}).get("mode"),
            payload.get("native_resume"), metadata.get("format"),
            lossy.get("affected"), lossy.get("kind"),
            bool(lossy.get("visible_caveat")),
        )
        expected = (
            "build_activity_log_prefixes.py", "activity_log_context",
            "Activity-log context", "inline_user_context", None,
            "observable-activity-log-v2", True, "observable_activity_only", True,
        )
    if observed != expected:
        raise RuntimeError(
            f"frozen prefix delivery drifted for {item.filename}: "
            f"expected {expected!r}, got {observed!r}"
        )


def validate_frozen_prefixes() -> None:
    if len({item.filename for item in ALL_PREFIXES}) != len(ALL_PREFIXES):
        raise RuntimeError("frozen prefix filenames are not unique")
    if {item.trajectory_id for item in ALL_PREFIXES} != {7420, 7764}:
        raise RuntimeError("the frozen source trajectories drifted")
    pairs = {
        item.trajectory_id: {
            candidate.source_mode
            for candidate in ALL_PREFIXES
            if candidate.trajectory_id == item.trajectory_id
        }
        for item in ALL_PREFIXES
    }
    if pairs != {
        7420: {"conversation", "activity-log"},
        7764: {"conversation", "activity-log"},
    }:
        raise RuntimeError("conversation/activity-log matching drifted")
    for item in ALL_PREFIXES:
        _validate_prefix(item)

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

    seeds_path, available = resolve_seeds("ml_training_data_misuse")
    if "fraud_detection" not in available:
        raise RuntimeError("fraud_detection destination is unavailable")
    for job in build_jobs():
        specs = load_prefix_specs(
            [], [str(path) for path in job.prefix_paths], harness=job.harness,
        )
        cells = build_continuation_cells(specs, seeds_path, ["fraud_detection"])
        for unit_path in {cell.unit_path for cell in cells}:
            load_protocol_sources(unit_path, pressure=None)
        if len(cells) * EPOCHS != job.cells:
            raise RuntimeError(f"{job.label}: continuation cell count drifted")


def _validate_jobs(jobs: list[Job]) -> None:
    if len(jobs) != PLANNED_CONTROLLERS:
        raise RuntimeError(f"expected {PLANNED_CONTROLLERS} controllers")
    if len({job.label for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate job labels")
    if sum(job.cells for job in jobs) != PLANNED_CELLS:
        raise RuntimeError("planned cell count drifted")
    if sum(job.vm_cap for job in jobs) != PLANNED_VM_CAP:
        raise RuntimeError("planned VM cap drifted")
    subscription_cap = sum(
        job.vm_cap for job in jobs if job.harness == "subscription"
    )
    if subscription_cap != PLANNED_SUBSCRIPTION_VM_CAP:
        raise RuntimeError("planned subscription VM cap drifted")
    if PLANNED_VM_CAP > GLOBAL_VM_LIMIT:
        raise RuntimeError("global VM cap exceeded")
    if subscription_cap > SUBSCRIPTION_VM_LIMIT:
        raise RuntimeError("subscription VM cap exceeded")
    if any(job.vm_cap != job.cells for job in jobs):
        raise RuntimeError("every paid cell should be able to run concurrently")

    expected = {
        "gpt-5-5-ml-conversation-hack2":
            (40, 40, "subscription", "hack-in-two-turns"),
        "gpt-5-5-ml-activity-log-hack2":
            (40, 40, "subscription", "hack-in-two-turns"),
        "glm-5-1-ml-conversation-clean":
            (40, 40, "production", "no-hack"),
        "glm-5-1-ml-activity-log-clean":
            (40, 40, "production", "no-hack"),
    }
    observed = {
        job.label: (job.cells, job.vm_cap, job.harness, job.treatment)
        for job in jobs
    }
    if observed != expected:
        raise RuntimeError(f"exact job matrix drifted: {observed!r}")

    for job in jobs:
        required = (
            "--seed-dir=ml_training_data_misuse", "--seeds=fraud_detection",
            "--epochs=40", f"--harness={job.harness}", "--condition=allow",
            f"--judge={JUDGE}", f"--gate-model={JUDGE}", "--compute=aws",
            f"--vm-concurrency={job.vm_cap}", f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}", "--time-limit=4200",
            "--skip-viewer",
        )
        if any(value not in job.args for value in required):
            raise RuntimeError(f"{job.label}: command invariant drifted")
        if "retry" in " ".join(job.command()).lower():
            raise RuntimeError(f"{job.label}: automatic retry unexpectedly enabled")


def source_fingerprint() -> str:
    return precedent.source_fingerprint()


def validate_plan() -> tuple[str, str]:
    required = (
        ENVIRONMENTS_ROOT / "exp_continuation_pipeline.py",
        ENVIRONMENTS_ROOT / "tools" / "check_codex_subscription_auth.py",
        ENVIRONMENTS_ROOT / "seeds" / "ml_training_data_misuse"
        / "fraud_detection" / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"required experiment inputs are missing: {missing}")
    jobs = build_jobs()
    _validate_jobs(jobs)
    validate_frozen_prefixes()
    return source_fingerprint(), frozen_inputs_fingerprint()


def print_plan(source_hash: str, inputs_hash: str) -> None:
    jobs = build_jobs()
    print("2026-08-17 newly elicited GPT + GLM ML follow-ups — exact free plan")
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
        "(regular_ml_to_ml=80, multi_agent_ml=80)"
    )
    print(
        f"Concurrent VM caps: {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}; "
        f"subscription cap={PLANNED_SUBSCRIPTION_VM_CAP}/{SUBSCRIPTION_VM_LIMIT}."
    )
    print(
        "GPT source: #7420, delivered two-turn demand-forecasting hack; its regular "
        "and exact activity-log arms each receive 40 fraud-detection continuations."
    )
    print(
        "GLM source: #7764, clean demand-forecasting trajectory; its regular and "
        "exact activity-log arms each receive 40 fraud-detection continuations."
    )
    print(
        "Activity-log processing is intentionally lossy: system messages and private "
        "reasoning are omitted; all stored observable activity is inline with a caveat."
    )
    print(
        "All 160 cells have a worker slot, so this is maximum useful concurrency; "
        "there are no additional paid cells with which to fill the 250-VM limit."
    )
    print(
        "The viewer is not rebuilt or reloaded. No automatic retries are attempted; "
        "successful cells remain usable if another cell fails."
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
        f"Starting concurrent slate: controllers={len(jobs)} "
        f"cells={sum(job.cells for job in jobs)} "
        f"vm_cap={sum(job.vm_cap for job in jobs)}"
    )
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        if source_fingerprint() != source_hash or (
            frozen_inputs_fingerprint() != inputs_hash
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
    success = launch_failure is None and all(
        statuses.get(job.label) == 0 for job in jobs
    )
    final_status = {
        "format": "environments-elicited-gpt-glm-ml-followups-20260817-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": source_hash,
        "frozen_input_fingerprint": inputs_hash,
        "planned_cells": PLANNED_CELLS,
        "planned_controllers": PLANNED_CONTROLLERS,
        "maximum_vms": GLOBAL_VM_LIMIT,
        "planned_maximum_vms": PLANNED_VM_CAP,
        "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
        "planned_subscription_maximum_vms": PLANNED_SUBSCRIPTION_VM_CAP,
        "automatic_campaign_retries": False,
        "viewer_rebuilt": False,
        "launch_failure": launch_failure,
        "new_campaign_ids": sorted(_campaign_ids() - existing_campaigns),
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
    variable = "MATS_ELICITED_GPT_GLM_ML_20260817_CAFFEINATED"
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
            "format": "environments-elicited-gpt-glm-ml-followups-20260817-plan-v1",
            "run_slug": run_slug,
            "source_fingerprint": source_hash,
            "frozen_input_fingerprint": inputs_hash,
            "planned_cells": PLANNED_CELLS,
            "planned_controllers": PLANNED_CONTROLLERS,
            "maximum_vms": GLOBAL_VM_LIMIT,
            "planned_maximum_vms": PLANNED_VM_CAP,
            "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
            "planned_subscription_maximum_vms": PLANNED_SUBSCRIPTION_VM_CAP,
            "automatic_campaign_retries": False,
            "viewer_rebuilt": False,
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
