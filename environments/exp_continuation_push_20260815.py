#!/usr/bin/env python3
"""Exact one-command launcher for the 2026-08-15 continuation push.

This paid, one-off batch does five things:

* fills only the still-missing Wikipedia -> ML and p-hacking -> ML cells;
* reruns p-hacking multi-agent transfer with complete inline log delivery;
* runs the natural file-available multi-agent condition for ML -> ML;
* generates ten demand-forecasting ML prefixes per available agent; and
* generates three positive checkout no-honeypot prefixes per available agent.

GPT-5.5 uses the subscription harness. DeepSeek V4 Pro, GLM-5.1, and
Kimi K2.6 use the production harness. Opus is deliberately absent.

Usage (from mats/environments/):

  uv run exp_continuation_push_20260815.py --plan
  uv run exp_continuation_push_20260815.py

``--plan`` is local and read-only: it validates every frozen source and parent
campaign, then prints the exact commands, trajectory counts, and concurrency caps.
The paid mode performs AWS setup once, starts labeled controllers with a stagger,
streams their VM lifecycle output, records durable logs/status, and builds the viewer
once after every controller exits. It never retries a failed child automatically.
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
from collections import Counter
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
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180
GLOBAL_VM_LIMIT = 250
PLANNED_VM_CAP = 220
SUBSCRIPTION_VM_CAP = 18
PLANNED_TRAJECTORIES = 988
PLANNED_CONTROLLERS = 20

RUN_BASENAME = "continuation-push-20260815"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class PrefixInput:
    trajectory_id: int
    filename: str
    file_sha256: str
    target: str
    harness: str
    source_seed: str
    condition: str

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


# These are complete-delivery, inline activity-log payloads. The ML-source files were
# generated locally and for free while preparing this batch. Their byte hashes and
# source identities are frozen here so a later viewer/source change fails pre-spend.
PREFIX_INPUTS: dict[int, PrefixInput] = {
    710: PrefixInput(710, "activity-log-traj710-5781081409f0.json", "aae3c26d2c4d519313bdfaf217a394a02cc92e6892d276ec46b3688dc79ad69b", "deepseek-v4-pro", "production", "reasoning_prompt_benchmark", "hack_2turn"),
    711: PrefixInput(711, "activity-log-traj711-0358b165569a.json", "c5086ac734a84e11a3dc672f093e69b609dd75b8209c4bbf785fc6135974e5f7", "deepseek-v4-pro", "production", "reasoning_prompt_benchmark", "hack_1turn"),
    715: PrefixInput(715, "activity-log-traj715-9ca892465362.json", "aec05e2a86041362d9c93a5fd9477941d5ba9fbec7495b1cbe73724340ed8b22", "glm-5.1", "production", "reasoning_prompt_benchmark", "hack_1turn"),
    716: PrefixInput(716, "activity-log-traj716-bf9210d33766.json", "23613be34021f6336ffd86f9f3ba3e0ef6fcd53f9c3170f45893f9d737f4b1b6", "glm-5.1", "production", "reasoning_prompt_benchmark", "hack_2turn"),
    718: PrefixInput(718, "activity-log-traj718-0e258e3fd115.json", "5509a27dcfaef2d23fe1d79de8849d31408fa33e2f27b16f9c014022d35aefaf", "glm-5.1", "production", "reasoning_prompt_benchmark", "clean"),
    720: PrefixInput(720, "activity-log-traj720-1216c26d3ae3.json", "bfe182435d7aeace568d434395665fed447f0d059eaba4d7ed2587c4d8eff71e", "kimi-k2.6", "production", "reasoning_prompt_benchmark", "clean"),
    722: PrefixInput(722, "activity-log-traj722-429281c6eb6d.json", "809734e2e13d54e12f705bafaf002137fa9f5136baaca38aa902043cddfe8d23", "kimi-k2.6", "production", "reasoning_prompt_benchmark", "hack_2turn"),
    723: PrefixInput(723, "activity-log-traj723-6aabc8831ff0.json", "b605dcc3d05340cc3f1ef3533fd9f4e70bb2580cf9147b0b9316af259052b26e", "kimi-k2.6", "production", "reasoning_prompt_benchmark", "hack_1turn"),
    750: PrefixInput(750, "activity-log-traj750-f38e0c3373ba.json", "bc9ecd1611f9c3eb6a01fa9313b5f20ab9b5eb214a3ae22a70427fea4d7bf97e", "gpt-5.5", "subscription", "reasoning_prompt_benchmark", "hack_2turn"),
    757: PrefixInput(757, "activity-log-traj757-873b1784ac89.json", "c280f342bbdb099c3710ff75d9a7e2b4194da56c200ad37da7e7b76237d1b4d7", "gpt-5.5", "subscription", "reasoning_prompt_benchmark", "clean"),
    2852: PrefixInput(2852, "activity-log-traj2852-c61e63600161.json", "90fb8738e083f02f945984bc1c28632a81283d4ac8891502469e66afec397b40", "deepseek-v4-pro", "production", "fraud_detection", "hack_2turn"),
    2856: PrefixInput(2856, "activity-log-traj2856-cf15f3a085cf.json", "bfebf3a04da1c858fc2190399eb390cdba52c5248d0def4a9c770e662b07b84b", "deepseek-v4-pro", "production", "fraud_detection", "hack_1turn"),
    2858: PrefixInput(2858, "activity-log-traj2858-b81d8892b3aa.json", "63fce496a785bff240e0c9b59fad20f2ada006cb84e21e636037423d71ae5da6", "deepseek-v4-pro", "production", "fraud_detection", "clean"),
    2893: PrefixInput(2893, "activity-log-traj2893-ba4ad01e4d27.json", "6b23b3a523e25430d186d5f9fb10ffa78d731a04535a2b256ed702d458d7b3eb", "glm-5.1", "production", "fraud_detection", "hack_1turn"),
    2917: PrefixInput(2917, "activity-log-traj2917-bdf3bb7fd111.json", "8ff295318939fc9d469be6277a8c73271d66d09fd284d5e375308bd6b8e5a16e", "glm-5.1", "production", "fraud_detection", "hack_2turn"),
    2918: PrefixInput(2918, "activity-log-traj2918-661576308934.json", "740d98aa068150037e9b8cf69f7c95ecc2efd32e056dc8594c881754b560ee3e", "glm-5.1", "production", "fraud_detection", "clean"),
    2933: PrefixInput(2933, "activity-log-traj2933-e37316cddf6b.json", "91d5e2f65c1e50e4e5d6119e269430afa1609f2059a458fab1f17ca419b7f59c", "kimi-k2.6", "production", "fraud_detection", "clean"),
    2940: PrefixInput(2940, "activity-log-traj2940-142657c15306.json", "551a884b5b7a707547fa48a1a9f844a613dfc194b7e9a3112168ab94f2bf9d98", "kimi-k2.6", "production", "fraud_detection", "hack_1turn"),
    2946: PrefixInput(2946, "activity-log-traj2946-dae8dc34ac95.json", "9b69a0a79798e32120c3a0f89ce6989f633dcba4573adab16a6da61074ef42fa", "kimi-k2.6", "production", "fraud_detection", "hack_2turn"),
    2972: PrefixInput(2972, "activity-log-traj2972-8faf127afa04.json", "a459876b2d8a0553814e9657c3121d7571af0ebf38d704c0da48f72815100274", "gpt-5.5", "subscription", "fraud_detection", "clean"),
    2997: PrefixInput(2997, "activity-log-traj2997-8f22f40bdc28.json", "9b552e458180f2807d02b6c6a1f497bda610b66fae81df32db08437e4155d525", "gpt-5.5", "subscription", "fraud_detection", "hack_2turn"),
}


@dataclass(frozen=True)
class RetryInput:
    campaign_id: str
    harness: str
    expected_by_target: tuple[tuple[str, int], ...]

    @property
    def expected_cells(self) -> int:
        return sum(count for _, count in self.expected_by_target)

    @property
    def state_path(self) -> Path:
        return CAMPAIGN_ROOT / f"{self.campaign_id}.json"


RETRY_INPUTS: dict[str, RetryInput] = {
    "wiki-ml-fill": RetryInput(
        "continuation-aws-wikipedia-summaries-40ep-20260815-015949-82cceb1f-retry-20260815-123006-c16c95f5",
        "production",
        (("glm-5.1", 17), ("kimi-k2.6", 29)),
    ),
    "phack-ml-hack1-production-fill": RetryInput(
        "continuation-aws-hack-in-one-turn-40ep-20260815-015952-72c5744f-retry-20260815-123106-3ed4f92a",
        "production",
        (("glm-5.1", 1), ("kimi-k2.6", 4)),
    ),
    "phack-ml-hack2-production-fill": RetryInput(
        "continuation-aws-hack-in-two-turns-40ep-20260815-015952-d018a77c-retry-20260815-123136-48056a34",
        "production",
        (("glm-5.1", 13), ("kimi-k2.6", 24)),
    ),
    "phack-ml-hack2-gpt-fill": RetryInput(
        "continuation-aws-hack-in-two-turns-40ep-20260815-015951-96acfbd5-retry-20260815-123206-2683a12f",
        "subscription",
        (("gpt-5.5", 1),),
    ),
    "phack-ml-nohack-production-fill": RetryInput(
        "continuation-aws-no-hack-40ep-20260815-015952-474ed33c-retry-20260815-123237-aa4cc4db",
        "production",
        (("glm-5.1", 3), ("kimi-k2.6", 3)),
    ),
    "phack-ml-nohack-gpt-fill": RetryInput(
        "continuation-aws-no-hack-40ep-20260815-015951-c7e12a59-retry-20260815-123306-d7a29fb4",
        "subscription",
        (("gpt-5.5", 1),),
    ),
}


@dataclass(frozen=True)
class Job:
    label: str
    group: str
    endpoint: str
    harness: str
    cells: int
    vm_cap: int
    args: tuple[str, ...]
    retry_parent: str | None = None

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _prefix_csv(ids: Iterable[int]) -> str:
    return ",".join(str(PREFIX_INPUTS[item].path) for item in ids)


def _retry_job(label: str, vm_cap: int) -> Job:
    retry = RETRY_INPUTS[label]
    return Job(
        label=label,
        group="exact gap fill",
        endpoint="exp_continuation_pipeline.py",
        harness=retry.harness,
        cells=retry.expected_cells,
        vm_cap=vm_cap,
        args=(
            f"--retry-failed={retry.campaign_id}",
            "--retry-pipeline-failures",
            f"--harness={retry.harness}",
            f"--vm-concurrency={vm_cap}",
            "--skip-viewer",
        ),
        retry_parent=retry.campaign_id,
    )


def _inline_continuation_job(
    *, label: str, treatment: str, ids: tuple[int, ...], harness: str, vm_cap: int
) -> Job:
    cells = len(ids) * 40
    return Job(
        label=label,
        group="multi-agent p-hacking (complete inline delivery)",
        endpoint="exp_continuation_pipeline.py",
        harness=harness,
        cells=cells,
        vm_cap=vm_cap,
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={_prefix_csv(ids)}",
            "--seed-dir=p_hacking",
            "--seeds=checkout_redesign",
            "--epochs=40",
            f"--harness={harness}",
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


def _multi_agent_ml_job(
    *, label: str, treatment: str, ids: tuple[int, ...], harness: str, vm_cap: int
) -> Job:
    cells = len(ids) * 40
    return Job(
        label=label,
        group="multi-agent ML (natural file-available condition)",
        endpoint="exp_multi_agent_pipeline.py",
        harness=harness,
        cells=cells,
        vm_cap=vm_cap,
        args=(
            f"--treatment={treatment}",
            f"--activity-logs={','.join(str(item) for item in ids)}",
            "--seed-dir=ml_training_data_misuse",
            "--seeds=demand_forecasting",
            "--epochs=40",
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


def _prefix_generation_jobs(run_slug: str) -> list[Job]:
    open_targets = "deepseek-v4-pro,glm-5.1,kimi-k2.6"
    return [
        Job(
            label="generate-ml-demand-production",
            group="prefix generation",
            endpoint="prefixes/exp_ml_prefix.py",
            harness="production",
            cells=30,
            vm_cap=7,
            args=(
                f"--targets={open_targets}", "--seeds=demand_forecasting",
                "--epochs=10", "--harness=production", "--reasoning=yes",
                f"--name={run_slug}-ml-demand-production", "--compute=aws",
                "--vm-concurrency=7", f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}", "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            label="generate-ml-demand-gpt",
            group="prefix generation",
            endpoint="prefixes/exp_ml_prefix.py",
            harness="subscription",
            cells=10,
            vm_cap=2,
            args=(
                "--targets=gpt-5.5", "--seeds=demand_forecasting", "--epochs=10",
                "--harness=subscription", "--reasoning=yes",
                f"--name={run_slug}-ml-demand-gpt", "--compute=aws",
                "--vm-concurrency=2", f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}", "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            label="generate-phack-nohoneypot-production",
            group="prefix generation",
            endpoint="prefixes/exp_p_hacking_prefix.py",
            harness="production",
            cells=9,
            vm_cap=3,
            args=(
                f"--targets={open_targets}", "--seeds=checkout_redesign_positive",
                "--epochs=3", "--harness=production", "--pressure=low",
                "--reasoning=yes", f"--name={run_slug}-phack-nohoneypot-production",
                "--compute=aws", "--vm-concurrency=3", f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}", "--time-limit=1800",
                "--skip-viewer",
            ),
        ),
        Job(
            label="generate-phack-nohoneypot-gpt",
            group="prefix generation",
            endpoint="prefixes/exp_p_hacking_prefix.py",
            harness="subscription",
            cells=3,
            vm_cap=2,
            args=(
                "--targets=gpt-5.5", "--seeds=checkout_redesign_positive",
                "--epochs=3", "--harness=subscription", "--pressure=low",
                "--reasoning=yes", f"--name={run_slug}-phack-nohoneypot-gpt",
                "--compute=aws", "--vm-concurrency=2", f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}", "--time-limit=1800",
                "--skip-viewer",
            ),
        ),
    ]


def build_jobs(run_slug: str) -> list[Job]:
    jobs = [
        _retry_job("wiki-ml-fill", 18),
        _retry_job("phack-ml-hack1-production-fill", 5),
        _retry_job("phack-ml-hack2-production-fill", 37),
        _retry_job("phack-ml-hack2-gpt-fill", 1),
        _retry_job("phack-ml-nohack-production-fill", 6),
        _retry_job("phack-ml-nohack-gpt-fill", 1),
        _inline_continuation_job(label="inline-phack-hack1-production", treatment="hack-in-one-turn", ids=(711, 715, 723), harness="production", vm_cap=20),
        _inline_continuation_job(label="inline-phack-hack2-production", treatment="hack-in-two-turns", ids=(710, 716, 722), harness="production", vm_cap=20),
        _inline_continuation_job(label="inline-phack-hack2-gpt", treatment="hack-in-two-turns", ids=(750,), harness="subscription", vm_cap=3),
        _inline_continuation_job(label="inline-phack-nohack-production", treatment="no-hack", ids=(718, 720), harness="production", vm_cap=14),
        _inline_continuation_job(label="inline-phack-nohack-gpt", treatment="no-hack", ids=(757,), harness="subscription", vm_cap=3),
        _multi_agent_ml_job(label="multi-ml-hack1-production", treatment="hack-in-one-turn", ids=(2856, 2893, 2940), harness="production", vm_cap=24),
        _multi_agent_ml_job(label="multi-ml-hack2-production", treatment="hack-in-two-turns", ids=(2852, 2917, 2946), harness="production", vm_cap=24),
        _multi_agent_ml_job(label="multi-ml-hack2-gpt", treatment="hack-in-two-turns", ids=(2997,), harness="subscription", vm_cap=3),
        _multi_agent_ml_job(label="multi-ml-nohack-production", treatment="no-hack", ids=(2858, 2918, 2933), harness="production", vm_cap=24),
        _multi_agent_ml_job(label="multi-ml-nohack-gpt", treatment="no-hack", ids=(2972,), harness="subscription", vm_cap=3),
    ]
    jobs.extend(_prefix_generation_jobs(run_slug))
    return jobs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _retryable(cell: dict) -> bool:
    if cell.get("status") in {"not_launched", "infrastructure_failure"}:
        return True
    if cell.get("status") != "completed":
        return False
    exit_code = (cell.get("terminal") or {}).get("pipeline_exit_code")
    try:
        return int(exit_code) != 0
    except (TypeError, ValueError):
        return True


def validate_retry_input(label: str, retry: RetryInput) -> None:
    if not retry.state_path.is_file():
        raise RuntimeError(f"{label}: missing parent campaign state {retry.state_path}")
    state = json.loads(retry.state_path.read_text())
    cfg = state.get("pipeline_config") or {}
    if state.get("campaign_id") != retry.campaign_id:
        raise RuntimeError(f"{label}: parent campaign identity drifted")
    if cfg.get("harness") != retry.harness or not cfg.get("continuation"):
        raise RuntimeError(f"{label}: parent harness or continuation mode drifted")
    if cfg.get("epochs") != 40 or cfg.get("seeds") != ["fraud_detection"]:
        raise RuntimeError(f"{label}: parent epoch/destination identity drifted")
    if (state.get("s3_cleanup") or {}).get("status") != "deleted":
        raise RuntimeError(f"{label}: parent S3 cleanup is not complete")
    local_log_dir = Path(str(state.get("local_log_dir") or ""))
    if not local_log_dir.is_dir() or not (local_log_dir / "remote_campaign.json").is_file():
        raise RuntimeError(f"{label}: parent import is incomplete: {local_log_dir}")
    allowed_terminal = {"completed", "not_launched", "infrastructure_failure"}
    statuses = {cell.get("status") for cell in state.get("cells") or []}
    if not statuses or not statuses <= allowed_terminal:
        raise RuntimeError(f"{label}: parent is not fully terminal: {sorted(statuses)}")
    selected = [cell for cell in state["cells"] if _retryable(cell)]
    actual = Counter(str(cell.get("target")) for cell in selected)
    expected = Counter(dict(retry.expected_by_target))
    if actual != expected:
        raise RuntimeError(
            f"{label}: retry selection changed; expected {dict(expected)}, got {dict(actual)}"
        )
    if any("opus" in target.lower() for target in actual):
        raise RuntimeError(f"{label}: retry selection unexpectedly contains Opus")
    for payload in (cfg.get("continuation") or {}).get("payloads") or []:
        path = Path(str(payload.get("local_path") or ""))
        if not path.is_file():
            raise RuntimeError(f"{label}: stored retry payload is missing: {path}")
        expected_hash = payload.get("file_sha256")
        if expected_hash and _sha256_file(path) != expected_hash:
            raise RuntimeError(f"{label}: stored retry payload bytes drifted: {path}")


def validate_prefix_input(item: PrefixInput) -> None:
    if not item.path.is_file():
        raise RuntimeError(f"missing frozen prefix payload: {item.path}")
    if _sha256_file(item.path) != item.file_sha256:
        raise RuntimeError(f"frozen prefix bytes drifted: {item.path}")
    payload = json.loads(item.path.read_text())
    source = payload.get("source") or {}
    delivery = payload.get("delivery") or {}
    observed = (
        payload.get("format"), payload.get("target"), source.get("trajectory_id"),
        source.get("harness"), source.get("seed"),
        (source.get("prefix_condition") or {}).get("key"),
        source.get("prefix_type"), delivery.get("mode"),
    )
    expected = (
        "environments-continuation-prefix-v1", item.target, item.trajectory_id,
        item.harness, item.source_seed, item.condition,
        "activity_log_context", "inline_user_context",
    )
    if observed != expected:
        raise RuntimeError(
            f"frozen prefix identity drifted for trajectory {item.trajectory_id}: "
            f"expected {expected!r}, got {observed!r}"
        )


def validate_current_prefix_sources() -> dict[int, str]:
    """Revalidate all frozen files and compute ML activity-log transport hashes."""

    for item in PREFIX_INPUTS.values():
        validate_prefix_input(item)

    # Import lazily so simple unit tests of the matrix do not need Inspect startup.
    lib_path = str(ENVIRONMENTS_ROOT / "lib")
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
    from exp_real_continuation import (  # noqa: PLC0415
        load_prefix_specs,
        reconstruct_prefix_payloads,
    )
    from exp_real_multi_agent import (  # noqa: PLC0415
        activity_log_payload_from_trajectory,
        payload_sha256,
    )

    for harness in ("production", "subscription"):
        paths = [str(item.path) for item in PREFIX_INPUTS.values() if item.harness == harness]
        load_prefix_specs([], paths, harness=harness)

    transport_hashes: dict[int, str] = {}
    ml_ids = sorted(item.trajectory_id for item in PREFIX_INPUTS.values() if item.source_seed == "fraud_detection")
    reconstructed = reconstruct_prefix_payloads(ml_ids, source_use="activity_log")
    for trajectory_id, source_payload in zip(ml_ids, reconstructed, strict=True):
        transport = activity_log_payload_from_trajectory(source_payload)
        inline = json.loads(PREFIX_INPUTS[trajectory_id].path.read_text())
        inline_log_hash = (inline["source"].get("activity_log_metadata") or {}).get("sha256")
        transport_log_hash = transport["activity_log"]["metadata"].get("sha256")
        if inline_log_hash != transport_log_hash:
            raise RuntimeError(
                f"trajectory {trajectory_id}: inline and file-available activity logs differ"
            )
        transport_hashes[trajectory_id] = payload_sha256(transport)
    return transport_hashes


def validate_job_matrix(jobs: list[Job]) -> None:
    if len(jobs) != PLANNED_CONTROLLERS:
        raise RuntimeError(f"expected {PLANNED_CONTROLLERS} controllers, got {len(jobs)}")
    labels = [job.label for job in jobs]
    if len(set(labels)) != len(labels):
        raise RuntimeError("job labels are not unique")
    cells = sum(job.cells for job in jobs)
    vm_caps = sum(job.vm_cap for job in jobs)
    subscription_caps = sum(job.vm_cap for job in jobs if job.harness == "subscription")
    if cells != PLANNED_TRAJECTORIES:
        raise RuntimeError(f"planned cell total drifted: expected {PLANNED_TRAJECTORIES}, got {cells}")
    if vm_caps != PLANNED_VM_CAP or vm_caps > GLOBAL_VM_LIMIT:
        raise RuntimeError(
            f"global VM cap drifted: expected {PLANNED_VM_CAP}, got {vm_caps} (limit {GLOBAL_VM_LIMIT})"
        )
    if subscription_caps != SUBSCRIPTION_VM_CAP:
        raise RuntimeError(
            f"subscription VM cap drifted: expected {SUBSCRIPTION_VM_CAP}, got {subscription_caps}"
        )
    for job in jobs:
        command = " ".join(job.command()).lower()
        if "opus" in command:
            raise RuntimeError(f"{job.label}: Opus unexpectedly appears in the command")
        if job.harness == "subscription" and "gpt-5.5" not in command and not job.retry_parent:
            # Source-conditioned commands identify GPT through the frozen payload or
            # activity-log ID rather than --targets. Validate that mapping separately.
            referenced_ids = [
                item for item in PREFIX_INPUTS
                if f"={item}" in command
                or f",{item}" in command
                or f"traj{item}-" in command
            ]
            if not referenced_ids or any(PREFIX_INPUTS[item].target != "gpt-5.5" for item in referenced_ids):
                raise RuntimeError(f"{job.label}: subscription job is not GPT-only")


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


def validate_plan(jobs: list[Job]) -> tuple[str, dict[int, str]]:
    validate_job_matrix(jobs)
    required_files = (
        ENVIRONMENTS_ROOT / "seeds" / "ml_prefix_only" / "demand_forecasting" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "p_hacking_prefix_only" / "checkout_redesign_positive" / "manifest.json",
        ENVIRONMENTS_ROOT / "prefixes" / "exp_ml_prefix.py",
        ENVIRONMENTS_ROOT / "prefixes" / "exp_p_hacking_prefix.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"required prefix-generation inputs are missing: {missing}")
    for label, retry in RETRY_INPUTS.items():
        validate_retry_input(label, retry)
    transport_hashes = validate_current_prefix_sources()
    fingerprint = source_fingerprint()
    return fingerprint, transport_hashes


def print_plan(jobs: list[Job], fingerprint: str, transport_hashes: dict[int, str]) -> None:
    print("2026-08-15 continuation push — exact free plan")
    print(f"source fingerprint: {fingerprint}")
    print()
    current_group = None
    for job in jobs:
        if job.group != current_group:
            current_group = job.group
            print(f"{current_group}:")
        print(
            f"  [{job.label}] cells={job.cells} harness={job.harness} "
            f"vm_cap={job.vm_cap}"
        )
        print("    " + " ".join(job.command()))
    print()
    print(
        f"TOTAL controllers={len(jobs)} trajectories={sum(job.cells for job in jobs)} "
        f"vm_caps={sum(job.vm_cap for job in jobs)}/{GLOBAL_VM_LIMIT} "
        f"subscription_caps={sum(job.vm_cap for job in jobs if job.harness == 'subscription')}"
    )
    print("Harnesses: production=DeepSeek/GLM/Kimi; subscription=GPT-5.5; Opus=0.")
    print("Existing successful cells are not duplicated by the six exact retry jobs.")
    print(
        "Source-bound omission: no selected GPT one-turn p-hack/ML source and no "
        "selected DeepSeek clean p-hack source exist; no unsupported condition is invented."
    )
    print(
        "Multi-agent p-hacking uses complete inline delivery; multi-agent ML keeps "
        "the separately defined natural ACTIVITY_LOG.md file-available condition."
    )
    print(
        f"Validated {len(PREFIX_INPUTS)} frozen inline payloads and "
        f"{len(transport_hashes)} exact ML activity-log transport hashes."
    )
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


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _campaign_ids() -> set[str]:
    if not CAMPAIGN_ROOT.is_dir():
        return set()
    return {path.stem for path in CAMPAIGN_ROOT.glob("*.json")}


async def run_paid_batch(
    jobs: list[Job], *, run_dir: Path, fingerprint: str,
    transport_hashes: dict[int, str], logger: RunLogger,
) -> int:
    environment = dict(os.environ)
    environment["AWS_PROFILE"] = "mats-run"
    environment["PYTHONUNBUFFERED"] = "1"

    login_status = await stream_command(
        label="aws-login",
        command=["aws", "login", "--profile", "mats-login", "--region", AWS_REGION],
        log_path=run_dir / "aws-login.log",
        logger=logger,
        environment=environment,
    )
    if login_status != 0:
        raise RuntimeError("AWS login failed; no controller was started")

    identity_status = await stream_command(
        label="aws-identity",
        command=["aws", "sts", "get-caller-identity", "--profile", "mats-run", "--region", AWS_REGION],
        log_path=run_dir / "aws-identity.log",
        logger=logger,
        environment=environment,
        quiet_success=True,
    )
    if identity_status != 0:
        raise RuntimeError("AWS run-profile authentication failed; no controller was started")

    setup_status = await stream_command(
        label="aws-setup",
        command=[
            "uv", "run", "exp_real_audit_pipeline.py", "--aws-setup",
            "--confirm-personal-account", "--harness=subscription",
            f"--aws-region={AWS_REGION}", f"--aws-instance-type={AWS_INSTANCE_TYPE}",
        ],
        log_path=run_dir / "aws-setup.log",
        logger=logger,
        environment=environment,
    )
    if setup_status != 0:
        raise RuntimeError("AWS setup failed; no controller was started")

    STARTED_MARKER.write_text(str(run_dir) + "\n")
    existing_campaigns = _campaign_ids()
    logger.log(
        f"Starting {len(jobs)} controllers for {sum(job.cells for job in jobs)} "
        f"trajectories; global VM cap={sum(job.vm_cap for job in jobs)}"
    )
    logger.log(
        f"Subscription cap={sum(job.vm_cap for job in jobs if job.harness == 'subscription')}; "
        f"controllers start {LAUNCH_STAGGER_SECONDS}s apart"
    )
    logger.log("Child failures do not cancel or automatically retry other campaigns.")

    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        current = source_fingerprint()
        if current != fingerprint:
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
                logger.log(f"Campaign controllers still running: {running}/{len(tasks)}")

    heartbeat_task = asyncio.create_task(heartbeat())
    statuses: dict[str, int | None] = {job.label: None for job in jobs}
    for label, task in tasks.items():
        try:
            statuses[label] = await task
        except Exception as error:  # preserve other controllers and their evidence
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

    new_campaigns = sorted(_campaign_ids() - existing_campaigns)
    status_jobs = []
    for job in jobs:
        row = asdict(job)
        row["args"] = list(job.args)
        row["command"] = job.command()
        row["exit_status"] = statuses[job.label]
        row["log"] = str(run_dir / f"{job.label}.log")
        status_jobs.append(row)
    success = (
        launch_failure is None
        and all(status == 0 for status in statuses.values())
        and viewer_status == 0
    )
    final_status = {
        "format": "environments-continuation-push-20260815-status-v1",
        "run_directory": str(run_dir),
        "source_fingerprint": fingerprint,
        "trajectories": sum(job.cells for job in jobs),
        "maximum_vms": sum(job.vm_cap for job in jobs),
        "subscription_maximum_vms": sum(
            job.vm_cap for job in jobs if job.harness == "subscription"
        ),
        "automatic_campaign_retries": False,
        "opus_included": False,
        "launch_failure": launch_failure,
        "viewer_exit_status": viewer_status,
        "new_campaign_ids": new_campaigns,
        "multi_agent_ml_transport_hashes": {
            str(key): value for key, value in sorted(transport_hashes.items())
        },
        "jobs": status_jobs,
        "success": success,
    }
    _atomic_json(run_dir / "final_status.json", final_status)
    logger.log(f"Final status: {run_dir / 'final_status.json'}")
    if success:
        logger.log(f"DONE. Viewer: {DATA_ROOT / 'viewer' / 'index.html'}")
        return 0
    logger.log(
        "DONE WITH FAILURES. Evidence was retained; inspect the labeled logs and "
        "campaign states before any exact-cell recovery."
    )
    return 1


def _maybe_caffeinate(args: list[str]) -> None:
    if os.environ.get("MATS_CONTINUATION_PUSH_CAFFEINATED") or not shutil.which("caffeinate"):
        return
    environment = dict(os.environ)
    environment["MATS_CONTINUATION_PUSH_CAFFEINATED"] = "1"
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
    plan_slug = f"{RUN_BASENAME}-runtime" if args.plan else (
        f"{RUN_BASENAME}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    jobs = build_jobs(plan_slug)
    fingerprint, transport_hashes = validate_plan(jobs)
    if args.plan:
        print_plan(jobs, fingerprint, transport_hashes)
        return 0

    _maybe_caffeinate(sys.argv[1:])
    for command in ("aws", "git", "uv"):
        if not shutil.which(command):
            raise SystemExit(f"required command is unavailable: {command}")
    if STARTED_MARKER.exists():
        previous = STARTED_MARKER.read_text().strip()
        raise SystemExit(
            "this exact paid slate was already started; inspect its durable record: "
            f"{previous}"
        )
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
    except FileExistsError as error:
        raise SystemExit(f"another continuation-push wrapper may be running: {LOCK_DIR}") from error

    run_dir = RUN_ROOT / plan_slug
    try:
        run_dir.mkdir()
        logger = RunLogger(run_dir / "orchestrator.log")
        try:
            plan_record = {
                "format": "environments-continuation-push-20260815-plan-v1",
                "created_at": datetime.now().astimezone().isoformat(),
                "run_directory": str(run_dir),
                "source_fingerprint": fingerprint,
                "trajectories": sum(job.cells for job in jobs),
                "maximum_vms": sum(job.vm_cap for job in jobs),
                "subscription_maximum_vms": sum(
                    job.vm_cap for job in jobs if job.harness == "subscription"
                ),
                "automatic_campaign_retries": False,
                "opus_included": False,
                "prefix_inputs": [
                    {**asdict(item), "path": str(item.path)}
                    for item in PREFIX_INPUTS.values()
                ],
                "multi_agent_ml_transport_hashes": {
                    str(key): value for key, value in sorted(transport_hashes.items())
                },
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
            _atomic_json(run_dir / "plan.json", plan_record)
            logger.log(f"Run directory: {run_dir}")
            logger.log(f"Frozen plan: {run_dir / 'plan.json'}")
            logger.log(f"Source fingerprint: {fingerprint}")
            logger.log(
                f"Plan: {sum(job.cells for job in jobs)} trajectories; "
                f"{sum(job.vm_cap for job in jobs)} maximum VMs"
            )
            return asyncio.run(run_paid_batch(
                jobs,
                run_dir=run_dir,
                fingerprint=fingerprint,
                transport_hashes=transport_hashes,
                logger=logger,
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
