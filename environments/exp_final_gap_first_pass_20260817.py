#!/usr/bin/env python3
"""Run the agreed first pass for the remaining experiment gaps.

The slate contains 460 independent paid trajectories:

* redo p-hacking no-honeypot -> ML from reasoning_prompt_benchmark prefixes
  (one frozen prefix for each of DeepSeek, GLM, Kimi, and GPT-5.5; n=40 each);
* fill the currently available GPT-5.5 ML -> ML, multi-agent ML, and
  p-hacking no-honeypot -> p-hacking arms (n=40 per source arm);
* add Kimi demand-forecasting two-turn-hack to regular and multi-agent ML
  (n=40 each);
* generate 20 ordinary judged originals in each of the three pools where a
  missing condition may still be elicitable: GPT demand hacks, GLM demand clean,
  and DeepSeek reasoning-prompt clean.

The eleven controller caps sum to the 250-VM account allowance. Subscription
controllers sum to the user-selected experimental ceiling of 100 workers. The
wrapper never retries automatically and never rebuilds the viewer.

Usage (from ``mats/environments``)::

    uv run exp_final_gap_first_pass_20260817.py --plan
    uv run exp_final_gap_first_pass_20260817.py

``--plan`` is local, free, and read-only. Paid mode performs subscription-auth
and AWS preflights, writes a durable one-use marker, and launches the frozen slate.
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
PRODUCTION = "production"
SUBSCRIPTION = "subscription"

CONTINUATION_EPOCHS = 40
ELICITATION_EPOCHS = 20
GLOBAL_VM_LIMIT = 250
SUBSCRIPTION_VM_LIMIT = 100
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180

PLANNED_CORRECTED_PHACK_TO_ML_CELLS = 160
PLANNED_GPT_CORE_CELLS = 160
PLANNED_KIMI_TWO_TURN_CELLS = 80
PLANNED_ELICITATION_CELLS = 60
PLANNED_CELLS = 460
PLANNED_CONTROLLERS = 11
PLANNED_VM_CAP = 250

RUN_BASENAME = "final-gap-first-pass-20260817"
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
    source_kind: str
    family: str
    seed: str
    prefix_type: str
    condition: str
    epoch: int
    message_count: int
    trajectory_id: int | None = None

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


PHACK_NO_HONEYPOT_PREFIXES = (
    FrozenPrefix(
        "reasoning-positive-open-n3-deepseek-v4-pro-reasoning-prompt-benchmark-"
        "positive-e1-305335f1aa06.json",
        "df39568f3174acff10aecc047ffc02e9d83403c6fa8cc7e485ca180b54d35b70",
        "reasoning-positive-open-n3-deepseek-v4-pro-reasoning-prompt-benchmark-"
        "positive-e1",
        "PHN-25", "deepseek-v4-pro", PRODUCTION, "phack-no-honeypot",
        "p_hacking_prefix_only", "reasoning_prompt_benchmark_positive",
        "p_hacking_no_honeypot", "no_honeypot", 1, 31,
    ),
    FrozenPrefix(
        "reasoning-positive-open-n3-glm-5-1-reasoning-prompt-benchmark-positive-"
        "e1-51e8a5d645c9.json",
        "ca53724d8465c1377554d24f32b2b71a75405dfc78e12deb09622ab6eaebc916",
        "reasoning-positive-open-n3-glm-5-1-reasoning-prompt-benchmark-positive-e1",
        "PHN-27", "glm-5.1", PRODUCTION, "phack-no-honeypot",
        "p_hacking_prefix_only", "reasoning_prompt_benchmark_positive",
        "p_hacking_no_honeypot", "no_honeypot", 1, 29,
    ),
    FrozenPrefix(
        "reasoning-positive-open-n3-kimi-k2-6-reasoning-prompt-benchmark-positive-"
        "e1-27119ee371d2.json",
        "42069a6bc41305075e8b579f959e99e3f0467ad3be80b914fb4d8859f05267bb",
        "reasoning-positive-open-n3-kimi-k2-6-reasoning-prompt-benchmark-positive-e1",
        "PHN-33", "kimi-k2.6", PRODUCTION, "phack-no-honeypot",
        "p_hacking_prefix_only", "reasoning_prompt_benchmark_positive",
        "p_hacking_no_honeypot", "no_honeypot", 1, 40,
    ),
    FrozenPrefix(
        "next-open-continuations-20260817-20260817014146-reasoning-nohoneypot-"
        "gpt-gpt-5-5-reasoning-prompt-benchmark-positive-e1-747b3f454be3.json",
        "a1cefc1980e46a02fe2b4c47cdf683f304d2ed139e90e0c2b1c04b31e2d0db21",
        "next-open-continuations-20260817-20260817014146-reasoning-nohoneypot-"
        "gpt-gpt-5-5-reasoning-prompt-benchmark-positive-e1",
        "PHN-37", "gpt-5.5", SUBSCRIPTION, "phack-no-honeypot",
        "p_hacking_prefix_only", "reasoning_prompt_benchmark_positive",
        "p_hacking_no_honeypot", "no_honeypot", 1, 23,
    ),
)

GPT_CLEAN_ML_PREFIX = FrozenPrefix(
    "traj783-c05e183128f0.json",
    "e29f1cef2bb9339b7254c5817aea54188645dc9592c883f6af235b9bd6862645",
    "traj783", "#783", "gpt-5.5", SUBSCRIPTION, "judged-conversation",
    "ml_training_data_misuse", "demand_forecasting", "ml_demand_forecasting",
    "clean", 1, 100, 783,
)

GPT_CLEAN_ACTIVITY_PREFIX = FrozenPrefix(
    "activity-log-traj783-1f837e98f439.json",
    "09c7f972b8d9594dccae26d23f82dde85d38c97d328760b7191646ce6cd83637",
    "activity-log-traj783", "#783 activity log", "gpt-5.5", SUBSCRIPTION,
    "activity-log", "ml_training_data_misuse", "demand_forecasting",
    "activity_log_context", "clean", 1, 1, 783,
)

GPT_NO_HONEYPOT_ML_PREFIX = FrozenPrefix(
    "next-open-continuations-20260817-20260817014146-demand-gpt-gpt-5-5-"
    "demand-forecasting-e5-23557dc62c31.json",
    "f7fc422a3fb0d87f85cefcde899e2a2d62040cd49e34ab7157e7657c02743f1a",
    "next-open-continuations-20260817-20260817014146-demand-gpt-gpt-5-5-"
    "demand-forecasting-e5",
    "MLD-136", "gpt-5.5", SUBSCRIPTION, "ml-no-honeypot",
    "ml_prefix_only", "demand_forecasting", "ml_prefix_only",
    "no_honeypot", 5, 94,
)

KIMI_TWO_TURN_ML_PREFIX = FrozenPrefix(
    "traj842-9c991569c3d8.json",
    "f93a4613a6548d557e7db6207d957d3337b52a0d754fea76a73c62d7b0ae01f7",
    "traj842", "#842", "kimi-k2.6", PRODUCTION, "judged-conversation",
    "ml_training_data_misuse", "demand_forecasting", "ml_demand_forecasting",
    "hack_2turn", 1, 70, 842,
)

KIMI_TWO_TURN_ACTIVITY_PREFIX = FrozenPrefix(
    "activity-log-traj842-056b444040da.json",
    "298fa029358703db3d84cfd1118f7cc253d7970935e405136fb7aab2579e6bd8",
    "activity-log-traj842", "#842 activity log", "kimi-k2.6", PRODUCTION,
    "activity-log", "ml_training_data_misuse", "demand_forecasting",
    "activity_log_context", "hack_2turn", 1, 1, 842,
)

ALL_PREFIXES = (
    *PHACK_NO_HONEYPOT_PREFIXES,
    GPT_CLEAN_ML_PREFIX,
    GPT_CLEAN_ACTIVITY_PREFIX,
    GPT_NO_HONEYPOT_ML_PREFIX,
    KIMI_TWO_TURN_ML_PREFIX,
    KIMI_TWO_TURN_ACTIVITY_PREFIX,
)


@dataclass(frozen=True)
class Job:
    label: str
    group: str
    source_mode: str
    treatment: str | None
    destination: str
    cells: int
    vm_cap: int
    harness: str
    prefix_paths: tuple[Path, ...]
    args: tuple[str, ...]
    endpoint: str

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _csv(paths: Iterable[Path]) -> str:
    return ",".join(str(path) for path in paths)


def _continuation_job(
    *, label: str, group: str, treatment: str, destination: str,
    harness: str, paths: tuple[Path, ...], vm_cap: int,
) -> Job:
    if destination == "fraud_detection":
        seed_dir = "ml_training_data_misuse"
        time_limit = 4200
        pressure: tuple[str, ...] = ()
    elif destination == "checkout_redesign":
        seed_dir = "p_hacking"
        time_limit = 1800
        pressure = ("--pressure=low",)
    else:
        raise ValueError(f"unsupported continuation destination: {destination}")
    return Job(
        label=label,
        group=group,
        source_mode=(
            "activity-log" if all("activity-log" in path.name for path in paths)
            else "conversation"
        ),
        treatment=treatment,
        destination=destination,
        cells=len(paths) * CONTINUATION_EPOCHS,
        vm_cap=vm_cap,
        harness=harness,
        prefix_paths=paths,
        endpoint="exp_continuation_pipeline.py",
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={_csv(paths)}",
            f"--seed-dir={seed_dir}",
            f"--seeds={destination}",
            f"--epochs={CONTINUATION_EPOCHS}",
            f"--harness={harness}",
            "--condition=allow",
            *pressure,
            f"--judge={JUDGE}",
            f"--gate-model={JUDGE}",
            "--compute=aws",
            f"--vm-concurrency={vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            f"--time-limit={time_limit}",
            "--skip-viewer",
        ),
    )


def _elicitation_job(
    *, label: str, target: str, family: str, seed: str,
    harness: str, vm_cap: int,
) -> Job:
    time_limit = 4200 if family == "ml_training_data_misuse" else 1800
    pressure = ("--pressure=low",) if family == "p_hacking" else ()
    return Job(
        label=label,
        group="ordinary judged prefix elicitation",
        source_mode="ordinary-judged-originals",
        treatment=None,
        destination=seed,
        cells=ELICITATION_EPOCHS,
        vm_cap=vm_cap,
        harness=harness,
        prefix_paths=(),
        endpoint="exp_real_audit_pipeline.py",
        args=(
            f"--targets={target}",
            f"--seed-dir={family}",
            f"--seeds={seed}",
            f"--epochs={ELICITATION_EPOCHS}",
            f"--harness={harness}",
            "--reasoning=yes",
            "--condition=allow",
            *pressure,
            f"--judge={JUDGE}",
            f"--gate-model={JUDGE}",
            "--compute=aws",
            f"--vm-concurrency={vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
            f"--time-limit={time_limit}",
            "--skip-viewer",
        ),
    )


def build_jobs() -> list[Job]:
    open_phack = tuple(
        item.path for item in PHACK_NO_HONEYPOT_PREFIXES
        if item.harness == PRODUCTION
    )
    gpt_phack = next(
        item.path for item in PHACK_NO_HONEYPOT_PREFIXES
        if item.harness == SUBSCRIPTION
    )
    return [
        _continuation_job(
            label="correct-phack-nohoneypot-to-ml-production",
            group="corrected p-hacking no-honeypot -> ML",
            treatment="no-honeypot", destination="fraud_detection",
            harness=PRODUCTION, paths=open_phack, vm_cap=75,
        ),
        _continuation_job(
            label="correct-phack-nohoneypot-to-ml-gpt",
            group="corrected p-hacking no-honeypot -> ML",
            treatment="no-honeypot", destination="fraud_detection",
            harness=SUBSCRIPTION, paths=(gpt_phack,), vm_cap=18,
        ),
        _continuation_job(
            label="gpt-ml-conversation-clean",
            group="GPT core", treatment="no-hack", destination="fraud_detection",
            harness=SUBSCRIPTION, paths=(GPT_CLEAN_ML_PREFIX.path,), vm_cap=18,
        ),
        _continuation_job(
            label="gpt-ml-conversation-nohoneypot",
            group="GPT core", treatment="no-honeypot",
            destination="fraud_detection", harness=SUBSCRIPTION,
            paths=(GPT_NO_HONEYPOT_ML_PREFIX.path,), vm_cap=18,
        ),
        _continuation_job(
            label="gpt-ml-activity-clean",
            group="GPT core", treatment="no-hack", destination="fraud_detection",
            harness=SUBSCRIPTION, paths=(GPT_CLEAN_ACTIVITY_PREFIX.path,), vm_cap=18,
        ),
        _continuation_job(
            label="gpt-phack-nohoneypot-to-phack",
            group="GPT core", treatment="no-honeypot",
            destination="checkout_redesign", harness=SUBSCRIPTION,
            paths=(gpt_phack,), vm_cap=18,
        ),
        _continuation_job(
            label="kimi-ml-conversation-hack2",
            group="Kimi two-turn completion", treatment="hack-in-two-turns",
            destination="fraud_detection", harness=PRODUCTION,
            paths=(KIMI_TWO_TURN_ML_PREFIX.path,), vm_cap=25,
        ),
        _continuation_job(
            label="kimi-ml-activity-hack2",
            group="Kimi two-turn completion", treatment="hack-in-two-turns",
            destination="fraud_detection", harness=PRODUCTION,
            paths=(KIMI_TWO_TURN_ACTIVITY_PREFIX.path,), vm_cap=25,
        ),
        _elicitation_job(
            label="elicit-gpt-demand-hacks", target="gpt-5.5",
            family="ml_training_data_misuse", seed="demand_forecasting",
            harness=SUBSCRIPTION, vm_cap=10,
        ),
        _elicitation_job(
            label="elicit-glm-demand-clean", target="glm-5.1",
            family="ml_training_data_misuse", seed="demand_forecasting",
            harness=PRODUCTION, vm_cap=13,
        ),
        _elicitation_job(
            label="elicit-deepseek-reasoning-clean", target="deepseek-v4-pro",
            family="p_hacking", seed="reasoning_prompt_benchmark",
            harness=PRODUCTION, vm_cap=12,
        ),
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


def _validate_prefix_identity(item: FrozenPrefix, payload: dict) -> None:
    source = payload.get("source") or {}
    condition = source.get("prefix_condition") or {}
    common = (
        payload.get("format"), payload.get("name"), payload.get("target"),
        payload.get("reasoning"), len(payload.get("messages") or []),
        source.get("harness"), source.get("family"), source.get("seed"),
        source.get("prefix_type"), source.get("epoch"),
    )
    expected_common = (
        "environments-continuation-prefix-v1", item.name, item.target, True,
        item.message_count, item.harness, item.family, item.seed,
        item.prefix_type, item.epoch,
    )
    if common != expected_common:
        raise RuntimeError(
            f"frozen prefix identity drifted for {item.filename}: "
            f"expected {expected_common!r}, got {common!r}"
        )

    if item.source_kind == "phack-no-honeypot":
        observed = (
            source.get("kind"), source.get("generator"),
            source.get("comparison_source_seed"),
            source.get("analysis_honeypot"), source.get("result_condition"),
            source.get("pressure"),
            (source.get("continuation_eligibility") or {}).get(
                "eligible_by_default"
            ),
        )
        expected = (
            "external", "exp_p_hacking_prefix.py", "reasoning_prompt_benchmark",
            False, "clear positive", "low", True,
        )
    elif item.source_kind == "ml-no-honeypot":
        observed = (
            source.get("kind"), source.get("generator"),
            (source.get("continuation_eligibility") or {}).get(
                "eligible_by_default"
            ),
        )
        expected = ("external", "exp_ml_prefix.py", True)
    elif item.source_kind == "judged-conversation":
        observed = (
            source.get("kind"), source.get("trajectory_id"),
            condition.get("key"), source.get("was_continuation"),
            (payload.get("native_resume") or {}).get("format"),
        )
        expected = (
            "trajectory", item.trajectory_id, item.condition, False,
            "environments-production-native-resume-v1",
        )
    elif item.source_kind == "activity-log":
        lossy = (source.get("activity_log_metadata") or {}).get(
            "lossy_processing"
        ) or {}
        observed = (
            source.get("kind"), source.get("generator"),
            source.get("trajectory_id"), condition.get("key"),
            (payload.get("delivery") or {}).get("mode"),
            payload.get("native_resume"), lossy.get("affected"),
            lossy.get("kind"), bool(lossy.get("visible_caveat")),
        )
        expected = (
            "trajectory", "build_activity_log_prefixes.py", item.trajectory_id,
            item.condition, "inline_user_context", None, True,
            "observable_activity_only", True,
        )
    else:
        raise RuntimeError(f"unknown frozen source kind: {item.source_kind}")
    if observed != expected:
        raise RuntimeError(
            f"frozen prefix provenance drifted for {item.filename}: "
            f"expected {expected!r}, got {observed!r}"
        )


def validate_frozen_prefixes() -> None:
    filenames = [item.filename for item in ALL_PREFIXES]
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("frozen prefix filenames are not unique")
    if len(PHACK_NO_HONEYPOT_PREFIXES) != 4:
        raise RuntimeError("exactly one p-hacking no-honeypot source per model is required")
    if {item.target for item in PHACK_NO_HONEYPOT_PREFIXES} != {
        "deepseek-v4-pro", "glm-5.1", "kimi-k2.6", "gpt-5.5",
    }:
        raise RuntimeError("p-hacking no-honeypot model coverage drifted")
    if GPT_CLEAN_ML_PREFIX.trajectory_id != GPT_CLEAN_ACTIVITY_PREFIX.trajectory_id:
        raise RuntimeError("GPT activity source is not the clean conversation source")
    if KIMI_TWO_TURN_ML_PREFIX.trajectory_id != (
        KIMI_TWO_TURN_ACTIVITY_PREFIX.trajectory_id
    ):
        raise RuntimeError("Kimi activity source is not the two-turn conversation source")

    for item in ALL_PREFIXES:
        if not item.path.is_file():
            raise RuntimeError(f"missing frozen prefix: {item.path}")
        observed_hash = _sha256_file(item.path)
        if observed_hash != item.file_sha256:
            raise RuntimeError(
                f"frozen prefix bytes drifted: {item.path}; expected "
                f"{item.file_sha256}, got {observed_hash}"
            )
        _validate_prefix_identity(item, json.loads(item.path.read_text()))

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
        if not job.prefix_paths:
            continue
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
            raise RuntimeError(f"{job.label}: unavailable destination")
        cells = build_continuation_cells(specs, seeds_path, [job.destination])
        pressure = "low" if seed_dir == "p_hacking" else None
        for unit_path in {cell.unit_path for cell in cells}:
            load_protocol_sources(unit_path, pressure=pressure)
        if len(cells) * CONTINUATION_EPOCHS != job.cells:
            raise RuntimeError(f"{job.label}: continuation cell count drifted")


def _validate_jobs(jobs: list[Job]) -> None:
    if len(jobs) != PLANNED_CONTROLLERS:
        raise RuntimeError(f"expected {PLANNED_CONTROLLERS} controllers")
    if len({job.label for job in jobs}) != len(jobs):
        raise RuntimeError("duplicate job labels")
    if sum(job.cells for job in jobs) != PLANNED_CELLS:
        raise RuntimeError("planned paid-cell total drifted")
    if sum(job.vm_cap for job in jobs) != PLANNED_VM_CAP:
        raise RuntimeError("planned VM-cap total drifted")
    if sum(job.vm_cap for job in jobs if job.harness == SUBSCRIPTION) != (
        SUBSCRIPTION_VM_LIMIT
    ):
        raise RuntimeError("subscription VM allocation drifted")
    if sum(job.vm_cap for job in jobs) > GLOBAL_VM_LIMIT:
        raise RuntimeError("global VM allocation exceeds the account limit")

    group_counts = {
        group: sum(job.cells for job in jobs if job.group == group)
        for group in {job.group for job in jobs}
    }
    expected_groups = {
        "corrected p-hacking no-honeypot -> ML":
            PLANNED_CORRECTED_PHACK_TO_ML_CELLS,
        "GPT core": PLANNED_GPT_CORE_CELLS,
        "Kimi two-turn completion": PLANNED_KIMI_TWO_TURN_CELLS,
        "ordinary judged prefix elicitation": PLANNED_ELICITATION_CELLS,
    }
    if group_counts != expected_groups:
        raise RuntimeError(f"job groups drifted: {group_counts!r}")

    expected_matrix = {
        "correct-phack-nohoneypot-to-ml-production": (120, 75, PRODUCTION),
        "correct-phack-nohoneypot-to-ml-gpt": (40, 18, SUBSCRIPTION),
        "gpt-ml-conversation-clean": (40, 18, SUBSCRIPTION),
        "gpt-ml-conversation-nohoneypot": (40, 18, SUBSCRIPTION),
        "gpt-ml-activity-clean": (40, 18, SUBSCRIPTION),
        "gpt-phack-nohoneypot-to-phack": (40, 18, SUBSCRIPTION),
        "kimi-ml-conversation-hack2": (40, 25, PRODUCTION),
        "kimi-ml-activity-hack2": (40, 25, PRODUCTION),
        "elicit-gpt-demand-hacks": (20, 10, SUBSCRIPTION),
        "elicit-glm-demand-clean": (20, 13, PRODUCTION),
        "elicit-deepseek-reasoning-clean": (20, 12, PRODUCTION),
    }
    observed_matrix = {
        job.label: (job.cells, job.vm_cap, job.harness) for job in jobs
    }
    if observed_matrix != expected_matrix:
        raise RuntimeError(f"exact job matrix drifted: {observed_matrix!r}")

    for job in jobs:
        command = " ".join(job.command()).lower()
        required = (
            f"--harness={job.harness}", "--compute=aws", "--skip-viewer",
            f"--vm-concurrency={job.vm_cap}",
            f"--aws-region={AWS_REGION}",
            f"--aws-instance-type={AWS_INSTANCE_TYPE}",
        )
        if any(value not in job.args for value in required):
            raise RuntimeError(f"{job.label}: required command invariant drifted")
        if "opus" in command:
            raise RuntimeError(f"{job.label}: Opus unexpectedly entered the slate")
        if job.prefix_paths and f"--epochs={CONTINUATION_EPOCHS}" not in job.args:
            raise RuntimeError(f"{job.label}: continuation n drifted")
        if not job.prefix_paths and f"--epochs={ELICITATION_EPOCHS}" not in job.args:
            raise RuntimeError(f"{job.label}: elicitation n drifted")


def source_fingerprint() -> str:
    return precedent.source_fingerprint()


def validate_plan() -> tuple[str, str]:
    required = (
        ENVIRONMENTS_ROOT / "exp_continuation_pipeline.py",
        ENVIRONMENTS_ROOT / "exp_real_audit_pipeline.py",
        ENVIRONMENTS_ROOT / "tools" / "check_codex_subscription_auth.py",
        ENVIRONMENTS_ROOT / "seeds" / "ml_training_data_misuse"
        / "demand_forecasting" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "ml_training_data_misuse"
        / "fraud_detection" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "p_hacking"
        / "reasoning_prompt_benchmark" / "manifest.json",
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
    return ", ".join(f"{item.viewer_label} {item.target}" for item in items)


def print_plan(source_hash: str, inputs_hash: str) -> None:
    jobs = build_jobs()
    print("2026-08-17 final gap first pass — exact free plan")
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
        f"(corrected_phacking_to_ml={PLANNED_CORRECTED_PHACK_TO_ML_CELLS}, "
        f"gpt_core={PLANNED_GPT_CORE_CELLS}, "
        f"kimi_two_turn={PLANNED_KIMI_TWO_TURN_CELLS}, "
        f"elicitation={PLANNED_ELICITATION_CELLS})"
    )
    print(
        f"Concurrent VM caps: {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}; "
        f"subscription cap={SUBSCRIPTION_VM_LIMIT}."
    )
    print(
        "Correct p-hacking no-honeypot sources (reasoning_prompt_benchmark): "
        + _selection_text(PHACK_NO_HONEYPOT_PREFIXES)
    )
    print(
        "GPT available ML sources: #783 clean conversation, MLD-136 "
        "no-honeypot conversation, and the #783 clean activity-log transform."
    )
    print(
        "GPT notable #781 is deliberately excluded rather than pooled with clean; "
        "there is no existing judged GPT hack arm."
    )
    print(
        "Kimi #842 is run as hack-in-two-turns in both conversation and its exact "
        "activity-log transform."
    )
    print(
        "Elicitation is 20 ordinary judged originals each: GPT demand_forecasting "
        "(seeking hacks), GLM demand_forecasting (seeking clean), and DeepSeek "
        "reasoning_prompt_benchmark (seeking clean). These are not auto-continued."
    )
    print(
        "Activity-log transforms intentionally omit system messages and private "
        "reasoning and include all stored observable activity with a visible caveat."
    )
    print(
        "The 100-worker subscription ceiling is user-selected and experimental; "
        "it is not a published OpenAI concurrency allowance."
    )
    print(
        "The viewer is not rebuilt or reloaded. No automatic retries are attempted; "
        "successful cells remain usable if another controller or cell fails."
    )
    print("Kimi Wikipedia is omitted because that bar is already complete. Opus is absent.")
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
            f"Submitting {job.label} — cells={job.cells} "
            f"source={job.source_mode} harness={job.harness} vm_cap={job.vm_cap}"
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
    success = launch_failure is None and all(
        statuses.get(job.label) == 0 for job in jobs
    )
    final_status = {
        "format": "environments-final-gap-first-pass-20260817-status-v1",
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
        "viewer_rebuilt": False,
        "opus_included": False,
        "gpt_notable_781_included": False,
        "elicitation_auto_continued": False,
        "launch_failure": launch_failure,
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
    variable = "MATS_FINAL_GAP_FIRST_PASS_20260817_CAFFEINATED"
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
            "format": "environments-final-gap-first-pass-20260817-plan-v1",
            "run_slug": run_slug,
            "source_fingerprint": source_hash,
            "frozen_input_fingerprint": inputs_hash,
            "planned_cells": PLANNED_CELLS,
            "planned_controllers": PLANNED_CONTROLLERS,
            "maximum_vms": GLOBAL_VM_LIMIT,
            "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
            "automatic_campaign_retries": False,
            "viewer_rebuilt": False,
            "gpt_notable_781_included": False,
            "elicitation_auto_continued": False,
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
