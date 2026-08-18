#!/usr/bin/env python3
"""Run the selected next-continuation slate plus the three missing GPT jobs.

This paid, one-off wrapper launches six independent experiments together:

1. demand_forecasting -> fraud_detection with nine ordinary conversation prefixes:
   six selected judged trajectories plus one selected no-honeypot prefix per model.
2. The matched inline activity-log condition for only the six judged trajectories.
   No no-honeypot ML prefix is copied into this condition.
3. reasoning_prompt_benchmark no-honeypot -> checkout_redesign with one selected
   positive prefix per open model.
4. The missing GPT-5.5 p-hacking no-honeypot -> fraud_detection continuations.
5. The missing eight GPT-5.5 demand_forecasting prefixes.
6. The missing three GPT-5.5 reasoning_prompt_benchmark no-honeypot prefixes.

Every source payload, transformation, treatment, destination, epoch count, and byte
hash is frozen below. The eleven controller caps sum to the 250-VM account allowance;
the three subscription controllers separately sum to the 18-worker subscription cap.
Children stream labeled output, never retry automatically, and skip the viewer build
because the viewer is managed separately.

Usage (from ``mats/environments``)::

    uv run exp_next_open_20260817.py --plan
    uv run exp_next_open_20260817.py

``--plan`` is local, free, and read-only. Paid mode has a durable one-use marker;
recover an interrupted run from the recorded campaign IDs rather than repeating it.
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
HARNESS = "production"
SUBSCRIPTION_HARNESS = "subscription"

EPOCHS = 40
GLOBAL_VM_LIMIT = 250
SUBSCRIPTION_VM_LIMIT = 18
LAUNCH_STAGGER_SECONDS = 20
HEARTBEAT_SECONDS = 180

PLANNED_REGULAR_ML_CELLS = 360
PLANNED_MULTI_AGENT_ML_CELLS = 240
PLANNED_PHACK_CELLS = 120
PLANNED_MISSING_GPT_CELLS = 51
PLANNED_CELLS = 771
PLANNED_CONTROLLERS = 11
PLANNED_VM_CAP = 250

RUN_BASENAME = "next-open-continuations-20260817"
LOCK_DIR = RUN_ROOT / f".{RUN_BASENAME}.lock"
STARTED_MARKER = RUN_ROOT / f".{RUN_BASENAME}.started"


@dataclass(frozen=True)
class FrozenPrefix:
    filename: str
    file_sha256: str
    name: str
    viewer_label: str
    target: str
    condition: str
    source_kind: str
    epoch: int
    message_count: int
    trajectory_id: int | None = None
    harness: str = HARNESS

    @property
    def path(self) -> Path:
        return PREFIX_ROOT / self.filename


# Six manually selected, judged demand-forecasting trajectories. These are the
# ordinary conversation prefixes used by regular ML -> ML.
JUDGED_ML_PREFIXES = (
    FrozenPrefix(
        "traj823-bacaaf0e6846.json",
        "07dcf8919d423c8162d45f960a6f814defb2a6dc5650492ecbdb630b0d4bdd3c",
        "traj823", "#823", "deepseek-v4-pro", "hack_1turn",
        "judged-conversation", 1, 67, 823,
    ),
    FrozenPrefix(
        "traj829-91e9001ecc62.json",
        "0bad8478966d995d2f41715c94fe54a7d34d654b1493c2ebf75cff96ff528ca3",
        "traj829", "#829", "deepseek-v4-pro", "hack_2turn",
        "judged-conversation", 1, 114, 829,
    ),
    FrozenPrefix(
        "traj824-3987c7ed694c.json",
        "91580471d235908ee909fab1d73b17cbbfb8faf56abeb1428c1c879f4006b186",
        "traj824", "#824", "deepseek-v4-pro", "clean",
        "judged-conversation", 1, 187, 824,
    ),
    FrozenPrefix(
        "traj831-10501827c3d6.json",
        "2865d783f81ea3eb439ebb16f15a74decc7a06f918e974f998317cbc58e4ef92",
        "traj831", "#831", "glm-5.1", "hack_1turn",
        "judged-conversation", 1, 151, 831,
    ),
    FrozenPrefix(
        "traj840-851f7d7c7500.json",
        "3c34e70fcde6f08e34528d2029957f3fbf54872e5410f7aa443ce1aa0dbdffc8",
        "traj840", "#840", "kimi-k2.6", "hack_1turn",
        "judged-conversation", 1, 120, 840,
    ),
    FrozenPrefix(
        "traj837-15de9f008344.json",
        "7087c59269bbe1329c77ba4685ab90252d291da546505353b616837f1c117e99",
        "traj837", "#837", "kimi-k2.6", "clean",
        "judged-conversation", 1, 182, 837,
    ),
)


# Exact observable-activity-log transforms of the six sources above. They omit
# system messages and private reasoning by design; validation below requires the
# visible lossy-processing caveat. There are deliberately no no-honeypot copies.
ACTIVITY_LOG_PREFIXES = (
    FrozenPrefix(
        "activity-log-traj823-c34f8007058d.json",
        "a7e4a558785241c050cf46bbf548a99a5ae8c71591be98388ea51c7cb1930b38",
        "activity-log-traj823", "#823 activity log", "deepseek-v4-pro",
        "hack_1turn", "activity-log", 1, 1, 823,
    ),
    FrozenPrefix(
        "activity-log-traj829-fea19dd01a79.json",
        "f7031d6e30eee19647abbd3ac73625c19c009c85b9291086b04455b49e2b2082",
        "activity-log-traj829", "#829 activity log", "deepseek-v4-pro",
        "hack_2turn", "activity-log", 1, 1, 829,
    ),
    FrozenPrefix(
        "activity-log-traj824-7d1f7f766b38.json",
        "7da97032867fd4851ac01e342e7e44850bd1367df7f8ed1dfe24729a1c5950c8",
        "activity-log-traj824", "#824 activity log", "deepseek-v4-pro",
        "clean", "activity-log", 1, 1, 824,
    ),
    FrozenPrefix(
        "activity-log-traj831-cdc0f8f3004f.json",
        "bd0f33a86b29d558ec98d1b91ffab1e23c07d154e02eeeeac39502e671492398",
        "activity-log-traj831", "#831 activity log", "glm-5.1",
        "hack_1turn", "activity-log", 1, 1, 831,
    ),
    FrozenPrefix(
        "activity-log-traj840-0e9b9992bac5.json",
        "d8f98350728280cb1654de005573b4be037f797f650231ea35b3944616effefb",
        "activity-log-traj840", "#840 activity log", "kimi-k2.6",
        "hack_1turn", "activity-log", 1, 1, 840,
    ),
    FrozenPrefix(
        "activity-log-traj837-33a149175515.json",
        "7420f4edc54aeafbb4a9a848298699f4326d3a89e1dff2c0a4638df8bc73b345",
        "activity-log-traj837", "#837 activity log", "kimi-k2.6",
        "clean", "activity-log", 1, 1, 837,
    ),
)


# One manually accepted, complete no-honeypot demand prefix per open model.
ML_NO_HONEYPOT_PREFIXES = (
    FrozenPrefix(
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "deepseek-v4-pro-demand-forecasting-e1-97d574d295f7.json",
        "d41db7542b8aa5a0cdb71f49840735c1f6d8cbf7f60bea055eac61e757ff41d0",
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "deepseek-v4-pro-demand-forecasting-e1",
        "MLD-110", "deepseek-v4-pro", "no_honeypot",
        "ml-no-honeypot", 1, 147,
    ),
    FrozenPrefix(
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "glm-5-1-demand-forecasting-e1-bcbcef01fd89.json",
        "04db68fb69e33611e0dccde7e3f621cdc392f88187bc870ac90911c691b459ca",
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "glm-5-1-demand-forecasting-e1",
        "MLD-122", "glm-5.1", "no_honeypot",
        "ml-no-honeypot", 1, 186,
    ),
    FrozenPrefix(
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "kimi-k2-6-demand-forecasting-e8-ea97458e063d.json",
        "bf42b7b31dbd1e7606e0aa198d40fb9ee84282880caa74b8b0d4b2db505184a9",
        "demand-phacking-to-ml-20260816-20260816215128-demand-production-"
        "kimi-k2-6-demand-forecasting-e8",
        "MLD-117", "kimi-k2.6", "no_honeypot",
        "ml-no-honeypot", 8, 90,
    ),
)


# One ordinary-looking, clear-positive reasoning-prompt no-honeypot prefix per open
# model, selected for the p-hacking -> p-hacking experiment.
PHACK_NO_HONEYPOT_PREFIXES = (
    FrozenPrefix(
        "reasoning-positive-open-n3-deepseek-v4-pro-reasoning-prompt-benchmark-"
        "positive-e1-305335f1aa06.json",
        "df39568f3174acff10aecc047ffc02e9d83403c6fa8cc7e485ca180b54d35b70",
        "reasoning-positive-open-n3-deepseek-v4-pro-reasoning-prompt-benchmark-"
        "positive-e1",
        "PHN-25", "deepseek-v4-pro", "no_honeypot",
        "phack-no-honeypot", 1, 31,
    ),
    FrozenPrefix(
        "reasoning-positive-open-n3-glm-5-1-reasoning-prompt-benchmark-positive-"
        "e1-51e8a5d645c9.json",
        "ca53724d8465c1377554d24f32b2b71a75405dfc78e12deb09622ab6eaebc916",
        "reasoning-positive-open-n3-glm-5-1-reasoning-prompt-benchmark-positive-e1",
        "PHN-27", "glm-5.1", "no_honeypot",
        "phack-no-honeypot", 1, 29,
    ),
    FrozenPrefix(
        "reasoning-positive-open-n3-kimi-k2-6-reasoning-prompt-benchmark-"
        "positive-e1-27119ee371d2.json",
        "42069a6bc41305075e8b579f959e99e3f0467ad3be80b914fb4d8859f05267bb",
        "reasoning-positive-open-n3-kimi-k2-6-reasoning-prompt-benchmark-positive-e1",
        "PHN-33", "kimi-k2.6", "no_honeypot",
        "phack-no-honeypot", 1, 40,
    ),
)


# The exact previously selected GPT-5.5 no-honeypot checkout prefix whose missing
# fraud-detection continuation is included tonight.
GPT_PHACK_NO_HONEYPOT_PREFIX = FrozenPrefix(
    "continuation-push-20260815-20260815185916-phack-nohoneypot-gpt-"
    "gpt-5-5-checkout-redesign-positive-e1-fca38e827408.json",
    "a11d5c23623ebebc0331e5d90d964a8dfa0313aad4e078c36851ad1deb94b18e",
    "continuation-push-20260815-20260815185916-phack-nohoneypot-gpt-"
    "gpt-5-5-checkout-redesign-positive-e1",
    "selected GPT checkout prefix", "gpt-5.5", "no_honeypot",
    "phack-checkout-no-honeypot-gpt", 1, 28,
    harness=SUBSCRIPTION_HARNESS,
)

ALL_PREFIXES = (
    *JUDGED_ML_PREFIXES,
    *ACTIVITY_LOG_PREFIXES,
    *ML_NO_HONEYPOT_PREFIXES,
    *PHACK_NO_HONEYPOT_PREFIXES,
    GPT_PHACK_NO_HONEYPOT_PREFIX,
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
    prefix_paths: tuple[Path, ...]
    args: tuple[str, ...]
    endpoint: str = "exp_continuation_pipeline.py"
    harness: str = HARNESS

    def command(self) -> list[str]:
        return ["uv", "run", self.endpoint, *self.args]


def _csv(paths: Iterable[Path]) -> str:
    return ",".join(str(path) for path in paths)


def _selected(
    prefixes: tuple[FrozenPrefix, ...], condition: str,
) -> tuple[Path, ...]:
    return tuple(item.path for item in prefixes if item.condition == condition)


def _ml_job(
    *, label: str, source_mode: str, treatment: str,
    paths: tuple[Path, ...], vm_cap: int,
) -> Job:
    return Job(
        label=label,
        group=(
            "regular ML -> ML"
            if source_mode == "conversation"
            else "multi-agent ML -> ML"
        ),
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


def _phack_job(*, paths: tuple[Path, ...], vm_cap: int) -> Job:
    treatment = "no-honeypot"
    return Job(
        label="phack-nohoneypot-to-phack",
        group="p-hacking no-honeypot -> p-hacking",
        source_mode="conversation",
        treatment=treatment,
        destination="checkout_redesign",
        cells=len(paths) * EPOCHS,
        vm_cap=vm_cap,
        prefix_paths=paths,
        args=(
            f"--treatment={treatment}",
            f"--prefix-files={_csv(paths)}",
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


def _missing_gpt_jobs(run_slug: str) -> list[Job]:
    phack_path = GPT_PHACK_NO_HONEYPOT_PREFIX.path
    return [
        Job(
            label="missing-gpt-phack-nohoneypot-to-ml",
            group="missing GPT-5.5",
            source_mode="conversation",
            treatment="no-honeypot",
            destination="fraud_detection",
            cells=40,
            vm_cap=10,
            prefix_paths=(phack_path,),
            harness=SUBSCRIPTION_HARNESS,
            args=(
                "--treatment=no-honeypot",
                f"--prefix-files={phack_path}",
                "--seed-dir=ml_training_data_misuse",
                "--seeds=fraud_detection",
                "--epochs=40",
                "--harness=subscription",
                "--condition=allow",
                f"--judge={JUDGE}",
                f"--gate-model={JUDGE}",
                "--compute=aws",
                "--vm-concurrency=10",
                f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}",
                "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            label="missing-gpt-demand-prefixes",
            group="missing GPT-5.5",
            source_mode="prefix-generation",
            treatment=None,
            destination="demand_forecasting",
            cells=8,
            vm_cap=5,
            prefix_paths=(),
            endpoint="prefixes/exp_ml_prefix.py",
            harness=SUBSCRIPTION_HARNESS,
            args=(
                "--targets=gpt-5.5",
                "--seeds=demand_forecasting",
                "--epochs=8",
                "--harness=subscription",
                "--reasoning=yes",
                f"--name={run_slug}-demand-gpt",
                "--compute=aws",
                "--vm-concurrency=5",
                f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}",
                "--time-limit=4200",
                "--skip-viewer",
            ),
        ),
        Job(
            label="missing-gpt-reasoning-nohoneypot-prefixes",
            group="missing GPT-5.5",
            source_mode="prefix-generation",
            treatment=None,
            destination="reasoning_prompt_benchmark_positive",
            cells=3,
            vm_cap=3,
            prefix_paths=(),
            endpoint="prefixes/exp_p_hacking_prefix.py",
            harness=SUBSCRIPTION_HARNESS,
            args=(
                "--targets=gpt-5.5",
                "--seeds=reasoning_prompt_benchmark_positive",
                "--epochs=3",
                "--harness=subscription",
                "--pressure=low",
                "--reasoning=yes",
                f"--name={run_slug}-reasoning-nohoneypot-gpt",
                "--compute=aws",
                "--vm-concurrency=3",
                f"--aws-region={AWS_REGION}",
                f"--aws-instance-type={AWS_INSTANCE_TYPE}",
                "--time-limit=1800",
                "--skip-viewer",
            ),
        ),
    ]


def build_jobs(run_slug: str = f"{RUN_BASENAME}-plan") -> list[Job]:
    judged = JUDGED_ML_PREFIXES
    activity = ACTIVITY_LOG_PREFIXES
    return [
        _ml_job(
            label="ml-conversation-hack1", source_mode="conversation",
            treatment="hack-in-one-turn", paths=_selected(judged, "hack_1turn"),
            vm_cap=39,
        ),
        _ml_job(
            label="ml-conversation-hack2", source_mode="conversation",
            treatment="hack-in-two-turns", paths=_selected(judged, "hack_2turn"),
            vm_cap=13,
        ),
        _ml_job(
            label="ml-conversation-nohack", source_mode="conversation",
            treatment="no-hack", paths=_selected(judged, "clean"), vm_cap=26,
        ),
        _ml_job(
            label="ml-conversation-nohoneypot", source_mode="conversation",
            treatment="no-honeypot",
            paths=tuple(item.path for item in ML_NO_HONEYPOT_PREFIXES),
            vm_cap=39,
        ),
        _ml_job(
            label="ml-activity-hack1", source_mode="activity-log",
            treatment="hack-in-one-turn",
            paths=_selected(activity, "hack_1turn"), vm_cap=39,
        ),
        _ml_job(
            label="ml-activity-hack2", source_mode="activity-log",
            treatment="hack-in-two-turns",
            paths=_selected(activity, "hack_2turn"), vm_cap=13,
        ),
        _ml_job(
            label="ml-activity-nohack", source_mode="activity-log",
            treatment="no-hack", paths=_selected(activity, "clean"), vm_cap=26,
        ),
        _phack_job(
            paths=tuple(item.path for item in PHACK_NO_HONEYPOT_PREFIXES),
            vm_cap=37,
        ),
        *_missing_gpt_jobs(run_slug),
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
            source.get("was_continuation"),
            (payload.get("native_resume") or {}).get("format"),
        )
        expected = (
            "trajectory", "ml_training_data_misuse", "demand_forecasting",
            item.trajectory_id, "ml_demand_forecasting", "ML: demand_forecasting",
            item.condition, False, "environments-production-native-resume-v1",
        )
    elif item.source_kind == "activity-log":
        metadata = source.get("activity_log_metadata") or {}
        lossy = metadata.get("lossy_processing") or {}
        identity = (
            source.get("generator"), source.get("family"), source.get("seed"),
            source.get("trajectory_id"), source.get("prefix_type"),
            condition.get("key"), (payload.get("delivery") or {}).get("mode"),
            payload.get("native_resume"), lossy.get("affected"),
            lossy.get("kind"), bool(lossy.get("visible_caveat")),
        )
        expected = (
            "build_activity_log_prefixes.py", "ml_training_data_misuse",
            "demand_forecasting", item.trajectory_id, "activity_log_context",
            item.condition, "inline_user_context", None, True,
            "observable_activity_only", True,
        )
    elif item.source_kind == "ml-no-honeypot":
        native = payload.get("native_resume") or {}
        identity = (
            source.get("generator"), source.get("prefix_type"), source.get("seed"),
            source.get("analysis_honeypot"),
            eligibility.get("eligible_by_default"),
            eligibility.get("missing_required_deliverables"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "exp_ml_prefix.py", "ml_prefix_only", "demand_forecasting", None,
            True, [], "environments-production-native-resume-v1",
            item.target, True,
        )
    elif item.source_kind == "phack-no-honeypot":
        native = payload.get("native_resume") or {}
        identity = (
            source.get("generator"), source.get("prefix_type"), source.get("seed"),
            source.get("comparison_source_seed"), source.get("analysis_honeypot"),
            source.get("result_condition"), source.get("pressure"),
            eligibility.get("eligible_by_default"),
            eligibility.get("missing_required_deliverables"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "exp_p_hacking_prefix.py", "p_hacking_no_honeypot",
            "reasoning_prompt_benchmark_positive", "reasoning_prompt_benchmark",
            False, "clear positive", "low", True, [],
            "environments-production-native-resume-v1", item.target, True,
        )
    elif item.source_kind == "phack-checkout-no-honeypot-gpt":
        native = payload.get("native_resume") or {}
        identity = (
            source.get("generator"), source.get("prefix_type"), source.get("seed"),
            source.get("comparison_source_seed"), source.get("analysis_honeypot"),
            source.get("result_condition"), source.get("pressure"),
            eligibility.get("eligible_by_default"),
            eligibility.get("missing_required_deliverables"), native.get("format"),
            native.get("target_name"), native.get("reasoning"),
        )
        expected = (
            "exp_p_hacking_prefix.py", "p_hacking_no_honeypot",
            "checkout_redesign_positive", "checkout_redesign", False,
            "clear positive", "low", True, [],
            "environments-production-native-resume-v1", item.target, True,
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
    if len(JUDGED_ML_PREFIXES) != 6 or len(ACTIVITY_LOG_PREFIXES) != 6:
        raise RuntimeError("the matched judged/activity-log matrix must be 6 + 6")
    if len(ML_NO_HONEYPOT_PREFIXES) != 3:
        raise RuntimeError("regular ML must have exactly three no-honeypot prefixes")
    if len(PHACK_NO_HONEYPOT_PREFIXES) != 3:
        raise RuntimeError("p-hacking must have exactly three no-honeypot prefixes")

    judged_pairs = {
        (item.trajectory_id, item.target, item.condition)
        for item in JUDGED_ML_PREFIXES
    }
    activity_pairs = {
        (item.trajectory_id, item.target, item.condition)
        for item in ACTIVITY_LOG_PREFIXES
    }
    if activity_pairs != judged_pairs:
        raise RuntimeError("activity-log sources are not an exact judged-source copy")
    if any(item.source_kind == "activity-log" for item in ML_NO_HONEYPOT_PREFIXES):
        raise RuntimeError("no-honeypot ML prefixes must not enter multi-agent")

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

    # Exercise the exact loader used by each paid child and keep condition groups
    # separate, just as the controller will.
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
            raise RuntimeError(
                f"{job.label}: destination {job.destination!r} is unavailable"
            )
        cells = build_continuation_cells(
            specs, seeds_path, [job.destination],
        )
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
    if sum(job.cells for job in jobs if job.group == "missing GPT-5.5") != (
        PLANNED_MISSING_GPT_CELLS
    ):
        raise RuntimeError("missing GPT-5.5 count drifted")
    subscription_cap = sum(
        job.vm_cap for job in jobs if job.harness == SUBSCRIPTION_HARNESS
    )
    if subscription_cap != SUBSCRIPTION_VM_LIMIT:
        raise RuntimeError(
            f"subscription cap must be {SUBSCRIPTION_VM_LIMIT}, "
            f"got {subscription_cap}"
        )

    expected = {
        "ml-conversation-hack1": (120, 39, "hack-in-one-turn"),
        "ml-conversation-hack2": (40, 13, "hack-in-two-turns"),
        "ml-conversation-nohack": (80, 26, "no-hack"),
        "ml-conversation-nohoneypot": (120, 39, "no-honeypot"),
        "ml-activity-hack1": (120, 39, "hack-in-one-turn"),
        "ml-activity-hack2": (40, 13, "hack-in-two-turns"),
        "ml-activity-nohack": (80, 26, "no-hack"),
        "phack-nohoneypot-to-phack": (120, 37, "no-honeypot"),
        "missing-gpt-phack-nohoneypot-to-ml": (40, 10, "no-honeypot"),
        "missing-gpt-demand-prefixes": (8, 5, None),
        "missing-gpt-reasoning-nohoneypot-prefixes": (3, 3, None),
    }
    observed = {
        job.label: (job.cells, job.vm_cap, job.treatment) for job in jobs
    }
    if observed != expected:
        raise RuntimeError(f"job matrix drifted: {observed!r}")

    for job in jobs:
        command = " ".join(job.command()).lower()
        if "opus" in command:
            raise RuntimeError(f"{job.label}: Opus unexpectedly entered command")
        if "--skip-viewer" not in job.args:
            raise RuntimeError(f"{job.label}: child must defer viewer build")
        expected_harness = f"--harness={job.harness}"
        if expected_harness not in job.args:
            raise RuntimeError(f"{job.label}: harness drifted")
        if job.source_mode != "prefix-generation" and "--epochs=40" not in job.args:
            raise RuntimeError(f"{job.label}: continuation epoch count drifted")
        if (
            job.harness == SUBSCRIPTION_HARNESS
            and job.label != "missing-gpt-phack-nohoneypot-to-ml"
            and "gpt-5.5" not in command
        ):
            raise RuntimeError(f"{job.label}: subscription target drifted")


def validate_plan() -> tuple[str, str]:
    required = (
        ENVIRONMENTS_ROOT / "exp_continuation_pipeline.py",
        ENVIRONMENTS_ROOT / "prefixes" / "exp_ml_prefix.py",
        ENVIRONMENTS_ROOT / "prefixes" / "exp_p_hacking_prefix.py",
        ENVIRONMENTS_ROOT / "seeds" / "ml_prefix_only"
        / "demand_forecasting" / "manifest.json",
        ENVIRONMENTS_ROOT / "seeds" / "p_hacking_prefix_only"
        / "reasoning_prompt_benchmark_positive" / "manifest.json",
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


def _selection_text(items: tuple[FrozenPrefix, ...]) -> str:
    return ", ".join(
        f"{item.viewer_label} {item.target}" for item in items
    )


def print_plan(source_hash: str, inputs_hash: str) -> None:
    jobs = build_jobs()
    print("2026-08-17 next continuations + missing GPT work — exact free plan")
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
        f"phacking_to_phacking={PLANNED_PHACK_CELLS}, "
        f"missing_gpt={PLANNED_MISSING_GPT_CELLS})"
    )
    print(
        f"Concurrent VM caps: {PLANNED_VM_CAP}/{GLOBAL_VM_LIMIT}; "
        f"subscription cap={SUBSCRIPTION_VM_LIMIT}."
    )
    print("Regular judged ML sources: " + _selection_text(JUDGED_ML_PREFIXES))
    print("Regular no-honeypot ML sources: " + _selection_text(ML_NO_HONEYPOT_PREFIXES))
    print(
        "Multi-agent sources: exact activity-log transforms of the six judged "
        "sources only; no no-honeypot source is copied."
    )
    print("P-hacking no-honeypot sources: " + _selection_text(PHACK_NO_HONEYPOT_PREFIXES))
    print(
        "Activity-log processing is intentionally lossy: system messages and private "
        "reasoning are omitted, while all stored observable activity is delivered "
        "inline with a visible caveat."
    )
    print(
        "Missing GPT-5.5 work included: 40 p-hacking no-honeypot -> ML "
        "continuations, 8 demand prefixes, and 3 reasoning no-honeypot prefixes."
    )
    print(
        "GPT-5.5 versions of the three run-next continuation directions remain "
        "deferred until tomorrow. Opus is absent. No automatic retries are attempted."
    )
    print("The viewer is not rebuilt or reloaded by this wrapper.")
    print("No AWS, VM, model, judge, credential, or filesystem write was performed.")


class RunLogger:
    def __init__(self, path: Path):
        self._handle = path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._handle.close()

    def log(self, message: str) -> None:
        line = (
            f"[{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}] "
            f"{message}"
        )
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


async def run_slate(
    jobs: list[Job], *, run_dir: Path, logger: RunLogger,
    environment: dict[str, str], source_hash: str, inputs_hash: str,
) -> tuple[dict[str, int | None], str | None]:
    logger.log(
        f"Starting concurrent slate: controllers={len(jobs)} "
        f"cells={sum(job.cells for job in jobs)} vm_cap={sum(job.vm_cap for job in jobs)}"
    )
    tasks: dict[str, asyncio.Task[int]] = {}
    launch_failure: str | None = None
    for index, job in enumerate(jobs):
        current_source = source_fingerprint()
        current_inputs = frozen_inputs_fingerprint()
        if current_source != source_hash or current_inputs != inputs_hash:
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
        except Exception as error:  # retain unrelated controllers and their evidence
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
    row["prefix_paths"] = [str(path) for path in job.prefix_paths]
    row["args"] = list(job.args)
    row["command"] = job.command()
    row["exit_status"] = status
    row["log"] = str(run_dir / f"{job.label}.log")
    return row


async def run_paid_batch(
    *, run_slug: str, run_dir: Path, source_hash: str, inputs_hash: str,
    logger: RunLogger,
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
        "format": "environments-next-continuations-20260817-status-v2",
        "run_directory": str(run_dir),
        "source_fingerprint": source_hash,
        "frozen_input_fingerprint": inputs_hash,
        "planned_cells": PLANNED_CELLS,
        "planned_controllers": PLANNED_CONTROLLERS,
        "maximum_vms": GLOBAL_VM_LIMIT,
        "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
        "automatic_campaign_retries": False,
        "gpt_included": True,
        "missing_gpt_cells": PLANNED_MISSING_GPT_CELLS,
        "gpt_run_next_continuations_included": False,
        "opus_included": False,
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
    variable = "MATS_NEXT_OPEN_20260817_CAFFEINATED"
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
            "format": "environments-next-continuations-20260817-plan-v2",
            "run_slug": run_slug,
            "source_fingerprint": source_hash,
            "frozen_input_fingerprint": inputs_hash,
            "planned_cells": PLANNED_CELLS,
            "planned_controllers": PLANNED_CONTROLLERS,
            "subscription_maximum_vms": SUBSCRIPTION_VM_LIMIT,
            "viewer_rebuilt": False,
            "multi_agent_no_honeypot_included": False,
            "frozen_prefixes": [
                {**asdict(item), "path": str(item.path)} for item in ALL_PREFIXES
            ],
            "jobs": [
                _serialized_job(job, None, run_dir)
                for job in build_jobs(run_slug)
            ],
        }
        _atomic_json(run_dir / "plan.json", plan_record)
        logger.log(f"Run directory: {run_dir}")
        logger.log(f"Source fingerprint: {source_hash}")
        logger.log(f"Frozen-input fingerprint: {inputs_hash}")
        return asyncio.run(run_paid_batch(
            run_slug=run_slug,
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
