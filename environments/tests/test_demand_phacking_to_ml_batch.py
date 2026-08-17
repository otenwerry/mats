"""Free checks for the exact 2026-08-16 prefix-to-ML wrapper."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))

import exp_demand_phacking_to_ml_20260816 as batch


def test_exact_matrix_maximizes_useful_concurrency_within_caps() -> None:
    jobs = batch.build_jobs("test-run")

    assert len(jobs) == 4
    assert sum(job.cells for job in jobs) == 192
    assert sum(job.cells for job in jobs if "generate-demand" in job.label) == 32
    assert sum(job.cells for job in jobs if "phacking" in job.label) == 160
    assert sum(job.vm_cap for job in jobs) == 162
    assert sum(
        job.vm_cap for job in jobs if job.harness == "subscription"
    ) == 18

    batch._validate_jobs(jobs)


def test_exact_commands_pin_experiment_semantics_and_exclude_opus() -> None:
    jobs = batch.build_jobs("test-run")
    commands = {job.label: job.command() for job in jobs}

    production_generation = commands["generate-demand-production"]
    assert "--targets=deepseek-v4-pro,glm-5.1,kimi-k2.6" in production_generation
    assert "--epochs=8" in production_generation
    assert "--seeds=demand_forecasting" in production_generation
    assert "--reasoning=yes" in production_generation

    for job in jobs:
        rendered = " ".join(job.command()).lower()
        assert "opus" not in rendered
        assert "--skip-viewer" in job.args
        if job.endpoint == "exp_continuation_pipeline.py":
            assert "--treatment=no-honeypot" in job.args
            assert "--seed-dir=ml_training_data_misuse" in job.args
            assert "--seeds=fraud_detection" in job.args
            assert "--epochs=40" in job.args
            assert "--time-limit=4200" in job.args

    production_continuation = next(
        job for job in jobs
        if job.label == "phacking-nohoneypot-to-fraud-production"
    )
    subscription_continuation = next(
        job for job in jobs
        if job.label == "phacking-nohoneypot-to-fraud-subscription"
    )
    assert production_continuation.cells == 3 * 40
    assert subscription_continuation.cells == 1 * 40
    assert len(production_continuation.args[1].split("=", 1)[1].split(",")) == 3
    assert len(subscription_continuation.args[1].split("=", 1)[1].split(",")) == 1


def test_one_frozen_phacking_prefix_per_agent_validates() -> None:
    assert len(batch.PHACK_PREFIXES) == 4
    assert {
        (item.target, item.epoch)
        for item in batch.PHACK_PREFIXES
    } == {
        (target, 1) for target in batch.ALL_TARGETS
    }
    batch.validate_frozen_phack_prefixes()


def test_read_only_plan_entrypoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["exp_demand_phacking_to_ml_20260816.py", "--plan"],
    )
    assert batch.main() == 0
    output = capsys.readouterr().out
    assert "paid_cells=192" in output
    assert "saved for manual inspection and are not continued" in output
    assert "viewer is not rebuilt or reloaded" in output
    assert "No AWS, VM, model, judge, credential, or filesystem write" in output


def test_paid_orchestrator_does_not_continue_demand_or_run_viewer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_labels: list[str] = []

    async def fake_stream_command(**kwargs: object) -> int:
        setup_labels.append(str(kwargs["label"]))
        return 0

    async def fake_run_wave(jobs: list[batch.Job], **_kwargs: object):
        return {job.label: 0 for job in jobs}, None

    class Logger:
        def log(self, _message: str) -> None:
            pass

    monkeypatch.setattr(batch, "STARTED_MARKER", tmp_path / "started")
    monkeypatch.setattr(batch, "stream_command", fake_stream_command)
    monkeypatch.setattr(batch, "run_wave", fake_run_wave)
    monkeypatch.setattr(batch, "_campaign_ids", lambda: set())

    assert asyncio.run(batch.run_paid_batch(
        run_slug="test-run",
        run_dir=tmp_path,
        fingerprint="test-fingerprint",
        logger=Logger(),  # type: ignore[arg-type]
    )) == 0
    assert setup_labels == ["aws-login", "aws-identity", "aws-setup"]

    status = json.loads((tmp_path / "final_status.json").read_text())
    assert status["demand_prefixes_continued"] is False
    assert status["viewer_rebuilt"] is False
    assert len(status["jobs"]) == 4
