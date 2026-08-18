"""Free checks for the exact 2026-08-17 overnight wrapper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))

import exp_next_open_20260817 as batch


def test_exact_matrix_uses_full_global_vm_cap() -> None:
    jobs = batch.build_jobs("test-run")

    assert len(jobs) == 11
    assert sum(job.cells for job in jobs) == 771
    assert sum(job.vm_cap for job in jobs) == 250
    assert {job.harness for job in jobs} == {"production", "subscription"}
    assert sum(job.vm_cap for job in jobs if job.harness == "subscription") == 18
    assert sum(
        job.cells for job in jobs if job.group == "regular ML -> ML"
    ) == 360
    assert sum(
        job.cells for job in jobs if job.group == "multi-agent ML -> ML"
    ) == 240
    assert sum(
        job.cells for job in jobs if job.group.startswith("p-hacking")
    ) == 120
    assert sum(job.cells for job in jobs if job.group == "missing GPT-5.5") == 51

    batch._validate_jobs(jobs)


def test_regular_and_activity_log_condition_counts_are_pinned() -> None:
    jobs = {job.label: job for job in batch.build_jobs("test-run")}

    assert jobs["ml-conversation-hack1"].cells == 120
    assert jobs["ml-conversation-hack2"].cells == 40
    assert jobs["ml-conversation-nohack"].cells == 80
    assert jobs["ml-conversation-nohoneypot"].cells == 120
    assert jobs["ml-activity-hack1"].cells == 120
    assert jobs["ml-activity-hack2"].cells == 40
    assert jobs["ml-activity-nohack"].cells == 80
    assert "ml-activity-nohoneypot" not in jobs

    judged = {
        (item.trajectory_id, item.target, item.condition)
        for item in batch.JUDGED_ML_PREFIXES
    }
    activity = {
        (item.trajectory_id, item.target, item.condition)
        for item in batch.ACTIVITY_LOG_PREFIXES
    }
    assert activity == judged
    assert len(activity) == 6


def test_commands_pin_destinations_and_include_only_missing_gpt_work() -> None:
    jobs = batch.build_jobs("test-run")

    for job in jobs:
        rendered = " ".join(job.command()).lower()
        assert "--skip-viewer" in job.args
        assert "opus" not in rendered
        assert f"--harness={job.harness}" in job.args

    ml_jobs = [
        job for job in jobs
        if job.destination == "fraud_detection" and job.prefix_paths
    ]
    assert len(ml_jobs) == 8
    assert all("--seed-dir=ml_training_data_misuse" in job.args for job in ml_jobs)
    assert all("--seeds=fraud_detection" in job.args for job in ml_jobs)
    assert all("--time-limit=4200" in job.args for job in ml_jobs)

    phack = next(job for job in jobs if job.destination == "checkout_redesign")
    assert "--seed-dir=p_hacking" in phack.args
    assert "--seeds=checkout_redesign" in phack.args
    assert "--pressure=low" in phack.args
    assert "--time-limit=1800" in phack.args

    gpt = [job for job in jobs if job.harness == "subscription"]
    assert len(gpt) == 3
    assert sum(job.cells for job in gpt) == 51
    assert all(
        "gpt-5.5" in " ".join(job.command()).lower()
        for job in gpt if job.source_mode == "prefix-generation"
    )
    continuation = next(
        job for job in gpt if job.label == "missing-gpt-phack-nohoneypot-to-ml"
    )
    assert continuation.prefix_paths == (batch.GPT_PHACK_NO_HONEYPOT_PREFIX.path,)
    assert {job.label for job in gpt} == {
        "missing-gpt-phack-nohoneypot-to-ml",
        "missing-gpt-demand-prefixes",
        "missing-gpt-reasoning-nohoneypot-prefixes",
    }


def test_every_frozen_payload_and_loader_path_validates() -> None:
    batch.validate_frozen_prefixes()
    assert len(batch.frozen_inputs_fingerprint()) == 64


def test_paid_plan_manifest_jobs_are_json_serializable(tmp_path: Path) -> None:
    jobs = [
        batch._serialized_job(job, None, tmp_path)
        for job in batch.build_jobs("test-run")
    ]
    rendered = json.dumps({"jobs": jobs})
    assert "PosixPath" not in rendered
    assert str(batch.JUDGED_ML_PREFIXES[0].path) in rendered


def test_paid_entrypoint_completes_with_all_external_commands_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "overnight_runs"
    campaign_root = tmp_path / "remote_campaigns"
    lock_dir = run_root / ".next-open.lock"
    started_marker = run_root / ".next-open.started"
    source_hash = "a" * 64
    inputs_hash = "b" * 64
    calls: list[str] = []

    async def fake_stream_command(**kwargs: object) -> int:
        calls.append(str(kwargs["label"]))
        return 0

    monkeypatch.setattr(sys, "argv", ["exp_next_open_20260817.py"])
    monkeypatch.setattr(batch, "RUN_ROOT", run_root)
    monkeypatch.setattr(batch, "CAMPAIGN_ROOT", campaign_root)
    monkeypatch.setattr(batch, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(batch, "STARTED_MARKER", started_marker)
    monkeypatch.setattr(batch, "LAUNCH_STAGGER_SECONDS", 0)
    monkeypatch.setattr(batch, "validate_plan", lambda: (source_hash, inputs_hash))
    monkeypatch.setattr(batch, "source_fingerprint", lambda: source_hash)
    monkeypatch.setattr(batch, "frozen_inputs_fingerprint", lambda: inputs_hash)
    monkeypatch.setattr(batch, "_maybe_caffeinate", lambda _args: None)
    monkeypatch.setattr(batch, "stream_command", fake_stream_command)

    assert batch.main() == 0

    run_dirs = [path for path in run_root.iterdir() if not path.name.startswith(".")]
    assert len(run_dirs) == 1
    plan = json.loads((run_dirs[0] / "plan.json").read_text())
    final = json.loads((run_dirs[0] / "final_status.json").read_text())
    assert len(plan["jobs"]) == 11
    assert final["success"] is True
    assert len(final["jobs"]) == 11
    assert final["planned_cells"] == 771
    assert started_marker.read_text().strip() == str(run_dirs[0])
    assert calls[:3] == ["aws-login", "aws-identity", "aws-setup"]
    assert set(calls[3:]) == {job.label for job in batch.build_jobs("test-run")}
    assert not lock_dir.exists()


def test_read_only_plan_entrypoint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["exp_next_open_20260817.py", "--plan"])
    monkeypatch.setattr(batch, "source_fingerprint", lambda: "a" * 64)

    assert batch.main() == 0
    output = capsys.readouterr().out
    assert "paid_cells=771" in output
    assert "regular_ml_to_ml=360" in output
    assert "multi_agent_ml=240" in output
    assert "phacking_to_phacking=120" in output
    assert "missing_gpt=51" in output
    assert "subscription cap=18" in output
    assert "Missing GPT-5.5 work included" in output
    assert "remain deferred until tomorrow" in output
    assert "no no-honeypot source is copied" in output
    assert "viewer is not rebuilt or reloaded" in output
    assert "No AWS, VM, model, judge, credential, or filesystem write" in output
