from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))

import exp_elicited_gpt_glm_ml_followups_20260817 as batch


def test_exact_four_arm_matrix_uses_maximum_useful_concurrency() -> None:
    jobs = batch.build_jobs()
    batch._validate_jobs(jobs)

    assert len(jobs) == 4
    assert sum(job.cells for job in jobs) == 160
    assert sum(job.vm_cap for job in jobs) == 160
    assert all(job.cells == job.vm_cap == 40 for job in jobs)
    assert sum(
        job.vm_cap for job in jobs if job.harness == "subscription"
    ) == 80
    assert {job.source_mode for job in jobs} == {"conversation", "activity-log"}
    assert {job.treatment for job in jobs} == {"hack-in-two-turns", "no-hack"}


def test_frozen_sources_are_exact_matched_pairs() -> None:
    batch.validate_frozen_prefixes()

    assert batch.GPT_CONVERSATION.trajectory_id == 7420
    assert batch.GPT_ACTIVITY.trajectory_id == 7420
    assert batch.GPT_CONVERSATION.condition == "hack_2turn"
    assert batch.GPT_ACTIVITY.condition == "hack_2turn"
    assert batch.GLM_CONVERSATION.trajectory_id == 7764
    assert batch.GLM_ACTIVITY.trajectory_id == 7764
    assert batch.GLM_CONVERSATION.condition == "clean"
    assert batch.GLM_ACTIVITY.condition == "clean"


def test_commands_pin_destination_and_skip_viewer_and_retries() -> None:
    for job in batch.build_jobs():
        assert "--seed-dir=ml_training_data_misuse" in job.args
        assert "--seeds=fraud_detection" in job.args
        assert "--epochs=40" in job.args
        assert "--skip-viewer" in job.args
        assert "retry" not in " ".join(job.command()).lower()


def test_plan_rows_are_json_serializable(tmp_path: Path) -> None:
    rows = [
        batch._serialized_job(job, None, tmp_path) for job in batch.build_jobs()
    ]
    encoded = json.dumps(rows, indent=2, sort_keys=True)

    assert "PosixPath" not in encoded
    assert "gpt-5-5-ml-conversation-hack2" in encoded
    assert "glm-5-1-ml-activity-log-clean" in encoded


def test_plan_text_is_explicit(capsys) -> None:
    batch.print_plan("source", "inputs")
    output = capsys.readouterr().out

    assert "paid_cells=160" in output
    assert "regular_ml_to_ml=80" in output
    assert "multi_agent_ml=80" in output
    assert "subscription cap=80/100" in output
    assert "maximum useful concurrency" in output
    assert "viewer is not rebuilt or reloaded" in output


def test_paid_entrypoint_with_every_external_command_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "overnight_runs"
    campaign_root = tmp_path / "remote_campaigns"
    lock_dir = run_root / ".elicited.lock"
    started_marker = run_root / ".elicited.started"
    source_hash = "a" * 64
    inputs_hash = "b" * 64
    calls: list[str] = []

    async def fake_stream_command(**kwargs: object) -> int:
        calls.append(str(kwargs["label"]))
        return 0

    monkeypatch.setattr(sys, "argv", ["wrapper.py"])
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
    assert len(plan["jobs"]) == 4
    assert final["success"] is True
    assert final["planned_cells"] == 160
    assert calls[:4] == [
        "codex-subscription-auth", "aws-login", "aws-identity", "aws-setup",
    ]
    assert set(calls[4:]) == {job.label for job in batch.build_jobs()}
    assert started_marker.read_text().strip() == str(run_dirs[0])
    assert not lock_dir.exists()


def test_full_free_validation() -> None:
    source_hash, inputs_hash = batch.validate_plan()

    assert len(source_hash) == 64
    assert len(inputs_hash) == 64
