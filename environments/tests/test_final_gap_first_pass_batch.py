from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))

import exp_final_gap_first_pass_20260817 as batch


def test_exact_matrix_fills_global_and_subscription_caps() -> None:
    jobs = batch.build_jobs()
    batch._validate_jobs(jobs)

    assert len(jobs) == 11
    assert sum(job.cells for job in jobs) == 460
    assert sum(job.vm_cap for job in jobs) == 250
    assert sum(
        job.vm_cap for job in jobs if job.harness == batch.SUBSCRIPTION
    ) == 100
    assert sum(
        job.cells for job in jobs
        if job.group == "corrected p-hacking no-honeypot -> ML"
    ) == 160
    assert sum(job.cells for job in jobs if job.group == "GPT core") == 160
    assert sum(
        job.cells for job in jobs if job.group == "Kimi two-turn completion"
    ) == 80
    assert sum(
        job.cells for job in jobs
        if job.group == "ordinary judged prefix elicitation"
    ) == 60


def test_prefix_selection_and_conditions_are_exact() -> None:
    batch.validate_frozen_prefixes()

    assert [item.viewer_label for item in batch.PHACK_NO_HONEYPOT_PREFIXES] == [
        "PHN-25", "PHN-27", "PHN-33", "PHN-37",
    ]
    assert {
        item.target for item in batch.PHACK_NO_HONEYPOT_PREFIXES
    } == {"deepseek-v4-pro", "glm-5.1", "kimi-k2.6", "gpt-5.5"}
    assert batch.GPT_CLEAN_ML_PREFIX.trajectory_id == 783
    assert batch.GPT_CLEAN_ACTIVITY_PREFIX.trajectory_id == 783
    assert batch.GPT_NO_HONEYPOT_ML_PREFIX.viewer_label == "MLD-136"
    assert batch.KIMI_TWO_TURN_ML_PREFIX.trajectory_id == 842
    assert batch.KIMI_TWO_TURN_ACTIVITY_PREFIX.trajectory_id == 842
    assert all(item.trajectory_id != 781 for item in batch.ALL_PREFIXES)


def test_commands_do_not_auto_continue_elicitations_or_rebuild_viewer() -> None:
    jobs = batch.build_jobs()
    continuation = [job for job in jobs if job.prefix_paths]
    elicitation = [job for job in jobs if not job.prefix_paths]

    assert len(continuation) == 8
    assert len(elicitation) == 3
    assert all("--epochs=40" in job.args for job in continuation)
    assert all("--epochs=20" in job.args for job in elicitation)
    assert all("--skip-viewer" in job.args for job in jobs)
    assert all("retry" not in " ".join(job.command()).lower() for job in jobs)
    assert all(job.endpoint == "exp_real_audit_pipeline.py" for job in elicitation)


def test_plan_rows_are_json_serializable(tmp_path: Path) -> None:
    rows = [
        batch._serialized_job(job, None, tmp_path) for job in batch.build_jobs()
    ]
    encoded = json.dumps(rows, indent=2, sort_keys=True)

    assert "PosixPath" not in encoded
    assert "correct-phack-nohoneypot-to-ml-production" in encoded


def test_plan_text_states_the_scientific_boundaries(capsys) -> None:
    batch.print_plan("source", "inputs")
    output = capsys.readouterr().out

    assert "paid_cells=460" in output
    assert "subscription cap=100" in output
    assert "GPT notable #781 is deliberately excluded" in output
    assert "These are not auto-continued" in output
    assert "Kimi Wikipedia is omitted" in output
    assert "viewer is not rebuilt or reloaded" in output


def test_paid_entrypoint_with_every_external_command_mocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "overnight_runs"
    campaign_root = tmp_path / "remote_campaigns"
    lock_dir = run_root / ".final-gap.lock"
    started_marker = run_root / ".final-gap.started"
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
    assert len(plan["jobs"]) == 11
    assert final["success"] is True
    assert final["planned_cells"] == 460
    assert started_marker.read_text().strip() == str(run_dirs[0])
    assert calls[:4] == [
        "codex-subscription-auth", "aws-login", "aws-identity", "aws-setup",
    ]
    assert set(calls[4:]) == {job.label for job in batch.build_jobs()}
    assert not lock_dir.exists()


def test_full_free_validation() -> None:
    source_hash, inputs_hash = batch.validate_plan()

    assert len(source_hash) == 64
    assert len(inputs_hash) == 64
