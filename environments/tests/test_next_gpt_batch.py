from __future__ import annotations

import json
import sys
from pathlib import Path

ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))

import exp_next_gpt_20260817 as batch


def test_exact_gpt_job_matrix() -> None:
    jobs = batch.build_jobs()
    batch._validate_jobs(jobs)

    assert len(jobs) == 4
    assert sum(job.cells for job in jobs) == 240
    assert sum(job.vm_cap for job in jobs) == 18
    assert {job.harness for job in jobs} == {"subscription"}
    assert all("--epochs=40" in job.args for job in jobs)
    assert all("--skip-viewer" in job.args for job in jobs)


def test_gpt_prefix_matrix_is_frozen_and_matched() -> None:
    batch.validate_frozen_prefixes()

    assert [item.trajectory_id for item in batch.JUDGED_ML_PREFIXES] == [781, 783]
    assert [item.trajectory_id for item in batch.ACTIVITY_LOG_PREFIXES] == [781, 783]
    assert batch.ML_NO_HONEYPOT_PREFIX.viewer_label == "MLD-128"
    assert batch.PHACK_NO_HONEYPOT_PREFIX.viewer_label == "PHN-37"
    assert batch.ML_NO_HONEYPOT_PREFIX.path not in {
        path
        for job in batch.build_jobs()
        if job.source_mode == "activity-log"
        for path in job.prefix_paths
    }


def test_plan_record_jobs_are_json_serializable(tmp_path: Path) -> None:
    rows = [
        batch._serialized_job(job, None, tmp_path)
        for job in batch.build_jobs()
    ]
    encoded = json.dumps(rows, indent=2, sort_keys=True)

    assert "PosixPath" not in encoded
    assert "ml-conversation-judged-gpt" in encoded
    assert all(isinstance(path, str) for row in rows for path in row["prefix_paths"])


def test_plan_text_is_explicit_about_missing_hack_arm(capsys) -> None:
    batch.print_plan("source", "inputs")
    output = capsys.readouterr().out

    assert "paid_cells=240" in output
    assert "subscription cap=18/18" in output
    assert "no successful reward-hack trajectory" in output
    assert "Exactly 40 continuations" in output
    assert "viewer is not rebuilt" in output


def test_full_free_plan_validation() -> None:
    source_hash, inputs_hash = batch.validate_plan()

    assert len(source_hash) == 64
    assert len(inputs_hash) == 64
