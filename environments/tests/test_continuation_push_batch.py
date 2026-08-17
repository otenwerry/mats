from __future__ import annotations

import exp_continuation_push_20260815 as batch


def test_exact_push_matrix_totals_and_global_caps() -> None:
    jobs = batch.build_jobs("test-run")

    assert len(jobs) == 20
    assert sum(job.cells for job in jobs) == 988
    assert sum(job.vm_cap for job in jobs) == 220
    assert sum(job.vm_cap for job in jobs if job.harness == "subscription") == 18
    assert all("opus" not in " ".join(job.command()).lower() for job in jobs)
    batch.validate_job_matrix(jobs)


def test_exact_gap_fill_does_not_repeat_successful_cells() -> None:
    jobs = {job.label: job for job in batch.build_jobs("test-run")}

    expected = {
        "wiki-ml-fill": 46,
        "phack-ml-hack1-production-fill": 5,
        "phack-ml-hack2-production-fill": 37,
        "phack-ml-hack2-gpt-fill": 1,
        "phack-ml-nohack-production-fill": 6,
        "phack-ml-nohack-gpt-fill": 1,
    }
    assert {label: jobs[label].cells for label in expected} == expected
    for label in expected:
        command = jobs[label].command()
        assert any(arg.startswith("--retry-failed=") for arg in command)
        assert "--retry-pipeline-failures" in command


def test_fresh_source_condition_matrices() -> None:
    jobs = {job.label: job for job in batch.build_jobs("test-run")}

    assert jobs["inline-phack-hack1-production"].cells == 120
    assert jobs["inline-phack-hack2-production"].cells == 120
    assert jobs["inline-phack-hack2-gpt"].cells == 40
    assert jobs["inline-phack-nohack-production"].cells == 80
    assert jobs["inline-phack-nohack-gpt"].cells == 40

    assert jobs["multi-ml-hack1-production"].cells == 120
    assert jobs["multi-ml-hack2-production"].cells == 120
    assert jobs["multi-ml-hack2-gpt"].cells == 40
    assert jobs["multi-ml-nohack-production"].cells == 120
    assert jobs["multi-ml-nohack-gpt"].cells == 40

    assert jobs["generate-ml-demand-production"].cells == 30
    assert jobs["generate-ml-demand-gpt"].cells == 10
    assert jobs["generate-phack-nohoneypot-production"].cells == 9
    assert jobs["generate-phack-nohoneypot-gpt"].cells == 3


def test_subscription_fresh_jobs_are_gpt_only() -> None:
    jobs = batch.build_jobs("test-run")

    for job in jobs:
        if job.harness != "subscription" or job.retry_parent:
            continue
        command = " ".join(job.command())
        referenced_ids = [
            trajectory_id
            for trajectory_id in batch.PREFIX_INPUTS
            if f"={trajectory_id}" in command
            or f",{trajectory_id}" in command
            or f"traj{trajectory_id}-" in command
        ]
        if "--targets=gpt-5.5" not in command:
            assert referenced_ids
            assert {
                batch.PREFIX_INPUTS[trajectory_id].target
                for trajectory_id in referenced_ids
            } == {"gpt-5.5"}


def test_frozen_local_sources_and_retry_selections_validate() -> None:
    fingerprint, transport_hashes = batch.validate_plan(
        batch.build_jobs("RUN-SLUG")
    )

    assert len(fingerprint) == 64
    assert set(transport_hashes) == {
        2852, 2856, 2858, 2893, 2917, 2918,
        2933, 2940, 2946, 2972, 2997,
    }
