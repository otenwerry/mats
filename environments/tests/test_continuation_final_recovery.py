"""Free exact-plan tests for the final 2026-08-16 recovery wrapper."""

from __future__ import annotations

from pathlib import Path
import sys


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_continuation_final_recovery_20260816 as recovery  # noqa: E402


def test_final_recovery_plan_is_exact_and_read_only(capsys) -> None:
    jobs = recovery.build_jobs()
    fingerprint = recovery.validate_plan(jobs)

    assert len(jobs) == 3
    assert sum(job.cells for job in jobs) == 46
    assert sum(job.vm_cap for job in jobs) == 24
    assert [job.cells for job in jobs] == [40, 1, 5]
    assert all(job.harness == "production" for job in jobs)
    assert all("--skip-viewer" in job.args for job in jobs)
    assert all("--retry-failed=" in " ".join(job.args) for job in jobs)
    assert "--retry-pipeline-failures" in jobs[0].args
    assert "--retry-pipeline-failures" in jobs[1].args
    assert "--retry-pipeline-failures" not in jobs[2].args

    recovery.print_plan(jobs, fingerprint)
    output = capsys.readouterr().out
    assert "trajectories=46" in output
    assert "subscription_caps=0" in output
    assert "40 Kimi OpenCode retries + 1 GLM timeout + 5 ML prefixes" in output
    assert "No AWS, VM, model, judge, credential, or filesystem write" in output
