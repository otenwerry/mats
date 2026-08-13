"""Free tests for the purpose-built prefix batch."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_prefix_batch as batch  # noqa: E402


def default_cfg() -> dict:
    return {
        "nq_tokens": 2_000,
        "nq_seed": 1234,
        "reasoning": True,
        "concurrency": 20,
        "simple_only": False,
        "resume_batch": None,
        "also_simple": False,
        "skip_viewer": False,
        "dry_run": False,
    }


def test_default_matrix_is_four_prefixes_by_five_models() -> None:
    jobs = batch.build_jobs(default_cfg(), python="python-test")
    assert len(jobs) == 20
    assert len({(job.prefix, job.model) for job in jobs}) == 20
    assert sum(job.harness == "production" for job in jobs) == 12
    assert sum(job.harness == "subscription" for job in jobs) == 8
    for job in jobs:
        expected_harness = dict(batch.MODEL_HARNESSES)[job.model]
        assert job.harness == expected_harness
        assert f"--harness={expected_harness}" in job.command
        assert "--reasoning=yes" in job.command


def test_only_nq_jobs_receive_tokens_and_seed() -> None:
    cfg = {**default_cfg(), "nq_tokens": 42_000, "nq_seed": 99}
    jobs = batch.build_jobs(cfg, python="python-test")
    for job in jobs:
        if job.prefix == "natural-questions":
            assert "--tokens=42000" in job.command
            assert "--seed=99" in job.command
        else:
            assert not any(value.startswith("--tokens=") for value in job.command)
            assert not any(value.startswith("--seed=") for value in job.command)


def test_simple_only_routes_all_twenty_jobs_to_simple() -> None:
    cfg = {**default_cfg(), "simple_only": True}
    jobs = batch.build_jobs(cfg, python="python-test")
    assert len(jobs) == 20
    assert {job.harness for job in jobs} == {"simple"}
    assert all("--harness=simple" in job.command for job in jobs)


def _interrupted_manifest(tmp_path: Path) -> Path:
    jobs = batch.build_jobs(default_cfg(), python="python-test")
    records = {}
    for job in jobs:
        unfinished = job.prefix == "natural-questions" or (
            job.model == "kimi-k2.6" and job.prefix != "natural-questions"
        )
        payload = None
        status = "pending" if unfinished else "succeeded"
        if not unfinished:
            payload_path = tmp_path / f"{job.key}.json"
            payload_path.write_text("{}")
            payload = str(payload_path)
        records[job.key] = {
            "prefix": job.prefix,
            "model": job.model,
            "harness": job.harness,
            "status": status,
            "payload_file": payload,
        }
    manifest_path = tmp_path / "interrupted" / "batch_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps({
        "format": "environments-prefix-batch-v1",
        "jobs": records,
    }))
    return manifest_path


def test_resume_plus_simple_selects_eight_unfinished_and_twenty_simple(
    tmp_path: Path,
) -> None:
    manifest_path = _interrupted_manifest(tmp_path)
    cfg = {
        **default_cfg(),
        "resume_batch": str(manifest_path),
        "also_simple": True,
        "concurrency": 5,
    }
    jobs = batch.build_jobs(cfg, python="python-test")

    assert len(jobs) == 28
    assert len({job.key for job in jobs}) == 28
    mixed = [job for job in jobs if job.harness != "simple"]
    simple = [job for job in jobs if job.harness == "simple"]
    assert len(mixed) == 8
    assert len(simple) == 20
    assert sum(job.prefix == "natural-questions" for job in mixed) == 5
    assert {
        job.prefix for job in mixed if job.model == "kimi-k2.6"
    } == {"natural-questions", "science-ethics", "general-ethics", "move-fast"}
    assert all(
        "--tokens=2000" in job.command
        for job in jobs
        if job.prefix == "natural-questions"
    )


def test_resume_refuses_to_skip_a_succeeded_job_with_no_payload(
    tmp_path: Path,
) -> None:
    manifest_path = _interrupted_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    succeeded = next(
        record
        for record in manifest["jobs"].values()
        if record["status"] == "succeeded"
    )
    Path(succeeded["payload_file"]).unlink()
    cfg = {**default_cfg(), "resume_batch": str(manifest_path)}

    try:
        batch.build_jobs(cfg, python="python-test")
    except SystemExit as error:
        assert "payload file is missing" in str(error)
    else:
        raise AssertionError("missing successful payload was not rejected")


def test_dry_run_does_not_create_a_batch_or_start_children(tmp_path: Path) -> None:
    argv = ["exp_prefix_batch.py", "--dry-run"]
    with (
        patch.object(sys, "argv", argv),
        patch.object(batch, "PREFIX_BATCH_ROOT", tmp_path / "batches"),
        patch.object(batch, "run_batch") as run_batch,
        patch.object(batch, "rebuild_viewer") as rebuild_viewer,
    ):
        batch.main()
    run_batch.assert_not_called()
    rebuild_viewer.assert_not_called()
    assert not (tmp_path / "batches").exists()


def test_batch_writes_logs_payload_paths_and_aggregate_cost(tmp_path: Path) -> None:
    paid_payload = tmp_path / "paid.json"
    paid_payload.write_text(json.dumps({
        "source": {
            "generation_cost": {
                "cost_usd": 1.25,
                "exact": True,
                "source": "test",
            }
        }
    }))
    subscription_payload = tmp_path / "subscription.json"
    subscription_payload.write_text(json.dumps({
        "source": {
            "generation_cost": {
                "cost_usd": None,
                "exact": False,
                "source": "subscription_not_metered",
            }
        }
    }))

    def job(name: str, payload: Path, harness: str) -> batch.Job:
        return batch.Job(
            prefix=name,
            entrypoint="test.py",
            model="test-model",
            harness=harness,
            command=(
                sys.executable,
                "-c",
                f"print('payload: {payload}')",
            ),
        )

    jobs = [
        job("paid", paid_payload, "production"),
        job("included", subscription_payload, "subscription"),
    ]
    cfg = {**default_cfg(), "concurrency": 2}
    with (
        patch.object(batch, "PREFIX_BATCH_ROOT", tmp_path / "batches"),
        patch.object(batch, "_batch_id", return_value="test-batch"),
    ):
        manifest, manifest_path = asyncio.run(batch.run_batch(cfg, jobs))

    assert manifest_path.is_file()
    assert manifest["summary"]["status_counts"] == {"succeeded": 2}
    assert manifest["summary"]["recorded_cost_usd"] == 1.25
    assert manifest["summary"]["included_subscription_jobs"] == 1
    assert all(
        Path(record["log_file"]).is_file()
        for record in manifest["jobs"].values()
    )
