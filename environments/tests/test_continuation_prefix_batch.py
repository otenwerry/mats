"""Free tests for the unattended purpose-built continuation batch."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS))
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

import exp_continuation_prefix_batch as batch  # noqa: E402


def default_cfg() -> dict:
    return {
        "epochs": 40,
        "vm_concurrency": 75,
        "subscription_vm_concurrency": 75,
        "infrastructure_retries": 1,
        "treatment": "purpose-built-prefix",
        "aws_region": "us-west-2",
        "aws_region_explicit": False,
        "aws_instance_type": "c7a.xlarge",
        "aws_instance_type_explicit": False,
        "aws_bucket": None,
        "aws_bucket_explicit": False,
        "aws_secret_env": [],
        "resume_batch": None,
        "native_only": False,
        "skip_viewer": False,
        "dry_run": False,
    }


def _payload(prefix_type: str, model: str, harness: str, generated: str) -> dict:
    source = {
        "kind": "external",
        "harness": harness,
        "generated_at": generated,
    }
    if prefix_type == "natural_questions":
        source.update({
            "generator": "exp_nq_prefix.py",
            "target_context_tokens": 2_000,
            "rng_seed": 1234,
            "reached_target_tokens": True,
        })
    else:
        source["prefix_type"] = prefix_type
    return {
        "format": "environments-continuation-prefix-v1",
        "name": f"{prefix_type}-{model}-{harness}-{generated}",
        "target": model,
        "reasoning": True,
        "source": source,
        "messages": [],
    }


def _selection(tmp_path: Path) -> dict:
    files = {}
    for prefix_type in batch.PREFIX_TYPES:
        for model, native_harness in batch.MODEL_HARNESSES:
            for harness in ("simple", native_harness):
                path = tmp_path / f"{prefix_type}-{model}-{harness}.json"
                path.write_text("{}")
                files[f"{prefix_type}/{model}/{harness}"] = str(path)
    return {"files": files, "selected_at": "now"}


def test_default_stages_are_ordered_and_cover_1600_trajectories(
    tmp_path: Path,
) -> None:
    stages = batch.build_stages(default_cfg(), _selection(tmp_path))

    assert [stage.key for stage in stages] == list(batch.STAGE_ORDER)
    assert [len(stage.prefix_files) for stage in stages] == [20, 12, 8]
    assert [stage.trajectories for stage in stages] == [800, 480, 320]
    assert [stage.vm_concurrency for stage in stages] == [75, 75, 75]
    assert sum(stage.trajectories for stage in stages) == 1_600


def test_native_only_stages_run_production_then_subscription(
    tmp_path: Path,
) -> None:
    cfg = {**default_cfg(), "native_only": True}

    stages = batch.build_stages(cfg, _selection(tmp_path))

    assert [stage.key for stage in stages] == ["production", "subscription"]
    assert [len(stage.prefix_files) for stage in stages] == [12, 8]
    assert [stage.trajectories for stage in stages] == [480, 320]
    assert sum(stage.trajectories for stage in stages) == 800


def test_discovery_selects_newest_exact_payload_and_requires_all_40(
    tmp_path: Path,
) -> None:
    for prefix_type in batch.PREFIX_TYPES:
        for model, native_harness in batch.MODEL_HARNESSES:
            for harness in ("simple", native_harness):
                old = tmp_path / f"old-{prefix_type}-{model}-{harness}.json"
                new = tmp_path / f"new-{prefix_type}-{model}-{harness}.json"
                old.write_text(json.dumps(
                    _payload(prefix_type, model, harness, "2026-01-01T00:00:00Z")
                ))
                new.write_text(json.dumps(
                    _payload(prefix_type, model, harness, "2026-01-02T00:00:00Z")
                ))

    selection = batch.discover_prefixes(tmp_path)

    assert len(selection["files"]) == 40
    assert all(Path(path).name.startswith("new-") for path in selection["files"].values())


def _cell(prefix: str, epoch: int, *, ok: bool, status: str = "completed") -> dict:
    return {
        "prefix_name": prefix,
        "seed": "checkout_redesign",
        "original_epoch": epoch,
        "status": status,
        "terminal": {"pipeline_exit_code": 0 if ok else 1},
    }


def test_stage_summary_replaces_infrastructure_failure_with_retry() -> None:
    first = {
        "cells": [
            _cell("p", 1, ok=True),
            _cell("p", 2, ok=False, status="infrastructure_failure"),
        ]
    }
    retry = {"cells": [_cell("p", 2, ok=True)]}

    assert batch._stage_summary([first, retry], 2) == {
        "expected": 2,
        "accounted_for": 2,
        "succeeded": 2,
        "trajectory_failures": 0,
        "infrastructure_failures": 0,
        "campaigns": 2,
    }


def test_campaign_cells_rotate_prefixes_before_advancing_epoch() -> None:
    class Spec:
        def __init__(self, name: str, target: str) -> None:
            self.name = name
            self.target_name = target
            self.target_model = f"provider/{target}"
            self.reasoning = True
            self.payload_path = Path(f"/{name}.json")
            self.sha256 = f"sha-{name}"

    stage = batch.Stage(
        key="subscription",
        harness="subscription",
        prefix_files=(Path("/opus.json"), Path("/gpt.json")),
        vm_concurrency=75,
        epochs=2,
    )
    specs = [Spec("opus-prefix", "opus-4.6"), Spec("gpt-prefix", "gpt-5.5")]
    with (
        patch.object(batch, "sha256_file", return_value="file-sha"),
        patch.object(batch, "resolve_target", side_effect=lambda value: f"provider/{value}"),
        patch.object(batch, "resolve_judge", return_value="judge/model"),
        patch.object(batch, "resolve_gate_model", return_value="gate/model"),
    ):
        cfg = batch._campaign_cfg(default_cfg(), stage, specs, campaign_id="campaign")

    assert cfg["_cell_selections"] == [
        ("opus-prefix", "checkout_redesign", 1),
        ("gpt-prefix", "checkout_redesign", 1),
        ("opus-prefix", "checkout_redesign", 2),
        ("gpt-prefix", "checkout_redesign", 2),
    ]


def test_resumed_stage_includes_cells_from_prior_campaigns(tmp_path: Path) -> None:
    cfg = default_cfg()
    stage = batch.Stage(
        key="simple",
        harness="simple",
        prefix_files=(tmp_path / "one.json", tmp_path / "two.json"),
        vm_concurrency=2,
        epochs=1,
    )
    first = {
        "campaign_id": "first",
        "cells": [_cell("one", 1, ok=True)],
    }
    resumed = {
        "campaign_id": "second",
        "cells": [_cell("two", 1, ok=True)],
    }
    (tmp_path / "first.json").write_text(json.dumps(first))
    (tmp_path / "second.json").write_text(json.dumps({
        "campaign_id": "second",
        "cells": [_cell("two", 1, ok=False, status="infrastructure_failure")],
    }))
    record = {
        "status": "failed",
        "started_at": "earlier",
        "campaign_ids": ["first", "second"],
    }
    manifest = {"batch_id": "batch", "stages": {"simple": record}}
    manifest_path = tmp_path / "batch_manifest.json"
    batch._write_manifest(manifest_path, manifest)

    with (
        patch.object(
            batch,
            "_campaign_state_path",
            side_effect=lambda cid: tmp_path / f"{cid}.json",
        ),
        patch.object(batch, "resume_campaign", return_value=resumed) as resume,
        patch.object(batch, "retry_failed") as retry,
    ):
        batch._run_stage(cfg, stage, [], record, manifest, manifest_path)

    resume.assert_called_once()
    retry.assert_not_called()
    assert record["status"] == "succeeded"
    assert record["summary"]["succeeded"] == 2
    assert record["summary"]["accounted_for"] == 2


def test_dry_run_writes_nothing_and_never_calls_aws(tmp_path: Path) -> None:
    cfg = {**default_cfg(), "dry_run": True}
    stages = batch.build_stages(cfg, _selection(tmp_path))
    argv = ["exp_continuation_prefix_batch.py", "--dry-run"]
    with (
        patch.object(sys, "argv", argv),
        patch.object(batch, "BATCH_ROOT", tmp_path / "batches"),
        patch.object(batch, "discover_prefixes", return_value=_selection(tmp_path)),
        patch.object(batch, "validate_stages", return_value={}),
        patch.object(batch, "build_stages", return_value=stages),
        patch.object(batch, "run_campaign") as run_campaign,
    ):
        batch.main()

    run_campaign.assert_not_called()
    assert not (tmp_path / "batches").exists()
