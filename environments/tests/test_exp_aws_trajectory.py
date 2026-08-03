"""Free tests for one-VM-per-trajectory AWS orchestration (no AWS calls)."""

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ENVIRONMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS / "lib"))

from exp_aws_trajectory import (  # noqa: E402
    AGENT_TIME_LIMIT_SECONDS,
    AMI_BUILDER_WATCHDOG_MINUTES,
    AWS_CLI_VERSION,
    FAILURE_PACKAGE_SECONDS,
    UNCONDITIONAL_TERMINATION_SECONDS,
    PERSONAL_REIMBURSEMENT_FUNDING,
    AwsTrajectoryError,
    _finalize_campaign_import,
    _iam_policy,
    _campaign_id,
    _builder_user_data,
    _monitor_campaign,
    _secure_extract_result,
    _vm_cost,
    _worker_user_data,
    account_id,
    build_cells,
    build_runtime_ami,
    build_source_bundle,
    campaign_ok,
    delete_campaign_s3_objects,
    ensure_bucket,
    import_campaign_results,
    preflight_aws,
    put_api_keys,
    retry_failed,
    run_campaign,
    setup_aws,
)


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


class SourceBundleTests(unittest.TestCase):
    def test_current_source_bytes_are_bundled_without_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "mats"
            environments = root / "environments"
            environments.mkdir(parents=True)
            git(root, "init")
            (root / "tracked.py").write_text("old\n")
            (root / ".env").write_text("OPENAI_API_KEY=secret\n")
            (root / "credentials").write_text("secret\n")
            git(root, "add", "tracked.py", ".env", "credentials")
            # The bundle must read the dirty working-tree bytes, not the git index.
            (root / "tracked.py").write_text("dirty current bytes\n")
            (root / "untracked.py").write_text("new source\n")
            (root / ".venv").mkdir()
            (root / ".venv" / "cache.py").write_text("cache\n")

            first = build_source_bundle(environments, Path(tmp) / "out1")
            second = build_source_bundle(environments, Path(tmp) / "out2")

            self.assertEqual(first["sha256"], second["sha256"])
            with tarfile.open(first["path"], "r:gz") as archive:
                names = set(archive.getnames())
                tracked = archive.extractfile("mats/tracked.py")
                untracked = archive.extractfile("mats/untracked.py")
                self.assertEqual(tracked.read().decode(), "dirty current bytes\n")
                self.assertEqual(untracked.read().decode(), "new source\n")
            self.assertNotIn("mats/.env", names)
            self.assertNotIn("mats/credentials", names)
            self.assertFalse(any(".venv" in name for name in names))


class CellPlanningTests(unittest.TestCase):
    def make_cfg(self):
        return {
            "targets": ["qwen3-32b", "deepseek-v4-pro"],
            "target_models": [
                "openrouter/qwen/qwen3-32b",
                "openrouter/deepseek/deepseek-v4-pro-20260423",
            ],
            "seeds": ["fraud_detection"],
            "seeds_path": "/repo/seeds/ml_training_data_misuse",
            "epochs": 2,
            "reasoning": True,
            "condition": "allow",
            "judge_resolved": "openai/gpt-5.6-luna",
            "gate_model": "openai/gpt-5.6-luna",
            "time_limit": AGENT_TIME_LIMIT_SECONDS,
            "aws_region": "us-west-2",
            "aws_instance_type": "c7a.xlarge",
            "aws_secret_env": [],
        }

    def test_one_unique_worker_cell_per_target_seed_epoch(self):
        cells = build_cells(
            self.make_cfg(),
            campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket",
            hourly_price=0.2,
        )

        self.assertEqual(len(cells), 4)
        self.assertEqual(len({cell["cell_id"] for cell in cells}), 4)
        self.assertTrue(all("--epochs=1" in cell["pipeline_args"] for cell in cells))
        self.assertTrue(all("--compute=local" in cell["pipeline_args"] for cell in cells))
        self.assertTrue(all(cell["task_name"].endswith(cell["cell_id"]) for cell in cells))

    def test_concurrent_campaign_ids_cannot_collide_within_one_second(self):
        cfg = self.make_cfg()

        self.assertNotEqual(_campaign_id(cfg), _campaign_id(cfg))

    def test_explicit_retry_selection_does_not_cross_product(self):
        cfg = self.make_cfg()
        cfg["_cell_selections"] = [
            ("qwen3-32b", "fraud_detection", 2),
        ]

        cells = build_cells(
            cfg, campaign_id="retry", source={"sha256": "b" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["original_epoch"], 2)

    def test_worker_has_independent_watchdogs_and_auto_shutdown(self):
        cell = build_cells(
            self.make_cfg(), campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        user_data = _worker_user_data(cell)

        self.assertIn(f"timeout --signal=TERM --kill-after=60 {FAILURE_PACKAGE_SECONDS}", user_data)
        self.assertIn(
            f"shutdown -h +{UNCONDITIONAL_TERMINATION_SECONDS // 60}", user_data
        )
        self.assertIn("shutdown -h now", user_data)
        self.assertIn("disable --now mats-ami-builder-watchdog.timer", user_data)
        self.assertIn("--failure-reason=source_checksum_failed", user_data)
        self.assertIn("--failure-reason=worker_watchdog_expired", user_data)
        self.assertNotIn("API_KEY", user_data)
        syntax = subprocess.run(
            ["bash", "-n"], input=user_data, text=True, capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_ami_builder_bootstrap_is_valid_bash(self):
        user_data = _builder_user_data(
            bucket="bucket", region="us-west-2", source_key="source",
            source_sha="a" * 64, runtime="b" * 64,
        )

        syntax = subprocess.run(
            ["bash", "-n"], input=user_data, text=True, capture_output=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        self.assertLessEqual(len(user_data.encode()), 16 * 1024)
        self.assertIn(f"shutdown -h +{AMI_BUILDER_WATCHDOG_MINUTES}", user_data)
        self.assertIn("mats-ami-builder-watchdog.timer", user_data)
        self.assertIn("product_uuid", user_data)
        self.assertIn(
            f"awscli-exe-linux-x86_64-{AWS_CLI_VERSION}.zip", user_data
        )
        self.assertNotIn("docker-compose-v2 awscli", user_data)
        self.assertIn('"state":"failed"', user_data)
        self.assertIn("builder.log", user_data)

    def test_ami_builder_reboots_for_image_and_has_termination_behavior(self):
        clients = {
            "s3": MagicMock(),
            "ssm": MagicMock(),
            "ec2": MagicMock(),
        }
        clients["ec2"].describe_images.return_value = {"Images": []}
        clients["ssm"].get_parameter.return_value = {
            "Parameter": {"Value": "ami-base"},
        }
        clients["ec2"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-builder"}],
        }
        clients["ec2"].describe_instances.return_value = {"Reservations": [{
            "Instances": [{"State": {"Name": "running"}}],
        }]}
        clients["ec2"].create_image.return_value = {"ImageId": "ami-runtime"}

        with patch("exp_aws_trajectory._s3_json_or_none", return_value={
            "runtime_hash": "runtime", "ready_at": "now",
        }):
            image = build_runtime_ami(
                clients,
                region="us-west-2",
                instance_type="c7a.xlarge",
                profile="profile",
                security_group="sg-1",
                subnet="subnet-1",
                bucket="bucket",
                bundle={"path": "/tmp/source.tar.gz", "sha256": "a" * 64},
                wanted_hash="runtime",
            )

        self.assertEqual(image, "ami-runtime")
        launch = clients["ec2"].run_instances.call_args.kwargs
        self.assertEqual(launch["InstanceInitiatedShutdownBehavior"], "terminate")
        self.assertIn("mats-ami-builder-watchdog.timer", launch["UserData"])
        self.assertNotIn("NoReboot", clients["ec2"].create_image.call_args.kwargs)
        clients["ec2"].terminate_instances.assert_called_once_with(
            InstanceIds=["i-builder"]
        )

    def test_ami_builder_reports_failed_stage_without_waiting_for_timeout(self):
        clients = {
            "s3": MagicMock(),
            "ssm": MagicMock(),
            "ec2": MagicMock(),
        }
        clients["ec2"].describe_images.return_value = {"Images": []}
        clients["ssm"].get_parameter.return_value = {"Parameter": {"Value": "ami-base"}}
        clients["ec2"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-builder"}],
        }
        clients["ec2"].describe_instances.return_value = {"Reservations": [{
            "Instances": [{"State": {"Name": "running"}}],
        }]}

        with (
            patch("exp_aws_trajectory._s3_json_or_none", return_value={
                "runtime_hash": "runtime",
                "state": "failed",
                "stage": "building-docker-image",
                "exit_code": 17,
            }),
            self.assertRaisesRegex(
                AwsTrajectoryError,
                r"failed during building-docker-image \(exit 17\)",
            ),
        ):
            build_runtime_ami(
                clients,
                region="us-west-2",
                instance_type="c7a.xlarge",
                profile="profile",
                security_group="sg-1",
                subnet="subnet-1",
                bucket="bucket",
                bundle={"path": "/tmp/source.tar.gz", "sha256": "a" * 64},
                wanted_hash="runtime",
            )

        clients["ec2"].create_image.assert_not_called()
        clients["ec2"].terminate_instances.assert_called_once_with(
            InstanceIds=["i-builder"]
        )

    def test_job_carries_preflight_secret_names_but_never_values(self):
        cfg = self.make_cfg()
        cfg["worker_allowed_secret_names"] = ["OPENROUTER_API_KEY", "CUSTOM_API_KEY"]

        cell = build_cells(
            cfg, campaign_id="campaign",
            source={"sha256": "a" * 64, "bytes": 100},
            bucket="bucket", hourly_price=0.2,
        )[0]

        self.assertEqual(
            cell["allowed_secret_names"],
            ["OPENROUTER_API_KEY", "CUSTOM_API_KEY"],
        )
        self.assertNotIn("secret-value", json.dumps(cell))


class ResultIntegrityTests(unittest.TestCase):
    def test_verified_result_extracts(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content = b"eval bytes"
            manifest = {"files": [{
                "path": "payload/run/task.eval",
                "sha256": hashlib.sha256(content).hexdigest(),
            }]}
            archive = root / "result.tar.gz"
            payload_file = root / "task.eval"
            payload_file.write_bytes(content)
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(payload_file, arcname="payload/run/task.eval")

            extracted = root / "out"
            extracted.mkdir()
            result = _secure_extract_result(archive, extracted)

            self.assertEqual(result, manifest)
            self.assertEqual((extracted / "payload/run/task.eval").read_bytes(), content)

    def test_result_rejects_an_unlisted_file(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            listed = root / "task.eval"
            listed.write_bytes(b"eval")
            extra = root / "extra.txt"
            extra.write_bytes(b"not checksummed")
            manifest = {"files": [{
                "path": "payload/run/task.eval",
                "sha256": hashlib.sha256(b"eval").hexdigest(),
            }]}
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            archive = root / "result.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(listed, arcname="payload/run/task.eval")
                output.add(extra, arcname="payload/run/extra.txt")
            extracted = root / "out"
            extracted.mkdir()

            with self.assertRaisesRegex(AwsTrajectoryError, "file set mismatch"):
                _secure_extract_result(archive, extracted)

    def test_verified_worker_result_imports_atomically_with_compute_sidecar(self):
        import hashlib
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_file = root / "task.eval"
            eval_file.write_bytes(b"eval bytes")
            manifest = {"files": [{
                "path": "payload/run/task.eval",
                "sha256": hashlib.sha256(b"eval bytes").hexdigest(),
            }]}
            manifest_file = root / "manifest.json"
            manifest_file.write_text(json.dumps(manifest))
            archive = root / "remote.tar.gz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(manifest_file, arcname="manifest.json")
                output.add(eval_file, arcname="payload/run/task.eval")
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            cell = {
                "status": "completed", "cell_id": "cell", "campaign_id": "campaign",
                "bucket": "bucket", "result_key": "result", "region": "us-west-2",
                "instance_type": "c7a.xlarge", "instance_id": "i-1",
                "hourly_price_usd": 0.2, "source_sha256": "source",
                "source_bytes": 100, "original_epoch": 1,
                "launched_at": "2026-08-01T00:00:00+00:00",
                "terminal": {
                    "task_name": "real_audit_task_cell",
                    "completed_at": "2026-08-01T02:00:00+00:00",
                    "result_sha256": archive_sha, "result_bytes": archive.stat().st_size,
                },
            }
            state = {
                "campaign_id": "campaign", "bucket": "bucket", "cells": [cell],
                "source": {"sha256": archive_sha, "bytes": archive.stat().st_size},
            }
            clients = {"s3": MagicMock()}
            clients["s3"].download_file.side_effect = (
                lambda _bucket, _key, destination: shutil.copy2(archive, destination)
            )

            destination = import_campaign_results(state, root / "data", clients)

            self.assertEqual((destination / "task.eval").read_bytes(), b"eval bytes")
            self.assertEqual(
                (destination / "remote_source.tar.gz").read_bytes(),
                archive.read_bytes(),
            )
            sidecar = json.loads((destination / "remote_campaign.json").read_text())
            compute = sidecar["task_compute"]["real_audit_task_cell"]
            self.assertAlmostEqual(compute["estimated_vm_cost_usd"], 0.4)
            self.assertTrue(compute["s3_cost_excluded"])
            self.assertTrue(compute["ebs_cost_excluded"])
            self.assertTrue(compute["public_ipv4_cost_excluded"])
            self.assertTrue(compute["internet_data_transfer_cost_excluded"])
            self.assertTrue(compute["shared_runtime_cost_excluded"])
            self.assertEqual(compute["root_volume_gb"], 16)
            self.assertFalse(any(path.name.startswith(".campaign.")
                                 for path in destination.parent.iterdir()))

    def test_campaign_s3_cleanup_is_exact_and_batches_large_campaigns(self):
        objects = [
            {"Key": f"campaigns/campaign/cell-{index}", "Size": index}
            for index in range(1001)
        ]
        s3 = MagicMock()
        s3.get_paginator.return_value.paginate.return_value = [
            {"Contents": objects[:700]},
            {"Contents": objects[700:]},
        ]
        s3.delete_objects.side_effect = [{}, {}]
        s3.list_objects_v2.return_value = {"KeyCount": 0}

        result = delete_campaign_s3_objects(
            s3, bucket="bucket", campaign_id="campaign"
        )

        self.assertEqual(result["prefix"], "campaigns/campaign/")
        self.assertEqual(result["objects_deleted"], 1001)
        self.assertEqual(result["bytes_deleted"], sum(range(1001)))
        self.assertEqual(s3.delete_objects.call_count, 2)
        first_delete = s3.delete_objects.call_args_list[0].kwargs["Delete"]
        second_delete = s3.delete_objects.call_args_list[1].kwargs["Delete"]
        self.assertEqual(len(first_delete["Objects"]), 1000)
        self.assertEqual(len(second_delete["Objects"]), 1)
        self.assertTrue(all(
            item["Key"].startswith("campaigns/campaign/")
            for call in s3.delete_objects.call_args_list
            for item in call.kwargs["Delete"]["Objects"]
        ))
        s3.list_objects_v2.assert_called_once_with(
            Bucket="bucket", Prefix="campaigns/campaign/", MaxKeys=1
        )

    def test_campaign_s3_cleanup_rejects_an_unsafe_prefix(self):
        s3 = MagicMock()

        with self.assertRaisesRegex(AwsTrajectoryError, "unsafe S3 campaign"):
            delete_campaign_s3_objects(
                s3, bucket="bucket", campaign_id="../another-campaign"
            )

        s3.get_paginator.assert_not_called()

    def test_final_import_is_recorded_locally_then_deleted_from_s3(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            source = imported / "remote_source.tar.gz"
            source.write_bytes(b"exact source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
                "task_compute": {},
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            clients = {"s3": MagicMock()}
            deletion = {
                "prefix": "campaigns/campaign/",
                "objects_deleted": 4,
                "bytes_deleted": 1234,
            }

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects",
                return_value=deletion,
            ) as delete:
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            delete.assert_called_once_with(
                clients["s3"], bucket="bucket", campaign_id="campaign"
            )
            self.assertEqual(result["s3_cleanup"]["status"], "deleted")
            self.assertEqual(result["s3_cleanup"]["objects_deleted"], 4)
            clients["s3"].put_object.assert_called_once()
            saved = json.loads(
                (data_root / "remote_campaigns" / "campaign.json").read_text()
            )
            sidecar = json.loads((imported / "remote_campaign.json").read_text())
            self.assertEqual(saved["s3_cleanup"], result["s3_cleanup"])
            self.assertEqual(sidecar["s3_cleanup"], result["s3_cleanup"])

    def test_failed_cleanup_keeps_the_valid_import_and_lifecycle_fallback(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            source = imported / "remote_source.tar.gz"
            source.write_bytes(b"exact source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            }
            clients = {"s3": MagicMock()}

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects",
                side_effect=AwsTrajectoryError("access denied"),
            ):
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            self.assertEqual(result["s3_cleanup"]["status"], "failed")
            self.assertEqual(result["s3_cleanup"]["fallback_lifecycle_days"], 7)
            self.assertIn("access denied", result["s3_cleanup"]["error"])
            self.assertTrue(source.is_file())
            sidecar = json.loads((imported / "remote_campaign.json").read_text())
            self.assertEqual(sidecar["s3_cleanup"]["status"], "failed")

    def test_changed_local_source_prevents_s3_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            imported = data_root / "logs" / "campaign"
            imported.mkdir(parents=True)
            (imported / "remote_source.tar.gz").write_bytes(b"changed source")
            (imported / "remote_campaign.json").write_text(json.dumps({
                "campaign_id": "campaign",
            }))
            state = {
                "campaign_id": "campaign",
                "bucket": "bucket",
                "cells": [],
                "source": {"sha256": "0" * 64},
            }
            clients = {"s3": MagicMock()}

            with patch(
                "exp_aws_trajectory.delete_campaign_s3_objects"
            ) as delete:
                result = _finalize_campaign_import(
                    state, data_root, clients, imported
                )

            delete.assert_not_called()
            self.assertEqual(result["s3_cleanup"]["status"], "failed")
            self.assertIn("checksum changed", result["s3_cleanup"]["error"])


class SafetyContractTests(unittest.TestCase):
    def test_setup_requires_exactly_one_funding_confirmation(self):
        with self.assertRaisesRegex(AwsTrajectoryError, "exactly one"):
            setup_aws({
                "confirm_approved_account": False,
                "confirm_personal_account": False,
            }, ENVIRONMENTS)
        with self.assertRaisesRegex(AwsTrajectoryError, "exactly one"):
            setup_aws({
                "confirm_approved_account": True,
                "confirm_personal_account": True,
            }, ENVIRONMENTS)

    def test_root_credentials_are_refused_before_setup_or_preflight(self):
        clients = {"sts": MagicMock()}
        clients["sts"].get_caller_identity.return_value = {
            "Account": "123",
            "Arn": "arn:aws:iam::123:root",
        }

        with self.assertRaisesRegex(AwsTrajectoryError, "root-user credentials"):
            account_id(clients)

    def test_personal_reimbursement_funding_is_truthfully_tagged(self):
        clients = {"s3": MagicMock()}

        ensure_bucket(
            clients,
            bucket="bucket",
            region="us-west-2",
            funding=PERSONAL_REIMBURSEMENT_FUNDING,
        )

        tag_set = clients["s3"].put_bucket_tagging.call_args.kwargs["Tagging"]["TagSet"]
        self.assertIn(
            {"Key": "Funding", "Value": PERSONAL_REIMBURSEMENT_FUNDING},
            tag_set,
        )

    def test_worker_policy_is_narrow_and_has_no_ec2_launch_permission(self):
        policy = json.dumps(_iam_policy("123", "bucket"))
        self.assertIn("campaigns/*", policy)
        self.assertIn("ssm:GetParameter", policy)
        self.assertNotIn("ec2:", policy)
        self.assertNotIn("s3:Delete", policy)

    def test_setup_secret_allow_list_rejects_invalid_environment_names(self):
        with self.assertRaisesRegex(AwsTrajectoryError, "invalid environment names"):
            put_api_keys({"ssm": MagicMock()}, extra_names=["bad-name"])

    def test_setup_rejects_an_explicit_secret_name_without_a_local_value(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(AwsTrajectoryError, "not set locally"),
        ):
            put_api_keys({"ssm": MagicMock()}, extra_names=["CUSTOM_API_KEY"])

    def test_new_secret_parameter_uses_tags_without_overwrite(self):
        missing = RuntimeError("missing")
        missing.response = {"Error": {"Code": "ParameterNotFound"}}
        ssm = MagicMock()
        ssm.get_parameter.side_effect = missing

        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            names = put_api_keys({"ssm": ssm}, extra_names=[])

        self.assertEqual(names, ["OPENAI_API_KEY"])
        request = ssm.put_parameter.call_args.kwargs
        self.assertEqual(request["Tags"], [{"Key": "Project", "Value": "mats-environments"}])
        self.assertNotIn("Overwrite", request)

    def test_existing_secret_parameter_uses_overwrite_without_tags(self):
        ssm = MagicMock()
        ssm.get_parameter.return_value = {"Parameter": {"Name": "existing"}}

        with patch.dict("os.environ", {"OPENAI_API_KEY": "secret"}, clear=True):
            put_api_keys({"ssm": ssm}, extra_names=[])

        request = ssm.put_parameter.call_args.kwargs
        self.assertTrue(request["Overwrite"])
        self.assertNotIn("Tags", request)

    def test_campaign_success_requires_every_cell_and_no_retry_magic(self):
        success = {"cells": [{
            "status": "completed", "terminal": {"pipeline_exit_code": 0},
        }]}
        infrastructure_failure = {"cells": [{
            "status": "infrastructure_failure", "terminal": {},
        }]}
        integrity_failure = {"cells": [{
            "status": "completed", "terminal": {"pipeline_exit_code": 1},
        }]}

        self.assertTrue(campaign_ok(success))
        self.assertFalse(campaign_ok(infrastructure_failure))
        self.assertFalse(campaign_ok(integrity_failure))

    def test_campaign_dry_run_never_uploads_or_launches(self):
        cfg = CellPlanningTests().make_cfg()
        cfg["vm_concurrency"] = 16
        clients = {"s3": MagicMock(), "ec2": MagicMock()}
        preflight = {
            "clients": clients, "bucket": "bucket", "ami": "ami-1",
            "funding": PERSONAL_REIMBURSEMENT_FUNDING,
            "runtime_hash": "runtime", "launch_template_id": "lt-1",
            "hourly_price_usd": 0.2, "quota_vcpus": 64,
            "required_vcpus": 16,
            "stored_secret_names": ["OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        }
        source = {
            "path": "/tmp/source.tar.gz", "sha256": "a" * 64,
            "bytes": 100, "files": 10,
        }
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory.build_source_bundle", return_value=source),
            patch(
                "exp_aws_trajectory.preflight_aws", return_value=preflight
            ) as preflight_mock,
        ):
            result = run_campaign(cfg, ENVIRONMENTS, Path(tmp), dry_run=True)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["vm_concurrency"], 4)
        self.assertEqual(result["required_vcpus"], 16)
        self.assertEqual(preflight_mock.call_args.args[0]["vm_concurrency"], 4)
        clients["s3"].upload_file.assert_not_called()
        clients["ec2"].run_instances.assert_not_called()

    def test_preflight_rejects_insufficient_vcpu_quota_before_launch(self):
        clients = {name: MagicMock() for name in (
            "sts", "s3", "ssm", "ec2", "quotas", "pricing",
        )}
        clients["sts"].get_caller_identity.return_value = {"Account": "123"}
        clients["s3"].get_bucket_tagging.return_value = {"TagSet": [
            {"Key": "Funding", "Value": PERSONAL_REIMBURSEMENT_FUNDING},
        ]}
        clients["ssm"].get_parameter.return_value = {"Parameter": {
            "Value": json.dumps({"OPENROUTER_API_KEY": "stored"}),
        }}
        clients["ec2"].describe_instance_types.return_value = {"InstanceTypes": [{
            "VCpuInfo": {"DefaultVCpus": 4}, "MemoryInfo": {"SizeInMiB": 8192},
        }]}
        clients["ec2"].describe_launch_templates.return_value = {"LaunchTemplates": [{
            "LaunchTemplateId": "lt-1", "DefaultVersionNumber": 3,
        }]}
        clients["ec2"].describe_launch_template_versions.return_value = {
            "LaunchTemplateVersions": [{"LaunchTemplateData": {
                "ImageId": "ami-1", "InstanceType": "c7a.xlarge",
            }}],
        }
        clients["quotas"].get_service_quota.return_value = {"Quota": {"Value": 32}}
        cfg = {
            "aws_region": "us-west-2", "aws_instance_type": "c7a.xlarge",
            "aws_bucket": None, "vm_concurrency": 16,
        }

        with (
            patch("exp_aws_trajectory.aws_clients", return_value=clients),
            patch("exp_aws_trajectory.runtime_hash", return_value="runtime"),
            patch("exp_aws_trajectory._find_runtime_ami", return_value="ami-1"),
        ):
            with self.assertRaisesRegex(AwsTrajectoryError, "require 64"):
                preflight_aws(
                    cfg, ENVIRONMENTS,
                    model_slugs=["openrouter/qwen/qwen3-32b"],
                )

        clients["ec2"].run_instances.assert_not_called()

    def test_terminating_instance_keeps_its_slot_until_fully_terminated(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "terminated"},
            }]}]},
        ]
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=None),
            patch("exp_aws_trajectory.time.sleep"),
        ):
            result = _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

        self.assertEqual(result["cells"][0]["status"], "infrastructure_failure")
        self.assertIn("terminated_at", result["cells"][0])
        clients["ec2"].terminate_instances.assert_not_called()

    def test_uploaded_completion_is_not_overwritten_while_vm_shuts_down(self):
        cell = {
            "campaign_id": "campaign", "cell_id": "cell", "bucket": "bucket",
            "complete_key": "complete", "failure_key": "failure",
            "instance_id": "i-1", "status": "running",
        }
        state = {
            "campaign_id": "campaign", "bucket": "bucket", "vm_concurrency": 1,
            "cells": [cell],
        }
        clients = {"ec2": MagicMock(), "s3": MagicMock()}
        clients["ec2"].describe_instances.side_effect = [
            {"Reservations": []},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "shutting-down"},
            }]}]},
            {"Reservations": [{"Instances": [{
                "InstanceId": "i-1", "State": {"Name": "terminated"},
            }]}]},
        ]
        marker = ("completed", {
            "pipeline_exit_code": 0,
            "completed_at": "2026-08-02T23:53:59+00:00",
        })
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("exp_aws_trajectory._terminal_marker", return_value=marker),
            patch("exp_aws_trajectory.time.sleep"),
        ):
            result = _monitor_campaign(state, Path(tmp), clients, allow_launch=False)

        completed = result["cells"][0]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["terminal"]["pipeline_exit_code"], 0)
        self.assertNotIn("reason", completed["terminal"])

    def test_vm_cost_uses_termination_time_not_earlier_upload_time(self):
        cost = _vm_cost({
            "hourly_price_usd": 0.2,
            "launched_at": "2026-08-01T00:00:00+00:00",
            "terminated_at": "2026-08-01T02:30:00+00:00",
            "terminal": {"completed_at": "2026-08-01T02:00:00+00:00"},
        })

        self.assertEqual(cost["billed_seconds_estimate"], 9000)
        self.assertAlmostEqual(cost["estimated_vm_cost_usd"], 0.5)

    def test_retry_inherits_original_custom_aws_location_without_explicit_override(self):
        old = {
            "campaign_id": "old", "bucket": "custom-bucket", "region": "eu-west-1",
            "instance_type": "c7a.xlarge", "vm_concurrency": 7,
            "pipeline_config": {
                "targets": ["qwen3-32b"],
                "target_models": ["openrouter/qwen/qwen3-32b"],
                "seeds": ["fraud_detection"],
                "seeds_path": "/repo/seeds/ml_training_data_misuse",
                "epochs": 1, "reasoning": True, "condition": "allow",
                "judge_resolved": "openai/gpt-5.6-luna",
                "gate_model": "openai/gpt-5.6-luna",
                "time_limit": AGENT_TIME_LIMIT_SECONDS, "aws_secret_env": [],
            },
            "cells": [{
                "status": "infrastructure_failure", "target": "qwen3-32b",
                "seed": "fraud_detection", "original_epoch": 1,
            }],
        }
        control = {
            "aws_region": "us-west-2", "aws_region_explicit": False,
            "aws_instance_type": "c7a.xlarge", "aws_instance_type_explicit": False,
            "vm_concurrency": 50, "vm_concurrency_explicit": False,
            "aws_bucket": None, "aws_bucket_explicit": False,
            "aws_secret_env": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            state_path = data / "remote_campaigns" / "old.json"
            state_path.parent.mkdir()
            state_path.write_text(json.dumps(old))
            with (
                patch("exp_aws_trajectory.aws_clients", return_value={}),
                patch("exp_aws_trajectory.run_campaign", return_value={"cells": []}) as run,
            ):
                retry_failed(control, ENVIRONMENTS, data, "old", dry_run=True)

        retry_cfg = run.call_args.args[0]
        self.assertEqual(retry_cfg["aws_region"], "eu-west-1")
        self.assertEqual(retry_cfg["aws_bucket"], "custom-bucket")
        self.assertEqual(retry_cfg["vm_concurrency"], 7)
        self.assertEqual(
            retry_cfg["_cell_selections"],
            [("qwen3-32b", "fraud_detection", 1)],
        )


if __name__ == "__main__":
    unittest.main()
