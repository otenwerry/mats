"""On-instance worker entrypoint for AWS trajectory cells.

The campaign controller imports this module only when its worker compatibility
functions are called. No AWS clients are constructed at import time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from urllib.request import Request, urlopen

from aws_runtime_contract import (
    AWS_SCHEMA_VERSION,
    DEFAULT_SECRET_NAMES,
    DEFAULT_WORKER_PIPELINE_SCRIPT,
    FAILURE_PACKAGE_SECONDS,
    SSM_PARAMETER_NAME,
    WORKER_PIPELINE_SCRIPTS,
    WORKER_PREFIX_PAYLOAD_PATH,
    WORKER_VERSION,
)
from aws_source_bundle import sha256_file


class WorkerRuntimeError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _imds(path: str) -> str | None:
    try:
        token_request = Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        )
        token = urlopen(token_request, timeout=2).read().decode()
        request = Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urlopen(request, timeout=2).read().decode()
    except Exception:
        return None


def _aws_cli(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["aws", *args],
        check=True,
        capture_output=capture,
        text=capture,
    )


def _worker_api_environment(job: dict) -> dict[str, str]:
    response = _aws_cli(
        "ssm", "get-parameter", "--name", SSM_PARAMETER_NAME,
        "--with-decryption", "--query", "Parameter.Value", "--output", "text",
        "--region", job["region"], capture=True,
    )
    secrets = json.loads(response.stdout)
    if not isinstance(secrets, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in secrets.items()
    ):
        raise WorkerRuntimeError("SSM API-key parameter has the wrong shape")
    allowed = set(job.get("allowed_secret_names") or DEFAULT_SECRET_NAMES)
    unexpected = set(secrets) - allowed
    if unexpected:
        raise WorkerRuntimeError(
            "SSM contains non-allow-listed names: " + ", ".join(unexpected)
        )
    return {**os.environ, **secrets}


def _result_manifest(payload_root: Path) -> dict:
    entries = []
    for path in sorted(payload_root.rglob("*")):
        if path.is_file():
            entries.append({
                "path": path.relative_to(payload_root.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {"schema_version": AWS_SCHEMA_VERSION, "files": entries}


def _package_worker_result(
    job: dict, worker_root: Path, run_dir: Path
) -> tuple[Path, dict]:
    package_root = worker_root / "package"
    if package_root.exists():
        shutil.rmtree(package_root)
    payload = package_root / "payload"
    payload.mkdir(parents=True)
    shutil.copytree(run_dir, payload / "run")
    for name in ("worker.log", "bootstrap.log"):
        path = worker_root / name
        if path.is_file():
            shutil.copy2(path, payload / name)
    manifest = _result_manifest(payload)
    (package_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    archive_path = worker_root / "result.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package_root / "manifest.json", arcname="manifest.json")
        archive.add(payload, arcname="payload")
    return archive_path, manifest


def smoke_worker_main(job: dict, job_path: Path) -> int:
    worker_root = job_path.parent
    instance_id = _imds("instance-id")
    started = _utc_now()
    checks = {}
    commands = {
        "docker_info_ok": ["docker", "info"],
        "compose_config_ok": [
            "docker", "compose", "-f",
            "/opt/supermats/mats/environments/sandbox/ml/compose.yaml", "config",
        ],
    }
    log_path = worker_root / "worker.log"
    with log_path.open("w") as log:
        for name, command in commands.items():
            result = subprocess.run(command, capture_output=True, text=True)
            checks[name] = result.returncode == 0
            log.write(f"{name}: exit {result.returncode}\n")
            if result.returncode:
                log.write((result.stderr or result.stdout)[-4000:] + "\n")
    run_dir = worker_root / "smoke-run"
    run_dir.mkdir()
    (run_dir / "smoke.json").write_text(json.dumps({
        **checks,
        "instance_id": instance_id,
        "checked_at": _utc_now(),
        "called_models": False,
    }, sort_keys=True))
    archive_path, manifest = _package_worker_result(job, worker_root, run_dir)
    _aws_cli(
        "s3", "cp", str(archive_path), f"s3://{job['bucket']}/{job['result_key']}",
        "--region", job["region"], "--sse", "AES256",
    )
    exit_code = 0 if all(checks.values()) else 1
    complete = {
        "schema_version": AWS_SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "campaign_id": job["campaign_id"],
        "cell_id": job["cell_id"],
        "instance_id": instance_id,
        "task_name": None,
        "pipeline_exit_code": exit_code,
        "started_at": started,
        "completed_at": _utc_now(),
        "result_key": job["result_key"],
        "result_sha256": sha256_file(archive_path),
        "result_bytes": archive_path.stat().st_size,
        "result_files": len(manifest["files"]),
    }
    marker = worker_root / "complete.json"
    marker.write_text(json.dumps(complete, sort_keys=True))
    # Marker is deliberately last: its presence means the result object is complete.
    _aws_cli(
        "s3", "cp", str(marker), f"s3://{job['bucket']}/{job['complete_key']}",
        "--region", job["region"], "--sse", "AES256",
    )
    return exit_code


def worker_main(job_path: Path) -> int:
    job = json.loads(job_path.read_text())
    if job.get("schema_version") != AWS_SCHEMA_VERSION:
        raise WorkerRuntimeError("worker job schema does not match this source bundle")
    if job.get("kind") == "smoke":
        return smoke_worker_main(job, job_path)
    worker_root = job_path.parent
    log_path = worker_root / "worker.log"
    instance_id = _imds("instance-id")
    compute = {
        "provider": "aws",
        "campaign_id": job["campaign_id"],
        "cell_id": job["cell_id"],
        "instance_id": instance_id,
        "instance_type": job["instance_type"],
        "funding": job.get("funding"),
        "region": job["region"],
        "source_sha256": job["source_sha256"],
        "original_epoch": job["original_epoch"],
        "worker_started_at": _utc_now(),
    }
    environment = _worker_api_environment(job)
    environment.update({
        "MATS_REMOTE_WORKER": "1",
        "MATS_REMOTE_RUN_DIR": job["worker_run_dir"],
        "MATS_REMOTE_TASK_SUFFIX": job["task_suffix"],
        "MATS_REMOTE_COMPUTE_JSON": json.dumps(compute, separators=(",", ":")),
    })
    script_name = job.get("pipeline_script") or DEFAULT_WORKER_PIPELINE_SCRIPT
    if script_name not in WORKER_PIPELINE_SCRIPTS:
        raise WorkerRuntimeError(
            f"job requests unsupported pipeline script {script_name!r}"
        )
    if job.get("prefix_payload_key"):
        payload_path = Path(WORKER_PREFIX_PAYLOAD_PATH)
        _aws_cli(
            "s3", "cp", f"s3://{job['bucket']}/{job['prefix_payload_key']}",
            str(payload_path), "--region", job["region"],
        )
        digest = sha256_file(payload_path)
        if digest != job["prefix_payload_sha256"]:
            failure = {
                "schema_version": AWS_SCHEMA_VERSION,
                "campaign_id": job["campaign_id"],
                "cell_id": job["cell_id"],
                "instance_id": instance_id,
                "reason": (
                    "downloaded continuation prefix payload failed its SHA-256 "
                    f"check ({digest[:12]} != {job['prefix_payload_sha256'][:12]})"
                ),
                "failure_code": "prefix_checksum_failed",
                "completed_at": _utc_now(),
            }
            marker = worker_root / "failure.json"
            marker.write_text(json.dumps(failure, sort_keys=True))
            _aws_cli(
                "s3", "cp", str(marker),
                f"s3://{job['bucket']}/{job['failure_key']}",
                "--region", job["region"], "--sse", "AES256",
            )
            return 1
    script = Path("/opt/supermats/mats/environments") / script_name
    command = [sys.executable, str(script), *job["pipeline_args"]]
    started = _utc_now()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
            cwd=script.parent,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    finished = _utc_now()
    data_root = Path("/opt/supermats/mats-local/environments")
    run_dir = data_root / "logs" / job["worker_run_dir"]
    if not run_dir.is_dir():
        failure = {
            "schema_version": AWS_SCHEMA_VERSION,
            "campaign_id": job["campaign_id"],
            "cell_id": job["cell_id"],
            "instance_id": instance_id,
            "reason": "pipeline produced no run directory",
            "pipeline_exit_code": exit_code,
            "started_at": started,
            "completed_at": finished,
        }
        temporary = worker_root / "failure.json"
        temporary.write_text(json.dumps(failure, sort_keys=True))
        _aws_cli(
            "s3", "cp", str(temporary), f"s3://{job['bucket']}/{job['failure_key']}",
            "--region", job["region"],
        )
        return 1
    archive_path, manifest = _package_worker_result(job, worker_root, run_dir)
    result_sha = sha256_file(archive_path)
    _aws_cli(
        "s3", "cp", str(archive_path), f"s3://{job['bucket']}/{job['result_key']}",
        "--region", job["region"], "--sse", "AES256",
    )
    eval_files = sorted(run_dir.glob("*.eval"))
    task_name = job["task_name"] if eval_files else None
    complete = {
        "schema_version": AWS_SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "campaign_id": job["campaign_id"],
        "cell_id": job["cell_id"],
        "instance_id": instance_id,
        "task_name": task_name,
        "pipeline_exit_code": exit_code,
        "started_at": started,
        "completed_at": finished,
        "result_key": job["result_key"],
        "result_sha256": result_sha,
        "result_bytes": archive_path.stat().st_size,
        "result_files": len(manifest["files"]),
    }
    marker = worker_root / "complete.json"
    marker.write_text(json.dumps(complete, sort_keys=True))
    _aws_cli(
        "s3", "cp", str(marker), f"s3://{job['bucket']}/{job['complete_key']}",
        "--region", job["region"], "--sse", "AES256",
    )
    return exit_code


def watchdog_failure_main(job_path: Path, reason: str) -> int:
    job = json.loads(job_path.read_text())
    result_fields = {}
    run_dir = (
        Path("/opt/supermats/mats-local/environments/logs") / job["worker_run_dir"]
    )
    if run_dir.is_dir():
        try:
            archive, manifest = _package_worker_result(job, job_path.parent, run_dir)
            _aws_cli(
                "s3", "cp", str(archive), f"s3://{job['bucket']}/{job['result_key']}",
                "--region", job["region"], "--sse", "AES256",
            )
            result_fields = {
                "result_key": job["result_key"],
                "result_sha256": sha256_file(archive),
                "result_bytes": archive.stat().st_size,
                "result_files": len(manifest["files"]),
            }
        except Exception as error:
            result_fields = {"partial_result_error": repr(error)[:500]}
    marker = job_path.parent / "watchdog-failure.json"
    marker.write_text(json.dumps({
        "schema_version": AWS_SCHEMA_VERSION,
        "campaign_id": job["campaign_id"],
        "cell_id": job["cell_id"],
        "instance_id": _imds("instance-id"),
        "reason": (
            f"worker exceeded the {FAILURE_PACKAGE_SECONDS}s packaging watchdog"
            if reason == "worker_watchdog_expired"
            else "downloaded source bundle failed its SHA-256 check"
        ),
        "failure_code": reason,
        "completed_at": _utc_now(),
        **result_fields,
    }, sort_keys=True))
    _aws_cli(
        "s3", "cp", str(marker), f"s3://{job['bucket']}/{job['failure_key']}",
        "--region", job["region"], "--sse", "AES256",
    )
    return 1
