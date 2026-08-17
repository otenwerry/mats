"""Generate the four current purpose-built prefixes for the five core agents.

This is a paid orchestration endpoint. Its default matrix launches Natural Questions,
science ethics, general ethics, and move-fast culture for each of the five core models.
Open models use the production harness; closed models use the subscription harness.
It can also resume unfinished jobs and add the full simple-harness matrix. Every child
payload retains its own usage and cost record.

Usage (from mats/environments/):
  uv run prefixes/exp_prefix_batch.py
  uv run prefixes/exp_prefix_batch.py --simple-only
  uv run prefixes/exp_prefix_batch.py --resume-batch=<batch-id> --also-simple
  uv run prefixes/exp_prefix_batch.py --dry-run  # FREE: print the complete plan

Flags:
  --nq-tokens=<N>   Natural Questions context target; default 2000.
  --nq-seed=<N>     Natural Questions question-order seed; default 1234.
  --reasoning=yes|no  default yes; forwarded to every builder.
  --concurrency=<N> default 20, so the complete matrix starts concurrently.
  --simple-only     Run all 20 jobs on the simple harness instead of the default
                    production-for-open/subscription-for-closed routing.
  --resume-batch=<batch-id-or-path>
                    Rerun every non-succeeded job from an earlier batch. Succeeded
                    jobs are skipped only after their payload files are verified.
  --also-simple     With --resume-batch, also run the complete 20-job simple matrix.
  --skip-viewer     Do not rebuild the free static viewer after generation.
  --dry-run         FREE: validate and print commands; no child processes or files.

Full child output, progress state, payload paths, and recorded cost summaries go under
``mats-local/environments/prefix_batches/<batch id>/``. Prefix payloads themselves go
to the ordinary continuation-prefix store and are ready for continuations.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PREFIX_BUILDERS_ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = PREFIX_BUILDERS_ROOT.parent
LIB = ENVIRONMENTS / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from project_paths import DATA_ROOT  # noqa: E402
from protocol_sources import resolve_reasoning  # noqa: E402


NQ_DEFAULT_TOKENS = 2_000
NQ_DEFAULT_SEED = 1234
DEFAULT_CONCURRENCY = 20
PREFIX_BATCH_ROOT = DATA_ROOT / "prefix_batches"

MODEL_HARNESSES = (
    ("opus-4.6", "subscription"),
    ("deepseek-v4-pro", "production"),
    ("glm-5.1", "production"),
    ("gpt-5.5", "subscription"),
    ("kimi-k2.6", "production"),
)
PREFIX_BUILDERS = (
    ("natural-questions", "exp_nq_prefix.py"),
    ("science-ethics", "exp_science_ethics_prefix.py"),
    ("general-ethics", "exp_general_ethics_prefix.py"),
    ("move-fast", "exp_move_fast_prefix.py"),
)

_VALUE_FLAGS = {
    "--nq-tokens",
    "--nq-seed",
    "--reasoning",
    "--concurrency",
    "--resume-batch",
}
_SWITCH_FLAGS = {"--also-simple", "--dry-run", "--simple-only", "--skip-viewer"}
_QUESTION_PROGRESS = re.compile(r"^q(?P<number>\d+)/(?P<total>\d+)\b")


@dataclass(frozen=True)
class Job:
    prefix: str
    entrypoint: str
    model: str
    harness: str
    command: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.prefix}__{self.model}__{self.harness}"

    @property
    def label(self) -> str:
        return f"{self.prefix}/{self.model}/{self.harness}"


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for argument in sys.argv[1:]:
        flag, separator, _value = argument.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(
                    f"{flag} requires a value in the form {flag}=<value>"
                )
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {argument!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next(
        (
            argument.split("=", 1)[1]
            for argument in sys.argv
            if argument.startswith(flag + "=")
        ),
        default,
    )


def _positive_int(flag: str, default: int) -> int:
    raw = _arg(flag)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}") from error
    if value < 1:
        raise SystemExit(f"{flag} must be >= 1, got {value}")
    return value


def parse_args() -> dict:
    _validate_cli_args()
    cfg = {
        "nq_tokens": _positive_int("--nq-tokens", NQ_DEFAULT_TOKENS),
        "nq_seed": _positive_int("--nq-seed", NQ_DEFAULT_SEED),
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "concurrency": _positive_int("--concurrency", DEFAULT_CONCURRENCY),
        "simple_only": "--simple-only" in sys.argv,
        "resume_batch": _arg("--resume-batch"),
        "also_simple": "--also-simple" in sys.argv,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "dry_run": "--dry-run" in sys.argv,
    }
    if cfg["simple_only"] and cfg["resume_batch"]:
        raise SystemExit("--simple-only cannot be combined with --resume-batch")
    if cfg["also_simple"] and not cfg["resume_batch"]:
        raise SystemExit("--also-simple requires --resume-batch")
    return cfg


def _make_job(
    cfg: dict,
    *,
    prefix: str,
    model: str,
    harness: str,
    executable: str,
) -> Job:
    builders = dict(PREFIX_BUILDERS)
    if prefix not in builders:
        raise SystemExit(f"unknown prefix type in resumed batch: {prefix!r}")
    if model not in dict(MODEL_HARNESSES):
        raise SystemExit(f"unknown model in resumed batch: {model!r}")
    if harness not in {"simple", "production", "subscription"}:
        raise SystemExit(f"unknown harness in resumed batch: {harness!r}")
    entrypoint = builders[prefix]
    script = PREFIX_BUILDERS_ROOT / entrypoint
    if not script.is_file():
        raise SystemExit(f"prefix builder is missing: {script}")
    reasoning = "yes" if cfg["reasoning"] else "no"
    command = [
        executable,
        str(script),
        f"--model={model}",
        f"--harness={harness}",
        f"--reasoning={reasoning}",
    ]
    if prefix == "natural-questions":
        command.extend((
            f"--tokens={cfg['nq_tokens']}",
            f"--seed={cfg['nq_seed']}",
        ))
    return Job(
        prefix=prefix,
        entrypoint=entrypoint,
        model=model,
        harness=harness,
        command=tuple(command),
    )


def _matrix_jobs(cfg: dict, *, simple: bool, executable: str) -> list[Job]:
    jobs: list[Job] = []
    for model, default_harness in MODEL_HARNESSES:
        harness = "simple" if simple else default_harness
        for prefix, _entrypoint in PREFIX_BUILDERS:
            jobs.append(_make_job(
                cfg,
                prefix=prefix,
                model=model,
                harness=harness,
                executable=executable,
            ))
    return jobs


def _resume_manifest_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and candidate.parent == Path("."):
        candidate = PREFIX_BATCH_ROOT / candidate
    if candidate.is_dir():
        candidate = candidate / "batch_manifest.json"
    if not candidate.is_file():
        raise SystemExit(f"resume batch manifest does not exist: {candidate}")
    return candidate.resolve()


def _resume_jobs(cfg: dict, *, executable: str) -> list[Job]:
    manifest_path = _resume_manifest_path(str(cfg["resume_batch"]))
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"could not read resume manifest {manifest_path}: {error}"
        ) from error
    if manifest.get("format") != "environments-prefix-batch-v1":
        raise SystemExit(f"not a supported prefix batch manifest: {manifest_path}")
    records = manifest.get("jobs")
    if not isinstance(records, dict) or not records:
        raise SystemExit(f"resume batch has no jobs: {manifest_path}")

    cfg["resume_manifest_path"] = str(manifest_path)
    jobs: list[Job] = []
    for key, record in records.items():
        if not isinstance(record, dict):
            raise SystemExit(f"invalid job record {key!r} in {manifest_path}")
        if record.get("status") == "succeeded":
            payload = record.get("payload_file")
            if not isinstance(payload, str) or not Path(payload).is_file():
                raise SystemExit(
                    f"cannot skip succeeded job {key!r}: its payload file is missing"
                )
            continue
        jobs.append(_make_job(
            cfg,
            prefix=str(record.get("prefix", "")),
            model=str(record.get("model", "")),
            harness=str(record.get("harness", "")),
            executable=executable,
        ))
    return jobs


def build_jobs(cfg: dict, *, python: str | None = None) -> list[Job]:
    executable = python or sys.executable
    if cfg.get("resume_batch"):
        jobs = _resume_jobs(cfg, executable=executable)
        if cfg.get("also_simple"):
            jobs.extend(_matrix_jobs(cfg, simple=True, executable=executable))
    else:
        jobs = _matrix_jobs(
            cfg,
            simple=bool(cfg.get("simple_only")),
            executable=executable,
        )

    keys = [job.key for job in jobs]
    if len(keys) != len(set(keys)):
        raise SystemExit(
            "selected jobs contain duplicate prefix/model/harness combinations"
        )
    if not jobs:
        raise SystemExit("no jobs selected: the resumed batch is already complete")
    return jobs


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _batch_id() -> str:
    return f"prefix-batch-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"


def _initial_manifest(batch_id: str, batch_dir: Path, cfg: dict, jobs: list[Job]) -> dict:
    return {
        "format": "environments-prefix-batch-v1",
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "created_at": _now(),
        "completed_at": None,
        "configuration": {
            "nq_tokens": cfg["nq_tokens"],
            "nq_seed": cfg["nq_seed"],
            "reasoning": cfg["reasoning"],
            "concurrency": cfg["concurrency"],
            "simple_only": cfg["simple_only"],
            "resume_manifest_path": cfg.get("resume_manifest_path"),
            "also_simple": cfg.get("also_simple", False),
        },
        "jobs": {
            job.key: {
                "prefix": job.prefix,
                "entrypoint": job.entrypoint,
                "model": job.model,
                "harness": job.harness,
                "command": list(job.command),
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "returncode": None,
                "log_file": None,
                "payload_file": None,
                "generation_cost": None,
                "error": None,
            }
            for job in jobs
        },
        "summary": None,
        "viewer_build": None,
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _echo_child_line(line: str) -> bool:
    stripped = line.strip()
    progress = _QUESTION_PROGRESS.match(stripped)
    if progress:
        number = int(progress.group("number"))
        total = int(progress.group("total"))
        return total <= 5 or number == 1 or number == total or number % 10 == 0
    return (
        stripped.startswith(("[generate]", "[done]", "WARNING", "Traceback"))
        or "payload:" in stripped
        or "generation cost:" in stripped
        or stripped.startswith("ERROR")
    )


def _payload_path_from_line(line: str) -> str | None:
    stripped = line.strip()
    marker = "payload:"
    if not stripped.startswith(marker):
        return None
    value = stripped[len(marker):].strip()
    return value or None


def _payload_cost(payload_path: str | None) -> dict | None:
    if not payload_path:
        return None
    try:
        payload = json.loads(Path(payload_path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cost = (payload.get("source") or {}).get("generation_cost")
    return cost if isinstance(cost, dict) else None


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
        return
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def _run_job(
    job: Job,
    *,
    semaphore: asyncio.Semaphore,
    manifest: dict,
    manifest_path: Path,
    manifest_lock: asyncio.Lock,
) -> None:
    record = manifest["jobs"][job.key]
    async with semaphore:
        log_path = manifest_path.parent / f"{job.key}.log"
        async with manifest_lock:
            record.update({
                "status": "running",
                "started_at": _now(),
                "log_file": str(log_path),
            })
            _write_manifest(manifest_path, manifest)
        print(f"[start] {job.label}")
        process: asyncio.subprocess.Process | None = None
        payload_path: str | None = None
        try:
            environment = dict(os.environ)
            environment["PYTHONUNBUFFERED"] = "1"
            process = await asyncio.create_subprocess_exec(
                *job.command,
                cwd=str(ENVIRONMENTS),
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            assert process.stdout is not None
            with log_path.open("w", encoding="utf-8") as log:
                while line_bytes := await process.stdout.readline():
                    line = line_bytes.decode(errors="replace")
                    log.write(line)
                    log.flush()
                    payload_path = _payload_path_from_line(line) or payload_path
                    if _echo_child_line(line):
                        print(f"[{job.label}] {line.strip()}")
            returncode = await process.wait()
            cost = _payload_cost(payload_path)
            succeeded = returncode == 0 and payload_path is not None
            async with manifest_lock:
                record.update({
                    "status": "succeeded" if succeeded else "failed",
                    "completed_at": _now(),
                    "returncode": returncode,
                    "payload_file": payload_path,
                    "generation_cost": cost,
                    "error": (
                        None
                        if succeeded
                        else (
                            "child exited successfully but printed no payload path"
                            if returncode == 0
                            else f"child exited with status {returncode}"
                        )
                    ),
                })
                _write_manifest(manifest_path, manifest)
            print(f"[{'done' if succeeded else 'failed'}] {job.label}")
        except asyncio.CancelledError:
            if process is not None:
                await _stop_process(process)
            async with manifest_lock:
                record.update({
                    "status": "cancelled",
                    "completed_at": _now(),
                    "returncode": process.returncode if process is not None else None,
                    "error": "batch interrupted",
                })
                _write_manifest(manifest_path, manifest)
            raise
        except Exception as error:
            if process is not None:
                await _stop_process(process)
            async with manifest_lock:
                record.update({
                    "status": "failed",
                    "completed_at": _now(),
                    "returncode": process.returncode if process is not None else None,
                    "error": f"{type(error).__name__}: {error}",
                })
                _write_manifest(manifest_path, manifest)
            print(f"[failed] {job.label}: {type(error).__name__}: {error}")


async def _heartbeat(manifest: dict) -> None:
    while True:
        await asyncio.sleep(30)
        counts: dict[str, int] = {}
        for record in manifest["jobs"].values():
            status = str(record["status"])
            counts[status] = counts.get(status, 0) + 1
        rendered = " · ".join(
            f"{status}={count}" for status, count in sorted(counts.items())
        )
        print(f"[status] {rendered}")


def _summarize(manifest: dict) -> dict:
    status_counts: dict[str, int] = {}
    recorded_cost_usd = 0.0
    costed_jobs = 0
    included_subscription_jobs = 0
    unpriced_jobs = 0
    for record in manifest["jobs"].values():
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != "succeeded":
            continue
        cost = record.get("generation_cost")
        if not isinstance(cost, dict):
            unpriced_jobs += 1
            continue
        amount = cost.get("cost_usd")
        if isinstance(amount, (int, float)):
            recorded_cost_usd += float(amount)
            costed_jobs += 1
        elif cost.get("source") == "subscription_not_metered":
            included_subscription_jobs += 1
        else:
            unpriced_jobs += 1
    return {
        "status_counts": status_counts,
        "recorded_cost_usd": recorded_cost_usd,
        "costed_jobs": costed_jobs,
        "included_subscription_jobs": included_subscription_jobs,
        "unpriced_jobs": unpriced_jobs,
    }


async def run_batch(cfg: dict, jobs: list[Job]) -> tuple[dict, Path]:
    batch_id = _batch_id()
    batch_dir = PREFIX_BATCH_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = batch_dir / "batch_manifest.json"
    manifest = _initial_manifest(batch_id, batch_dir, cfg, jobs)
    _write_manifest(manifest_path, manifest)
    print(f"[batch] logs + manifest: {batch_dir}")

    semaphore = asyncio.Semaphore(min(cfg["concurrency"], len(jobs)))
    manifest_lock = asyncio.Lock()
    heartbeat = asyncio.create_task(_heartbeat(manifest))
    try:
        await asyncio.gather(*(
            _run_job(
                job,
                semaphore=semaphore,
                manifest=manifest,
                manifest_path=manifest_path,
                manifest_lock=manifest_lock,
            )
            for job in jobs
        ))
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat

    manifest["completed_at"] = _now()
    manifest["summary"] = _summarize(manifest)
    _write_manifest(manifest_path, manifest)
    return manifest, manifest_path


def rebuild_viewer(manifest: dict, manifest_path: Path) -> int:
    command = [sys.executable, str(ENVIRONMENTS / "viewer.py")]
    started_at = _now()
    print("\n[viewer] rebuilding the free static viewer ...")
    completed = subprocess.run(command, cwd=ENVIRONMENTS, check=False)
    manifest["viewer_build"] = {
        "command": command,
        "started_at": started_at,
        "completed_at": _now(),
        "returncode": completed.returncode,
        "status": "succeeded" if completed.returncode == 0 else "failed",
    }
    _write_manifest(manifest_path, manifest)
    return completed.returncode


def _print_plan(cfg: dict, jobs: list[Job]) -> None:
    print("=" * 78)
    print(f"PREFIX BATCH  ({len(jobs)} paid generation jobs)")
    print("=" * 78)
    print(
        f"  NQ target={cfg['nq_tokens']:,} tokens  seed={cfg['nq_seed']}  "
        f"reasoning={'on' if cfg['reasoning'] else 'off'}  "
        f"concurrency={min(cfg['concurrency'], len(jobs))}"
    )
    if cfg.get("resume_batch"):
        route = f"unfinished jobs from {cfg['resume_manifest_path']}"
        if cfg.get("also_simple"):
            route += " · plus the full simple matrix"
        print(f"  selection: {route}")
    elif cfg["simple_only"]:
        print("  harness routing: all simple")
    else:
        print("  harness routing: closed agents subscription · open agents production")
    for index, job in enumerate(jobs, start=1):
        print(f"\n  {index:>2}. {job.label}\n      {shlex.join(job.command)}")


def main() -> None:
    cfg = parse_args()
    jobs = build_jobs(cfg)
    _print_plan(cfg, jobs)
    if cfg["dry_run"]:
        print("\n[dry-run] no child processes, model calls, or files created.")
        return
    try:
        manifest, manifest_path = asyncio.run(run_batch(cfg, jobs))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    summary: dict[str, Any] = manifest["summary"] or {}
    viewer_returncode = (
        0 if cfg["skip_viewer"] else rebuild_viewer(manifest, manifest_path)
    )
    print("\n" + "=" * 78)
    print("PREFIX BATCH COMPLETE")
    print("=" * 78)
    print(f"  status: {summary.get('status_counts')}")
    print(f"  recorded API cost: ${summary.get('recorded_cost_usd', 0.0):.4f}")
    print(
        "  included-subscription jobs: "
        f"{summary.get('included_subscription_jobs', 0)}"
    )
    print(f"  unpriced jobs: {summary.get('unpriced_jobs', 0)}")
    print(f"  manifest: {manifest_path}")
    if summary.get("status_counts", {}).get("failed", 0) or viewer_returncode != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
