"""End-to-end real-environment pipeline: audit, structured judge, then viewer.

Everything imported by this endpoint is owned by ``environments/`` or by Inspect. The
stage-one and final judge use the environment-specific structured schema. A continued
run is judged in a fresh context over its complete observable transcript.

v1 scope: fraud_detection + demand_forecasting + rating_prediction (ml, CPU-only) +
reasoning_prompt_benchmark + checkout_redesign + retrieval_practice (p_hacking),
--condition=allow only.

Usage (from mats/environments/):
  uv run exp_real_audit_pipeline.py --targets=qwen3-32b --seed-dir=ml_training_data_misuse \
      --seeds=fraud_detection --epochs=3
  uv run exp_real_audit_pipeline.py --targets=qwen3-32b --seed-dir=p_hacking \
      --seeds=reasoning_prompt_benchmark --epochs=3

Flags:
  --targets=<a,b,..>   REQUIRED. target names from TARGET_CHOICES.
  --seed-dir=<sub>     REQUIRED. a pinned seed dir or collection; every selected member
                       needs a generated real_env/workspace/ tree (envgen/gen_*.py).
  --seeds=<a,b,..>     REQUIRED. member names, or `all` (= every PORTED member).
  --epochs=<N>         REQUIRED. epochs per (target, seed) cell.
  --reasoning=yes|no   optional (default yes). Same semantics as the simulated pipeline.
  --condition=<c>      defaults to allow; v1 supports allow only (correct is a designed
                       seam, rejected loudly).
  --judge=<m>          alignment judge (default gpt-5.6-luna; see
                       lib/judge_selection.py, or set $ENVIRONMENTS_JUDGE).
  --gate-model=<m>     first-submission judge; defaults to and must equal --judge.
  --concurrency=<N>    LLM parallelism (default 50).
  --sandbox-concurrency=<N>  max live Docker containers (default 8).
  --time-limit=<sec>   initial wall clock; default depends on the seed family
                       (ml 7200 / p_hacking 1800, else 3600). There is no turn cap.
                       ML's first pass is fixed at 7200; its follow-up resets to 3600.
  --skip-viewer        don't rebuild the viewer at the end.
  --compute=aws|local  ML defaults to aws; p-hacking defaults to local. AWS runs one
                       trajectory on each VM.
  --vm-concurrency=<N> max active trajectory VMs (default 50).
  --dry-run            free AWS preflight: cells, source SHA/size, AMI/quota, price,
                       and worst-case compute cost; no VM or model calls.
  --aws-setup (--confirm-approved-account|--confirm-personal-account)
                       create/update the private bucket, encrypted secret, no-ingress
                       worker role/network, pinned AMI, and launch template. The first
                       confirmation records team funding; the second records personal
                       payment with expected reimbursement.
  --aws-smoke-test     paid VM but no LLM: verify Docker, S3 handoff, and termination.
  --resume-campaign=<id> recover/download a campaign without relaunching.
  --retry-failed=<id>  explicitly retry infrastructure failures in a new campaign.

Local compute requires a running Docker daemon (checked before any spend). AWS setup
requires an explicit, truthfully recorded team-funded or personal/reimbursement account.
Costs money (target + stage-one/final judge calls, and EC2 for AWS runs). The viewer
stage and AWS dry-run are free.
"""

import asyncio
import json
import os
import pathlib
import shutil
import subprocess
import sys
import traceback
from datetime import datetime

_ENVIRONMENTS = pathlib.Path(__file__).resolve().parent
for _p in (str(_ENVIRONMENTS / "lib"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from model_catalog import (
    REASONING_EFFORT,
    TARGET_CHOICES,
    route,
)
from protocol_sources import (
    reject_retired_fixed_system_prompt_flag,
    resolve_condition,
    resolve_reasoning,
    resolve_seeds,
)
from exp_real_audit import (
    DEFAULT_SANDBOX_CONCURRENCY,
    build_real_tasks,
    ported_members,
    resolve_gate_model,
    resolve_time_limit,
)
from exp_inspect_runner import dead_targets, run_eval
from judge_selection import resolve_judge
from project_paths import DATA_ROOT as DATA, VIEWER_ROOT as OUT
from exp_aws_trajectory import (
    DEFAULT_INSTANCE_TYPE,
    DEFAULT_REGION,
    DEFAULT_VM_CONCURRENCY,
    AwsTrajectoryError,
    campaign_ok,
    resume_campaign,
    retry_failed,
    run_campaign,
    smoke_test,
    setup_aws,
)


DEFAULT_CONCURRENCY = 50


def reasoning_tag(reasoning: bool) -> str:
    return " [reasoning:on]" if reasoning else " [reasoning:off]"


def env_viewer():
    """Load this project's free viewer by its explicit path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "environments_viewer", _ENVIRONMENTS / "viewer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_VALUE_FLAGS = {
    "--targets", "--seed-dir", "--seeds", "--epochs", "--reasoning", "--condition",
    "--concurrency", "--sandbox-concurrency", "--time-limit",
    "--judge", "--gate-model", "--compute",
    "--vm-concurrency", "--aws-region", "--aws-instance-type", "--aws-bucket",
    "--aws-secret-env", "--resume-campaign", "--retry-failed",
}
_SWITCH_FLAGS = {
    "--skip-viewer",
    "--dry-run", "--aws-setup", "--aws-smoke-test", "--confirm-approved-account",
    "--confirm-personal-account",
}


def _validate_cli_args() -> None:
    valid = sorted(_VALUE_FLAGS | _SWITCH_FLAGS)
    for arg in sys.argv[1:]:
        flag, separator, _ = arg.partition("=")
        if flag in _VALUE_FLAGS:
            if not separator:
                raise SystemExit(f"{flag} requires a value in the form {flag}=<value>")
            continue
        if flag in _SWITCH_FLAGS:
            if separator:
                raise SystemExit(f"{flag} is a switch and does not take a value")
            continue
        raise SystemExit(f"unknown argument {arg!r}; valid flags: {valid}")


def _arg(flag: str, default: str | None = None) -> str | None:
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")), default)


def _posint(flag: str, default: int | None) -> int | None:
    raw = _arg(flag)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise SystemExit(f"{flag} must be an integer, got {raw!r}")
    if v < 1:
        raise SystemExit(f"{flag} must be >= 1, got {v}")
    return v


def require_docker() -> None:
    """Abort BEFORE any spend if no Docker daemon is reachable (each trajectory needs
    its own container). This machine currently has no Docker installed -- install
    Docker Desktop or OrbStack first."""
    docker = shutil.which("docker")
    if docker is None:
        raise SystemExit(
            "docker is not installed (or not on PATH). Real-environment audits run "
            "every trajectory in a container; install Docker Desktop/OrbStack and retry."
        )
    probe = subprocess.run([docker, "info"], capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            "docker is installed but the daemon is not reachable "
            f"({(probe.stderr or probe.stdout).strip().splitlines()[:1]}). "
            "Start Docker and retry."
        )


def _parse_args() -> dict:
    reject_retired_fixed_system_prompt_flag()
    _validate_cli_args()

    targets_arg = _arg("--targets")
    if targets_arg is None:
        raise SystemExit(f"--targets is required (no default); choices: {sorted(TARGET_CHOICES)}")
    targets = list(dict.fromkeys(t.strip() for t in targets_arg.split(",") if t.strip()))
    unknown = [t for t in targets if t not in TARGET_CHOICES]
    if unknown:
        raise SystemExit(f"unknown --targets {unknown}; choices: {sorted(TARGET_CHOICES)}")

    seed_dir_arg = _arg("--seed-dir")
    if seed_dir_arg is None:
        raise SystemExit("--seed-dir is required (e.g. ml_training_data_misuse or p_hacking)")
    seeds_path, available_seeds = resolve_seeds(seed_dir_arg)
    ported = ported_members(seeds_path)
    seeds_arg = _arg("--seeds")
    if seeds_arg is None:
        raise SystemExit(
            "--seeds is required; use --seeds=all for every PORTED member, or a comma "
            f"list. Ported members here: {ported or 'none'}"
        )
    if seeds_arg.strip() == "all":
        seeds = list(ported)
        if not seeds:
            raise SystemExit(
                f"no member of {seed_dir_arg} has a real_env workspace yet; generate one "
                "with the matching envgen/gen_*.py script"
            )
    else:
        seeds = [s.strip() for s in seeds_arg.split(",") if s.strip()]
        unknown_s = [s for s in seeds if s not in available_seeds]
        if unknown_s:
            raise SystemExit(f"unknown --seeds {unknown_s}; available: {available_seeds}")

    epochs = _posint("--epochs", None)
    if epochs is None:
        raise SystemExit("--epochs is required (no default); positive integer")

    condition = resolve_condition(_arg("--condition"), seeds_path)
    resolved_seed_path = pathlib.Path(seeds_path)
    family = (
        resolved_seed_path.parent.name
        if (resolved_seed_path / "manifest.json").is_file()
        else resolved_seed_path.name
    )
    compute = _arg("--compute") or (
        "aws" if family == "ml_training_data_misuse" else "local"
    )
    if compute not in {"aws", "local"}:
        raise SystemExit("--compute must be aws or local")
    if compute == "aws" and family != "ml_training_data_misuse":
        raise SystemExit("--compute=aws is currently scoped to ML real environments only")
    time_limit = resolve_time_limit(_arg("--time-limit"), family)
    if compute == "aws" and time_limit != 7200:
        raise SystemExit("AWS ML trajectories use the fixed two-hour --time-limit=7200")
    judge_arg = _arg("--judge")

    return {
        "targets": targets,
        "seeds": seeds,
        "seeds_path": seeds_path,
        "epochs": epochs,
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "condition": condition,
        "judge": judge_arg,
        "judge_resolved": resolve_judge(judge_arg),
        "gate_model": resolve_gate_model(_arg("--gate-model"), judge_arg),
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        "sandbox_concurrency": _posint("--sandbox-concurrency", DEFAULT_SANDBOX_CONCURRENCY),
        "time_limit": time_limit,
        "skip_viewer": "--skip-viewer" in sys.argv,
        "compute": compute,
        "vm_concurrency": _posint("--vm-concurrency", DEFAULT_VM_CONCURRENCY),
        "aws_region": _arg("--aws-region", DEFAULT_REGION),
        "aws_instance_type": _arg("--aws-instance-type", DEFAULT_INSTANCE_TYPE),
        "aws_bucket": _arg("--aws-bucket"),
        "aws_secret_env": [
            name.strip() for name in (_arg("--aws-secret-env", "") or "").split(",")
            if name.strip()
        ],
        "dry_run": "--dry-run" in sys.argv,
    }


def _parse_aws_control_args() -> dict:
    """AWS-only modes do not require target/seed experiment arguments."""
    _validate_cli_args()
    return {
        "aws_region": _arg("--aws-region", DEFAULT_REGION),
        "aws_region_explicit": _arg("--aws-region") is not None,
        "aws_instance_type": _arg("--aws-instance-type", DEFAULT_INSTANCE_TYPE),
        "aws_instance_type_explicit": _arg("--aws-instance-type") is not None,
        "aws_bucket": _arg("--aws-bucket"),
        "aws_bucket_explicit": _arg("--aws-bucket") is not None,
        "aws_secret_env": [
            name.strip() for name in (_arg("--aws-secret-env", "") or "").split(",")
            if name.strip()
        ],
        "vm_concurrency": _posint("--vm-concurrency", DEFAULT_VM_CONCURRENCY),
        "vm_concurrency_explicit": _arg("--vm-concurrency") is not None,
        "confirm_approved_account": "--confirm-approved-account" in sys.argv,
        "confirm_personal_account": "--confirm-personal-account" in sys.argv,
        "dry_run": "--dry-run" in sys.argv,
        "concurrency": _posint("--concurrency", DEFAULT_CONCURRENCY),
        "skip_viewer": "--skip-viewer" in sys.argv,
    }


def gate_integrity_failures(logs: list) -> list[dict]:
    """Samples whose scripted branch was decided by a FAILED gate (verdict=error) --
    the run ended conservatively and must not be pooled as a normal trajectory."""
    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            real_env = (sample.metadata or {}).get("real_env") or {}
            protocol = real_env.get("protocol") or {}
            gate_errors = [
                g for g in real_env.get("gates", []) if g.get("verdict") == "error"
            ]
            if gate_errors or protocol.get("ended_reason") == "gate_error_end":
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": ["gate_error"]
                              + [str(g.get("error"))[:120] for g in gate_errors],
                })
    return failures


def audit_integrity_failures(logs: list) -> list[dict]:
    """Samples whose real workspace record did not finalize safely."""

    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            real_env = (sample.metadata or {}).get("real_env") or {}
            issues = []
            if not real_env.get("protocol"):
                issues.append("missing_protocol_record")
            if not real_env.get("artifacts"):
                issues.append("missing_final_artifact_record")
            if real_env.get("finalize_error"):
                issues.append(f"finalize_error:{real_env['finalize_error']}")
            if issues:
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": issues,
                })
    return failures


def judge_procedure_integrity_failures(logs: list) -> list[dict]:
    """Samples without one valid owned structured official judgment."""

    failures: list[dict] = []
    for log in logs or []:
        for sample in log.samples or []:
            score = (sample.scores or {}).get("environment_judge")
            metadata = getattr(score, "metadata", None) or {} if score else {}
            envelope = metadata.get("environment_judge") or {}
            issues = []
            if score is None:
                issues.append("missing_environment_judge_score")
            if not isinstance(envelope.get("result"), dict):
                issues.append("missing_structured_result")
            if envelope.get("post_validation") != "passed":
                issues.append(
                    f"judge_post_validation:{envelope.get('post_validation') or 'missing'}"
                )
            stage = envelope.get("official_stage")
            if stage not in {"stage_1", "final"}:
                issues.append(f"invalid_official_stage:{stage or 'missing'}")
            if stage == "final" and envelope.get("fresh_call") is not True:
                issues.append("final_judgment_not_marked_fresh")
            if issues:
                failures.append({
                    "task": log.eval.task,
                    "sample": str(sample.id),
                    "epoch": sample.epoch,
                    "issues": issues,
                })
    return failures


def run_real_audit_stage(cfg: dict):
    targets, seeds, epochs = cfg["targets"], cfg["seeds"], cfg["epochs"]
    target_models = [route(TARGET_CHOICES[t]) for t in targets]
    expected_n = len(targets) * len(seeds) * epochs
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = targets[0] if len(targets) == 1 else f"{len(targets)}targets"
    remote_run_name = os.environ.get("MATS_REMOTE_RUN_DIR")
    log_dir = DATA / "logs" / (
        remote_run_name
        or f"real-v3-{label}-{cfg['condition']}-{epochs}ep-{timestamp}"
    )
    task_suffix = os.environ.get("MATS_REMOTE_TASK_SUFFIX")
    execution_metadata = None
    if os.environ.get("MATS_REMOTE_COMPUTE_JSON"):
        try:
            execution_metadata = json.loads(os.environ["MATS_REMOTE_COMPUTE_JSON"])
        except json.JSONDecodeError as ex:
            raise SystemExit(f"invalid MATS_REMOTE_COMPUTE_JSON: {ex}")

    print("=" * 72)
    print(f"STAGE 1/2  REAL AUDIT + STRUCTURED JUDGE  ->  {log_dir.name}")
    print("=" * 72)
    print(f"  targets ({len(targets)}): "
          + ", ".join(f"{t}{reasoning_tag(cfg['reasoning'])}" for t in targets))
    print(f"  seeds ({len(seeds)}): {seeds}  condition={cfg['condition']}")
    print(f"  judge={resolve_judge(cfg['judge'])}  gate={cfg['gate_model']}")
    print(f"  epochs={epochs}  concurrency={cfg['concurrency']}  "
          f"sandbox_concurrency={cfg['sandbox_concurrency']}  "
          f"time_limit={cfg['time_limit']}s (no turn cap)")
    if cfg["reasoning"]:
        print(f"  reasoning=ON effort={REASONING_EFFORT}; '<thinking> tags' SP "
              "instruction stripped")
    print(f"  expected trajectories: {len(targets)} x {len(seeds)} x {epochs} = {expected_n}")
    print("  environment: REAL docker sandboxes; user turns fully scripted; the only "
          "in-run AI decision is the first-submission gate\n")

    try:
        tasks = build_real_tasks(
            targets, seeds, log_dir.name,
            reasoning=cfg["reasoning"], condition=cfg["condition"],
            gate_model=cfg["gate_model"], judge=cfg["judge"],
            seeds_path=cfg["seeds_path"],
            artifacts_root=log_dir / "real_artifacts",
            task_id_suffix=task_suffix,
            execution_metadata=execution_metadata,
        )
        success, logs = run_eval(
            tasks, epochs, cfg["concurrency"], log_dir,
            max_sandboxes=cfg["sandbox_concurrency"], time_limit=cfg["time_limit"],
        )
    except SystemExit:
        raise
    except Exception as e:
        print("\n!! REAL AUDIT STAGE CRASHED (continuing to viewer on existing logs):")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, log_dir, expected_n, False

    print(f"\neval_set finished, success={success}")
    actual_n = 0
    for log in logs:
        n = len(log.samples or [])
        actual_n += n
        print(f"  {log.eval.task}: status={log.status}, samples={n}")

    dead = dead_targets(logs, target_models)
    if dead:
        print("\n" + "!" * 72)
        print(f"WARNING: target(s) {dead} produced 0 output tokens -- they never ran.")
        print("!" * 72)

    if actual_n != expected_n:
        print(f"\nWARNING: expected {expected_n} trajectories but eval_set wrote {actual_n}.")
    else:
        print(f"\nall {expected_n} trajectories present.")

    integrity_failures = audit_integrity_failures(logs)
    gate_failures = gate_integrity_failures(logs)
    judge_failures = judge_procedure_integrity_failures(logs)
    for name, failures in (("data-integrity", integrity_failures),
                           ("gate", gate_failures),
                           ("judge-procedure", judge_failures)):
        if failures:
            print("\n" + "!" * 72)
            print(f"WARNING: {len(failures)} trajectory(ies) have {name} failures:")
            for failure in failures:
                print(f"  {failure['task']}/{failure['sample']} epoch {failure['epoch']}: "
                      f"{', '.join(failure['issues'])}")
            print("!" * 72)

    integrity_ok = (bool(success) and actual_n == expected_n and not dead
                    and not integrity_failures and not gate_failures
                    and not judge_failures)
    return logs, log_dir, expected_n, integrity_ok


async def run_env_post_stages(cfg: dict) -> bool:
    """Build the free environment-owned viewer after a run or AWS import."""
    ok = True
    if not cfg["skip_viewer"]:
        print("\n" + "=" * 72)
        print("STAGE 2/2  VIEWER  (rebuild this project's static viewer, free)")
        print("=" * 72)
        try:
            await env_viewer().main()
        except Exception as e:
            ok = False
            print(f"\n!! VIEWER STAGE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
    else:
        print("\n(skipping STAGE 3 viewer: --skip-viewer)")
    return ok


def main() -> None:
    try:
        if "--aws-setup" in sys.argv:
            control = _parse_aws_control_args()
            if control["dry_run"]:
                raise SystemExit("--dry-run is not supported with --aws-setup")
            setup_aws(control, _ENVIRONMENTS)
            return
        if "--aws-smoke-test" in sys.argv:
            control = _parse_aws_control_args()
            state = smoke_test(
                control, _ENVIRONMENTS, DATA, dry_run=control["dry_run"]
            )
            if not control["dry_run"] and not campaign_ok(state):
                raise SystemExit(1)
            return

        resume_id = _arg("--resume-campaign")
        retry_id = _arg("--retry-failed")
        if resume_id and retry_id:
            raise SystemExit("--resume-campaign and --retry-failed are mutually exclusive")
        if resume_id or retry_id:
            control = _parse_aws_control_args()
            if resume_id:
                if control["dry_run"]:
                    raise SystemExit("--dry-run is not meaningful with --resume-campaign")
                state = resume_campaign(control, _ENVIRONMENTS, DATA, resume_id)
            else:
                state = retry_failed(
                    control, _ENVIRONMENTS, DATA, retry_id,
                    dry_run=control["dry_run"],
                )
                if control["dry_run"]:
                    return
            post_cfg = {"skip_viewer": control["skip_viewer"]}
            post_ok = asyncio.run(run_env_post_stages(post_cfg))
            if not (campaign_ok(state) and post_ok):
                raise SystemExit(1)
            return

        cfg = _parse_args()
        if cfg["compute"] == "aws":
            cfg["target_models"] = [
                route(TARGET_CHOICES[target]) for target in cfg["targets"]
            ]
            state = run_campaign(
                cfg, _ENVIRONMENTS, DATA, dry_run=cfg["dry_run"]
            )
            if cfg["dry_run"]:
                return
            audit_ok = campaign_ok(state)
        else:
            if cfg["dry_run"]:
                raise SystemExit("--dry-run is an AWS campaign preflight")
            require_docker()
            _logs, _log_dir, _expected_n, audit_ok = run_real_audit_stage(cfg)
        post_ok = asyncio.run(run_env_post_stages(cfg))
    except AwsTrajectoryError as ex:
        raise SystemExit(f"AWS trajectory error: {ex}") from ex
    print("\n" + "=" * 72)
    if audit_ok and post_ok:
        print(f"PIPELINE DONE.  open {OUT / 'index.html'}")
    else:
        print("PIPELINE FINISHED WITH INTEGRITY FAILURES. "
              f"Inspect {OUT / 'index.html'} and the warnings above.")
    print("=" * 72)
    if not (audit_ok and post_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
