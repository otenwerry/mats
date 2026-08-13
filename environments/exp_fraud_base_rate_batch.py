"""Run the fixed four-agent fraud-detection base-rate batch.

One invocation runs two AWS campaigns sequentially so their VM allowances never
overlap:

1. DeepSeek V4 Pro, GLM-5.1, and Kimi K2.6 through the production harness.
2. GPT-5.5 through the subscription harness.

The default is 160 trajectories: 4 agents x 40 epochs. Both child campaigns skip
their own viewer build; this wrapper rebuilds the viewer once after both campaigns,
including when one campaign returns a nonzero status after saving partial results.

Usage (from mats/environments/):
  uv run exp_fraud_base_rate_batch.py
  uv run exp_fraud_base_rate_batch.py --dry-run

Flags:
  --epochs=<N>            epochs per agent; default 40.
  --vm-concurrency=<N>    max active AWS trajectory VMs; default 75.
  --reasoning=yes|no      default yes.
  --judge=<model>         defaults through the child pipeline to gpt-5.6-luna.
  --aws-region=<region>   optional child-pipeline override.
  --aws-instance-type=<t> optional child-pipeline override.
  --aws-bucket=<name>     optional child-pipeline override.
  --aws-secret-env=<a,b>  optional child-pipeline override.
  --skip-viewer           do not perform the final free viewer build.
  --dry-run               free: run both AWS preflights without VM/model calls.

This endpoint costs money unless --dry-run is supplied. GPT agent calls consume
included subscription quota; production agents, judges, and EC2 are billable.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ENVIRONMENTS = Path(__file__).resolve().parent
LIB = ENVIRONMENTS / "lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from exp_aws_trajectory import DEFAULT_VM_CONCURRENCY  # noqa: E402
from pipeline_cli import positive_int_arg, validate_cli_args, value_arg  # noqa: E402
from protocol_sources import resolve_reasoning  # noqa: E402


DEFAULT_EPOCHS = 40
SEED_DIR = "ml_training_data_misuse"
SEED = "fraud_detection"
PRODUCTION_TARGETS = ("deepseek-v4-pro", "glm-5.1", "kimi-k2.6")
SUBSCRIPTION_TARGETS = ("gpt-5.5",)

_VALUE_FLAGS = {
    "--epochs",
    "--vm-concurrency",
    "--reasoning",
    "--judge",
    "--aws-region",
    "--aws-instance-type",
    "--aws-bucket",
    "--aws-secret-env",
}
_SWITCH_FLAGS = {"--skip-viewer", "--dry-run"}


@dataclass(frozen=True)
class Stage:
    key: str
    harness: str
    targets: tuple[str, ...]

    def command(self, cfg: dict, *, python: str) -> tuple[str, ...]:
        command = [
            python,
            str(ENVIRONMENTS / "exp_real_audit_pipeline.py"),
            f"--targets={','.join(self.targets)}",
            f"--seed-dir={SEED_DIR}",
            f"--seeds={SEED}",
            f"--epochs={cfg['epochs']}",
            f"--harness={self.harness}",
            f"--reasoning={'yes' if cfg['reasoning'] else 'no'}",
            "--condition=allow",
            "--compute=aws",
            f"--vm-concurrency={cfg['vm_concurrency']}",
            "--skip-viewer",
        ]
        for flag, key in (
            ("--judge", "judge"),
            ("--aws-region", "aws_region"),
            ("--aws-instance-type", "aws_instance_type"),
            ("--aws-bucket", "aws_bucket"),
            ("--aws-secret-env", "aws_secret_env"),
        ):
            if cfg.get(key) is not None:
                command.append(f"{flag}={cfg[key]}")
        if cfg["dry_run"]:
            command.append("--dry-run")
        return tuple(command)

    def trajectories(self, epochs: int) -> int:
        return len(self.targets) * epochs


STAGES = (
    Stage("open production", "production", PRODUCTION_TARGETS),
    Stage("GPT subscription", "subscription", SUBSCRIPTION_TARGETS),
)


def _arg(flag: str) -> str | None:
    return value_arg(sys.argv, flag)


def _posint(flag: str, default: int) -> int:
    value = positive_int_arg(sys.argv, flag, default)
    assert value is not None
    return value


def parse_args() -> dict:
    validate_cli_args(
        sys.argv,
        value_flags=_VALUE_FLAGS,
        switch_flags=_SWITCH_FLAGS,
    )
    return {
        "epochs": _posint("--epochs", DEFAULT_EPOCHS),
        "vm_concurrency": _posint(
            "--vm-concurrency", DEFAULT_VM_CONCURRENCY
        ),
        "reasoning": resolve_reasoning(_arg("--reasoning")),
        "judge": _arg("--judge"),
        "aws_region": _arg("--aws-region"),
        "aws_instance_type": _arg("--aws-instance-type"),
        "aws_bucket": _arg("--aws-bucket"),
        "aws_secret_env": _arg("--aws-secret-env"),
        "skip_viewer": "--skip-viewer" in sys.argv,
        "dry_run": "--dry-run" in sys.argv,
    }


def build_commands(cfg: dict, *, python: str | None = None) -> list[tuple[str, ...]]:
    executable = python or sys.executable
    return [stage.command(cfg, python=executable) for stage in STAGES]


def _run_command(label: str, command: tuple[str, ...]) -> int:
    print(f"\n{'=' * 72}")
    print(f"STARTING {label}")
    print(" ".join(command))
    print("=" * 72, flush=True)
    try:
        completed = subprocess.run(command, cwd=ENVIRONMENTS, check=False)
    except OSError as error:
        print(f"FAILED TO START {label}: {type(error).__name__}: {error}")
        return 1
    print(f"FINISHED {label}: exit={completed.returncode}", flush=True)
    return int(completed.returncode)


def run_batch(cfg: dict) -> int:
    expected = sum(stage.trajectories(cfg["epochs"]) for stage in STAGES)
    print("Fraud-detection base-rate batch")
    print(f"  epochs per agent: {cfg['epochs']}")
    print(f"  expected trajectories: {expected}")
    print(f"  maximum concurrent VMs: {cfg['vm_concurrency']} (one stage at a time)")
    print(f"  mode: {'FREE DRY RUN' if cfg['dry_run'] else 'PAID'}")

    results: list[tuple[str, int]] = []
    for stage, command in zip(STAGES, build_commands(cfg), strict=True):
        results.append((stage.key, _run_command(stage.key, command)))

    if not cfg["dry_run"] and not cfg["skip_viewer"]:
        viewer_command = (sys.executable, str(ENVIRONMENTS / "viewer.py"))
        results.append(("viewer build", _run_command("viewer build", viewer_command)))

    print("\nBatch summary")
    for label, returncode in results:
        print(f"  {label}: {'ok' if returncode == 0 else f'failed (exit {returncode})'}")
    failures = [(label, code) for label, code in results if code != 0]
    if failures:
        print("One or more stages failed; successful/partial campaign results remain saved.")
        return 1
    return 0


def main() -> None:
    try:
        raise SystemExit(run_batch(parse_args()))
    except KeyboardInterrupt:
        print("\nInterrupted; no later stage will be started.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
