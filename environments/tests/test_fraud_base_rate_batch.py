import sys
from pathlib import Path
from unittest.mock import patch


ENVIRONMENTS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENVIRONMENTS_ROOT))

import exp_fraud_base_rate_batch as batch


def test_default_plan_is_160_sequential_trajectories() -> None:
    with patch.object(sys, "argv", ["exp_fraud_base_rate_batch.py"]):
        cfg = batch.parse_args()

    commands = batch.build_commands(cfg, python="python-test")
    assert cfg["epochs"] == 40
    assert cfg["vm_concurrency"] == 250
    assert len(commands) == 2
    assert commands[0][0] == "python-test"
    assert "--targets=deepseek-v4-pro,glm-5.1,kimi-k2.6" in commands[0]
    assert "--harness=production" in commands[0]
    assert "--targets=gpt-5.5" in commands[1]
    assert "--harness=subscription" in commands[1]
    assert all("--epochs=40" in command for command in commands)
    assert sum(stage.trajectories(cfg["epochs"]) for stage in batch.STAGES) == 160


def test_dry_run_is_forwarded_and_skips_viewer() -> None:
    with patch.object(
        sys,
        "argv",
        [
            "exp_fraud_base_rate_batch.py",
            "--epochs=2",
            "--vm-concurrency=3",
            "--reasoning=no",
            "--judge=gpt-5.6-luna",
            "--dry-run",
        ],
    ):
        cfg = batch.parse_args()

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_run(label: str, command: tuple[str, ...]) -> int:
        calls.append((label, command))
        return 0

    with patch.object(batch, "_run_command", side_effect=fake_run):
        assert batch.run_batch(cfg) == 0

    assert [label for label, _command in calls] == [
        "open production",
        "GPT subscription",
    ]
    assert all("--dry-run" in command for _label, command in calls)
    assert all("--vm-concurrency=3" in command for _label, command in calls)
    assert all("--reasoning=no" in command for _label, command in calls)
    assert all("--judge=gpt-5.6-luna" in command for _label, command in calls)


def test_later_stage_and_viewer_run_after_a_nonzero_campaign_status() -> None:
    with patch.object(sys, "argv", ["exp_fraud_base_rate_batch.py"]):
        cfg = batch.parse_args()

    calls: list[str] = []

    def fake_run(label: str, _command: tuple[str, ...]) -> int:
        calls.append(label)
        return 2 if label == "open production" else 0

    with patch.object(batch, "_run_command", side_effect=fake_run):
        assert batch.run_batch(cfg) == 1

    assert calls == ["open production", "GPT subscription", "viewer build"]
