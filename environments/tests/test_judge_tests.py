"""The old Petri-shaped candidate replay endpoint must fail before spending money."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]


def test_retired_candidate_replay_fails_before_importing_a_model() -> None:
    result = subprocess.run(
        [sys.executable, str(ENVIRONMENTS / "exp_judge_tests.py")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "temporarily retired" in (result.stdout + result.stderr)

