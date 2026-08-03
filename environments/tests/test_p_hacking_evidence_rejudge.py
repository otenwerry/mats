"""The lossy one-off p-hacking rejudge must remain retired."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ENVIRONMENTS = pathlib.Path(__file__).resolve().parents[1]


def test_retired_rejudge_fails_before_any_api_call() -> None:
    result = subprocess.run(
        [sys.executable, str(ENVIRONMENTS / "exp_rejudge_p_hacking_evidence.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "retired" in (result.stdout + result.stderr).lower()

