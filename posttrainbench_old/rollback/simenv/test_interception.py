"""Offline end-to-end test of the sim-env interception pipeline. FREE — no
model API key; the policy is mock_policy (scripted tool calls) and the
environment daemon is the canned mock_daemon.

Asserts, by running a REAL `opencode run` (pinned 1.1.59 via npx):
  1. the plugin loads and routes bash calls through the daemon;
  2. a GPU-ish command (nvidia-smi) returns the SIMULATED output, and the
     real command never produced it;
  3. a cheap command (echo) executed for REAL;
  4. the final stream ends normally (session reaches idle).

Usage:  python -m rollback.simenv.test_interception
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import mock_daemon, mock_policy

OPENCODE_PKG = "opencode-ai@1.1.59"
PKG_DIR = Path(__file__).resolve().parent


def main() -> int:
    policy = mock_policy.serve(8377)
    daemon = mock_daemon.serve(8378)
    sandbox = Path(tempfile.mkdtemp(prefix="simenv_test_"))
    work = sandbox / "task"
    work.mkdir()
    (sandbox / ".config" / "opencode").mkdir(parents=True)
    (sandbox / ".config" / "opencode" / "opencode.json").write_text(json.dumps({
        "$schema": "https://opencode.ai/config.json",
        "permission": "allow",
        "plugin": [f"file://{PKG_DIR / 'sim_env.js'}"],
        "provider": {
            "mock": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "MockPolicy",
                "options": {"baseURL": "http://127.0.0.1:8377/v1", "apiKey": "x"},
                "models": {"policy": {"name": "policy", "tool_call": True}},
            }
        },
    }, indent=2))

    env = os.environ.copy()
    env.update({
        "XDG_DATA_HOME": str(sandbox / ".local" / "share"),
        "XDG_CONFIG_HOME": str(sandbox / ".config"),
        "SIM_DAEMON_URL": "http://127.0.0.1:8378",
    })
    r = subprocess.run(
        ["npx", "-y", OPENCODE_PKG, "run", "--model", "mock/policy",
         "--format", "json", "check the gpu then echo something"],
        cwd=work, env=env, capture_output=True, text=True, timeout=300)
    out = r.stdout

    checks = {
        "ran to completion (exit 0)": r.returncode == 0,
        "simulated nvidia-smi output present": "SIMULATED" in out and "H100 80GB" in out,
        "real echo executed": "real-command-ran" in out,
        "final text turn arrived": "Smoke test complete." in out,
    }
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    if not all(checks.values()):
        print("\n--- stdout tail ---\n", out[-3000:])
        print("\n--- stderr tail ---\n", r.stderr[-2000:])
        print(f"\nsandbox kept: {sandbox}")
        return 1
    shutil.rmtree(sandbox, ignore_errors=True)
    policy.shutdown(), daemon.shutdown()
    print("\ninterception pipeline OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
