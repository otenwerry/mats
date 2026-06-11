#!/usr/bin/env python3
"""exp_local_smoke: one-real-step resume test of a reconstructed OpenCode
session, on this Mac, with your real OPENROUTER_API_KEY. Costs ~a cent.

What it does:
  1. copies a prepared cell (default backward_control) into a throwaway sandbox
     so the real build dir is never mutated;
  2. resumes the session with the cell's actual resume prompt via a local
     opencode-ai@1.1.59 (npx); the model sees the full reconstructed history;
  3. kills the agent after its FIRST completed step (or --max-seconds), prints
     that step, and checks it looks like a coherent continuation of the task
     (not a confused "what session?" reply).

The agent's bash tool runs LOCALLY during that one step — inside the sandbox
cwd, briefly, and almost always something harmless like `timer.sh` or `ls`
(what the original agent did at this point). Don't extend --max-steps far on
the Mac: there is no GPU here and the full loop belongs in the container.

Usage:
    OPENROUTER_API_KEY=sk-or-v1-... python -m rollback.run.exp_local_smoke \
        [--cell backward_control_cut39] [--max-steps 1] [--max-seconds 180]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import config

OPENCODE_PKG = "opencode-ai@1.1.59"  # match the container pin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="backward_control_cut39")
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--max-seconds", type=int, default=180)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY", "").startswith("sk-or-"):
        print("Set OPENROUTER_API_KEY (sk-or-v1-...) first."); return 2

    cell_dir = (config.BUILD_ROOT / config.TRAJECTORY.run_name / args.cell)
    if not (cell_dir / "run_config.json").exists():
        print(f"cell not prepared: {cell_dir}\n"
              "run: python -m rollback.run.orchestrate --bash-mode skip")
        return 2
    cfg = json.loads((cell_dir / "run_config.json").read_text())

    sandbox = Path(tempfile.mkdtemp(prefix="rollback_smoke_"))
    shutil.copytree(cell_dir, sandbox / "cell")
    cell = sandbox / "cell"
    print(f"sandbox: {cell}")

    env = os.environ.copy()
    env["XDG_DATA_HOME"] = str(cell / ".local" / "share")
    env["XDG_CONFIG_HOME"] = str(cell / ".config")

    cmd = ["npx", "-y", OPENCODE_PKG, "run",
           "--session", cfg["session_id"], "--model", cfg["policy_model"],
           "--format", "json", cfg["resume_prompt"]]
    print(f"resuming session {cfg['session_id']} with model {cfg['policy_model']}")
    print(f"prompt: {cfg['resume_prompt']!r}\n")

    proc = subprocess.Popen(cmd, cwd=cell / "task", env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    t0 = time.time()
    steps = 0
    saw_content = False
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            part = d.get("part") or {}
            if d.get("type") == "text":
                saw_content = True
                print(f"[assistant text] {part.get('text', '')[:500]}")
            elif d.get("type") == "tool_use":
                saw_content = True
                st = part.get("state") or {}
                print(f"[tool {part.get('tool')}] input={json.dumps(st.get('input'))[:200]}")
                print(f"  -> {str(st.get('output'))[:200]}")
            elif d.get("type") == "error":
                print(f"[ERROR] {json.dumps(d.get('error'))[:400]}")
                proc.kill()
                return 1
            elif d.get("type") == "step_finish":
                steps += 1
                print(f"[step {steps} finished] tokens={json.dumps((part.get('tokens') or {}))[:120]}")
                if steps >= args.max_steps:
                    print("\nmax steps reached; interrupting agent.")
                    proc.send_signal(signal.SIGINT)
                    break
            if time.time() - t0 > args.max_seconds:
                print("\nmax seconds reached; killing agent.")
                proc.kill()
                break
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        proc.kill()
    finally:
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    print(f"\nresult: {'OK — agent produced a continuation step' if saw_content else 'NO content streamed (investigate)'}")
    print(f"sandbox kept for inspection: {cell}")
    print("a second invocation of this script tests the crash-relaunch path "
          "(the session now contains these extra turns) — use a fresh run "
          "for clean storage.")
    return 0 if saw_content else 1


if __name__ == "__main__":
    sys.exit(main())
