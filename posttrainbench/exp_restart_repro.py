"""Reproduce the mid-run restart mechanism seen in the PTB claude runs.

PAID (one small claude call, ~$0.05 with haiku).

Hypothesis (see ISSUES.md, "unexplained restarts"): the mid-run `system:init`
events in the claude/qwen runs are Claude Code itself re-entering the session
to deliver background-task completion notifications — the agent had spawned
background training jobs, declared itself done, and every post-restart reply
addresses a specific stale task id.

The test: run `claude -p` with a prompt that spawns a background job and
finishes immediately, then observe (1) whether the stream emits a second
`system:init` after the job completes, (2) whether the process lingers past
its answer, and (3) what the session file records for any re-entry — the
injected turn format is exactly what reconstruction would need to insert at
restart points instead of the (probably wrong) time-nudge template.

  uv run python posttrainbench/exp_restart_repro.py                # current CLI
  uv run python posttrainbench/exp_restart_repro.py --cli "npx @anthropic-ai/claude-code@2.1.76"
      (pin the run-era CLI version — behavior may have changed since)

Runs in a scratch dir with a fresh CLAUDE_CONFIG_DIR (auth via
ANTHROPIC_API_KEY). Prints every stream event as it arrives with a wall-clock
timestamp, then dumps the session-file tail.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import runs  # noqa: E402

PROMPT = ("Using your Bash tool, start this command as a BACKGROUND task "
          "(run_in_background=true): `sleep 45 && echo BACKGROUND_DONE`. "
          "Then, without waiting for it, reply with exactly the word FINISHED "
          "and stop. Do not poll or wait for the background task.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default="claude",
                    help="CLI command to test (e.g. 'npx @anthropic-ai/claude-code@2.1.76')")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--wait", type=int, default=120,
                    help="seconds to keep reading after launch (background job "
                         "finishes at ~45s; re-entry, if any, follows)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: export ANTHROPIC_API_KEY (fresh config dir has no auth)")
        sys.exit(2)

    out_dir = runs.OUT_ROOT / "restart_repro" / datetime.datetime.now().strftime("%m%d_%H%M%S")
    cwd = out_dir / "cwd"
    config = out_dir / "claude_config"
    cwd.mkdir(parents=True)
    config.mkdir(parents=True)

    cmd = [*args.cli.split(), "--print", "--verbose", "--output-format", "stream-json",
           "--model", args.model, "--dangerously-skip-permissions", PROMPT]
    print(f"launching: {' '.join(cmd[:-1])} <prompt>")
    print(f"outputs: {out_dir}\n")
    t0 = datetime.datetime.now()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            env={**os.environ, "CLAUDE_CONFIG_DIR": str(config)})

    events, n_init = [], 0
    stream_path = out_dir / "stream.jsonl"
    with open(stream_path, "w") as f:
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            f.write(line)
            dt = (datetime.datetime.now() - t0).total_seconds()
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(d)
            kind = f"{d.get('type')}:{d.get('subtype')}" if d.get("subtype") else d.get("type")
            if d.get("type") == "system" and d.get("subtype") == "init":
                n_init += 1
                kind += f"  <-- INIT #{n_init}"
            print(f"  [{dt:6.1f}s] {kind}")
            if (datetime.datetime.now() - t0).total_seconds() > args.wait:
                print(f"  ... {args.wait}s elapsed, terminating")
                proc.terminate()
                break

    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"\nprocess exited after {dt:.1f}s with code {proc.returncode}; "
          f"{len(events)} events, {n_init} init(s)")

    print("\n--- session file(s) under the scratch config ---")
    for sess in sorted((config / "projects").rglob("*.jsonl")):
        lines = sess.read_text().splitlines()
        print(f"{sess.name}: {len(lines)} records; tail:")
        for ln in lines[-6:]:
            d = json.loads(ln)
            m = d.get("message") or {}
            c = m.get("content")
            txt = c if isinstance(c, str) else " | ".join(
                b.get("type", "?") + ": " + str(b.get("text", ""))[:60]
                for b in (c or []) if isinstance(b, dict))
            print(f"  {d.get('type'):12} {'isMeta' if d.get('isMeta') else '':6} {str(txt)[:120]}")

    print(f"\nVERDICT hints: >1 init or a post-FINISHED task-notification turn in the "
          f"session file supports the CLI-internal notification hypothesis; a clean "
          f"single-init exit means something EXTERNAL restarted the PTB agents.")


if __name__ == "__main__":
    main()
