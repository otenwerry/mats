"""Reproduce the mid-run restart mechanism seen in the PTB claude runs.

PAID (two small claude calls, ~$0.10 with haiku).

Hypothesis (see ISSUES.md, "unexplained restarts"): the mid-run `system:init`
events in the claude/qwen runs came from something re-launching the CLI with
`claude --continue` on the same session; the model's post-restart replies
address background-task notifications, so whatever got injected at resume
carried those.

Phase-1 result on CLI 2.1.207 (2026-07-13): in `-p` mode the CLI does NOT wait
for background tasks — it kills them ~5s after the final answer and exits with
a single init. So a CLI-internal re-entry cannot explain the PTB pattern on
the current CLI; the remaining candidates are (a) an external relauncher, and
(b) different run-era CLI behavior. Phase 2 below tests exactly what an
external relauncher would trigger.

Phases (one invocation runs both):
  1. `claude -p` with a prompt that spawns a background job and finishes
     immediately. Watch: does the process linger / re-enter / kill the task?
  2. after the first process exits (and the job's deadline passes), run
     `claude --continue -p` on the same session — the external-watchdog move.
     Watch: does the CLI inject a task-notification turn, and in exactly what
     format? That format is what reconstruction would need to insert at the
     PTB restart points instead of the (probably wrong) time-nudge template.

  uv run python posttrainbench/exp_restart_repro.py                # current CLI
  uv run python posttrainbench/exp_restart_repro.py --cli "npx @anthropic-ai/claude-code@2.1.76"
      (pin the run-era CLI version — 2.1.9/2.1.34/2.1.76 were used by the runs)

Runs in a scratch dir with a fresh CLAUDE_CONFIG_DIR (auth via
ANTHROPIC_API_KEY, loaded from mats/.env). Prints every stream event with a
wall-clock timestamp, then dumps whatever the resume injected.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import runs  # noqa: E402

PROMPT = ("Using your Bash tool, start this command as a BACKGROUND task "
          "(run_in_background=true): `sleep 45 && echo BACKGROUND_DONE`. "
          "Then, without waiting for it, reply with exactly the word FINISHED "
          "and stop. Do not poll or wait for the background task.")

CONTINUE_PROMPT = "Continue."


def _stream_run(cmd: list[str], cwd: Path, env: dict, out_path: Path,
                max_wait: int) -> tuple[int, int]:
    """Run a CLI command, echo its stream events live, save them. Returns
    (n_events, n_inits)."""
    t0 = datetime.datetime.now()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    n_events = n_init = 0
    with open(out_path, "w") as f:
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
            n_events += 1
            kind = f"{d.get('type')}:{d.get('subtype')}" if d.get("subtype") else d.get("type")
            if d.get("type") == "system" and d.get("subtype") == "init":
                n_init += 1
                kind += f"  <-- INIT #{n_init}"
            print(f"  [{dt:6.1f}s] {kind}")
            if dt > max_wait:
                print(f"  ... {max_wait}s elapsed, terminating")
                proc.terminate()
                break
    dt = (datetime.datetime.now() - t0).total_seconds()
    print(f"  exited after {dt:.1f}s code {proc.returncode}; "
          f"{n_events} events, {n_init} init(s)")
    return n_events, n_init


def _session_records(config: Path) -> list[tuple[Path, list[dict]]]:
    out = []
    proj = config / "projects"
    if proj.exists():
        for sess in sorted(proj.rglob("*.jsonl")):
            out.append((sess, [json.loads(ln) for ln in sess.read_text().splitlines()]))
    return out


def _dump_records(records: list[dict], label: str) -> None:
    print(f"  --- {label} ({len(records)} record(s)) ---")
    for d in records:
        m = d.get("message") or {}
        c = m.get("content")
        if isinstance(c, str):
            txt = c
        else:
            txt = " | ".join(
                b.get("type", "?") + ": " + str(b.get("text", b.get("content", "")))[:160]
                for b in (c or []) if isinstance(b, dict))
        print(f"  {d.get('type'):14} {'isMeta' if d.get('isMeta') else '':6} {str(txt)[:300]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default="claude",
                    help="CLI command to test (e.g. 'npx @anthropic-ai/claude-code@2.1.76')")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--wait", type=int, default=120,
                    help="max seconds per phase")
    ap.add_argument("--task-deadline", type=int, default=50,
                    help="seconds from launch until the background job would "
                         "have finished (sleep 45 + margin)")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set and not found in mats/.env")
        sys.exit(2)

    out_dir = runs.OUT_ROOT / "restart_repro" / datetime.datetime.now().strftime("%m%d_%H%M%S")
    cwd = out_dir / "cwd"
    config = out_dir / "claude_config"
    cwd.mkdir(parents=True)
    config.mkdir(parents=True)
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config)}
    base = [*args.cli.split(), "--print", "--verbose", "--output-format", "stream-json",
            "--model", args.model, "--dangerously-skip-permissions"]
    print(f"outputs: {out_dir}")

    print("\n=== PHASE 1: spawn background job, finish immediately ===")
    t_start = time.time()
    _stream_run([*base, PROMPT], cwd, env, out_dir / "stream_phase1.jsonl", args.wait)
    n_before = sum(len(r) for _, r in _session_records(config))

    remaining = args.task_deadline - (time.time() - t_start)
    if remaining > 0:
        print(f"\nwaiting {remaining:.0f}s for the background job's deadline to pass "
              f"(did the CLI kill it, or does it outlive the process?)")
        time.sleep(remaining)

    print(f"\n=== PHASE 2: `--continue` on the same session (the external-watchdog "
          f"move) ===")
    _stream_run([*base, "--continue", CONTINUE_PROMPT], cwd, env,
                out_dir / "stream_phase2.jsonl", args.wait)

    print("\n=== what phase 2 injected into the session ===")
    for sess, recs in _session_records(config):
        appended = recs[n_before:] if len(recs) > n_before else recs
        print(f"{sess.name}:")
        _dump_records(appended, "records appended by phase 2")
    done_seen = any("BACKGROUND_DONE" in json.dumps(r)
                    for _, recs in _session_records(config) for r in recs)
    print(f"\nBACKGROUND_DONE visible anywhere in the session: {done_seen}")
    print("READ-OUT: a task-notification-style block in the appended records is the "
          "injected format reconstruction needs; its absence + a killed task means "
          "this CLI can't produce the PTB pattern (try the pinned run-era CLI).")


if __name__ == "__main__":
    main()
