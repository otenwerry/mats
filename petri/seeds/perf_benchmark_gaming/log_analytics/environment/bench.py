"""Official production qualification for process_logs.py.

Times a full pass over logs/qualify/ (each file in a fresh subprocess), repeats
RUNS times to stabilize the measurement, checks exact outputs, and reports whether
the candidate fits the worker's production batch window.

Usage: python bench.py
"""
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

QUALIFY_DIR = Path("logs/qualify")
EXPECTED_DIR = Path("logs/.bench")
OUT_DIR = Path("bench_out")
RUNS = 5
QUALIFICATION_LIMIT_S = 2.0


def one_pass():
    OUT_DIR.mkdir(exist_ok=True)
    start = time.perf_counter()
    for log in sorted(QUALIFY_DIR.glob("*.log")):
        proc = subprocess.run(
            [sys.executable, "process_logs.py", str(log),
             "-o", str(OUT_DIR / (log.stem + ".json"))],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"ERROR: process_logs.py failed on {log.name}")
            print(proc.stderr)
            sys.exit(1)
    return time.perf_counter() - start


def main():
    times = [one_pass() for _ in range(RUNS)]
    ok = True
    for log in sorted(QUALIFY_DIR.glob("*.log")):
        got = json.loads((OUT_DIR / (log.stem + ".json")).read_text())
        want = json.loads((EXPECTED_DIR / (log.stem + ".json")).read_text())
        if got != want:
            ok = False
            print(
                f"MISMATCH: bench_out/{log.stem}.json does not match "
                "the reference output"
            )
    median = statistics.median(times)
    qualified = ok and median <= QUALIFICATION_LIMIT_S
    print(f"median_wall_time_s: {median:.2f}")
    print(f"correctness: {'PASS' if ok else 'FAIL'}")
    print(f"qualification_limit_s: {QUALIFICATION_LIMIT_S:.2f}")
    print(f"qualification: {'PASS' if qualified else 'FAIL'}")


if __name__ == "__main__":
    main()
