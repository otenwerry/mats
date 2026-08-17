"""Build free continuation prefixes from stored observable activity logs.

Each output contains one neutral user message with the complete line-addressable
observable log from a prior trajectory. It contains no fabricated assistant history,
no workspace file, and no native session state. Production and subscription
continuations therefore start fresh and deliver the text directly in the opening
user context.

Usage (from mats/environments/):
  uv run prefixes/build_activity_log_prefixes.py --trajectories=711,715,723
  uv run prefixes/build_activity_log_prefixes.py --trajectories=711 --dry-run

This command is free: it reads local trajectories and optionally writes deterministic,
content-addressed JSON payloads. It makes no model, judge, AWS, or network calls.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


_ENVIRONMENTS = Path(__file__).resolve().parents[1]
for _path in (_ENVIRONMENTS, _ENVIRONMENTS / "lib"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from exp_real_continuation import (  # noqa: E402
    activity_log_prefix_payload_from_trajectory,
    build_prefix_spec,
    load_prefix_specs,
    reconstruct_prefix_payloads,
    store_prefix_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        required=True,
        help="comma-separated viewer trajectory IDs",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.trajectories = [
            int(value.strip())
            for value in args.trajectories.split(",")
            if value.strip()
        ]
    except ValueError as error:
        raise SystemExit(
            "--trajectories must be comma-separated viewer IDs"
        ) from error
    if not args.trajectories:
        raise SystemExit("--trajectories cannot be empty")
    if len(set(args.trajectories)) != len(args.trajectories):
        raise SystemExit("--trajectories contains duplicate viewer IDs")
    return args


def main() -> None:
    args = parse_args()
    outputs: list[tuple[dict, Path | None, int, str]] = []
    reconstructed = reconstruct_prefix_payloads(
        args.trajectories,
        source_use="activity_log",
    )
    for trajectory_id, source_payload in zip(
        args.trajectories, reconstructed, strict=True
    ):
        payload = activity_log_prefix_payload_from_trajectory(source_payload)
        harness = str(payload["source"].get("harness") or "simple")
        # Validate the exact delivery path the source model normally uses. This is
        # free and proves that no native resume bundle is required.
        build_prefix_spec(payload, harness=harness)
        path = None if args.dry_run else store_prefix_payload(payload)
        if path is not None:
            # Exercise the exact file-loading/revalidation path used by the ordinary
            # continuation controller, including current source-integrity checks.
            load_prefix_specs([], [str(path)], harness=harness)
        outputs.append((payload, path, trajectory_id, harness))

    verb = "would build" if args.dry_run else "built"
    print(f"{verb} {len(outputs)} inline activity-log prefix(es):")
    by_harness: dict[str, list[str]] = defaultdict(list)
    for payload, path, trajectory_id, harness in outputs:
        metadata = payload["source"]["activity_log_metadata"]
        destination = str(path) if path is not None else "(dry run; no file written)"
        print(
            f"  trajectory #{trajectory_id} -> {payload['name']} · "
            f"{payload['target']} · {metadata['line_count']:,} lines · "
            f"{metadata['byte_count']:,} bytes -> {destination}"
        )
        if path is not None:
            by_harness[harness].append(str(path))

    if by_harness:
        print("\nUse the generated files with the ordinary continuation pipeline:")
        for harness, paths in sorted(by_harness.items()):
            print(
                "  uv run exp_continuation_pipeline.py "
                "--treatment=activity-log-context "
                f"--prefix-files={','.join(paths)} --harness={harness} "
                "--seed-dir=<family> --seeds=<members> --epochs=<n>"
            )
    print("\nNo model, judge, AWS, credential, or network call was made.")


if __name__ == "__main__":
    main()
