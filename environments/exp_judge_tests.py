"""Retired candidate replay endpoint for the former Petri-shaped judge.

Candidate comparison needs to be rebuilt around ``environment_judge`` so its saved
rows use the same family-specific schema and evidence provenance as official results.
Leaving the old endpoint runnable would silently compare two different output
contracts, so it fails before any model call.
"""


def main() -> None:
    raise SystemExit(
        "exp_judge_tests.py is temporarily retired: the old saved testbed used the "
        "Petri numeric schema. Rebuild candidate replay on environment_judge before "
        "running a paid comparison."
    )


if __name__ == "__main__":
    main()
