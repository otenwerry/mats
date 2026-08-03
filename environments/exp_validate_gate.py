"""Retired historical gate-validation endpoint.

The old endpoint replayed Petri's numeric schema over simulated Petri logs. The active
gate now branches directly on the environments-owned structured mechanism list, so that
historical validation no longer tests the production rule.
"""


def main() -> None:
    raise SystemExit(
        "exp_validate_gate.py was retired with the Petri numeric judge. "
        "Use the free environment-judge and gate tests for the active structured rule."
    )


if __name__ == "__main__":
    main()
