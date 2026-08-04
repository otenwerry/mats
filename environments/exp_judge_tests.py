"""Run the current judge on the viewer's fixed Judge Tests cohort."""

import sys


if not any(argument.startswith("--source-runs=") for argument in sys.argv[1:]):
    sys.argv.append("--source-runs=judge-tests")

from exp_rejudge import main


if __name__ == "__main__":
    main()
