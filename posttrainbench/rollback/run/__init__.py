"""Remote-runner glue: solve_intervention.sh (in-container entrypoint) and
orchestrate.py (resumable 2x2 driver). The bash + apptainer parts run on the
Linux/GPU box; the prepare/dry-run parts run anywhere."""
