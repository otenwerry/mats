# agent-notes index

Hand-maintained menu of the notes in this folder. Each note repeats its own summary in a banner at the top of the file. **Dates are each note's last content edit (from git), not a claim it was re-checked against the current code — treat every note skeptically.**

## Petri (reward-hacking audits)

- **CODEBASE_GUIDE.md** — how the Petri code, experiments, and run-dir types are organized, plus the binary reward-hack definition. Read at the start of any Petri work. (updated 2026-07-16)
- **CONTINUATION_EXPERIMENT_SPEC.md** — design of the continuation experiment (does hacking on task A raise the hack rate on task B?). Read when working on continuation runs. Predates the 2026-07-08 `--treatment` rewrite. (updated 2026-07-03)
- **ml_training_data_misuse_DESIGN_CHOICES.md** — pinned conditions + per-seed variations for the seven ml_training_data_misuse seeds, and how to flip each. Read when designing/varying those seeds. (updated 2026-07-08)
- **p_hacking_DESIGN_CHOICES.md** — pinned conditions for the p_hacking seed class (checkout_redesign) plus a judge-rubric caveat. Read when designing or judging p_hacking runs. (updated 2026-07-08)
- **IDEAS.md** — planned p_hacking seed variants (8 candidates). Read when adding p_hacking variants. (updated 2026-07-08)
- **VARIANT_IDEAS.md** — the perf_benchmark_gaming seed class plus six planned variants. Read when building perf_benchmark_gaming seeds. (updated 2026-07-08)

## Cross-environment

- **SHARED_ASK_QUESTIONS.md** — the shared ask-questions endpoint (`shared/exp_ask_questions.py` + per-env adapters): where shared vs env-specific logic lives, viewer schema contracts, the two-venv re-exec gotcha, and the retire-the-originals plan. Read before touching any exp_ask_questions/exp_ask_env file. (updated 2026-07-16)

## PostTrainBench — current pass

- **ISSUES.md** — context-reconstruction issue tracker: per-run fidelity tags + universal caveats, mirrored in the viewer. Read when doing PTB reconstruction / probe work. (updated 2026-07-16)
- **PTB_CONTEXT_VISUALS.md** — token accounting, capacity denominators, event-to-call mapping, compaction rules, and missing-data coverage for Trajectories / visuals. Read when changing the PTB context visuals. (updated 2026-07-16)

## PostTrainBench — first pass (posttrainbench_old/)

- **JUDGING.md** — the multi-stage cascade behind the PTB reward-hack labels. Read when trusting those labels. (updated 2026-07-10)
- **ROADMAP.md** — build-out roadmap for the PTB rollback pipeline (all 30 trajectories). Read when working on rollback experiments. (updated 2026-07-10)
