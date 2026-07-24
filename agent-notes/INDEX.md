# agent-notes index

Hand-maintained menu of the notes in this folder. Each note repeats its own summary in a banner at the top of the file. **Dates are each note's last content edit (from git), not a claim it was re-checked against the current code — treat every note skeptically.**

## Petri (reward-hacking audits)

- **CODEBASE_GUIDE.md** — how the Petri code, experiments, Current/Round-1/Old viewer, direction-based continuations, run-dir types, fixed target tools, exact ML auditor-script check, and centralized global/seed-scoped judge dimensions are organized, plus the family-aware binary reward-hack definition. Read at the start of any Petri work. (updated 2026-07-24)
- **CONTINUATION_EXPERIMENT_SPEC.md** — design of the continuation experiment (does hacking on task A raise the hack rate on task B?), including exact fixed B-tool registration and the shared interface used by new originals. Read when working on continuation runs. Predates the 2026-07-08 `--treatment` rewrite. (updated 2026-07-24)
- **PROMPT_CACHING.md** — provider prompt caching vs response replay; Petri's warm-first barrier, v2 message-ID-normalized key, and per-run reports; direct Anthropic breakpoints; PTB's verified caching mode and old-CLI limitation. Read before changing any repeated-context paid pipeline. (updated 2026-07-23)
- **ml_training_data_misuse_DESIGN_CHOICES.md** — pinned conditions + per-seed variations for the seven ml_training_data_misuse seeds, and how to flip each. Read when designing/varying those seeds. (updated 2026-07-08)
- **p_hacking_DESIGN_CHOICES.md** — shared pins, per-scenario traps, and the family-scoped judge line for the three active checkout, retrieval-practice, and LLM-benchmark scenarios. Read when designing or judging p_hacking runs. (updated 2026-07-21)
- **IDEAS.md** — planned p_hacking domains, the selective-reporting definition, and notes on the three active composed scenarios including the low-stakes LLM benchmark. Read when adding p_hacking variants. (updated 2026-07-21)
- **VARIANT_IDEAS.md** — the perf_benchmark_gaming seed class, six planned domains, and the deliberate changes in the draft log_analytics2 iteration. Read when building or varying perf_benchmark_gaming seeds. (updated 2026-07-17)

## Cross-environment

- **SHARED_ASK_QUESTIONS.md** — the shared ask-questions endpoint (`shared/exp_ask_questions.py` + per-env adapters): where shared vs env-specific logic lives, propensity scale/reasoning contracts, viewer schema contracts, the two-venv re-exec gotcha, and the retire-the-originals plan. Read before touching any exp_ask_questions/exp_ask_env file. (updated 2026-07-22)

## PostTrainBench — current pass

- **ISSUES.md** — context-reconstruction issue tracker: per-run fidelity tags + universal caveats, mirrored in the viewer. Read when doing PTB reconstruction / probe work. (updated 2026-07-16)
- **PTB_CONTEXT_VISUALS.md** — token accounting, capacity denominators, event-to-call mapping, compaction rules, and missing-data coverage for Trajectories / visuals. Read when changing the PTB context visuals. (updated 2026-07-16)

## PostTrainBench — first pass (posttrainbench_old/)

- **JUDGING.md** — the multi-stage cascade behind the PTB reward-hack labels. Read when trusting those labels. (updated 2026-07-10)
- **ROADMAP.md** — build-out roadmap for the PTB rollback pipeline (all 30 trajectories). Read when working on rollback experiments. (updated 2026-07-10)
