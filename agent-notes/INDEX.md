# agent-notes index

Hand-maintained menu of the notes in this folder. Each note repeats its own summary in a banner at the top of the file. **Dates are each note's last content edit (from git), not a claim it was re-checked against the current code — treat every note skeptically.**

## Petri (reward-hacking audits)

- **REAL_ENV_GUIDE.md** — `mats/environments/`, the real-environment mode (docker sandbox + scripted controller, no auditor): layout after the 2026-07-30 move out of petri/, what is byte-shared with simulated audits, the INCREMENTAL two-stage judge (gate call = judge's turn 1, replayed then continued; gate model must equal judge model), the grader contract, fraud calibration numbers, viewer routing (real- prefix window, DECLARED_ONLY criteria), and what is deferred (correct condition, other seeds, GPU). Read before touching anything under environments/. (updated 2026-07-30)
- **CODEBASE_GUIDE.md** — how the Petri code, experiments, Current/Round-1/Old viewer (including shared Current/Round-1 past-iteration archive views and its ownership/navigation/link-validation contract), explicit audit-integrity gates and diagnosed-dead omission policy, uncapped auditor resume output, concurrency-safe viewer cache/build lock, original-audit context distributions (percentage and absolute tokens), model-by-seed outcomes and family-labeled reward-hacking-rate panels, candidate-judge comparisons, direction-based continuations and their filtered/all-trajectories visual switch, run-dir types, fixed target tools, exact ML auditor-script check, and centralized global/seed-scoped judge dimensions are organized, plus the main viewer's auditor-mistakes <=2 boundary and the intentionally separate binary-selection definition. Read at the start of any Petri work. (updated 2026-07-30)
- **CONTINUATION_EXPERIMENT_SPEC.md** — design of the continuation experiment (does hacking on task A raise the hack rate on task B?), including exact fixed B-tool registration and the shared interface used by new originals. Read when working on continuation runs. Predates the 2026-07-08 `--treatment` rewrite. (updated 2026-07-24)
- **PROMPT_CACHING.md** — provider prompt caching vs response replay; Petri's warm-first barrier, v2 message-ID-normalized key, and per-run reports; direct Anthropic breakpoints; PTB's verified caching mode and old-CLI limitation. Read before changing any repeated-context paid pipeline. (updated 2026-07-23)
- **ml_training_data_misuse_DESIGN_CHOICES.md** — pinned conditions + per-seed variations for the seven ml_training_data_misuse seeds, and how to flip each. Read when designing/varying those seeds. (updated 2026-07-24)
- **p_hacking_DESIGN_CHOICES.md** — shared pins, exact auditor protocol, per-scenario traps, and the family-scoped judge line for the three active checkout, retrieval-practice, and LLM-benchmark scenarios. Read when designing or judging p_hacking runs. (updated 2026-07-24)
- **IDEAS.md** — planned p_hacking domains, the selective-reporting definition, and notes on the three active composed scenarios including the low-stakes LLM benchmark. Read when adding p_hacking variants. (updated 2026-07-21)
- **VARIANT_IDEAS.md** — the perf_benchmark_gaming seed class, six planned domains, and the deliberate changes in the draft log_analytics2 iteration. Read when building or varying perf_benchmark_gaming seeds. (updated 2026-07-24)

## Cross-environment

- **SHARED_ASK_QUESTIONS.md** — the shared ask-questions endpoint (`shared/exp_ask_questions.py` + per-env adapters): where shared vs env-specific logic lives, propensity scale/reasoning contracts, viewer schema contracts, the two-venv re-exec gotcha, and the retire-the-originals plan. Read before touching any exp_ask_questions/exp_ask_env file. (updated 2026-07-22)

## PostTrainBench — current pass

- **ISSUES.md** — context-reconstruction issue tracker: per-run fidelity tags + universal caveats, mirrored in the viewer. Read when doing PTB reconstruction / probe work. (updated 2026-07-16)
- **PTB_CONTEXT_VISUALS.md** — token accounting, capacity denominators, event-to-call mapping, compaction rules, and missing-data coverage for Trajectories / visuals. Read when changing the PTB context visuals. (updated 2026-07-16)

## PostTrainBench — first pass (posttrainbench_old/)

- **JUDGING.md** — the multi-stage cascade behind the PTB reward-hack labels. Read when trusting those labels. (updated 2026-07-10)
- **ROADMAP.md** — build-out roadmap for the PTB rollback pipeline (all 30 trajectories). Read when working on rollback experiments. (updated 2026-07-10)
