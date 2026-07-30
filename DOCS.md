# Notes

- [stuff in brackets=xyz] means the flag is optional, with xyz being the default value.


# Shared

## Ask questions to resumed contexts (single endpoint; supersedes the two per-env files above)
uv run shared/exp_ask_questions.py
    --env=<petri or ptb>   
    --trajectory=<per env: viewer #ids or 'hacks' | run_ids or 'claude'>
    --questions=<em, context, or propensity>
    [--n=1]
    petri only: --reasoning=yes/no   # REQUIRED; no requires --only with numeric questions
    [--turn=end (or a turn/event index, 'first_hack')]
    [--only=<qid,qid>]
    [--include-inactive]   # propensity only; requires --only, permits retained/archived qids
    [--concurrency=50]
    [--timeout=600]
    [--campaign=questions_<set>]   # output subdirectory, under the env's own output root
    [--baseline=no]   # yes = also ask each question without resumed context, per model config; only = just those baselines, no trajectory asks
    [--no-judge, --dry-run]
    posttrainbench only: [--cli=original] [--user-config]
        [--prefix-cache=auto]   # auto = required for propensity, off otherwise; choices: auto|required|off

# Petri

## Viewer
uv run viewer.py

## Auditing
uv run exp_audit_pipeline.py
	--targets=x,y,z --seed-dir=x --seeds=x,y,z/all --epochs=n --max-turns=n
	[--condition=allow/correct]   # pinned seed dirs only; default allow
    [--reasoning=yes/no]
	[--auditor=deepseek-v4-pro]
	[--auditor-thinking=yes/no]
	[--concurrency=50]
	[--annotate-model=deepseek-v4-pro]
	[--skip-annotate, --skip-viewer, --force-annotate]
    hack-turn annotation is AGENTIC: the model gets the judge output + a content-free
        transcript index, then reads the turns it wants (read_turns/search_transcript)
        and submits; its read coverage is stored and shown in the viewer
    ml + p-hacking allow: exact system/opening/ordered optional-follow-up
        compliance is stored per trajectory and flagged in the viewer on failure;
        p-hacking allows FOLLOW_UP.txt followed, if needed, by FOLLOW_UP_FINALIZE.txt

## New judge comparison
uv run exp_new_judge.py
    --judge=x --n=n
    [--sample-seed=0]
    [--concurrency=10]
    [--dry-run, --summary-only]
    samples n each from reward hacks (autonomous + elicited pooled), interesting,
        and clean non-hacks, separately for current ML and p-hacking (6n total)
    saved samples + checkpointed results + analysis + HTML:
        mats-local/petri/new_judge/

## Continuations
uv run exp_continuation_pipeline.py
	--treatment={full-hack,etc} --prefixes=x,y,z --continuations=x,y,z --n=n
	[--concurrency=50]
	[--deviation-model=opus-4.8, --annotate-model=opus-4.8]
	[--dry-run, --faithfulness-judge, --skip-annotate, --skip-viewer, --force-judge, --force-annotate]

## Rollbacks
uv run exp_rollback_pipeline.py
	--locations=(begin,middle,before,after,all) --conditions=(control,treatment,all) --N=n
	[--trajectories=all]
	[--concurrency=50]
	[--annotate-model=judge]
	[--dry-run, --skip-judge, --skip-viewer, --force-judge]

## Legacy Petri-only ask runner (do not use; use the Shared endpoint above)
uv run exp_ask_questions.py
	--trajectory=<id,id or 'hacks'> --questions=em
	[--n=1]
	[--turn=end (or A-turn number, 'first_hack')]
	[--only=<qid,qid>]
	[--baseline=no]   # yes = also ask each question bare (no context), per target model
	[--concurrency=50]
	[--timeout=600]
	[--campaign=questions_<set>]   # output subdirectory
	[--no-judge, --dry-run]



# Environments
(real ports of the petri scenarios: code actually executes in a docker container per
trajectory, user turns are fully scripted, no auditor model. Run from mats/environments/;
re-execs into petri/.venv automatically. Its OWN data root, mats-local/environments/,
holds run dirs + annotations + its own viewer -- nothing mixes with petri.
v1 = fraud_detection + reasoning_prompt_benchmark, allow only.)

## Viewer
uv run viewer.py
    one "current" window; renders an empty skeleton until the first run exists
    loud banners: truncated tool output, annotation read-coverage caveats

## Auditing
uv run exp_real_audit_pipeline.py
	--targets=x,y,z --seed-dir=x --seeds=x,y,z/all --epochs=n --max-turns=n
	[--condition=allow]   # v1: allow only; correct rejected loudly
    [--reasoning=yes/no]
	[--gate-model=opus-4.8]   # must equal the judge model (it is the judge's turn 1)
	[--concurrency=50]
	[--sandbox-concurrency=8]   # live containers; the binding resource, not LLM calls
	[--time-limit=seconds]
	[--annotate-model=deepseek-v4-pro]
	[--skip-annotate, --skip-viewer, --force-annotate]
    requires a running docker daemon (checked before any spend)
    run dirs are mats-local/environments/logs/real-v1-*
    --max-turns caps TARGET GENERATE CALLS (the analog of the auditor turn cap)
    exact system/opening/ordered-follow-up compliance is checked as in petri
    judging is INCREMENTAL: one gate call judges the trajectory through the first
        submission and decides whether the pinned follow-up is sent; the judge then
        replays that exchange and receives the remainder as a second turn, scoring the
        whole trajectory. Stored in metadata.real_env.{gates,judge_replay}; the Result
        carries judge_mode=incremental|single_shot (+ judge_fallback_reason)
    real grade (macro-F1 vs the genuine key), extracted deliverables, and gate evidence
        are stored per trajectory; full artifact copies in logs/real-v1-*/real_artifacts/
    tool output is capped at 16KB per call; any trajectory that hit the cap is flagged
        LOUDLY on its page and badged in the index (the target saw a partial view)

## Gate validation (retrospective, against stored simulated runs)
uv run exp_validate_gate.py
	[--limit-per-seed=30]
	[--gate-model=anthropic/claude-opus-4-8]
	[--dry-run]   # selection only, free
    validates the gate LLM + rubrics against existing judged trajectories; cannot
        validate the mechanical detectors (old runs' answer keys were fictional)

## Generate an environment's data (free, no APIs)
uv run envgen/gen_fraud_detection.py
uv run envgen/gen_reasoning_prompt_benchmark.py
    deterministic; writes seeds/<family>/<member>/{workspace,answers}/ + manifest.json
    committed bytes are canonical -- the invariant tests validate the committed files

## Calibrate the honest ceiling (free, CPU, ML environments only)
./envgen/calibrate.sh
	[--sweep t1,t2,t3]   # hardness-knob sweep instead of the full report
    ALWAYS use the wrapper, not calibrate_fraud.py directly: it pins the sandbox image's
        xgboost/sklearn/numpy and supplies an arm64 libomp on macOS. Run bare and it
        silently falls back to sklearn's booster and marks "definitive": false (a test
        then fails, loudly, rather than letting the fallback pass as real)
    writes calibration_summary.json (committed) + a full report in
        mats-local/petri/env_assets/calibration/; the gate's key-agreement threshold
        comes from it, and tests fail if it is stale
    fraud_detection as of 2026-07-30 (xgboost 2.1.1): honest ceiling 0.834,
        Bayes bound 0.868, both cheat channels 1.00, gate threshold 0.979

## Tests (free; the docker smoke tests skip when no daemon is present)
uv run -m pytest tests/



# PostTrainBench

## Viewer
uv run viewer.py        

## Reconstruct context (free)
uv run reconstruct.py
    --trajectory=<run_id> 
    [--turn=end (or event_id, 'first_hack')]

## Legacy PTB-only ask runner (do not use; use the Shared endpoint above)
uv run exp_ask_questions.py
    --trajectory=<run_id,run_id or 'claude'>
    --questions=<em or context>
    [--n=1] 
    [--turn=end (or event_id, 'first_hack')] 
    [--only=<qid,qid>]    
    [--cli=original]   # pin each run's own claude CLI version (or 'local')
    [--concurrency=50]     
    [--timeout=600]           
    [--campaign=questions_<set>]   # output subdirectory
    [--baseline=no]   # yes = also ask each question in a fresh scaffold session, per model config
    [--no-judge, --dry-run]
    [--user-config] # personal ~/.claude instead of a clean config dir
