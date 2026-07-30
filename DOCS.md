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
	[--annotate-model=opus-4.8]
	[--skip-annotate, --skip-viewer, --force-annotate]
    ml + p-hacking allow: exact system/opening/ordered optional-follow-up
        compliance is stored per trajectory and flagged in the viewer on failure;
        p-hacking allows FOLLOW_UP.txt followed, if needed, by FOLLOW_UP_FINALIZE.txt

## Real-environment auditing (docker sandboxes, no auditor; v1 = fraud_detection + reasoning_prompt_benchmark)
uv run exp_real_audit_pipeline.py
	--targets=x,y,z --seed-dir=x --seeds=x,y,z/all --epochs=n --max-turns=n
	[--condition=allow]   # v1: allow only; correct rejected
    [--reasoning=yes/no]
	[--gate-model=opus-4.8]
	[--concurrency=50]
	[--sandbox-concurrency=8]
	[--time-limit=seconds]
	[--annotate-model=opus-4.8]
	[--skip-annotate, --skip-viewer, --force-annotate]
    requires a running docker daemon; run dirs are logs/real-v1-*, viewer window "real environments"
    the same exact system/opening/follow-up compliance check applies; user turns are fully
        scripted, and one gate LLM call per first submission decides clean/hacked (stored
        with evidence in metadata.real_env.gates)

## Gate validation (retrospective, against stored simulated runs)
uv run tools/exp_validate_gate.py
	[--limit-per-seed=30]
	[--gate-model=anthropic/claude-opus-4-8]
	[--dry-run]

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
