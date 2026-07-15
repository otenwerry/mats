# Notes

- [stuff in brackets=xyz] means the flag is optional, with xyz being the default value.

# Petri

## Viewer
uv run viewer.py

## Auditing
uv run exp_audit_pipeline.py
	--targets=x,y,z --seed-dir=x --seeds=x,y,z/all --epochs=n --condition=allow/correct 
    [--reasoning=yes/no]
	[--max-turns=60]
	[--auditor=deepseek-v4-pro]
	[--auditor-thinking=no]
	[--concurrency=50]
	[--annotate-model=opus-4.8]
	[--skip-annotate, --skip-viewer, --force-annotate]

## Continuations
uv run exp_continuation_pipeline.py
	--treatment={full-hack,etc} --prefixes=x,y,z --continuations=x,y,z --n=n
	[--concurrency=50]
	[--deviation-model=opus-4.8, --annotate-model=opus-4.8]
	[--dry-run, --skip-judge, --skip-annotate, --skip-viewer, --force-judge, --force-annotate]

## Rollbacks
uv run exp_rollback_pipeline.py
	--locations=(begin,middle,before,after,all) --conditions=(control,treatment,all) --N=n
	[--trajectories=all]
	[--concurrency=50]
	[--annotate-model=judge]
	[--dry-run, --skip-judge, --skip-viewer, --force-judge]

## Ask propensity questions to resumed targets (auto-judged by the EM judge)
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

## Ask questions to resumed agents (em = auto-judged; context = the reconstruction probe)
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



# Shared (works on every environment)

## Ask questions to resumed contexts (single endpoint; supersedes the two per-env files above)
uv run shared/exp_ask_questions.py
    --env=<petri or posttrainbench>   # 'ptb' works too; picks the right venv itself
    --trajectory=<per env: viewer #ids or 'hacks' | run_ids or 'claude'>
    --questions=<em or context>
    [--n=1]
    [--turn=end (or a turn/event index, 'first_hack')]
    [--only=<qid,qid>]
    [--concurrency=50]
    [--timeout=600]
    [--campaign=questions_<set>]   # output subdirectory, under the env's own output root
    [--baseline=no]   # yes = also ask each question without resumed context, per model config
    [--no-judge, --dry-run]
    posttrainbench only: [--cli=original] [--user-config]