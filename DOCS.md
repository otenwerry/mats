# Documentation rule

This is where I keep track of how each experiment runs. If you edit some code and make the structure of one of these files change, update this file, but be careful to preserve my format exactly. This means don't add any explanatory language, just keep it to the structural records. If there's stuff you need to keep track of for yourself, write it down elsewhere.

# Notes

- [stuff in brackets=xyz] means the flag is optional, with xyz being the default value.
- Petri judge: petri/lib/judge_models.py or $PETRI_JUDGE; default gpt-5.6-luna.
- Environments judge choices: environments/lib/judge_selection.py; paid endpoints require --judge.

# Environments

## Viewer

uv run viewer.py

## Auditing

uv run exp_real_audit_pipeline.py
    --targets=x,y,z
    --seed-dir=x
    --seeds=x,y,z/all
    --epochs=n
    --harness=simple/production/subscription
    [--condition=allow]
    [--reasoning=yes/no]
    --judge=x
    [--gate-model=<judge>]
    [--concurrency=50]
    [--compute=aws/local]  # default ML aws; p_hacking local
    [--vm-concurrency=50]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--sandbox-concurrency=8]
    [--time-limit=<sec>]  # ML 7200; p_hacking 1800
    [--dry-run]
    [--skip-viewer]

## Continuations

uv run exp_continuation_pipeline.py
    --treatment=<slug>
    --seed-dir=x
    --seeds=x,y,z/all
    --epochs=n
    --harness=simple/production/subscription
    --prefixes=<viewer ids> and/or --prefix-files=<paths>
    [--condition=allow]
    --judge=x
    [--gate-model=<judge>]
    [--concurrency=50]
    [--sandbox-concurrency=8]
    [--compute=aws/local]  # default ML aws; p_hacking local
    [--vm-concurrency=50]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--time-limit=<sec>]  # ML 7200; p_hacking 1800
    [--resume-campaign=<id>]
    [--retry-failed=<id>]
    [--dry-run]
    [--skip-viewer]

## Natural Questions prefix

uv run exp_nq_prefix.py
    --model=<target>
    --tokens=<n>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--seed=1234]
    [--name=<slug>]
    [--max-questions=500]
    [--dry-run]

## Rejudge stored trajectories

uv run exp_rejudge.py
    --source-runs=<all, judge-tests, or real-v* run-dir names>
    [--family=all/ml_training_data_misuse/p_hacking]
    --judge=x
    [--concurrency=10]
    [--dry-run]
    [--force]
    [--skip-viewer]

## Fixed judge-test cohort

uv run exp_judge_tests.py
    [--family=all/ml_training_data_misuse/p_hacking]
    --judge=x
    [--concurrency=10]
    [--dry-run]
    [--force]
    [--skip-viewer]

## AWS setup

aws login --profile mats-environments --region us-west-2
export AWS_PROFILE=mats-environments

uv run exp_real_audit_pipeline.py
    --aws-setup
    (--confirm-approved-account | --confirm-personal-account)
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]

## AWS smoke test

uv run exp_real_audit_pipeline.py
    --aws-smoke-test
    [--dry-run]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]

## AWS campaign recovery

uv run exp_real_audit_pipeline.py --resume-campaign=<id>
uv run exp_real_audit_pipeline.py --retry-failed=<id> --harness=simple/production/subscription

## Generate environment data

uv run envgen/gen_fraud_detection.py
uv run envgen/gen_demand_forecasting.py
uv run envgen/gen_rating_prediction.py
uv run envgen/gen_reasoning_prompt_benchmark.py
uv run envgen/gen_checkout_redesign.py
uv run envgen/gen_retrieval_practice.py

## Calibrate ML environments

./envgen/calibrate.sh <fraud_detection|demand_forecasting|rating_prediction>
    [--sweep t1,t2,t3]

## Tests

uv run -m pytest tests/


# Shared (petri + PTB)

## Ask questions to resumed contexts

uv run shared/exp_ask_questions.py
    --env=<petri or ptb>
    --trajectory=<viewer ids, hacks, run ids, or claude>
    --questions=<em, context, or propensity>
    [--n=1]
    petri only: --reasoning=yes/no  # REQUIRED; no requires --only with numeric questions
    [--turn=end (or a turn/event index, first_hack)]
    [--only=<qid,qid>]
    [--include-inactive]  # propensity only; requires --only
    [--concurrency=50]
    [--timeout=600]
    [--campaign=questions_<set>]
    [--baseline=no/yes/only]
    [--no-judge]
    [--dry-run]
    [--cli=original]  # PostTrainBench only
    [--user-config]  # PostTrainBench only
    [--prefix-cache=auto/required/off]  # PostTrainBench only

# Petri

## Viewer

uv run viewer.py

## Auditing

uv run exp_audit_pipeline.py
    --targets=x,y,z
    --seed-dir=x
    --seeds=x,y,z/all
    --epochs=n
    --max-turns=n
    [--condition=allow/correct]  # default allow
    [--reasoning=yes/no]
    [--auditor=deepseek-v4-pro]
    [--auditor-thinking=yes/no]
    [--judge=gpt-5.6-luna]
    [--concurrency=50]
    [--annotate-model=deepseek-v4-pro]
    [--annotate]
    [--skip-viewer]
    [--force-annotate]

## New judge comparison

uv run exp_new_judge.py
    --judge=x
    --n=n
    [--sample-seed=0]
    [--concurrency=10]
    [--dry-run]
    [--summary-only]

## Continuations

uv run exp_continuation_pipeline.py
    --treatment=<full-hack,etc>
    --prefixes=x,y,z
    --continuations=x,y,z
    --n=n
    [--concurrency=50]
    [--deviation-model=gpt-5.6-luna]
    [--annotate-model=deepseek-v4-pro]
    [--dry-run]
    [--faithfulness-judge]
    [--skip-annotate]
    [--skip-viewer]
    [--force-judge]
    [--force-annotate]

## Rollbacks

uv run exp_rollback_pipeline.py
    --locations=<begin,middle,before,after,all>
    --conditions=<control,treatment,all>
    --N=n
    [--trajectories=all]
    [--concurrency=50]
    [--annotate-model=judge]
    [--dry-run]
    [--skip-judge]
    [--skip-viewer]
    [--force-judge]

## Legacy Petri-only ask runner

uv run exp_ask_questions.py
    --trajectory=<id,id or hacks>
    --questions=em
    [--n=1]
    [--turn=end (or an A-turn number, first_hack)]
    [--only=<qid,qid>]
    [--baseline=no/yes/only]
    [--concurrency=50]
    [--timeout=600]
    [--campaign=questions_<set>]
    [--no-judge]
    [--dry-run]

# PostTrainBench

## Viewer

uv run viewer.py

## Reconstruct context

uv run reconstruct.py
    --trajectory=<run_id>
    [--turn=end (or event_id, first_hack)]

## Legacy PostTrainBench-only ask runner

uv run exp_ask_questions.py
    --trajectory=<run_id,run_id or claude>
    --questions=<em or context>
    [--n=1]
    [--turn=end (or event_id, first_hack)]
    [--only=<qid,qid>]
    [--cli=original]
    [--concurrency=50]
    [--timeout=600]
    [--campaign=questions_<set>]
    [--baseline=no/yes/only]
    [--no-judge]
    [--dry-run]
    [--user-config]
