# Documentation rule

This is where I keep track of how each experiment runs. If you edit some code and make the structure of one of these files change, update this file, but be careful to preserve my format exactly. This means don't add any explanatory language, just keep it to the structural records. If there's stuff you need to keep track of for yourself, write it down elsewhere.

# Notes

- [stuff in brackets=xyz] means the flag is optional, with xyz being the default value.
- Petri judge: petri/lib/judge_models.py or $PETRI_JUDGE; default gpt-5.6-luna.
- Environments judge: environments/lib/judge_selection.py or $ENVIRONMENTS_JUDGE; default gpt-5.6-luna.

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
    [--judge=gpt-5.6-luna]
    [--condition=allow]
    [--pressure=low/high]  # p_hacking only; default low
    [--reasoning=yes/no]
    [--gate-model=<judge>]
    [--concurrency=50]
    [--compute=aws/local]  # default aws for all seed families
    [--vm-concurrency=75]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--sandbox-concurrency=8]
    [--time-limit=<sec>]  # ML 4200; p_hacking 1800
    [--dry-run]
    [--skip-viewer]

## Fraud detection base-rate batch

uv run exp_fraud_base_rate_batch.py
    [--epochs=40]
    [--vm-concurrency=75]
    [--reasoning=yes/no]
    [--judge=gpt-5.6-luna]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--skip-viewer]
    [--dry-run]

## Continuations

uv run exp_continuation_pipeline.py
    --treatment=<slug>
    --seed-dir=x
    --seeds=x,y,z/all
    --epochs=n
    --harness=simple/production/subscription
    --prefixes=<viewer ids> and/or --prefix-files=<paths>
    [--condition=allow]
    [--pressure=low/high]  # p_hacking only; default low
    [--judge=gpt-5.6-luna]
    [--gate-model=<judge>]
    [--concurrency=50]
    [--sandbox-concurrency=8]
    [--compute=aws/local]  # default aws for all seed families
    [--vm-concurrency=75]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--time-limit=<sec>]  # ML 4200; p_hacking 1800
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

## Fixed-question prefixes

uv run exp_science_ethics_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

uv run exp_general_ethics_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

uv run exp_move_fast_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

## Prefix batch

uv run exp_prefix_batch.py
    [--nq-tokens=2000]
    [--nq-seed=1234]
    [--reasoning=yes/no]
    [--concurrency=20]
    [--simple-only]
    [--resume-batch=<batch-id-or-manifest-path>]
    [--also-simple]
    [--skip-viewer]
    [--dry-run]

## Continuation prefix batch

uv run exp_continuation_prefix_batch.py
    [--epochs=40]
    [--vm-concurrency=75]
    [--subscription-vm-concurrency=75]
    [--infrastructure-retries=1]
    [--treatment=purpose-built-prefix]
    [--native-only]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--resume-batch=<batch-id-or-manifest-path>]
    [--skip-viewer]
    [--dry-run]

## Rejudge stored trajectories

uv run exp_rejudge.py
    --source-runs=<old, all, judge-tests, or real-v* run-dir names>
    [--family=all/ml_training_data_misuse/p_hacking]
    [--judge=gpt-5.6-luna]
    [--concurrency=10]
    [--dry-run]
    [--force]
    [--skip-viewer]

## Fixed judge-test cohort

uv run exp_judge_tests.py
    [--family=all/ml_training_data_misuse/p_hacking]
    [--judge=gpt-5.6-luna]
    [--concurrency=10]
    [--dry-run]
    [--force]
    [--skip-viewer]

## AWS setup

aws login --profile mats-login --region us-west-2

uv run exp_real_audit_pipeline.py
    --aws-setup
    (--confirm-approved-account | --confirm-personal-account)
    [--harness=subscription]
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
