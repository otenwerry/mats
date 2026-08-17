# Documentation rule

This is where I keep track of how each experiment runs. If you edit some code and make the structure of one of these files change, update this file, but be careful to preserve my format exactly. This means don't add any explanatory language, just keep it to the structural records. If there's stuff you need to keep track of for yourself, write it down elsewhere.

# Notes

- [stuff in brackets=xyz] means the flag is optional, with xyz being the default value.
- Petri judge: petri/lib/judge_models.py or $PETRI_JUDGE; default gpt-5.6-luna.
- Environments judge: environments/lib/judge_selection.py or $ENVIRONMENTS_JUDGE; default gpt-5.6-luna.

# Environments

## Viewer

uv run viewer.py

[pre-launch prefix selections: mats-local/environments/prefix_selections.json]

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
    [--vm-concurrency=250]
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
    [--vm-concurrency=250]
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
    [--vm-concurrency=250]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--time-limit=<sec>]  # ML 4200; p_hacking 1800
    [--resume-campaign=<id>]
    [--retry-failed=<id>]
    [--retry-pipeline-failures]  # only with --retry-failed
    [--allow-incomplete-prefixes]
    [--dry-run]
    [--skip-viewer]

## Multi-agent activity-log transfer

uv run exp_multi_agent_pipeline.py
    --treatment=<slug>
    --seed-dir=x
    --seeds=x,y,z/all
    --epochs=n
    --harness=simple/production/subscription
    --activity-logs=<viewer ids> and/or --activity-log-files=<paths>
    [--condition=allow]
    [--pressure=low/high]  # p_hacking only; default low
    [--judge=gpt-5.6-luna]
    [--gate-model=<judge>]
    [--concurrency=50]
    [--sandbox-concurrency=8]
    [--compute=aws/local]  # default aws for all seed families
    [--vm-concurrency=250]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--time-limit=<sec>]  # ML 4200; p_hacking 1800
    [--resume-campaign=<id>]
    [--retry-failed=<id>]
    [--retry-pipeline-failures]  # only with --retry-failed
    [--dry-run]
    [--skip-viewer]

## Natural Questions prefix

uv run prefixes/exp_nq_prefix.py
    --model=<target>
    --tokens=<n>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--seed=1234]
    [--name=<slug>]
    [--max-questions=500]
    [--dry-run]

## Trajectory prefix export

uv run prefixes/export_trajectory_prefixes.py
    --prefixes=x,y,z
    --prefix-type=<slug>
    --prefix-type-label=<label>
    [--cutoff-before-user-turn=2]
    [--dry-run]

## Activity-log continuation prefix export

uv run prefixes/build_activity_log_prefixes.py
    --trajectories=x,y,z
    [--dry-run]

## Fixed-question prefixes

uv run prefixes/exp_science_ethics_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

uv run prefixes/exp_general_ethics_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

uv run prefixes/exp_move_fast_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--dry-run]

## Wikipedia summary prefix

uv run prefixes/exp_wikipedia_prefix.py
    --model=<target>
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<slug>]
    [--refresh-articles]
    [--dry-run]

## ML prefix-only control

uv run prefixes/exp_ml_prefix.py
    --targets=x,y,z
    --seeds=demand_forecasting/fraud_detection/all
    --epochs=n
    --harness=simple/production/subscription
    [--reasoning=yes/no]
    [--name=<base-slug>]
    [--concurrency=50]
    [--sandbox-concurrency=8]
    [--time-limit=4200]
    [--compute=aws/local]  # default aws
    [--vm-concurrency=250]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--resume-campaign=<id>]
    [--retry-failed=<id>]
    [--skip-viewer]
    [--dry-run]

For this prefix-only endpoint, `--retry-failed` retries both infrastructure failures
and cells whose AWS worker completed with a nonzero or missing pipeline exit (including
agent timeouts). It does not broaden retry policy for ordinary audits or continuations.

To recover a historical prefix whose terminal OpenCode response provably completed
before the deadline but whose CLI handoff exited afterward (free; read-only unless
`--apply`; original logs remain unchanged):

uv run tools/maintenance/recover_ml_prefix_deadline.py <run-dir> [--apply]

uv run exp_continuation_pipeline.py
    --treatment=no-honeypot
    --prefix-files=<ml-prefix-payload>
    --seed-dir=p_hacking
    --seeds=checkout_redesign
    --epochs=n
    --harness=simple/production/subscription
    [--pressure=low/high]
    [--judge=gpt-5.6-luna]
    [--compute=aws/local]
    [--dry-run]

## P-hacking prefix-only controls

uv run prefixes/exp_p_hacking_prefix.py
    --targets=x,y,z
    --seeds=checkout_redesign_positive/all
    --epochs=n
    --harness=simple/production/subscription
    [--pressure=low/high]
    [--reasoning=yes/no]
    [--name=<base-slug>]
    [--concurrency=50]
    [--sandbox-concurrency=8]
    [--time-limit=1800]
    [--compute=aws/local]  # default aws
    [--vm-concurrency=250]
    [--aws-region=us-west-2]
    [--aws-instance-type=c7a.xlarge]
    [--aws-bucket=<name>]
    [--aws-secret-env=NAME,...]
    [--resume-campaign=<id>]
    [--retry-failed=<id>]
    [--skip-viewer]
    [--dry-run]

## Prefix batch

uv run prefixes/exp_prefix_batch.py
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
    [--vm-concurrency=250]
    [--subscription-vm-concurrency=250]
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

## Continuation push batch (2026-08-15)

uv run exp_continuation_push_20260815.py [--plan]

## Demand prefixes + p-hacking no-honeypot to ML batch (2026-08-16)

uv run exp_demand_phacking_to_ml_20260816.py [--plan]

## Rejudge stored trajectories

uv run exp_rejudge.py
    --source-runs=<old, all, judge-tests, or real-v* run-dir names>
    [--family=all/ml_training_data_misuse/p_hacking]
    [--judge=gpt-5.6-luna]
    [--concurrency=10]
    [--only-interrupted-native]
    [--dry-run]
    [--force]
    [--skip-viewer]

Successful rejudges remain comparison-only unless their run directory is explicitly
listed under `promoted_rejudge_run_directories` in `viewer_old_runs.json`. Promotion
replaces the canonical judgment on the source row without modifying either raw log.

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
uv run exp_real_audit_pipeline.py --retry-failed=<id> --retry-pipeline-failures \
    --harness=simple/production/subscription

Resume never launches planned cells. It reconciles uploaded terminal markers even when
EC2 has already aged the terminated worker ID out of `DescribeInstances`, then imports
the retained result and marks untouched planned cells `not_launched` for exact retry.

## Generate environment data

uv run envgen/gen_fraud_detection.py
uv run envgen/gen_demand_forecasting.py
uv run envgen/gen_rating_prediction.py
uv run envgen/gen_reasoning_prompt_benchmark.py
uv run envgen/gen_checkout_redesign.py
uv run envgen/gen_checkout_redesign_no_honeypot.py
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
