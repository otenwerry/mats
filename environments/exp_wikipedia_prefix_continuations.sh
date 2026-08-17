#!/usr/bin/env zsh

# Run every current Wikipedia-summary prefix into the established
# checkout_redesign continuation task. This is a paid experiment.

set -u

task_script_dir="${0:A:h}"
cd "$task_script_dir" || exit 1

# Refresh the short-lived AWS login. The ordinary campaign preflight below checks
# that the reusable AMI is still current; this script does not build one.
aws login --profile mats-login --region us-west-2 || exit 1

task_prefix_root=../../mats-local/environments/continuation_prefixes

task_production_prefixes=(
  "$task_prefix_root/wikipedia-summaries-deepseek-v4-pro-20260813152931-879fdd0a1177.json"
  "$task_prefix_root/wikipedia-summaries-glm-5-1-20260813152925-57a4301a03d3.json"
  "$task_prefix_root/wikipedia-summaries-kimi-k2-6-20260813152919-e60b6b671381.json"
)

task_subscription_prefixes=(
  "$task_prefix_root/wikipedia-summaries-gpt-5-5-20260813152916-a594dd77d52d.json"
  "$task_prefix_root/wikipedia-summaries-opus-4-6-20260813151057-7da4422e28a7.json"
)

task_pids=()

run_continuations() {
  local task_harness="$1"
  local task_vm_concurrency="$2"
  local task_prefix_csv="$3"

  uv run exp_continuation_pipeline.py \
    --treatment=wikipedia-summaries \
    --prefix-files="$task_prefix_csv" \
    --seed-dir=p_hacking \
    --seeds=checkout_redesign \
    --epochs=40 \
    --harness="$task_harness" \
    --condition=allow \
    --pressure=low \
    --judge=gpt-5.6-luna \
    --gate-model=gpt-5.6-luna \
    --compute=aws \
    --vm-concurrency="$task_vm_concurrency" \
    --aws-region=us-west-2 \
    --aws-instance-type=c7a.xlarge \
    --time-limit=1800 \
    --skip-viewer &

  task_pids+=($!)
}

# The two campaigns run together. Their 120 + 80 cells fit under the 250-VM
# account budget, so every cell can start immediately.
run_continuations \
  production 120 \
  "${(j:,:)task_production_prefixes}"

run_continuations \
  subscription 80 \
  "${(j:,:)task_subscription_prefixes}"

task_status=0
for task_pid in "${task_pids[@]}"; do
  wait "$task_pid" || task_status=1
done

uv run viewer.py || task_status=1
exit "$task_status"
