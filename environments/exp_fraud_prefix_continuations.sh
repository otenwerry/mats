#!/usr/bin/env zsh

# Run every current ML fraud-detection trajectory prefix into the established
# checkout_redesign continuation task. This is a paid experiment.

set -u

task_script_dir="${0:A:h}"
cd "$task_script_dir" || exit 1

aws login --profile mats-login --region us-west-2 || exit 1

uv run exp_real_audit_pipeline.py \
  --aws-setup \
  --confirm-personal-account \
  --harness=subscription || exit 1

task_prefix_root=../../mats-local/environments/continuation_prefixes

task_hack1_prod_prefixes=(
  "$task_prefix_root/traj2856-732044f418d1.json"
  "$task_prefix_root/traj2893-07aaa935470f.json"
  "$task_prefix_root/traj2940-329c6a5944c0.json"
)

task_hack2_prod_prefixes=(
  "$task_prefix_root/traj2852-410ea0c1076e.json"
  "$task_prefix_root/traj2917-d90b9d331294.json"
  "$task_prefix_root/traj2946-e107bb12a383.json"
)

task_hack2_sub_prefixes=(
  "$task_prefix_root/traj2997-1459e9cf8b54.json"
)

task_clean_prod_prefixes=(
  "$task_prefix_root/traj2858-44ac51fb0476.json"
  "$task_prefix_root/traj2858-cutoff-u2-c56b00675b91.json"
  "$task_prefix_root/traj2918-651552867f73.json"
  "$task_prefix_root/traj2918-cutoff-u2-95b45f426e0c.json"
  "$task_prefix_root/traj2933-5d6f23fc89ee.json"
  "$task_prefix_root/traj2933-cutoff-u2-44b08e4b2147.json"
)

task_clean_sub_prefixes=(
  "$task_prefix_root/traj2972-8d96424f4675.json"
  "$task_prefix_root/traj2972-cutoff-u2-a30630612ab7.json"
)

task_pids=()

run_continuations() {
  local task_treatment="$1"
  local task_harness="$2"
  local task_vm_concurrency="$3"
  local task_prefix_csv="$4"

  uv run exp_continuation_pipeline.py \
    --treatment="$task_treatment" \
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

# The five simultaneous campaigns preserve the prior relative allocation while
# sharing the full 250-VM account budget (57 + 63 + 10 + 97 + 23).
run_continuations \
  hack-in-one-turn production 57 \
  "${(j:,:)task_hack1_prod_prefixes}"

run_continuations \
  hack-in-two-turns production 63 \
  "${(j:,:)task_hack2_prod_prefixes}"

run_continuations \
  hack-in-two-turns subscription 10 \
  "${(j:,:)task_hack2_sub_prefixes}"

run_continuations \
  no-hack production 97 \
  "${(j:,:)task_clean_prod_prefixes}"

run_continuations \
  no-hack subscription 23 \
  "${(j:,:)task_clean_sub_prefixes}"

task_status=0
for task_pid in "${task_pids[@]}"; do
  wait "$task_pid" || task_status=1
done

uv run viewer.py || task_status=1
exit "$task_status"
