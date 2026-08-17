#!/usr/bin/env zsh

# HISTORICAL ONE-OFF — NOT A TEMPLATE AND NOT REUSABLE. Its durable start marker
# intentionally prevents a second run. The run exposed a cross-controller AWS login
# refresh stampede that is documented in agent-notes/ENVIRONMENT_BATCH_WRAPPERS.md;
# core credential handling was fixed afterward. This file is retained only to preserve
# the exact 2026-08-15 recovery matrix and command provenance.
#
# It retried the exact failed or never-launched cells from the 2026-08-15 slate,
# excluding every Opus campaign. This is a paid experiment. --plan remains a free local
# verification of the immutable parent states and exact retry counts.

set -u
setopt pipefail
setopt no_bg_nice

task_script_path="${0:A}"
task_script_dir="${task_script_path:h}"
cd "$task_script_dir" || exit 1

task_mode="run"
if (( $# > 1 )); then
  print -u2 "Usage: ${0:t} [--plan]"
  exit 2
elif (( $# == 1 )); then
  [[ "$1" == "--plan" ]] || {
    print -u2 "Usage: ${0:t} [--plan]"
    exit 2
  }
  task_mode="plan"
fi

# Controllers must remain awake to launch later VM waves, import results, and
# clean their S3 campaign prefixes.
if [[ "$task_mode" == "run" && -z "${MATS_NONOPUS_RECOVERY_CAFFEINATED:-}" ]] && \
   command -v caffeinate >/dev/null; then
  exec env MATS_NONOPUS_RECOVERY_CAFFEINATED=1 caffeinate -i \
    "$task_script_path" "$@"
fi

task_data_root="${task_script_dir}/../../mats-local/environments"
task_data_root="${task_data_root:A}"
task_campaign_root="$task_data_root/remote_campaigns"
task_run_root="$task_data_root/overnight_runs"
task_stamp="$(date '+%Y%m%d%H%M%S')"
task_run_slug="tonight-nonopus-recovery-20260815-${task_stamp}"
task_run_dir="$task_run_root/$task_run_slug"
task_lock_dir="$task_run_root/.tonight-nonopus-recovery-20260815.lock"
task_started_marker="$task_run_root/.tonight-nonopus-recovery-20260815.started"
task_launch_stagger_seconds=30

# These are the finalized original campaign states. retry_failed reconstructs
# exact (payload, seed, original_epoch) selections, so successful cells are not
# duplicated. --retry-pipeline-failures additionally opts in the completed cells
# whose imported pipeline exit code was nonzero.
typeset -a task_labels=(
  wiki-production
  wiki-gpt
  phack-ml-hack1-production
  phack-ml-hack2-production
  phack-ml-hack2-gpt
  phack-ml-nohack-production
  phack-ml-nohack-gpt
  multi-hack1-production
  multi-hack2-production
  multi-hack2-gpt
  multi-nohack-production
  multi-nohack-gpt
)
typeset -a task_endpoints=(
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_continuation_pipeline.py
  exp_multi_agent_pipeline.py
  exp_multi_agent_pipeline.py
  exp_multi_agent_pipeline.py
  exp_multi_agent_pipeline.py
  exp_multi_agent_pipeline.py
)
typeset -a task_campaign_ids=(
  continuation-aws-wikipedia-summaries-40ep-20260815-015949-82cceb1f
  continuation-aws-wikipedia-summaries-40ep-20260815-015950-7257e74c
  continuation-aws-hack-in-one-turn-40ep-20260815-015952-72c5744f
  continuation-aws-hack-in-two-turns-40ep-20260815-015952-d018a77c
  continuation-aws-hack-in-two-turns-40ep-20260815-015951-96acfbd5
  continuation-aws-no-hack-40ep-20260815-015952-474ed33c
  continuation-aws-no-hack-40ep-20260815-015951-c7e12a59
  multi-agent-aws-hack-in-one-turn-40ep-20260815-015952-0e5b9342
  multi-agent-aws-hack-in-two-turns-40ep-20260815-015952-a0d4e8f1
  multi-agent-aws-hack-in-two-turns-40ep-20260815-015952-031d8132
  multi-agent-aws-no-hack-40ep-20260815-015952-c6efb52c
  multi-agent-aws-no-hack-40ep-20260815-015951-8e389955
)
typeset -a task_harnesses=(
  production subscription production production subscription production
  subscription production production subscription production subscription
)
typeset -a task_expected_cells=(118 10 33 115 19 45 17 95 120 40 52 40)
typeset -a task_vm_caps=(42 8 33 42 6 28 6 27 27 2 27 2)

task_expected_total=704
task_vm_cap_total=250

log() {
  print -- "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

die() {
  log "STOPPED: $*"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null || die "required command is unavailable: $1"
}

retryable_count() {
  jq -r '
    [.cells[] | select(
      .status == "not_launched"
      or .status == "infrastructure_failure"
      or (
        .status == "completed"
        and ((try (.terminal.pipeline_exit_code | tonumber) catch 1) != 0)
      )
    )] | length
  ' "$1"
}

validate_parent() {
  local task_index="$1"
  local task_label="$task_labels[$task_index]"
  local task_campaign_id="$task_campaign_ids[$task_index]"
  local task_harness="$task_harnesses[$task_index]"
  local task_expected="$task_expected_cells[$task_index]"
  local task_state="$task_campaign_root/$task_campaign_id.json"
  local task_actual

  [[ -f "$task_state" ]] || die "missing parent state for $task_label: $task_state"
  jq -e \
    --arg task_campaign_id "$task_campaign_id" \
    --arg task_harness "$task_harness" '
      .campaign_id == $task_campaign_id
      and .pipeline_config.harness == $task_harness
      and ([.cells[].target] | all(test("opus"; "i") | not))
      and ([.cells[].status] | all(
        . == "completed"
        or . == "not_launched"
        or . == "infrastructure_failure"
      ))
    ' "$task_state" >/dev/null || die \
      "parent identity, harness, non-Opus scope, or terminal state changed: $task_state"

  task_actual="$(retryable_count "$task_state")" || die \
    "could not count retryable cells in $task_state"
  [[ "$task_actual" == "$task_expected" ]] || die \
    "$task_label now selects $task_actual cells; pinned recovery expects $task_expected"

  # Retry payloads are immutable local inputs referenced by the stored campaign.
  jq -r '
    .pipeline_config.continuation.payloads[]?.local_path,
    .pipeline_config.multi_agent.payloads[]?.local_path
  ' "$task_state" | while IFS= read -r task_payload; do
    [[ -z "$task_payload" || -f "$task_payload" ]] || die \
      "stored retry payload is missing: $task_payload"
  done
}

(( ${#task_labels} == ${#task_endpoints} )) || die "internal endpoint count mismatch"
(( ${#task_labels} == ${#task_campaign_ids} )) || die "internal campaign count mismatch"
(( ${#task_labels} == ${#task_harnesses} )) || die "internal harness count mismatch"
(( ${#task_labels} == ${#task_expected_cells} )) || die "internal cell count mismatch"
(( ${#task_labels} == ${#task_vm_caps} )) || die "internal VM-cap count mismatch"
task_computed_cells=0
task_computed_vm_caps=0
for task_value in $task_expected_cells; do
  (( task_computed_cells += task_value ))
done
for task_value in $task_vm_caps; do
  (( task_computed_vm_caps += task_value ))
done
(( task_computed_cells == task_expected_total )) || die \
  "internal retry total is not $task_expected_total"
(( task_computed_vm_caps == task_vm_cap_total )) || die \
  "internal VM-cap total is not $task_vm_cap_total"

for (( task_index = 1; task_index <= ${#task_labels}; task_index++ )); do
  validate_parent "$task_index"
done

if [[ "$task_mode" == "plan" ]]; then
  for (( task_index = 1; task_index <= ${#task_labels}; task_index++ )); do
    print -- "[$task_labels[$task_index]] retry_cells=$task_expected_cells[$task_index] vm_cap=$task_vm_caps[$task_index]"
    print -- "  uv run $task_endpoints[$task_index] --retry-failed=$task_campaign_ids[$task_index] --retry-pipeline-failures --harness=$task_harnesses[$task_index] --vm-concurrency=$task_vm_caps[$task_index] --skip-viewer"
  done
  print
  print -- "TOTAL controllers=${#task_labels} exact_retry_cells=$task_expected_total vm_caps=$task_vm_cap_total"
  print -- "Scope: non-Opus only; 552 never launched + 152 completed pipeline failures."
  print -- "No AWS, VM, model, judge, or filesystem write was performed."
  exit 0
fi

for task_command in aws uv jq tee; do
  require_command "$task_command"
done

mkdir -p "$task_run_root" || die "could not create $task_run_root"
if [[ -f "$task_started_marker" ]]; then
  die "this exact recovery slate was already started; see $(<"$task_started_marker")"
fi
if ! mkdir "$task_lock_dir" 2>/dev/null; then
  die "another non-Opus recovery wrapper may be running: $task_lock_dir"
fi
cleanup_lock() {
  rmdir "$task_lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT

mkdir -p "$task_run_dir" || die "could not create $task_run_dir"
task_orchestrator_log="$task_run_dir/orchestrator.log"
exec > >(tee -a "$task_orchestrator_log") 2>&1

log "Run directory: $task_run_dir"
log "Exact recovery: $task_expected_total non-Opus cells; maximum $task_vm_cap_total VMs"
log "Selection: 552 never launched + 152 completed pipeline failures"
log "Controllers start $task_launch_stagger_seconds seconds apart to avoid an EC2 launch burst"
log "Output streams live with campaign labels and is retained in per-campaign logs"

# The user authenticates the login profile once. Controllers use mats-run through
# the refreshable AWS CLI export provider; no separate shell export is required.
log "Opening/refreshing AWS login profile mats-login."
aws login --profile mats-login --region us-west-2 || die "AWS login failed"
export AWS_PROFILE=mats-run
aws sts get-caller-identity \
  --profile mats-run \
  --region us-west-2 \
  >/dev/null || die "AWS run-profile authentication failed for mats-run"

log "Refreshing the AWS runtime contract, worker secrets, and subscription credentials."
(
  setopt pipefail
  PYTHONUNBUFFERED=1 uv run exp_real_audit_pipeline.py \
    --aws-setup \
    --confirm-personal-account \
    --harness=subscription \
    --aws-region=us-west-2 \
    --aws-instance-type=c7a.xlarge \
    2>&1 | tee "$task_run_dir/aws-setup.log"
) || die "AWS setup failed; no recovery controller was started"
print -r -- "$task_run_dir" > "$task_started_marker"

typeset -a task_pids
typeset -a task_logs

start_controller() {
  local task_index="$1"
  local task_label="$task_labels[$task_index]"
  local task_endpoint="$task_endpoints[$task_index]"
  local task_campaign_id="$task_campaign_ids[$task_index]"
  local task_harness="$task_harnesses[$task_index]"
  local task_cells="$task_expected_cells[$task_index]"
  local task_cap="$task_vm_caps[$task_index]"
  local task_log_path="$task_run_dir/$task_label.log"

  task_logs+=("$task_log_path")
  log "Starting $task_label — retry_cells=$task_cells vm_cap=$task_cap log=$task_log_path"
  (
    setopt pipefail
    PYTHONUNBUFFERED=1 uv run "$task_endpoint" \
      --retry-failed="$task_campaign_id" \
      --retry-pipeline-failures \
      --harness="$task_harness" \
      --vm-concurrency="$task_cap" \
      --skip-viewer \
      2>&1 | while IFS= read -r task_line; do
        print -r -- "[$task_label] $task_line"
      done | tee "$task_log_path"
    task_status=$pipestatus[1]
    print -- "[$task_label] controller exited — status=$task_status"
    exit "$task_status"
  ) &
  task_pids+=($!)
}

for (( task_index = 1; task_index <= ${#task_labels}; task_index++ )); do
  start_controller "$task_index"
  if (( task_index < ${#task_labels} )); then
    sleep "$task_launch_stagger_seconds"
  fi
done

log "All ${#task_pids} recovery controllers have started."
log "A failed controller will not cancel or automatically rerun any other controller."

report_running_controllers() {
  local task_running
  while true; do
    sleep 180
    task_running=0
    for task_pid in $task_pids; do
      if kill -0 "$task_pid" 2>/dev/null; then
        (( task_running += 1 ))
      fi
    done
    (( task_running > 0 )) || return 0
    log "Recovery controllers still running: $task_running/${#task_pids}"
  done
}
report_running_controllers &
task_reporter_pid=$!

typeset -a task_statuses
task_any_failure=0
for (( task_index = 1; task_index <= ${#task_pids}; task_index++ )); do
  if wait "$task_pids[$task_index]"; then
    task_status=0
  else
    task_status=$?
    task_any_failure=1
  fi
  task_statuses+=("$task_status")
done
kill "$task_reporter_pid" 2>/dev/null || true
wait "$task_reporter_pid" 2>/dev/null || true

for (( task_index = 1; task_index <= ${#task_labels}; task_index++ )); do
  log "Finished $task_labels[$task_index] — exit=$task_statuses[$task_index]"
done

log "All recovery controllers exited. Building the viewer once."
(
  setopt pipefail
  PYTHONUNBUFFERED=1 uv run viewer.py 2>&1 | tee "$task_run_dir/viewer.log"
)
task_viewer_status=$?
(( task_viewer_status == 0 )) || task_any_failure=1
log "Viewer exited — status=$task_viewer_status"

task_status_rows=()
for (( task_index = 1; task_index <= ${#task_labels}; task_index++ )); do
  task_status_rows+=("$(jq -n \
    --arg label "$task_labels[$task_index]" \
    --arg parent_campaign "$task_campaign_ids[$task_index]" \
    --arg log "$task_logs[$task_index]" \
    --argjson cells "$task_expected_cells[$task_index]" \
    --argjson vm_cap "$task_vm_caps[$task_index]" \
    --argjson exit_status "$task_statuses[$task_index]" \
    '{label:$label, parent_campaign:$parent_campaign, retry_cells:$cells,
      vm_cap:$vm_cap, exit_status:$exit_status, log:$log}')")
done

print -rl -- $task_status_rows | jq -s \
  --arg run_directory "$task_run_dir" \
  --argjson retry_cells "$task_expected_total" \
  --argjson maximum_vms "$task_vm_cap_total" \
  --argjson viewer_exit_status "$task_viewer_status" '
    {
      format: "environments-nonopus-recovery-status-v1",
      run_directory: $run_directory,
      retry_cells: $retry_cells,
      maximum_vms: $maximum_vms,
      opus_deferred: true,
      automatic_campaign_retries: false,
      viewer_exit_status: $viewer_exit_status,
      jobs: .,
      success: (([.[].exit_status] | all(. == 0)) and $viewer_exit_status == 0)
    }
  ' > "$task_run_dir/final_status.json"

log "Final status: $task_run_dir/final_status.json"
if (( task_any_failure != 0 )); then
  log "DONE WITH FAILURES. Evidence was retained; inspect the labeled logs before any further exact-cell retry."
  exit 1
fi
log "DONE. All exact non-Opus recovery campaigns completed successfully."
