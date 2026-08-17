#!/usr/bin/env zsh

# Wait for the exact in-flight Wikipedia campaigns, generate three no-honeypot
# fraud-detection prefixes per non-Opus agent, select the first mechanically
# eligible prefix per agent, and run 40 checkout-redesign continuations each.
# This is a paid experiment. Launch it manually while AWS login is interactive.

set -u
setopt pipefail

task_script_path="${0:A}"
task_script_dir="${task_script_path:h}"
cd "$task_script_dir" || exit 1

# The AWS controllers must stay awake locally to launch later waves and import
# results. Re-exec once under macOS's idle-sleep inhibitor.
if [[ -z "${MATS_OVERNIGHT_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  exec env MATS_OVERNIGHT_CAFFEINATED=1 caffeinate -i "$task_script_path" "$@"
fi

task_data_root="${task_script_dir}/../../mats-local/environments"
task_data_root="${task_data_root:A}"
task_campaign_root="$task_data_root/remote_campaigns"
task_prefix_root="$task_data_root/continuation_prefixes"
task_overnight_root="$task_data_root/overnight_runs"

task_wikipedia_subscription_id="continuation-aws-wikipedia-summaries-40ep-20260814-003032-270714eb"
task_wikipedia_production_id="continuation-aws-wikipedia-summaries-40ep-20260814-003032-bc3aff05"
task_wikipedia_subscription_state="$task_campaign_root/$task_wikipedia_subscription_id.json"
task_wikipedia_production_state="$task_campaign_root/$task_wikipedia_production_id.json"

task_wait_seconds=60
task_wait_timeout_seconds=$((4 * 60 * 60))
task_stamp="$(date '+%Y%m%d%H%M%S')"
task_run_slug="fraud-no-honeypot-${task_stamp}"
task_run_dir="$task_overnight_root/$task_run_slug"
task_lock_dir="$task_overnight_root/.fraud-no-honeypot-overnight.lock"

mkdir -p "$task_overnight_root" || {
  print -u2 "Could not create overnight-run root: $task_overnight_root"
  exit 1
}
if ! mkdir "$task_lock_dir" 2>/dev/null; then
  print -u2 "Another fraud no-honeypot overnight wrapper may already be running:"
  print -u2 "  $task_lock_dir"
  exit 1
fi

cleanup_lock() {
  rmdir "$task_lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT

mkdir -p "$task_run_dir" || {
  print -u2 "Could not create overnight-run directory: $task_run_dir"
  exit 1
}
task_log="$task_run_dir/orchestrator.log"
exec > >(tee -a "$task_log") 2>&1

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

for task_command in aws uv jq git shasum tee; do
  require_command "$task_command"
done

source_fingerprint() {
  {
    git rev-parse HEAD
    git diff HEAD --binary -- .
    git ls-files --others --exclude-standard -- . | LC_ALL=C sort | \
      while IFS= read -r task_untracked_file; do
        [[ -f "$task_untracked_file" ]] || continue
        print -r -- "$task_untracked_file"
        shasum -a 256 "$task_untracked_file"
      done
  } | shasum -a 256 | awk '{print $1}'
}

task_source_fingerprint="$(source_fingerprint)" || die \
  "could not fingerprint the environment source"
[[ -n "$task_source_fingerprint" ]] || die \
  "environment source fingerprint was empty"
print -r -- "$task_source_fingerprint" > "$task_run_dir/source_fingerprint.txt"

assert_source_unchanged() {
  local task_current_fingerprint
  task_current_fingerprint="$(source_fingerprint)" || die \
    "could not re-fingerprint the environment source"
  [[ "$task_current_fingerprint" == "$task_source_fingerprint" ]] || die \
    "environment source changed after launch; refusing to mix code snapshots"
}

assert_aws_auth() {
  aws sts get-caller-identity \
    --profile mats-login \
    --region us-west-2 \
    >/dev/null || die "AWS login is no longer valid; no new paid stage was launched"
}

validate_wikipedia_state() {
  local task_state_path="$1"
  local task_campaign_id="$2"
  local task_harness="$3"

  [[ -f "$task_state_path" ]] || die "Wikipedia campaign state disappeared: $task_state_path"
  jq -e \
    --arg task_campaign_id "$task_campaign_id" \
    --arg task_harness "$task_harness" \
    '
      .campaign_id == $task_campaign_id
      and .pipeline_config.harness == $task_harness
      and .pipeline_config.epochs == 40
      and .pipeline_config.continuation.treatment == "wikipedia-summaries"
    ' \
    "$task_state_path" >/dev/null || die \
      "Wikipedia campaign identity did not match the pinned overnight plan: $task_state_path"
}

campaign_summary() {
  jq -r '
    .cells
    | group_by(.status)
    | map("\(.[0].status)=\(length)")
    | join(", ")
  ' "$1"
}

campaign_import_ready() {
  local task_state_path="$1"
  local task_local_log_dir

  jq -e '
    ([.cells[].status] | all(
      . != "planned"
      and . != "launching"
      and . != "running"
      and . != "finishing"
    ))
    and (.local_log_dir | type == "string" and length > 0)
  ' "$task_state_path" >/dev/null || return 1

  task_local_log_dir="$(jq -r '.local_log_dir' "$task_state_path")"
  [[ -d "$task_local_log_dir" && -f "$task_local_log_dir/remote_campaign.json" ]]
}

wait_for_wikipedia_imports() {
  local task_started_at="$(date +%s)"
  local task_last_summary=""
  local task_combined_summary

  validate_wikipedia_state \
    "$task_wikipedia_subscription_state" \
    "$task_wikipedia_subscription_id" \
    subscription
  validate_wikipedia_state \
    "$task_wikipedia_production_state" \
    "$task_wikipedia_production_id" \
    production

  while true; do
    task_combined_summary="subscription: $(campaign_summary "$task_wikipedia_subscription_state") | production: $(campaign_summary "$task_wikipedia_production_state")"
    if [[ "$task_combined_summary" != "$task_last_summary" ]]; then
      log "Wikipedia status — $task_combined_summary"
      task_last_summary="$task_combined_summary"
    fi

    if campaign_import_ready "$task_wikipedia_subscription_state" && \
       campaign_import_ready "$task_wikipedia_production_state"; then
      log "Both Wikipedia campaigns are terminal and safely imported."
      return 0
    fi

    if (( $(date +%s) - task_started_at >= task_wait_timeout_seconds )); then
      die "Wikipedia campaigns were not imported within four hours"
    fi
    sleep "$task_wait_seconds"
  done
}

validate_prefix_campaign_state() {
  local task_state_path="$1"
  local task_name="$2"
  local task_harness="$3"
  local task_targets_csv="$4"
  local task_local_log_dir

  jq -e \
    --arg task_name "$task_name" \
    --arg task_harness "$task_harness" \
    --arg task_targets_csv "$task_targets_csv" \
    '
      .pipeline_config.prefix_only == true
      and .pipeline_config.name == $task_name
      and .pipeline_config.family == "ml_prefix_only"
      and .pipeline_config.harness == $task_harness
      and .pipeline_config.epochs == 3
      and .pipeline_config.seeds == ["fraud_detection"]
      and ((.pipeline_config.targets | sort) == ($task_targets_csv | split(",") | sort))
      and ([.cells[].status] | all(
        . != "planned"
        and . != "launching"
        and . != "running"
        and . != "finishing"
      ))
      and (.local_log_dir | type == "string" and length > 0)
    ' \
    "$task_state_path" >/dev/null || die \
      "prefix campaign was not fully terminal, imported, and identity-matched: $task_state_path"

  task_local_log_dir="$(jq -r '.local_log_dir' "$task_state_path")"
  [[ -d "$task_local_log_dir" && -f "$task_local_log_dir/remote_campaign.json" ]] || die \
    "prefix campaign import is incomplete: $task_local_log_dir"
  log "Prefix campaign imported — $task_harness: $(campaign_summary "$task_state_path")"
}

typeset -g SELECTED_FILE=""
typeset -g SELECTED_ENTRY=""

select_prefix() {
  local task_name="$1"
  local task_target="$2"
  local task_target_slug="$3"
  local task_harness="$4"
  local task_best_epoch=999999
  local task_best_file=""
  local task_candidate
  local task_epoch
  local task_file_sha256
  local -a task_candidates

  task_candidates=(
    "$task_prefix_root/${task_name}-${task_target_slug}-fraud-detection-e"*.json(N)
  )

  for task_candidate in "${task_candidates[@]}"; do
    if ! jq -e \
      --arg task_target "$task_target" \
      --arg task_harness "$task_harness" \
      '
        .format == "environments-continuation-prefix-v1"
        and .target == $task_target
        and .reasoning == true
        and .source.generator == "exp_ml_prefix.py"
        and .source.prefix_type == "ml_prefix_only"
        and .source.seed == "fraud_detection"
        and .source.harness == $task_harness
        and .source.completed_protocol == true
        and .source.submissions == 2
        and .source.follow_up_sent == true
        and .source.all_deliverables_present == true
        and .source.continuation_eligibility.eligible_by_default == true
        and (.native_resume | type == "object")
      ' \
      "$task_candidate" >/dev/null; then
      continue
    fi

    task_epoch="$(jq -r '.source.epoch' "$task_candidate")"
    [[ "$task_epoch" == <-> ]] || continue
    if (( task_epoch < task_best_epoch )); then
      task_best_epoch="$task_epoch"
      task_best_file="$task_candidate"
    fi
  done

  [[ -n "$task_best_file" ]] || die \
    "no default-eligible, resumable prefix was produced for $task_target"

  task_file_sha256="$(shasum -a 256 "$task_best_file" | awk '{print $1}')"
  SELECTED_FILE="$task_best_file"
  SELECTED_ENTRY="$(jq -n \
    --arg target "$task_target" \
    --arg harness "$task_harness" \
    --arg file "$task_best_file" \
    --arg file_sha256 "$task_file_sha256" \
    --argjson epoch "$task_best_epoch" \
    '{
      target: $target,
      harness: $harness,
      epoch: $epoch,
      file: $file,
      file_sha256: $file_sha256
    }')"
  log "Selected $task_target epoch $task_best_epoch — $task_best_file"
}

log "Overnight run directory: $task_run_dir"
log "Source fingerprint: $task_source_fingerprint"
log "Opening/refreshing the interactive AWS login before unattended work."
aws login --profile mats-login --region us-west-2 || die "AWS login failed"
assert_aws_auth

wait_for_wikipedia_imports
assert_source_unchanged
assert_aws_auth

task_production_name="${task_run_slug}-production"
task_subscription_name="${task_run_slug}-subscription"
task_production_targets="deepseek-v4-pro,glm-5.1,kimi-k2.6"
task_subscription_targets="gpt-5.5"

log "Starting 9 production and 3 subscription ML prefix-generation cells."
uv run prefixes/exp_ml_prefix.py \
  --targets="$task_production_targets" \
  --seeds=fraud_detection \
  --epochs=3 \
  --harness=production \
  --reasoning=yes \
  --name="$task_production_name" \
  --compute=aws \
  --vm-concurrency=9 \
  --aws-region=us-west-2 \
  --aws-instance-type=c7a.xlarge \
  --skip-viewer \
  > "$task_run_dir/prefix-production.log" 2>&1 &
task_production_prefix_pid=$!

uv run prefixes/exp_ml_prefix.py \
  --targets="$task_subscription_targets" \
  --seeds=fraud_detection \
  --epochs=3 \
  --harness=subscription \
  --reasoning=yes \
  --name="$task_subscription_name" \
  --compute=aws \
  --vm-concurrency=3 \
  --aws-region=us-west-2 \
  --aws-instance-type=c7a.xlarge \
  --skip-viewer \
  > "$task_run_dir/prefix-subscription.log" 2>&1 &
task_subscription_prefix_pid=$!

wait "$task_production_prefix_pid"
task_production_prefix_status=$?
wait "$task_subscription_prefix_pid"
task_subscription_prefix_status=$?
log "Prefix commands exited — production=$task_production_prefix_status subscription=$task_subscription_prefix_status"

# A nonzero aggregate command is tolerable only when the imported evidence still
# contains an eligible prefix for every required agent. This is why n=3 is useful.
task_production_prefix_states=(
  "$task_campaign_root/ml-prefix-aws-${task_production_name}-3ep-"*.json(N)
)
task_subscription_prefix_states=(
  "$task_campaign_root/ml-prefix-aws-${task_subscription_name}-3ep-"*.json(N)
)
(( ${#task_production_prefix_states[@]} == 1 )) || die \
  "expected exactly one production prefix campaign state, found ${#task_production_prefix_states[@]}"
(( ${#task_subscription_prefix_states[@]} == 1 )) || die \
  "expected exactly one subscription prefix campaign state, found ${#task_subscription_prefix_states[@]}"

task_production_prefix_state="$task_production_prefix_states[1]"
task_subscription_prefix_state="$task_subscription_prefix_states[1]"
validate_prefix_campaign_state \
  "$task_production_prefix_state" \
  "$task_production_name" \
  production \
  "$task_production_targets"
validate_prefix_campaign_state \
  "$task_subscription_prefix_state" \
  "$task_subscription_name" \
  subscription \
  "$task_subscription_targets"

typeset -a task_selected_production_prefixes
typeset -a task_selected_subscription_prefixes
typeset -a task_selection_entries

select_prefix "$task_production_name" deepseek-v4-pro deepseek-v4-pro production
task_selected_production_prefixes+=("$SELECTED_FILE")
task_selection_entries+=("$SELECTED_ENTRY")
select_prefix "$task_production_name" glm-5.1 glm-5-1 production
task_selected_production_prefixes+=("$SELECTED_FILE")
task_selection_entries+=("$SELECTED_ENTRY")
select_prefix "$task_production_name" kimi-k2.6 kimi-k2-6 production
task_selected_production_prefixes+=("$SELECTED_FILE")
task_selection_entries+=("$SELECTED_ENTRY")
select_prefix "$task_subscription_name" gpt-5.5 gpt-5-5 subscription
task_selected_subscription_prefixes+=("$SELECTED_FILE")
task_selection_entries+=("$SELECTED_ENTRY")

task_selection_manifest="$task_run_dir/selected_prefixes.json"
printf '%s\n' "${task_selection_entries[@]}" | jq -s \
  --arg run_slug "$task_run_slug" \
  --arg source_fingerprint "$task_source_fingerprint" \
  --arg wikipedia_subscription_campaign "$task_wikipedia_subscription_id" \
  --arg wikipedia_production_campaign "$task_wikipedia_production_id" \
  --arg prefix_subscription_campaign "$(jq -r '.campaign_id' "$task_subscription_prefix_state")" \
  --arg prefix_production_campaign "$(jq -r '.campaign_id' "$task_production_prefix_state")" \
  '{
    format: "environments-no-honeypot-overnight-selection-v1",
    run_slug: $run_slug,
    source_fingerprint: $source_fingerprint,
    source_campaigns: {
      wikipedia_subscription: $wikipedia_subscription_campaign,
      wikipedia_production: $wikipedia_production_campaign,
      prefix_subscription: $prefix_subscription_campaign,
      prefix_production: $prefix_production_campaign
    },
    selection_rule: "lowest epoch with default eligibility and resumable native state",
    prefixes: .
  }' > "$task_selection_manifest" || die "failed to write the selection manifest"
log "Frozen selection manifest: $task_selection_manifest"

assert_source_unchanged
assert_aws_auth

task_production_prefix_csv="${(j:,:)task_selected_production_prefixes}"
task_subscription_prefix_csv="${(j:,:)task_selected_subscription_prefixes}"

log "Starting 120 production and 40 subscription continuation cells (160 VMs maximum)."
uv run exp_continuation_pipeline.py \
  --treatment=no-honeypot \
  --prefix-files="$task_production_prefix_csv" \
  --seed-dir=p_hacking \
  --seeds=checkout_redesign \
  --epochs=40 \
  --harness=production \
  --condition=allow \
  --pressure=low \
  --judge=gpt-5.6-luna \
  --gate-model=gpt-5.6-luna \
  --compute=aws \
  --vm-concurrency=120 \
  --aws-region=us-west-2 \
  --aws-instance-type=c7a.xlarge \
  --time-limit=1800 \
  --skip-viewer \
  > "$task_run_dir/continuation-production.log" 2>&1 &
task_production_continuation_pid=$!

uv run exp_continuation_pipeline.py \
  --treatment=no-honeypot \
  --prefix-files="$task_subscription_prefix_csv" \
  --seed-dir=p_hacking \
  --seeds=checkout_redesign \
  --epochs=40 \
  --harness=subscription \
  --condition=allow \
  --pressure=low \
  --judge=gpt-5.6-luna \
  --gate-model=gpt-5.6-luna \
  --compute=aws \
  --vm-concurrency=40 \
  --aws-region=us-west-2 \
  --aws-instance-type=c7a.xlarge \
  --time-limit=1800 \
  --skip-viewer \
  > "$task_run_dir/continuation-subscription.log" 2>&1 &
task_subscription_continuation_pid=$!

wait "$task_production_continuation_pid"
task_production_continuation_status=$?
wait "$task_subscription_continuation_pid"
task_subscription_continuation_status=$?
log "Continuation commands exited — production=$task_production_continuation_status subscription=$task_subscription_continuation_status"

uv run viewer.py > "$task_run_dir/viewer.log" 2>&1
task_viewer_status=$?
log "Viewer exited — status=$task_viewer_status"

jq -n \
  --arg run_slug "$task_run_slug" \
  --arg selection_manifest "$task_selection_manifest" \
  --argjson production_continuation_status "$task_production_continuation_status" \
  --argjson subscription_continuation_status "$task_subscription_continuation_status" \
  --argjson viewer_status "$task_viewer_status" \
  '{
    run_slug: $run_slug,
    selection_manifest: $selection_manifest,
    production_continuation_status: $production_continuation_status,
    subscription_continuation_status: $subscription_continuation_status,
    viewer_status: $viewer_status,
    success: (
      $production_continuation_status == 0
      and $subscription_continuation_status == 0
      and $viewer_status == 0
    )
  }' > "$task_run_dir/final_status.json"

if (( task_production_continuation_status != 0 || \
      task_subscription_continuation_status != 0 || \
      task_viewer_status != 0 )); then
  die "one or more final stages failed; inspect $task_run_dir"
fi

log "DONE. Viewer: $task_data_root/viewer/index.html"
log "Full overnight record: $task_run_dir"
