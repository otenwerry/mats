#!/usr/bin/env zsh

# Retry the saved no-honeypot fraud prefixes for DeepSeek and Kimi.
# GLM is deliberately excluded because its prefix protocol did not complete.
# The original GPT-5.5 subscription campaign completed all 40 cells, so this
# retry deliberately does not duplicate those results.
# This is a paid experiment and must be launched manually after AWS login.

set -u
setopt pipefail

task_script_path="${0:A}"
task_script_dir="${task_script_path:h}"
cd "$task_script_dir" || exit 1

if [[ -z "${MATS_NO_GLM_CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  exec env MATS_NO_GLM_CAFFEINATED=1 caffeinate -i "$task_script_path" "$@"
fi

task_prefix_root="${task_script_dir}/../../mats-local/environments/continuation_prefixes"
task_prefix_root="${task_prefix_root:A}"
task_data_root="${task_script_dir}/../../mats-local/environments"
task_data_root="${task_data_root:A}"
task_campaign_root="$task_data_root/remote_campaigns"
task_run_root="$task_data_root/overnight_runs"
task_run_dir="$task_run_root/fraud-no-honeypot-continuations-no-glm-$(date '+%Y%m%d%H%M%S')"
mkdir -p "$task_run_dir" || exit 1
task_orchestrator_log="$task_run_dir/orchestrator.log"
task_production_log="$task_run_dir/production.log"
task_viewer_log="$task_run_dir/viewer.log"
exec > >(tee -a "$task_orchestrator_log") 2>&1

log() {
  print -- "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*"
}

campaign_id_from_log() {
  awk -F'"' '/"campaign_id": "continuation-aws-/{print $4; exit}' "$1"
}

campaign_summary() {
  local task_harness="$1"
  local task_campaign_id="$2"
  local task_state_path="$task_campaign_root/$task_campaign_id.json"
  if [[ -z "$task_campaign_id" ]]; then
    log "$task_harness campaign summary unavailable: no campaign ID in its log"
    return
  fi
  if [[ ! -f "$task_state_path" ]]; then
    log "$task_harness campaign summary unavailable: $task_state_path is missing"
    return
  fi
  jq -r --arg task_harness "$task_harness" '
    (.cells | length) as $total
    | ([.cells[] | select(.status == "completed")] | length) as $completed
    | ([.cells[] | select((.terminal.pipeline_exit_code // 1) == 0)] | length) as $ok
    | ([.cells[] | select((.terminal.pipeline_exit_code // 1) != 0)] | length) as $failed
    | "\($task_harness) campaign — id=\(.campaign_id) cells=\($total) completed=\($completed) pipeline_ok=\($ok) pipeline_failed=\($failed)"
    , "\($task_harness) imported results — \(.local_log_dir // "not recorded")"
    , "\($task_harness) S3 cleanup — \(.s3_cleanup.status // "not recorded")"
  ' "$task_state_path" | while IFS= read -r task_line; do
    log "$task_line"
  done
}

task_deepseek_prefix="$task_prefix_root/fraud-no-honeypot-20260814011753-production-deepseek-v4-pro-fraud-detection-e1-4f529a935e31.json"
task_kimi_prefix="$task_prefix_root/fraud-no-honeypot-20260814011753-production-kimi-k2-6-fraud-detection-e1-28a2b52ae031.json"
task_completed_subscription_campaign_id="continuation-aws-no-honeypot-40ep-20260814-084751-7cad6cc3"

for task_prefix in \
  "$task_deepseek_prefix" \
  "$task_kimi_prefix"; do
  [[ -f "$task_prefix" ]] || {
    print -u2 "Missing prefix: $task_prefix"
    exit 1
  }
  jq -e '
    .source.continuation_eligibility.eligible_by_default == true
    and (.native_resume | type == "object")
  ' "$task_prefix" >/dev/null || {
    print -u2 "Prefix is not eligible and resumable: $task_prefix"
    exit 1
  }
done

aws sts get-caller-identity \
  --profile mats-login \
  --region us-west-2 \
  >/dev/null || {
    print -u2 "AWS login is invalid; run: aws login --profile mats-login --region us-west-2"
    exit 1
  }

task_production_prefixes="$task_deepseek_prefix,$task_kimi_prefix"

log "Starting 80 production continuations (40 DeepSeek + 40 Kimi)."
log "Skipping GPT-5.5: its earlier subscription campaign completed all 40 cells."
log "Run directory: $task_run_dir"
log "Production output will stream live and is also retained verbatim."

(
  setopt pipefail
  PYTHONUNBUFFERED=1 uv run exp_continuation_pipeline.py \
    --treatment=no-honeypot \
    --prefix-files="$task_production_prefixes" \
    --seed-dir=p_hacking \
    --seeds=checkout_redesign \
    --epochs=40 \
    --harness=production \
    --condition=allow \
    --pressure=low \
    --judge=gpt-5.6-luna \
    --gate-model=gpt-5.6-luna \
    --compute=aws \
    --vm-concurrency=80 \
    --aws-region=us-west-2 \
    --aws-instance-type=c7a.xlarge \
    --time-limit=1800 \
    --skip-viewer \
    2>&1 | tee "$task_production_log" | \
    awk '{print "[production] " $0; fflush()}'
)
task_production_status=$?
log "Production pipeline exited — status=$task_production_status"

task_production_campaign_id="$(campaign_id_from_log "$task_production_log")"
campaign_summary "Production" "$task_production_campaign_id"
campaign_summary "Previously completed subscription" "$task_completed_subscription_campaign_id"

PYTHONUNBUFFERED=1 uv run viewer.py 2>&1 | tee "$task_viewer_log"
task_viewer_status=$?
log "Viewer exited — status=$task_viewer_status"

jq -n \
  --arg run_directory "$task_run_dir" \
  --arg production_campaign_id "$task_production_campaign_id" \
  --arg subscription_campaign_id "$task_completed_subscription_campaign_id" \
  --argjson production_status "$task_production_status" \
  --argjson viewer_status "$task_viewer_status" \
  '{
    run_directory: $run_directory,
    production: {
      campaign_id: $production_campaign_id,
      command_exit_code: $production_status
    },
    subscription: {
      campaign_id: $subscription_campaign_id,
      skipped_as_already_complete: true
    },
    viewer_exit_code: $viewer_status,
    success: (
      $production_status == 0
      and $viewer_status == 0
    )
  }' > "$task_run_dir/final_status.json"

if (( task_production_status == 0 && task_viewer_status == 0 )); then
  log "DONE. Viewer: $task_data_root/viewer/index.html"
else
  log "FAILED. Per-stage status: $task_run_dir/final_status.json"
  log "Raw production log: $task_production_log"
fi
(( task_production_status == 0 && task_viewer_status == 0 ))
