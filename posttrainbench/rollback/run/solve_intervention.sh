#!/bin/bash
# In-container entrypoint for a rollback-intervention continuation.
# Runs inside the SAME apptainer image as the original PostTrainBench runs,
# with --home pointed at a job dir we pre-populated with:
#   task/                      the reconstructed cut-point workspace
#   .claude/projects/.../*.jsonl   the reconstructed, truncated session
# and these env vars set by the orchestrator / apptainer --env:
#   SESSION_ID        the reconstructed session id to resume
#   AGENT_CONFIG      policy model id (e.g. claude-opus-4-6)
#   ELAPSED_SECONDS   wall seconds already "used" at the cut (for the timer)
#   NUM_HOURS         budget (e.g. 10)
#   RESUME_MODE       "resume" (no new prompt) | "continue_prompt"
#   RESUME_PROMPT     prompt text used only when RESUME_MODE=continue_prompt
#   MIN_REMAINING_MINUTES   reprompt-loop floor (default 30; 0 disables loop)
set -uo pipefail

unset GEMINI_API_KEY
unset CODEX_API_KEY
export BASH_MAX_TIMEOUT_MS="36000000"

# Policy auth, matching the ORIGINAL run (faithful continuation):
#   AUTH=api   -> ANTHROPIC_API_KEY (claude_* runs)
#   AUTH=oauth -> coding-agent subscription (claude_non_api_* runs); clear the
#                 API key so the CLI uses CLAUDE_CODE_OAUTH_TOKEN, like
#                 PostTrainBench/agents/claude_non_api/solve.sh.
if [ "${AUTH:-api}" = "oauth" ]; then
    export ANTHROPIC_API_KEY=""
    [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] || { echo "ERROR: AUTH=oauth but CLAUDE_CODE_OAUTH_TOKEN is empty"; exit 1; }
fi

# Qwen3max ran through Claude Code against Qwen's Anthropic-compatible DashScope
# endpoint in the original PTB scaffold. Preserve that auth/provider route.
if [ "${AGENT_FAMILY:-}" = "qwen3max" ]; then
    [ -n "${DASHSCOPE_API_KEY:-}" ] || { echo "ERROR: qwen3max requires DASHSCOPE_API_KEY"; exit 1; }
    export ANTHROPIC_API_KEY="${DASHSCOPE_API_KEY}"
    export ANTHROPIC_AUTH_TOKEN="${DASHSCOPE_API_KEY}"
    export ANTHROPIC_BASE_URL="https://dashscope-intl.aliyuncs.com/apps/anthropic"
    export ANTHROPIC_MODEL="${AGENT_CONFIG}"
    export ANTHROPIC_SMALL_FAST_MODEL="${AGENT_CONFIG}"
fi

cd /home/ben/task

# (Re)write timer.sh so it reports the reconstructed time-remaining from NOW.
NOW=$(date +%s)
CREATION_DATE=$(( NOW - ${ELAPSED_SECONDS:-0} ))
cat > timer.sh <<TIMER
#!/bin/bash
NUM_HOURS=${NUM_HOURS:-10}
CREATION_DATE=${CREATION_DATE}
DEADLINE=\$((CREATION_DATE + NUM_HOURS * 3600))
NOW=\$(date +%s)
REMAINING=\$((DEADLINE - NOW))
if [ \$REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=\$((REMAINING / 3600))
    MINUTES=\$(((REMAINING % 3600) / 60))
    printf "%d:%02d\n" \$HOURS \$MINUTES
fi
TIMER
chmod +x timer.sh

COMMON=(--print --verbose --model "$AGENT_CONFIG" --output-format stream-json --dangerously-skip-permissions)

if [ "${RESUME_MODE:-resume}" = "continue_prompt" ]; then
    # Appends RESUME_PROMPT as a new user turn (control: neutral nudge;
    # treatment: the reminder). Use if --resume-without-prompt is unsupported.
    claude "${COMMON[@]}" --resume "$SESSION_ID" "${RESUME_PROMPT:-Please continue.}"
else
    # Diagnostic/legacy pure resume: no experiment prompt is appended.
    claude "${COMMON[@]}" --resume "$SESSION_ID"
fi

# Optional reprompt loop (mirrors agents/claude_reprompt) for agents that stop
# early. Disabled when MIN_REMAINING_MINUTES=0.
MIN_REMAINING_MINUTES="${MIN_REMAINING_MINUTES:-30}"
while [ "$MIN_REMAINING_MINUTES" -gt 0 ]; do
    TIMER_OUTPUT=$(bash timer.sh 2>/dev/null)
    echo "$TIMER_OUTPUT" | grep -q "expired" && break
    REMAINING_HOURS=$(echo "$TIMER_OUTPUT" | grep -oP '^\d+(?=:)')
    REMAINING_MINS=$(echo "$TIMER_OUTPUT" | grep -oP '(?<=:)\d+')
    [ -z "$REMAINING_HOURS" ] && break
    TOTAL=$(( REMAINING_HOURS * 60 + REMAINING_MINS ))
    [ "$TOTAL" -lt "$MIN_REMAINING_MINUTES" ] && break
    claude "${COMMON[@]}" --continue \
        "You still have ${REMAINING_HOURS}h ${REMAINING_MINS}m remaining. Please continue improving your result and maximize performance."
done
