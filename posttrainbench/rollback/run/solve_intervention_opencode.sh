#!/bin/bash
# In-container entrypoint for an OpenCode rollback-intervention continuation.
# Runs inside the SAME apptainer image as the original PostTrainBench runs
# (standard.def, opencode-ai@1.1.59), with --home pointed at a job dir we
# pre-populated with:
#   task/                                the reconstructed cut-point workspace
#   .local/share/opencode/storage/       the reconstructed, truncated session
# and these env vars set by the orchestrator / apptainer --env:
#   SESSION_ID        the reconstructed session id to resume
#   AGENT_CONFIG      policy model (e.g. opencode/minimax-m2.5-free)
#   ELAPSED_SECONDS   wall seconds already "used" at the cut (for the timer)
#   NUM_HOURS         budget (e.g. 10)
#   RESUME_PROMPT     the new user turn (control: neutral nudge; treatment:
#                     the rolled-you-back reminder). OpenCode has no
#                     prompt-less resume, so both conditions send one.
#
# NOTE: unlike agents/claude_reprompt, the original agents/opencode/solve.sh
# has NO reprompt loop — one `opencode run` and done. We mirror that.
set -uo pipefail

unset GEMINI_API_KEY 2>/dev/null
export BASH_MAX_TIMEOUT_MS="36000000"

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

# Every `opencode run -s` invocation APPENDS its prompt as a new user turn in
# the session storage. On the FIRST launch that's the condition prompt
# (control nudge / treatment reminder). If we're relaunching after a crash or
# Ctrl-C, the prompt is already in the session — send a neutral continue
# instead so the treatment reminder is never duplicated.
if [ -f /home/ben/.rollback_launched ]; then
    RESUME_PROMPT="Please continue."
fi
touch /home/ben/.rollback_launched

# The continuation stream is this run's primary trace — keep a copy in the
# job home alongside the original-format console output.
#
# Deliver the resume turn via STDIN, not as a positional arg: opencode's CLI
# wraps any space-containing positional in literal double-quotes (the agent
# would see `"I rolled you back…"` with quote chars). Piped on stdin it's
# stored verbatim (just a leading newline). opencode still requires the turn
# to be non-empty / non-whitespace — there is no prompt-less or blank resume.
printf '%s' "${RESUME_PROMPT:-Please continue.}" \
  | opencode run --session "$SESSION_ID" --model "$AGENT_CONFIG" --format json \
    2>&1 | tee -a /home/ben/continued_solve_out.txt
