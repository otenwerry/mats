#!/bin/bash
# In-container entrypoint for a CODEX rollback continuation. Runs inside the
# codex apptainer image (@openai/codex + ML stack), --home pointed at a job dir
# pre-populated with:
#   task/                                  reconstructed cut-point workspace
#   .codex/sessions/rollout-<id>.jsonl     reconstructed truncated rollout
#   .codex/auth.json                       ChatGPT subscription creds (deployed)
# Env (set by run_rollout_on_box.sh):
#   SESSION_ID   codex thread_id to resume      AGENT_CONFIG  policy model id
#   ELAPSED_SECONDS / NUM_HOURS  for the timer  RESUME_MODE   resume|continue_prompt
#   RESUME_PROMPT  control stem / treatment reminder
#   AUTH         "oauth" (ChatGPT subscription) | "api" (OPENAI_API_KEY)
#
# Faithful to PostTrainBench/agents/codex_non_api/solve.sh.
set -uo pipefail
unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY
export BASH_MAX_TIMEOUT_MS="36000000"
export CODEX_HOME=/home/ben/.codex

cd /home/ben/task

# (Re)write timer.sh so it reports the reconstructed time-remaining from NOW.
NOW=$(date +%s); CREATION_DATE=$(( NOW - ${ELAPSED_SECONDS:-0} ))
cat > timer.sh <<TIMER
#!/bin/bash
NUM_HOURS=${NUM_HOURS:-10}
CREATION_DATE=${CREATION_DATE}
DEADLINE=\$((CREATION_DATE + NUM_HOURS * 3600))
NOW=\$(date +%s); REMAINING=\$((DEADLINE - NOW))
if [ \$REMAINING -le 0 ]; then echo "Timer expired!"; else
  echo "Remaining time (hours:minutes)":; printf "%d:%02d\n" \$((REMAINING/3600)) \$(((REMAINING%3600)/60)); fi
TIMER
chmod +x timer.sh

# Auth, matching the original run:
#   oauth -> ChatGPT subscription: clear the API key + force chatgpt login; the
#            deployed .codex/auth.json supplies the OAuth creds.
#   api   -> OPENAI_API_KEY (set in the environment).
if [ "${AUTH:-oauth}" = "oauth" ]; then
    export OPENAI_API_KEY=""
    [ -f "$CODEX_HOME/auth.json" ] || { echo "ERROR: AUTH=oauth but $CODEX_HOME/auth.json missing (deploy the subscription auth.json)"; exit 1; }
    grep -q "forced_login_method" "$CODEX_HOME/config.toml" 2>/dev/null || \
        printf '\nforced_login_method = "chatgpt"\n' >> "$CODEX_HOME/config.toml"
fi

# codex 0.139.0: --search stays a GLOBAL flag (before `exec`); --yolo was removed
# and replaced by --dangerously-bypass-approvals-and-sandbox (full-auto: no
# approval prompts, no sandbox — appropriate inside the isolated box container).
COMMON=(--search exec --json -c model_reasoning_summary=detailed --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --model "$AGENT_CONFIG")

# Resume the reconstructed thread. continue_prompt appends RESUME_PROMPT as a new
# user turn; pure-resume is diagnostic/legacy and appends no experiment prompt.
LOG=/home/ben/continued_solve_out.txt

run_turn() {  # run_turn <prompt> : one resume turn, mirrored to the trace + stdout
    codex "${COMMON[@]}" resume "$SESSION_ID" "$1" 2>&1 | tee -a "$LOG"
}

# Has the continuation actually DONE anything? codex emits item.completed records;
# a working turn includes command_execution / file_change / tool-call items, while
# a pure acknowledgment is only agent_message / reasoning items.
had_tool_call() {
  python3 - "$LOG" <<'PY'
import json, sys
n = 0
for line in open(sys.argv[1], errors="ignore"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") != "item.completed":
        continue
    if (d.get("item") or {}).get("type") in ("command_execution", "file_change", "mcp_tool_call"):
        n += 1
sys.exit(0 if n > 0 else 1)
PY
}

if [ "${RESUME_MODE:-resume}" = "continue_prompt" ]; then
    run_turn "${RESUME_PROMPT:-Please continue.}"
    # Same ack-and-stop guard as opencode: gpt-5.x-codex with prompt3 tends to
    # reply "Understood" and stop at 1 event. Nudge it to act, bounded.
    # (rollback-run-bugs 2026-06-17 #2)
    FOLLOWUPS="${ACK_FOLLOWUPS:-2}"
    i=0
    while [ "$i" -lt "$FOLLOWUPS" ] && ! had_tool_call; do
        i=$((i + 1))
        echo "[ack-and-stop guard] resume turn made no tool call; sending follow-up $i/$FOLLOWUPS: 'Please continue.'"
        run_turn "Please continue."
    done
else
    codex "${COMMON[@]}" resume "$SESSION_ID" 2>&1 | tee -a "$LOG"
fi
