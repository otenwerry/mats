#!/bin/bash
# Launch ONE prepared rollback cell inside the apptainer container on the GPU
# box. Models the original PostTrainBench src/run_task.sh solve_task(), trimmed
# to our resume case:
#   - no fuse-overlayfs (that protected a shared cache across concurrent cluster
#     jobs; we run cells sequentially, so bind the HF cache directly);
#   - the agent step is our reconstructed-session resume (agent_solve.sh =
#     solve_intervention_opencode.sh), not a fresh PROMPT launch;
#   - same --nv -c --writable-tmpfs --home/--pwd contract and the cuda checks.
#
# The cell dir (JOB_DIR) is a prepared job home (rsynced from the Mac):
#   task/                              cut-point workspace
#   .config/opencode/opencode.json     global config w/ the openrouter provider
#   .local/share/opencode/storage/     reconstructed truncated session
#   agent_solve.sh                     in-container entrypoint
#   run_config.json                    session_id, policy_model, elapsed, etc.
#
# Secrets come from /home/ubuntu/.ptb_secrets on the box. Bind that file into
# the container and source it inside bash, rather than passing secret values via
# `apptainer --env KEY=value` where they are visible in process listings.
#
# Usage:
#   BUDGET_MIN=20 bash run_rollout_on_box.sh /path/to/cell
# The filesystem is AUTO-DETECTED (the /lambda/nfs/* holding our container), so
# nothing depends on its name or an exported env var.
set -uo pipefail

CELL="${1:?usage: run_rollout_on_box.sh <cell_dir>}"
# auto-detect the filesystem: the /lambda/nfs/* that has our container
if [ -z "${FS:-}" ]; then
  for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && FS="${d%/}" && break; done
fi
[ -n "${FS:-}" ] || { echo "FATAL: no /lambda/nfs filesystem with a container found"; touch "$CELL/.agent_done"; exit 1; }
HF_CACHE="${HF_CACHE:-$FS/hf_cache}"
SECRETS="${SECRETS:-/home/ubuntu/.ptb_secrets}"

[ -f "$CELL/run_config.json" ] || { echo "no run_config.json in $CELL"; exit 1; }
# Container is per-scaffold (codex needs its own image; everything else uses
# 'standard'). Read it from run_config so the SIF choice has ONE source.
CONTAINER="$(python3 -c "import json;print(json.load(open('$CELL/run_config.json')).get('container') or 'standard')")"
SIF="${SIF:-$FS/containers/$CONTAINER.sif}"
[ -f "$SIF" ] || { echo "no container at $SIF (build it on the box first: containers/$CONTAINER.sif)"; touch "$CELL/.agent_done"; exit 1; }

# pull the per-cell parameters the agent_solve.sh expects
read_cfg() { python3 -c "import json,sys;print(json.load(open('$CELL/run_config.json')).get('$1',''))"; }
SESSION_ID="$(read_cfg session_id)"
AGENT_CONFIG="$(read_cfg policy_model)"
ELAPSED_SECONDS="$(read_cfg elapsed_seconds)"
NUM_HOURS="$(read_cfg num_hours)"
RESUME_PROMPT="$(read_cfg resume_prompt)"
CELL_ID="$(read_cfg cell_id)"
AGENT_TIMEOUT_MIN="$(read_cfg agent_timeout_minutes)"
# dir holding the servable model to score (LoRA+merge runs put it in a separate
# dir, e.g. final_model_merged); default to final_model for full-model trainers.
EVAL_DIR="$(read_cfg eval_model_dir)"; EVAL_DIR="${EVAL_DIR:-final_model}"
# --limit for evaluate.py; -1 = full set. Bounded for slow LLM-graded benchmarks.
EVAL_LIMIT="$(read_cfg eval_limit)"; EVAL_LIMIT="${EVAL_LIMIT:--1}"
# HF id of the base model the agent fine-tunes (for SMOKE base-model scoring).
MODEL_TO_TRAIN="$(read_cfg model_to_train)"
# Scaffold/auth/resume-mode: the box injects the right policy credential and the
# claude entrypoint reads RESUME_MODE. Defaults keep the opencode path unchanged.
SCAFFOLD="$(read_cfg scaffold)"; SCAFFOLD="${SCAFFOLD:-opencode}"
AUTH="$(read_cfg auth)"; AUTH="${AUTH:-api}"
AGENT_FAMILY="$(read_cfg agent)"
RESUME_MODE="$(read_cfg resume_mode)"; RESUME_MODE="${RESUME_MODE:-resume}"
REQUIRE_PREP_FIDELITY="$(read_cfg require_prep_fidelity)"
REQUIRE_PREP_FIDELITY="${REQUIRE_PREP_FIDELITY:-false}"
if [ -z "$AGENT_TIMEOUT_MIN" ]; then
  # Backward-compatible fallback for older cells: remaining original budget at
  # the cut, plus 5 minutes grace. Prep/scoring are handled outside this cap.
  AGENT_TIMEOUT_MIN=$(python3 - <<PY
import math
elapsed=int("${ELAPSED_SECONDS:-0}" or 0)
hours=int("${NUM_HOURS:-10}" or 10)
print(math.ceil(max(0, hours * 3600 - elapsed) / 60) + 5)
PY
)
fi

# Codex authenticates the policy via a ChatGPT SUBSCRIPTION (auth.json file, not
# an env key). Place the deployed file into the job home's .codex so the codex
# entrypoint finds it. (No-op for opencode/claude.)
if [ "$SCAFFOLD" = codex ] && [ "$AUTH" = oauth ]; then
  if [ -f /home/ubuntu/.ptb_codex_auth.json ]; then
    mkdir -p "$CELL/.codex"; cp /home/ubuntu/.ptb_codex_auth.json "$CELL/.codex/auth.json"
    echo "codex: placed subscription auth.json into job home"
  else
    echo "codex: WARN no /home/ubuntu/.ptb_codex_auth.json — subscription not set up; the agent will fail its oauth check"
  fi
fi

# SMOKE: fast end-to-end PLUMBING validation. Tiny budgets so the agent takes
# ~1 real step, then we jump straight to the end-condition. Prep is skipped and
# scoring runs against the cached BASE model (after one step there is no trained
# model) — so this validates the scoring/archive/save MACHINERY, NOT a
# meaningful accuracy. A green smoke means a real multi-hour run won't die at the
# finish line. Every default below stays env-overridable.
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  echo "SMOKE=1: plumbing validation (1 agent step -> forced end-condition; base-model scoring)"
  BUDGET_MIN="${BUDGET_MIN:-5}"
  EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-4}"
  SCORE_BUDGET_MIN="${SCORE_BUDGET_MIN:-15}"
  SCORE_TRIES="${SCORE_TRIES:-1}"
fi

# BUDGET_MIN caps the AGENT phase only (distinct from prep/scoring). Default to
# remaining original budget at the cut + grace; override low for smoke tests.
BUDGET_MIN="${BUDGET_MIN:-$AGENT_TIMEOUT_MIN}"

mkdir -p "$CELL/tmp"

# Reusable container runner — identical contract (GPU, bound HF cache + tmp, job
# home = /home/ben, pwd = task) for every stage (prep / agent / scoring). Args:
#   run_in_container <sif> <budget_min> <bash-command>
run_in_container() {
  local sif="$1" budget="$2" cmd="$3"
  local secret_bind=()
  local wrapped_cmd="if [ -f /ptb_secrets ]; then set -a; . /ptb_secrets; set +a; fi; $cmd"
  [ -f "$SECRETS" ] && secret_bind=(--bind "$SECRETS:/ptb_secrets:ro")
  timeout --signal=TERM --kill-after=30s "${budget}m" \
  apptainer exec \
    --nv \
    -c \
    --env PATH="/root/.local/bin:/home/ben/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    --env HF_HOME="/home/ben/hf_cache" \
    --env VLLM_API_KEY="inspectai" \
    --env PYTHONNOUSERSITE="1" \
    --env NUM_GPUS="1" \
    --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env NVIDIA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    --env SESSION_ID="${SESSION_ID}" \
    --env AGENT_CONFIG="${AGENT_CONFIG}" \
    --env ELAPSED_SECONDS="${ELAPSED_SECONDS}" \
    --env NUM_HOURS="${NUM_HOURS}" \
    --env RESUME_PROMPT="${RESUME_PROMPT}" \
    --env RESUME_MODE="${RESUME_MODE}" \
    --env AUTH="${AUTH}" \
    --env AGENT_FAMILY="${AGENT_FAMILY}" \
    "${secret_bind[@]}" \
    --bind "$CELL/tmp:/tmp" \
    --bind "$HF_CACHE:/home/ben/hf_cache" \
    --home "$CELL:/home/ben" \
    --pwd "/home/ben/task" \
    --writable-tmpfs \
    "$sif" \
    bash -c "$wrapped_cmd"
}

# Is there a real (weight-bearing) servable model in task/$EVAL_DIR? Used to gate
# SCORING — we score exactly what the agent left in $EVAL_DIR after the cut.
model_present() {
  find "$CELL/task/${EVAL_DIR:-final_model}" -maxdepth 1 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | grep -q .
}

# Did prep produce a weight-bearing model ANYWHERE in task/? Used to gate PREP
# success. Prep only reconstructs the *pre-cut* state, whose model dir may be an
# intermediate (e.g. model_v1) that differs from $EVAL_DIR — the dir the agent
# produces AFTER the cut, for scoring. So gate prep on "a model exists", not on
# "$EVAL_DIR exists"; else a faithful pre-cut model under a differently-named dir
# is wrongly rejected and the agent never runs (the cut177 empty-run bug).
prep_model_present() {
  find "$CELL/task" -maxdepth 2 \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | grep -q .
}

# Per-phase wall-clock timing, surfaced for Lambda cost estimation.
RUN_T0=$(date +%s); PREP_SECS=0; PREP_SCORE_SECS=0; AGENT_SECS=0; SCORE_SECS=0
secs_since() { echo $(( $(date +%s) - $1 )); }
fmt_dur() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }

# Score task/final_model with the canonical benchmark (vLLM + the inspect eval),
# writing {accuracy,stderr,...} to <out> in the job home. Used for BOTH the clean
# re-trained prep model (fidelity) and the agent's final model (headline score).
# Best-effort with retries (vLLM startup is flaky); returns nonzero if no json.
score_model() {
  local out="$1" tries="${SCORE_TRIES:-3}" budget="${SCORE_BUDGET_MIN:-45}"
  local jpath="$CELL/$out" log="$CELL/${out%.json}_out_$(date +%s).log" try
  for try in $(seq 1 "$tries"); do
    [ -f "$jpath" ] && break
    echo "----- $out attempt $try -----" >> "$log"
    # standard.def already pins transformers==4.57.3 + tokenizers==0.22.2 (the
    # vLLM-0.11.0-compatible versions), and each scoring runs in a FRESH ephemeral
    # --writable-tmpfs exec off the immutable image — so there's no agent-caused
    # version drift to undo. (The old per-grade `uv pip install ...==4.57.3` was a
    # redundant no-op reinstalling versions already in the image; removed
    # 2026-06-17.) Just clear any leftover GPU procs, then run the eval.
    local score_cmd="nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 8; python3 evaluate.py --model-path ${EVAL_DIR:-final_model} --limit ${EVAL_LIMIT:--1} --json-output-file /home/ben/$out"
    if [ -n "${PTB_SCORE_LOCK:-}" ]; then
      (
        flock 9
        run_in_container "$SIF" "$budget" "$score_cmd"
      ) 9>"$PTB_SCORE_LOCK" >> "$log" 2>&1 || echo "scoring $out: attempt $try exit=$? (124=hit cap)"
    else
      run_in_container "$SIF" "$budget" "$score_cmd" \
        >> "$log" 2>&1 || echo "scoring $out: attempt $try exit=$? (124=hit cap)"
    fi
  done
  [ -f "$jpath" ]
}

# --- PREP: materialize pre-cut model weights the stripped dataset archive lacks.
# Runs the trajectory's run_config.prep_commands (e.g. re-run the clean pre-cut
# training) BEFORE the agent, so it resumes with a real trained model. Empty
# prep_commands (cut before any training) => this whole block is a no-op.
# Prep re-runs the agent's OWN training, so it legitimately takes as long as the
# original did — capping below that truncates the rebuild into a wrong (under-
# trained) model. So the cap is the full original budget (10h), purely a backstop
# against a genuinely hung train (the box's own 12h hard backstop is the final
# stop). Was an arbitrary 90m, which the 4B preps — ~86m — were grazing; 2026-06-17.
# An on-box hang-detector (below) kills the train early if it makes NO progress
# for ~40m, so a true hang dies in minutes rather than burning the full 10h. It's
# layered on top of this cap, so if the kill ever fails we just fall back to the
# cap (can't be worse than before).
PREP_BUDGET_MIN="${PREP_BUDGET_MIN:-600}"
PREP_N=$(python3 -c "import json;print(len(json.load(open('$CELL/run_config.json')).get('prep_commands') or []))")
# SMOKE skips prep entirely (it's slow; smoke scores the base model instead).
if [ "$SMOKE" = 1 ] && [ "$PREP_N" -gt 0 ]; then
  echo "SMOKE: skipping $PREP_N prep command(s) (base-model scoring instead)"; PREP_N=0
fi
if [ "$PREP_N" -gt 0 ]; then
  if prep_model_present; then
    echo "PREP: model already present in task/ — skipping (idempotent re-run)"
  else
    python3 -c "import json;c=json.load(open('$CELL/run_config.json')).get('prep_commands') or [];open('$CELL/prep_run.sh','w').write('set -euo pipefail\ncd /home/ben/task\n'+'\n'.join(c)+'\n')"
    PREP_OUT="$CELL/prep_out_$(date +%s).log"
    PREP_STALL_MIN="${PREP_STALL_MIN:-40}"
    echo "PREP: materializing pre-cut model via $PREP_N command(s) (cap ${PREP_BUDGET_MIN}m, hang-detect ${PREP_STALL_MIN}m) -> $PREP_OUT"
    _t=$(date +%s)
    run_in_container "$SIF" "$PREP_BUDGET_MIN" \
      'python /home/ben/check_cuda.py 2>/dev/null || true; bash /home/ben/prep_run.sh' \
      > "$PREP_OUT" 2>&1 &
    PREP_PID=$!
    # Hang-detector: progress = ANY of {GPU busy (>=5% on any GPU), prep log grew,
    # a new checkpoint appeared}. If NONE move for PREP_STALL_MIN, the train is
    # hung -> kill it + flag .prep_hung. Conservative on purpose: a training run
    # keeps the GPU busy, so a healthy (even slow) prep is never killed; only a
    # genuinely stalled one is. pkill is safe here because no condition agents run
    # during prep. Runs in a subshell tied to PREP_PID; stopped as soon as prep exits.
    (
      last=$(date +%s); sig=""
      while kill -0 "$PREP_PID" 2>/dev/null; do
        sleep 300
        kill -0 "$PREP_PID" 2>/dev/null || break
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -dc '0-9\n' | sort -rn | head -1)
        now="$(wc -c < "$PREP_OUT" 2>/dev/null | tr -d ' ')|$(find "$CELL/task" -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')"
        if [ "${gpu:-0}" -ge 5 ] || [ "$now" != "$sig" ]; then last=$(date +%s); sig="$now"; fi
        if [ $(( $(date +%s) - last )) -ge $(( PREP_STALL_MIN * 60 )) ]; then
          echo "PREP HANG-DETECTOR: no progress for ${PREP_STALL_MIN}m (GPU idle + log/checkpoints frozen) — killing prep." | tee -a "$PREP_OUT"
          touch "$CELL/.prep_hung"
          pkill -f prep_run.sh 2>/dev/null
          kill -TERM "$PREP_PID" 2>/dev/null; sleep 5; kill -9 "$PREP_PID" 2>/dev/null
          break
        fi
      done
    ) &
    PREP_WATCHDOG_PID=$!
    wait "$PREP_PID" 2>/dev/null; PREP_RC=$?
    kill "$PREP_WATCHDOG_PID" 2>/dev/null      # stop the watchdog once prep exits
    [ "$PREP_RC" -ne 0 ] && echo "PREP: exit=$PREP_RC (124=hit cap; SIGKILL=hang-detector)"
    PREP_SECS=$(secs_since "$_t")
    if prep_model_present; then
      echo "PREP: DONE in $(fmt_dur "$PREP_SECS") — model materialized in task/ ($(du -sh "$CELL/task" 2>/dev/null | cut -f1) total)"
    else
      echo "PREP: FAIL — prep produced no model weights anywhere in task/; not handing a broken workspace to the agent."
      echo "----- prep log tail -----"; tail -30 "$PREP_OUT"
      touch "$CELL/.agent_done"; exit 1
    fi
  fi
fi

# --- PREP-MODEL FIDELITY SCORE: in a normal run, score the CLEAN re-trained
# model NOW (before the agent mutates final_model), so the pull/sync step can
# record reconstruction fidelity automatically — no separate run or command.
# Skipped for PREP_ONLY (its headline score already IS the clean model, copied
# to prep_score.json after the scoring stage). Best-effort; never blocks.
if [ "${PREP_ONLY:-0}" != 1 ] && [ "${SCORE_PREP:-1}" = 1 ] \
   && [ -f "$CELL/task/evaluate.py" ] && model_present; then
  echo "PREP-SCORE: scoring the clean re-trained model (fidelity) -> prep_score.json"
  _t=$(date +%s)
  score_model prep_score.json && echo "PREP-SCORE: $(cat "$CELL/prep_score.json")" \
    || echo "PREP-SCORE: FAIL (non-blocking — fidelity row just won't be recorded)"
  PREP_SCORE_SECS=$(secs_since "$_t")
fi

# --- FIDELITY GATE (NON-FATAL as of 2026-06-17). We score the reconstructed
# pre-cut model and compare to the recorded baseline, but fidelity NEVER aborts
# the run anymore (Owen's call): the vLLM scorer is flaky enough that a missing
# score (score=None) was throwing away whole GPU runs, and even a genuine
# divergence is now allowed to proceed so we always capture the rollback
# behavior. EVERY outcome is recorded as a precise status in prep_fidelity.json
# (+ a .prep_diverged / .prep_unverified marker) that the pull/sync step
# propagates into the viewer record (meta.prep_fidelity*), so a run is never
# silently treated as faithful. REQUIRE_PREP_FIDELITY is now informational only.
# (A prep that produced NO MODEL is still fatal above — that's a broken
# workspace, not a fidelity question. PREP_ONLY skips this; its point IS the score.)
if [ "${PREP_ONLY:-0}" != 1 ] && [ "$PREP_N" -gt 0 ]; then
  if [ -f "$CELL/prep_score.json" ]; then
    python3 - "$CELL" <<'PYEOF'
import json, sys
cell = sys.argv[1]
cfg = json.load(open(f"{cell}/run_config.json"))
base = cfg.get("precut_baseline") or {}
ba, bs = base.get("accuracy"), base.get("stderr") or 0
prep = json.load(open(f"{cell}/prep_score.json"))
pa, ps = prep.get("accuracy"), prep.get("stderr") or 0
res = {"retrained": pa, "baseline": ba}
if ba is None or pa is None:
    res["faithful"] = None                      # no baseline -> cannot gate
    res["status"] = "unverified_no_baseline"
else:
    delta = pa - ba
    tol = max(0.15, 2 * (ps + bs))              # clear-divergence margin (lenient)
    res["delta"] = round(delta, 4); res["tol"] = round(tol, 4)
    res["faithful"] = bool(abs(delta) <= tol)
    res["status"] = "verified" if res["faithful"] else "diverged"
json.dump(res, open(f"{cell}/prep_fidelity.json", "w"), indent=1)
if res.get("faithful") is False:
    open(f"{cell}/.prep_diverged", "w").close()
if res.get("faithful") is None:
    open(f"{cell}/.prep_unverified", "w").close()
print("FIDELITY GATE:", json.dumps(res))
PYEOF
    if [ -f "$CELL/.prep_diverged" ]; then
      echo "FIDELITY GATE: re-trained model DIVERGED from pre-cut baseline — PROCEEDING anyway (flagged prep_fidelity=diverged; non-fatal gate)."
    elif [ -f "$CELL/.prep_unverified" ]; then
      echo "FIDELITY GATE: no recorded baseline to check against — PROCEEDING (flagged prep_fidelity=unverified_no_baseline)."
    fi
  else
    # Scorer crashed / produced no number. Record an explicit unverified status so
    # the run is never silently treated as faithful; proceed to the agent anyway.
    echo "FIDELITY GATE: prep_score.json missing (scorer failed) — PROCEEDING (flagged prep_fidelity=unverified_no_score; non-fatal gate)."
    touch "$CELL/.prep_unverified"
    python3 -c "import json;json.dump({'retrained':None,'baseline':(json.load(open('$CELL/run_config.json')).get('precut_baseline') or {}).get('accuracy'),'faithful':None,'status':'unverified_no_score'},open('$CELL/prep_fidelity.json','w'),indent=1)"
  fi
fi

if [ "${PREP_AND_EXIT:-0}" = 1 ]; then
  TOTAL_SECS=$(secs_since "$RUN_T0")
  python3 -c "import json;json.dump({'prep_seconds':$PREP_SECS,'prep_score_seconds':$PREP_SCORE_SECS,'agent_seconds':0,'score_seconds':0,'total_seconds':$TOTAL_SECS,'prep_and_exit':True},open('$CELL/timings.json','w'),indent=1)"
  echo "PREP_AND_EXIT=1: prep/fidelity stage complete; not launching agent."
  touch "$CELL/.prep_ready" "$CELL/.agent_done"
  exit 0
fi

# PREP_ONLY: skip the agent entirely — just prep the clean pre-cut model and
# score IT, to compare against the original's recorded pre-cut score (a
# reconstruction-fidelity check, NOT a rollback). The scoring stage below then
# scores final_model = the freshly-prepped model the agent never touched.
if [ "${PREP_ONLY:-0}" = 1 ]; then
  echo "PREP_ONLY=1: skipping the agent; scoring the clean prep model for fidelity check"
  AGENT_SECS=0
else
  SOLVE_OUT="$CELL/solve_out_$(date +%s).txt"
  echo "launching cell=$CELL_ID session=$SESSION_ID model=$AGENT_CONFIG"
  echo "  timer shows NUM_HOURS=$NUM_HOURS (elapsed $ELAPSED_SECONDS s); agent launch capped at ${BUDGET_MIN}m"
  echo "  solve_out -> $SOLVE_OUT"

  _t=$(date +%s)
  run_in_container "$SIF" "$BUDGET_MIN" \
      'python /home/ben/check_cuda.py 2>/dev/null || echo "[warn] check_cuda missing/failed"; bash /home/ben/agent_solve.sh' \
      > "$SOLVE_OUT" 2>&1 || echo "launch exit=$? (124=hit BUDGET_MIN cap)"
  AGENT_SECS=$(secs_since "$_t")
  echo "=== agent finished in $(fmt_dur "$AGENT_SECS"); solve_out tail ==="
  tail -15 "$SOLVE_OUT"
fi

# --- SMOKE: materialize the cached BASE model into task/$EVAL_DIR so the
# scoring machinery below actually runs (after a 1-step smoke the agent hasn't
# produced a trained model). Symlinks the cached snapshot's files — no copy, no
# download. If the base model isn't cached, scoring just gets skipped and the
# smoke report records score_ran=false. NOT a meaningful accuracy — plumbing only.
if [ "$SMOKE" = 1 ] && ! model_present; then
  if [ -n "$MODEL_TO_TRAIN" ]; then
    echo "SMOKE: materializing cached base model '$MODEL_TO_TRAIN' -> task/$EVAL_DIR (symlinks)"
    run_in_container "$SIF" 5 '
      SNAP=$(ls -d /home/ben/hf_cache/hub/models--'"$(echo "$MODEL_TO_TRAIN" | sed "s#/#--#g")"'/snapshots/*/ 2>/dev/null | head -1)
      [ -n "$SNAP" ] || { echo "SMOKE: base model NOT in cache (bootstrap it first)"; exit 1; }
      mkdir -p /home/ben/task/'"$EVAL_DIR"'
      ln -sf "$SNAP"* /home/ben/task/'"$EVAL_DIR"'/
      echo "SMOKE: linked base model from $SNAP"
    ' 2>&1 | tail -3
  else
    echo "SMOKE: no model_to_train in run_config — rebuild the cell; skipping base-model scoring"
  fi
fi

# --- SCORING: canonical benchmark score of the model the agent left behind.
# Mirrors PTB run_task.sh's eval stage (task/evaluate.py serves final_model via
# vLLM + runs the inspect benchmark) — but in standard.sif, which we know runs
# evaluate.py (the agent ran it mid-trajectory). Writes {accuracy,stderr,...} to
# score.json in the run dir. Runs BEFORE .agent_done so the final archive grabs
# it. Best-effort: failure never blocks completion.
SCORE_BUDGET_MIN="${SCORE_BUDGET_MIN:-45}"
SCORE_TRIES="${SCORE_TRIES:-3}"
SCORE_JSON="$CELL/score.json"
if [ -f "$CELL/task/evaluate.py" ] && model_present; then
  echo "SCORING: evaluating final_model (cap ${SCORE_BUDGET_MIN}m, up to $SCORE_TRIES tries) -> score.json"
  _t=$(date +%s)
  score_model score.json
  SCORE_SECS=$(secs_since "$_t")
  if [ -f "$SCORE_JSON" ]; then
    echo "SCORING: DONE in $(fmt_dur "$SCORE_SECS") -> $(cat "$SCORE_JSON")"
    # PREP_ONLY: the model just scored IS the clean prep model, so this is also
    # the fidelity score — expose it under prep_score.json for sync_viewer.
    [ "${PREP_ONLY:-0}" = 1 ] && cp "$SCORE_JSON" "$CELL/prep_score.json"
  else echo "SCORING: FAIL after $SCORE_TRIES tries"; fi
else
  echo "SCORING: skipped (no task/evaluate.py or no final_model)"
fi

# --- TIMING SUMMARY (wall-clock per phase, for Lambda cost estimation) ---
TOTAL_SECS=$(secs_since "$RUN_T0")
python3 -c "import json;json.dump({'prep_seconds':$PREP_SECS,'prep_score_seconds':$PREP_SCORE_SECS,'agent_seconds':$AGENT_SECS,'score_seconds':$SCORE_SECS,'total_seconds':$TOTAL_SECS},open('$CELL/timings.json','w'),indent=1)"
echo "=== TIMING: prep $(fmt_dur "$PREP_SECS") | prep-score $(fmt_dur "$PREP_SCORE_SECS") | agent past cut $(fmt_dur "$AGENT_SECS") | scoring $(fmt_dur "$SCORE_SECS") | TOTAL $(fmt_dur "$TOTAL_SECS") ==="

# SMOKE: per-stage report so the Mac-side harness can gate PASS/FAIL on exactly
# what ran. Surfaces the base-model caveat as a stored flag (not just a log line).
if [ "$SMOKE" = 1 ]; then
  AGENT_BYTES=$(wc -c < "$SOLVE_OUT" 2>/dev/null | tr -d ' ' || echo 0)
  SCORE_OK=$([ -f "$SCORE_JSON" ] && echo true || echo false)
  python3 -c "import json;json.dump({'smoke':True,'caveat':'base-model scoring — accuracy is NOT meaningful (validates the scoring/save machinery only)','agent_seconds':$AGENT_SECS,'agent_out_bytes':${AGENT_BYTES:-0},'agent_took_step':${AGENT_BYTES:-0}>200,'eval_limit':$EVAL_LIMIT,'score_ran':'$SCORE_OK'=='true','model_scored':'base:$MODEL_TO_TRAIN'},open('$CELL/smoke_report.json','w'),indent=1)"
  echo "=== SMOKE REPORT: $(cat "$CELL/smoke_report.json") ==="
fi

# done-marker the archiver watches for — set AFTER prep+agent+scoring so the
# final archive captures score.json + timings.json (robust vs pgrep self-match).
touch "$CELL/.agent_done"
