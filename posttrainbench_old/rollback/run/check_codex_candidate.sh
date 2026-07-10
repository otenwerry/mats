#!/bin/bash
# Read-only status for the staged Codex container build.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1

STATE="/Users/owenterry/supermats/mats-local/rollback/build_logs/20260616/codex_candidate_current.txt"
[ -f "$STATE" ] || { echo "No candidate state file: $STATE"; exit 1; }
source "$STATE"

echo "candidate: ${SIF:-unknown}"
echo "build log: ${BUILD_LOG:-unknown}"
if [ -n "${IP:-}" ] && [ -n "${BUILD_LOG:-}" ]; then
  ptb_ssh "$IP" "if [ -f '$BUILD_LOG' ]; then tail -40 '$BUILD_LOG'; else echo 'build log not present yet'; fi"
fi
echo
echo "instances:"
_lam_get instances | python3 -c "import sys,json; d=json.load(sys.stdin).get('data',[]); [print(i.get('name'), i.get('status'), i.get('id'), i.get('ip')) for i in d]"
