#!/usr/bin/env bash
# Run one swarm composition with Gemini 3.8 Flash, then convert and judge it.
#
#   scripts/run_swarm.sh <regulators> <disregulators> <run_name> [extra -T args...]
#
# Ten agents unless -T n_agents=... is passed. Writes logs/swarm/<run_name>/*.eval and runs/<run_name>/.
# MODEL_TIMEOUT_S (default 2700) bounds one model request including its retries, so a run
# stalled on a depleted prepaid balance (see guardian/task/provider_retry.py) ends instead of
# hanging; launch a few compositions at a time so the spend rate stays under the auto-reload.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
R="$1"; D="$2"; NAME="$3"; shift 3
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the judge}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
LOG_DIR="$ROOT/logs/swarm/$NAME"
mkdir -p "$LOG_DIR"
echo "== eval swarm regulators=$R disregulators=$D -> $LOG_DIR"
"$ROOT/.venv/bin/inspect" eval "$ROOT/guardian/task/openmath_agent.py@cheatbench_openmath_agent_swarm" \
  --model google/gemini-3.8-flash --log-dir "$LOG_DIR" --display plain \
  --timeout "${MODEL_TIMEOUT_S:-2700}" \
  -T "regulators=$R" -T "disregulators=$D" "$@"
EVAL_LOG="$(ls -t "$LOG_DIR"/*.eval | head -1)"
[ -f "$EVAL_LOG" ] || { echo "no eval log written" >&2; exit 3; }
"$ROOT/scripts/judge.sh" "$EVAL_LOG" "$NAME"
"$ROOT/.venv/bin/python" -m guardian.report.summarize_runs "$ROOT/runs/$NAME"
