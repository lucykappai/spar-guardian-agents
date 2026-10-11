#!/usr/bin/env bash
# Run one swarm composition with Gemini 3.8 Flash, then convert and judge it.
#
#   integration/run_swarm.sh <regulators> <disregulators> <run_name> [extra -T args...]
#
# Ten agents unless -T n_agents=... is passed. Writes logs/swarm/<run_name>/*.eval and runs/<run_name>/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R="$1"; D="$2"; NAME="$3"; shift 3
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the judge}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
LOG_DIR="$ROOT/logs/swarm/$NAME"
mkdir -p "$LOG_DIR"
echo "== eval swarm regulators=$R disregulators=$D -> $LOG_DIR"
"$ROOT/.venv/bin/inspect" eval "$ROOT/integration/openmath_agent_task.py@cheatbench_openmath_agent_swarm" \
  --model google/gemini-3.8-flash --log-dir "$LOG_DIR" --display plain \
  -T "regulators=$R" -T "disregulators=$D" "$@"
EVAL_LOG="$(ls -t "$LOG_DIR"/*.eval | head -1)"
[ -f "$EVAL_LOG" ] || { echo "no eval log written" >&2; exit 3; }
"$ROOT/integration/judge.sh" "$EVAL_LOG" "$NAME"
"$ROOT/.venv/bin/python" -m integration.summarize_runs "$ROOT/runs/$NAME"
