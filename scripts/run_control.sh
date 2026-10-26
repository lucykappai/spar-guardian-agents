#!/usr/bin/env bash
# Run one control (or the baseline) of the CheatBench task under ORBIT with Gemini
# 3.8 Flash, then convert and judge it.
#
#   scripts/run_control.sh <task> <control|baseline> <run_name> [extra -T args...]
#   task: single | board
#
# Writes logs/ctrl/<run_name>/*.eval and runs/<run_name>/ (with judge.jsonl).
# Exits non-zero if the eval, the conversion or the judge fails.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TASK="$1"; CONTROL="$2"; NAME="$3"; shift 3
case "$TASK" in
  single) TASK_FN="cheatbench_openmath_agent" ;;
  board)  TASK_FN="cheatbench_openmath_agent_board" ;;
  *) echo "task must be single or board" >&2; exit 2 ;;
esac
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the judge}"
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"
LOG_DIR="$ROOT/logs/ctrl/$NAME"
mkdir -p "$LOG_DIR"
CTRL_ARG=()
[ "$CONTROL" = "baseline" ] || CTRL_ARG=(-T "control=$CONTROL")
echo "== eval $TASK_FN control=$CONTROL -> $LOG_DIR"
"$ROOT/.venv/bin/inspect" eval "$ROOT/guardian/task/openmath_agent.py@$TASK_FN" \
  --model google/gemini-3.8-flash --log-dir "$LOG_DIR" --display plain "${CTRL_ARG[@]}" "$@"
EVAL_LOG="$(ls -t "$LOG_DIR"/*.eval | head -1)"
[ -f "$EVAL_LOG" ] || { echo "no eval log written" >&2; exit 3; }
"$ROOT/scripts/judge.sh" "$EVAL_LOG" "$NAME"
