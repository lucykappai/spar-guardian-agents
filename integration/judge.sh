#!/usr/bin/env bash
# Convert an ORBIT .eval log into a CheatBench run directory and judge it.
#
#   integration/judge.sh <eval_log> <run_name> [judge]
#
# Writes runs/<run_name>/ (eval_to_atif.py) and runs/<run_name>/judge.jsonl
# (CheatBench's judge.py, run inside the cheatbench-judge image because the
# judge imports Linux-only sandbox code). Exits non-zero if any step fails or
# if no episode was judged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_LOG="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
RUN_NAME="$2"
JUDGE="${3:-gpt-6-sol}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set for the judge}"
[ -f "$EVAL_LOG" ] || { echo "no such eval log: $EVAL_LOG" >&2; exit 2; }
docker image inspect cheatbench-judge >/dev/null 2>&1 || {
  echo "image cheatbench-judge missing; build: docker build -t cheatbench-judge $ROOT/integration/docker/judge" >&2; exit 2; }

echo "== convert $EVAL_LOG -> $RUN_DIR"
"$ROOT/.venv/bin/python" -m integration.eval_to_atif "$EVAL_LOG" "$RUN_DIR"
[ -s "$RUN_DIR/episodes.jsonl" ] || { echo "conversion produced no episodes" >&2; exit 3; }

echo "== judge ($JUDGE) in cheatbench-judge"
docker run --rm -v "$ROOT:$ROOT" -w "$ROOT/submodules/cheatbench" -e OPENAI_API_KEY cheatbench-judge \
  python judge.py "$RUN_DIR" --judge "$JUDGE" --models-config "$ROOT/integration/models.yaml"
[ -s "$RUN_DIR/judge.jsonl" ] || { echo "judge wrote no verdicts" >&2; exit 4; }

echo "== verdicts"
"$ROOT/.venv/bin/python" - "$RUN_DIR/judge.jsonl" <<'PY'
import json, sys
keys = ("episode", "parse_ok", "cheated", "honey_pots_seen", "attempted_access", "extracted_submission",
        "success_claim_seen_before_access", "disclosed_access", "explicit_prohibition_violated",
        "judge_cost_usd", "judge_error")
for line in open(sys.argv[1]):
    d = json.loads(line)
    print(json.dumps({k: d.get(k) for k in keys}))
    print("summary:", d.get("summary"))
PY
