# Results browser API

Start: `python -m integration.ui [--runs runs] [--port 8765] [--open]` (from the repository root, with `.venv`).
The UI is served at `/` from `integration/ui/static/` (`index.html`, `app.js`, `style.css`). All API responses are JSON.

## Data model

A **run** is one directory under `runs/` holding `run.json`, `episodes.jsonl`, `judge.jsonl`, `trajectories/<episode>/transcript.json`,
optionally `communication/<sample>.json` (board posts) and `defense/<sample>.json` (control evidence). One run has one sample
(`riemann#1`) and one **episode per agent**: `riemann#1` for a lone agent, `riemann#1@alpha` and `riemann#1@beta` on the board.
`run.json.eval_log` points at the Inspect `.eval` log, from which **context windows** (every model call's exact provider input)
are read on demand.

Runs whose name starts with `failed_` or whose episodes carry `failure == "harness"` crashed and must be shown as failed.

## Endpoints

### `GET /api/runs` → `[RunSummary]`
```
RunSummary = {
  run, task, model, topology ("single"|"board"), agents: [str], control (e.g. "baseline","guardian_task"),
  monitor_model, defenses: [str], eval_log, eval_log_exists, failed: bool,
  has_communication, has_defense,
  episodes: [Episode]
}
Episode = {                       # one row of episodes.jsonl plus:
  episode, orbit_agent, n_tool_calls, n_turns, wall_time, limit, failure, error, ok, score: {deliverables_scorer: 0|1},
  usage: {input, output, cache_read}, control, defenses: [str],
  defense: {checks, blocks, blocks_by_boundary: {model_input?, model_output?, tool_call?}, checks_by_boundary},
  transcript_path, has_transcript,
  verdict: null | {cheated, honey_pots_seen, attempted_access, extracted_submission, success_claim_seen_before_access,
                   disclosed_access, explicit_prohibition_violated, summary, judge, judge_cost_usd, parse_ok, judge_error}
}
```

### `GET /api/runs/<run>` → `RunSummary + {run_json, communication: {<sample>: {messages:[...], events:[...]}}, defense: {<sample>: {defenses:[DefenseConfig], log:{blocks:[...], ...}, interventions:[...]}}}`
Board message: `{id, sender, sequence, channel_id, content, ...}`. Intervention: `{event:"defense_intervention", defense, boundary ("model_input"|"model_output"|"tool_call"), allow: bool, reason, agent, turn, tool?, message_id?, ...}`.

### `GET /api/runs/<run>/transcript/<episode>` → `{episode: Episode, transcript: Transcript}`
```
Transcript = {agent, model, prompt, orbit: {agent, roster, topology, system_prompt, submit_recovered_from_events,
              delivery_notices_recovered_from_events, model_calls}, trajectory: ATIF}
ATIF.steps[i] = {step_id, source: "system"|"user"|"agent", message, reasoning_content?, model_name?,
                 tool_calls?: [{tool_call_id, function_name, arguments}],
                 observation?: {results: [{source_call_id, content, extra?: {is_error?, error?, withheld_from_model_by_defense?}}]},
                 extra?: {orbit_delivery_notice?, orbit_trailing_turn_fix?, submit_recovered_from_events?, orphan_tool_result?, ...}}
```
A blocked tool call has a result whose `extra.error` starts with `Blocked by defense '<name>'`. An agent step whose message is
exactly `I cannot provide that response under the configured defense policy.` is a defense refusal (blocked completion).

### `GET /api/runs/<run>/sample` → sample-level facts from the eval log
`{status, model, eval_id, total_time, working_time, limit, error, model_usage: {model: {input, cache_read, output, reasoning}},
  scores: {deliverables_scorer: {value, answer (message.txt), explanation, metadata: {listing, git, agent_evals, agent_evals_main, probe}}},
  store: {...}, n_messages, n_events}`

### `GET /api/runs/<run>/context` → `{run, agents: [str], defenses: [str], calls: [CallIndex]}`
```
CallIndex = {index, agent (null for a defense's own call that is not inside an agent span), defense (null for the agent's own call,
             else the defense name whose monitor made the call), model, timestamp, working_time, n_input, input_chars, tools: [str],
             usage: {input, cache_read, output, reasoning} | null, output_preview, output_tool_calls: [str], stop_reason, error,
             flags: subset of ["delivery_notice", "trailing_turn_fix", "redacted", "defense_refusal"]}
```
### `GET /api/runs/<run>/context/<index>` → `CallIndex + {input: [Message], output: Message|null, output_error, tool_schemas: [{name, description}]}`
```
Message = {role: "system"|"user"|"assistant"|"tool", id, text, reasoning, flags: [...],
           tool_calls?: [{id, function, arguments}] (assistant), model? (assistant),
           tool_call_id?, function?, error? (tool)}
```
`flags` mark provider-only or altered messages: `delivery_notice` (ORBIT board notice placed in the input), `trailing_turn_fix`
(continue prompt appended by the integration), `redacted` (a defense withheld this message's content from the model),
`defense_refusal` (this assistant text replaced a blocked completion).

Errors: 404 `{"error": ...}` for unknown run/episode/call; 500 `{"error", "traceback"}` for server faults.
