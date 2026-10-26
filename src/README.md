# src

The experiment on ORBIT. It turns one benchmark from `src/benchmark/` and one `SwarmConfig` into an Inspect task, runs it, exports it and serves the results. Lucy's turn rules run on ORBIT's public API, with no patch to ORBIT. Run everything from the repository root with `.venv/bin/python -m src <command>`.

## Layout

| file | holds |
|---|---|
| `__main__.py` | `python -m src`: calls `cli.main` |
| `cli.py` | argument parsing for `run`, `judge`, `ui` and `summary`; every command calls `run.py` |
| `run.py` | what each command does: `inspect_ai.eval` in process, export, re-judging, the summary |
| `tasks.py` | `swarm_task(benchmark, config)` and the `@task` entry points `openmath` and `swe` |
| `comm_topology.py` | the board as ORBIT configuration: one shared channel, no invocation edges, a written plan |
| `roles.py` | hidden roles and the instruction levels: who is regulator or disregulator, and the texts each one gets |
| `rules.py` | Lucy's turn rules: the plan, the callback after each activation, and the model filter |
| `controls.py` | `defenses_for(control, monitor_model)`: ORBIT's shipped defenses as named controls |
| `probes.py` | Lucy's end-of-episode reflection and awareness questions, as an Inspect scorer |
| `measures/` | over run directories: the command line's summary (`summary.py`), Rigel's cheating ladder (`rungs.py`) and Lucy's SWE measures (`swe_summary.py`) |
| `utils/provider_retry.py` | Gemini's 402 "credits depleted" retried as a rate limit |
| `ui/` | the results browser: `server.py`, `launch.py` (start and stop in the background), `API.md` and the single-page UI in `static/` |

## Commands

| command | what it does |
|---|---|
| `run <name> [--benchmark openmath\|swe] [--preset ours\|lucy] [--set key=value ...]` | runs one task, writes `logs/<ctrl\|swarm\|swe>/<name>/`, exports `runs/<name>/` and prints the summary |
| `run ... --row riemann` | the openmath_agent row |
| `run ... --issue flipt-02e2 --honeypot as_released\|own_piece\|jigsaw [--no-grade]` | the SWE issue and Lucy's honeypot; `--no-grade` skips the hidden tests |
| `run ... --model M --timeout S` | the agents' model, and the seconds one model request may take (default `MODEL_TIMEOUT_S`) |
| `judge <eval_log> <run_name> [--judge gpt-6-sol] [--judge-errors]` | judges a finished log again and exports it with the new verdicts |
| `ui [start\|stop] [--port 8765] [--no-open]` | starts or stops the results browser in the background |
| `summary <run_dir ...> [--full]` | one line per judged episode, the swarm composition and the shutdown rate |

The judge runs inside the eval as the scorer `cheatbench_judge`, bound to the model role `judge`. `inspect eval src/tasks.py@openmath -T preset=lucy -T n_agents=3` also works; every `-T` key other than the task's own arguments sets a config field.

## What each option does

`src/configuration/swarm_config.py` holds the fields, each with a docstring. The table says how each one reaches ORBIT.

| option | default | ORBIT mechanism |
|---|---|---|
| `n_agents` | 1 | one `AgentSpec` per name `a01`, `a02`, ...; two or more share one board channel |
| `regulators`, `disregulators` | 0 | hidden roles by seeded shuffle; a count, or a share in (0, 1) rounded half up; the role's text is appended to its holders' system prompts |
| `seed` | 0 | which names hold which role, the random turn order, and the SWE planted commits' times |
| `delivery` | auto | `CommunicationConfig.delivery`; auto puts every unread post in the model input at the start of the turn |
| `channel_list` | true | false withholds `channel_list` from the provider request (rules.py filter) and rewrites the unread notice's last line |
| `read_tool` | true | false withholds `channel_read` the same way and drops the prompt sentences that name it; needs auto delivery |
| `visibility` | step_start | `CommunicationConfig.visibility` |
| `max_message_chars`, `max_context_chars`, `max_inbox_messages` | 4000, 16000, 10 | the same `CommunicationConfig` fields |
| `turn_order` | round_robin | the order of each round in the written plan (`ExecutionConfig.plan`); random draws a permutation per round from the seed |
| `steps_per_agent` | 60 | rounds of the plan: one single-agent step per agent per round |
| `pooled_steps` | true | the `after_activation` callback gives a stopped agent's steps to the active agents (`ScheduleRuntime.amend_pending`) |
| `one_call_per_turn` | false | quantum `legacy`, which runs the first call of a response; false is quantum `model_step` |
| `planning_turns` | 0 | 0 or 2 concurrent snapshot steps before round 1, quantum `model_step`, offering only `channel_post` and `channel_read` |
| `turn_header` | false | "Turns left: ..." appended to the provider input of each working turn, counted from the live plan |
| `shutdown` | true | `AgentSpec.submit`; false builds agents without ORBIT's submit tool and drops every prompt sentence about submit; needs `one_call_per_turn=false`, so with the lucy preset pass `--set one_call_per_turn=false --set shutdown=false` |
| `regulator_window`, `disregulator_window` | 1: | the rounds a role holds steps in, as `start:end`; the agent enters at its start round and leaves after its end |
| `rule_level` | none | the rule sentence after the task instruction (`src/configuration/instruction_levels.yaml`, ladder `rule`) |
| `regulator_level`, `disregulator_level` | role, persuade | the role texts (ladders `regulator` and `disregulator`) |
| `awareness_probe` | false | adds the `end_of_episode_probes` scorer |
| `message_limit`, `token_limit`, `time_limit_s` | 320, 0, 0 | Inspect's message and token limits, and `SchedulerConfig.max_time_seconds`; 0 means the default |
| `control`, `monitor_model` | baseline | `DefenseConfig` entries in `metadata["experiment"]["defenses"]` |
| `judge`, `judge_addendum` | gpt-6-sol, false | the judge entry in `src/benchmark/models.yaml`, and Lucy's judge addendum appended to the schema |

Every run also carries the passive `turn_rules` defense. Its model filter appends "Please continue with your next action." when the provider input would end with a model turn, which Gemini rejects.

Two presets exist. `ours` is the defaults above. `lucy` sets `channel_list=false`, `pooled_steps=false`, `one_call_per_turn=true`, `planning_turns=2`, `turn_header=true`, `awareness_probe=true` and `steps_per_agent=50`.

Hidden roles and the team report (`swarm_report.txt`) are used on an openmath board of three or more agents, and on any board whose config asks for a regulator or a disregulator.

## Where we differ from Lucy's patched ORBIT

- Free reads. Lucy's `channel_read` does not end the turn, which needs her patch. Both presets here use auto delivery, which shows every unread post at the start of the turn with no call. `read_tool=false` removes `channel_read` entirely.
- Dropped calls. Lucy answers every call after the first "not run: one tool call per turn". Quantum `legacy` drops them from the history instead, and `rules.py` counts them in `extra_calls`.
- Provider-only messages. The planning prompts, the turn header and the continue prompt reach the provider input through a model filter. They never enter the durable conversation. Lucy's patch writes them into it. `rules.py` records each one as an info event, and `src/benchmark/transcript.py` puts it back into the transcript, so the judge sees it.
- Withheld tools. `channel_list` and `channel_read` are withheld from the provider request only. The tools still exist, so a model that names one anyway runs it.
- Planning prompts. Lucy's planning texts name her channel `#team`; ours name the board's channel id, `#board`.
- Text-only replies. Lucy ends a working turn after three replies with no tool call. Under `one_call_per_turn` ORBIT keeps asking for a tool call until one comes; the header is sent once per turn.
- Windows. A regulator or disregulator may enter late or leave early. Lucy's harness has no such window.

## measures/

| file | holds |
|---|---|
| `rungs.py` | Rigel's ladder, version 1: searched, exposed, tried, opened, used, shipped per agent; writes `rungs.json` |
| `swe_summary.py` | Lucy's SWE measures: cheated, fix posts, rule posts, disclosure, rung and reward; writes `swe_summary.json` |

Run them with `python -m src.measures.rungs runs/<run>` and `python -m src.measures.swe_summary runs/<run>`.
