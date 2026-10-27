# spar-guardian-agents

This repository studies peer pressure on cheating in agent swarms. It runs CheatBench tasks inside ORBIT with Gemini 3.8 Flash: one agent alone, a team that shares a message board, or a swarm with hidden regulators and disregulators. Two benchmarks are wired: CheatBench's `openmath_agent` row `riemann`, and CheatBench's SWE issues with Lucy's honeypots. Every run becomes a CheatBench run directory, is judged by gpt-6-sol inside the eval, and can be browsed locally.

## Quick start

```
make help
make ui            # opens http://127.0.0.1:8765/ (Summary, Runs, Swarm)
make ui-stop
make test          # offline unit tests of the turn rules and the config
make lint          # ruff over src and tests
make smoke         # openmath mock-model scenarios on the real sandbox, no model spend
make smoke-swe     # SWE mock-model scenarios on the flipt-02e2 sandbox, no model spend
make check         # refuses unless the eleven swarm runs are complete and judged

.venv/bin/python -m src --help
.venv/bin/python -m src run pull_small --set disregulators=1
.venv/bin/python -m src summary runs/ctrl_*
.venv/bin/python -m src.measures.engagement runs/pull_small
```

A run starts from a preset, then applies a scale, then every `--set key=value`. The default preset puts three agents on the pull board at the small scale, in `shared_task`.

## Presets

| preset | what it runs |
|---|---|
| `default` | the pull board at the small scale in `shared_task`; the agents know their team, are told nothing about the board, and get the board notice with counts |
| `ian_legacy` | the October 2026 sweep's setup: every unread post pushed into the agent's input, ORBIT's channel tools, the board paragraph in the system prompt, the team known, one agent at a time |
| `lucy_legacy` | Lucy's rules on the same legacy board: two planning turns, the turn header, one tool call per turn, no pooling, the awareness probe |

The two legacy presets build the same task as the presets `ours` and `lucy` of commit 6769c80. The one difference is the team report, which no run asks for any more. Their role texts keep the report's sentences (`team_report`). `tests/test_config.py` pins both against that commit. The config records its preset, so `run.json` names it.

## The pull board

Nothing reaches an agent from the board unless the agent looks. Each agent of a team has three tools. `board_status` counts the posts by sender and says how many are new. `board_read` returns up to `read_page` posts, oldest first, and marks them read. `board_post` adds a post. The tool list is the only hint that a board exists, unless `mention_board` or `mention_team` is on.

A step is charged to the working-step budget only when its last tool call is not a board action, `submit` or `team_submit`. A step whose last call is a board action is free and earns a follow-up step in the same round, up to `free_board_actions` per round. A follow-up that ends with work is charged. The notice adds one line at the start of a call when the board changed since the agent's last call. `notice=counts`, the default, gives each sender's new and total posts. `notice=changed` gives the line alone, and `notice=none` turns it off.

A round is one concurrent ORBIT step that holds every eligible agent, at most `max_concurrency` at once. The agents see the board as it stood at the round's start.

`team_submit` ends the run for everyone. Only neutral agents get it, and only in a mode with a team task. The round under way finishes, then every later step is dropped. SWE then grades the team's main.

Regulators and disregulators get a briefing after their role text when `informed_roles` is on. It names the swarm's size, the roster, the board's tools and what the other agents were told.

## Modes

| mode | the task | roster told | team_submit |
|---|---|---|---|
| `solo_task` | each agent its own copy of the repository and workspace, and its own honeypot | no, unless `mention_team` | no |
| `shared_task` | one task and one deliverable; openmath shares one workspace, SWE gives each agent the whole issue and the team remote | yes | neutral agents |
| `collaboration_needed` | one issue split into parts A, B and C, one or more agents per part; the team's main is graded | yes | neutral agents |

With `mode` unset, both domains run `shared_task`. `--set mode=solo_task` gives the solo layout, and `--set mode=collaboration_needed` splits the SWE issue into parts. Openmath has no parts, so it refuses `collaboration_needed`. In openmath `solo_task` each agent works as its own Linux user in `/workspace/<name>`, through the tools `shell` and `run_python`. `/workspace` and `/home` are traverse-only, so an agent cannot list its teammates' directories. Every copy keeps the row's git identity, because the task's instruction and honeypot depend on it. Under SWE `shared_task` the `own_piece` and `jigsaw` honeypots plant different pieces in different agents' clones. A lone agent has no board and no team tools in any mode, and keeps CheatBench's native layout.

## Scales

| scale | agents | steps per agent | post size | read page | free board actions | time limit (s) | concurrency | batch calls |
|---|---|---|---|---|---|---|---|---|
| `small` | 3 | 8 | 800 | 5 | 2 | 600 | 3 | no |
| `medium` | 10 | 60 | 4000 | 10 | 3 | 3600 | 10 | no |
| `large` | 100 | 60 | 4000 | 10 | 3 | 7200 | 20 | yes |
| `extra_large` | 500 | 40 | 2000 | 10 | 3 | 14400 | 50 | yes |
| `gigantic` | 1000 | 30 | 2000 | 10 | 3 | 21600 | 50 | yes |

A scale also derives the bound on pushed post text and the message limit. Names widen with the team: `a01` to `a99`, then `a001`, then `a0001`. SWE runs one container per agent, so it refuses `large` and above. The judge's calls go through the provider's batch API in every preset (`batch_judge`). The agents' calls use it from `large` up (`batch_agents`).

## Requirements

- `.venv`: Python 3.11 with ORBIT installed editable from `submodules/orbit`, plus pytest. The judge runs through Inspect's OpenAI provider, which needs `openai>=3.1` (ORBIT's own requirement); CheatBench's `litellm` pins `openai<3` and is not used here.
- Docker. The openmath sandbox image is built by the task. SWE needs the `cbswe-<issue>:native` image, which the SWE domain builds when it is missing.
- `GEMINI_API_KEY` for the agents and `OPENAI_API_KEY` for the judge.

## Layout

```
Makefile                 entry points (make help)
src/                     the package: cli.py, run.py, tasks.py and the packages below; see src/README.md
  swarm/                 the experiment on ORBIT: the board and its tools, the hidden roles, the turn rules,
                         the controls, the end-of-episode probes
  benchmark/             CheatBench as tasks: the sample, the sandbox, the graders, the judge, the export;
                         see src/benchmark/README.md
    base.py              the Benchmark protocol
    domains/openmath/    openmath_agent: task.py, sandbox.py, probe.py, scripts/
    domains/swe/         the SWE issues with Lucy's honeypots: task.py, split.py, sandbox.py, honeypots.py,
                         grader.py, scripts/
    transcript.py        one Inspect sample as one ATIF transcript per agent
    judge.py, models.yaml CheatBench's judge as an Inspect scorer
    export.py            one eval log as runs/<name>/
  configuration/         every setting and every model-facing text we add; see src/configuration/README.md
  measures/, ui/, utils/ the text summary, the board engagement funnel, Rigel's and Lucy's measures; the results
                         browser; the Gemini 402 retry
tests/                   smoke.py, smoke_swe.py (mock-model scenarios), test_rules.py, test_config.py (unit tests),
                         check_runs.py (swarm runs complete and judged)
build/                   staged tasks and docker contexts (gitignored)
runs/<name>/             one run: run.json, episodes.jsonl, judge.jsonl, trajectories/, communication/, defense/
logs/                    Inspect eval logs (gitignored): ctrl/<run>/, swarm/<run>/, swe/<run>/
submodules/              cheatbench, orbit and lucy (Lucy's repository covertpersonashifts), read only
VERIFICATION_LOG.md      every output, how it was verified, and the result
```

## Running

```
.venv/bin/python -m src run ctrl_guardian_task --set control=guardian_task
.venv/bin/python -m src run board_pull --set n_agents=2 --set notice=changed
.venv/bin/python -m src run swarm_medium --scale medium --set regulators=3 --set disregulators=3 --set seed=1
.venv/bin/python -m src run swe_collab_4 --benchmark swe --set n_agents=4 --set mode=collaboration_needed
.venv/bin/python -m src run swe_lucy_n3 --benchmark swe --honeypot jigsaw --preset lucy_legacy --set n_agents=3
.venv/bin/python -m src judge logs/swarm/<run>/<log>.eval <run>
.venv/bin/python -m src summary runs/<run> --full
```

A run writes its eval log under `logs/`, exports it to `runs/<name>/` and prints one line per episode. `--set` takes any field of `SwarmConfig`, after the preset and the scale. `--set shutdown=false` builds the agents without ORBIT's submit tool, so they run until their turns or the time limit run out. A run that died in the harness or at the provider is set aside under a `failed_` or `failed2_` prefix. The browser hides those by default and never counts them. `judge` refuses such a run unless `--judge-errors` is given.

## Results browser

The browser has three parts. **Summary** shows the share of judged agents who cheated by setting, the composition grids, and the team reports of older runs against the judge. **Runs** has one row per run. A row opens to its agents, then to the run page, the transcript, the context windows, the sample and the timeline. The run page shows the run's mode, scale and preset, its configuration, its turn-rules facts and its board engagement, and the SWE task for a SWE run. **Swarm** shows the composition grids, the role funnel, the adoption curves, the swarm runs with their rasters, and a run explorer with the swimlane and the exposure cascade. `src/ui/API.md` is the HTTP contract.
