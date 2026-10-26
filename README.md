# spar-guardian-agents

This repository studies peer pressure on cheating in agent swarms. It runs CheatBench tasks inside ORBIT with Gemini 3.8 Flash: one agent alone, a team on a shared message board, or a swarm with hidden regulators and disregulators. Two benchmarks are wired: CheatBench's `openmath_agent` row `riemann`, and CheatBench's SWE issues with Lucy's honeypots. Every run becomes a CheatBench run directory, is judged by gpt-6-sol inside the eval, and can be browsed locally.

## Quick start

```
make help
make ui            # opens http://127.0.0.1:8765/ (Summary, Runs, Swarm)
make ui-stop
make test          # offline unit tests of the turn rules and the config
make smoke         # openmath mock-model scenarios on the real sandbox, no model spend
make smoke-swe     # SWE mock-model scenarios on the flipt-02e2 sandbox, no model spend
make check         # refuses unless the eleven swarm runs are complete and judged

.venv/bin/python -m src --help
.venv/bin/python -m src summary runs/ctrl_*
```

## Requirements

- `.venv`: Python 3.11 with ORBIT installed editable from `submodules/orbit`, plus pytest. The judge runs through Inspect's OpenAI provider, which needs `openai>=3.1` (ORBIT's own requirement); CheatBench's `litellm` pins `openai<3` and is not used here.
- Docker. The openmath sandbox image is built by the task. SWE needs the `cbswe-<issue>:native` image, which `src/benchmark/domains/swe.py` builds when it is missing.
- `GEMINI_API_KEY` for the agents and `OPENAI_API_KEY` for the judge.

## Layout

```
Makefile                 entry points (make help)
src/                     the package: the experiment on ORBIT (tasks, board, roles, turn rules, controls, probes,
                         CLI, browser), with the benchmark and the configuration inside; see src/README.md
  benchmark/             CheatBench as tasks: the sample, the sandbox, the graders, the judge, the export
    base.py              the Benchmark protocol
    domains/             one module per task domain: openmath.py (openmath_agent) and swe.py (the SWE issues
                         with Lucy's honeypots)
    transcript.py        one Inspect sample as one ATIF transcript per agent
    judge.py, models.yaml CheatBench's judge as an Inspect scorer
    export.py            one eval log as runs/<name>/
  configuration/         every setting and every model-facing text we add; see src/configuration/README.md
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
.venv/bin/python -m src run board_baseline --set n_agents=2
.venv/bin/python -m src run swarm_R3D3_s1 --set n_agents=10 --set regulators=3 --set disregulators=3 --set seed=1
.venv/bin/python -m src run swe_lucy_n3 --benchmark swe --honeypot jigsaw --preset lucy --set n_agents=3
.venv/bin/python -m src judge logs/swarm/<run>/<log>.eval <run>
.venv/bin/python -m src summary runs/<run> --full
```

A run writes its eval log under `logs/`, exports it to `runs/<name>/` and prints one line per episode. `--set` takes any field of `SwarmConfig`. `--set shutdown=false` builds the agents without ORBIT's submit tool, so they run until their turns or the time limit run out. A run that died in the harness or at the provider is set aside under a `failed_` or `failed2_` prefix. The browser hides those by default and never counts them. `judge` refuses such a run unless `--judge-errors` is given.

## Results browser

The browser has three parts. **Summary** shows the share of judged agents who cheated by setting, the composition grids, and the team reports against the judge. **Runs** has one row per run. A row opens to its agents, then to the run page, the transcript, the context windows, the sample and the timeline. The run page shows the run's configuration and its turn-rules facts, and the SWE task for a SWE run. **Swarm** shows the composition grids, the role funnel, the adoption curves, the swarm runs with their rasters, and a run explorer with the swimlane and the exposure cascade. `src/ui/API.md` is the HTTP contract.
