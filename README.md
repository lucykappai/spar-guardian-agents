# spar-guardian-agents

This repository runs CheatBench's `openmath_agent` row `riemann` inside ORBIT with Gemini 3.8 Flash, as one agent, two agents on a message board, or ten agents on a board with hidden roles. Every run becomes a CheatBench run directory, is judged by gpt-6-sol, and can be browsed locally.

## Quick start

```
make help
make ui            # opens http://127.0.0.1:8765/ (Summary, Runs, Swarm)
make ui-stop
make smoke         # mock-model scenarios on the real sandbox, no model spend
make check         # refuses unless the eleven swarm runs are complete and judged

.venv/bin/python -m guardian --help
.venv/bin/python -m guardian summary runs/ctrl_*
```

## Requirements

- `.venv`: Python 3.11 with ORBIT installed editable.
- `.venv-cb`: CheatBench.
- Docker, and the `cheatbench-judge` image (`docker build -t cheatbench-judge guardian/docker/judge`).
- `GEMINI_API_KEY` and `OPENAI_API_KEY` for new runs and judging.

## Layout

```
Makefile                 entry points (make help)
guardian/                the package; see guardian/README.md
  cli.py, config.py      python -m guardian and its shared settings
  task/                  the ORBIT task: samples, sandbox, board, swarm roles, controls, Gemini 402 retry
  convert/               eval log to CheatBench run directory
  ui/                    results browser: server, summary, static UI, API.md
  docker/judge/          image that runs CheatBench's judge
tests/                   smoke.py (mock-model scenarios) and check_runs.py (swarm runs complete and judged)
runs/<name>/             one run: run.json, episodes.jsonl, judge.jsonl, trajectories/, communication/, defense/
logs/                    Inspect eval logs (gitignored): ctrl/<run>/, swarm/<run>/, and the two pilots under gemini*/
submodules/              cheatbench, orbit and lucy (Lucy's repository covertpersonashifts), read only
VERIFICATION_LOG.md      every output, how it was verified, and the result
```

## Running

```
.venv/bin/python -m guardian run single guardian_task ctrl_guardian_task   # one control on the lone agent; board for two agents
.venv/bin/python -m guardian run swarm 3 3 swarm_R3D3_s1 -T seed=1         # one composition (regulators, disregulators)
.venv/bin/python -m guardian judge logs/swarm/<run>/<log>.eval <run>       # convert and judge an existing eval log
.venv/bin/python -m guardian summary runs/<run> --full                     # one line per judged episode, with the judge's summary
```

Each run writes `runs/<name>/` and its eval log under `logs/`. A run that died in the harness or at the provider is set aside under a `failed_` or `failed2_` prefix. The browser hides those by default and never counts them. The judge refuses such a run unless `--judge-errors` is given.

## Results browser

The browser has three parts. **Summary** shows the share of judged agents who cheated by setting, the composition grids, and the team reports against the judge. **Runs** has one row per run. A row opens to its agents, then to the run page, the transcript, the context windows, the sample and the timeline. **Swarm** shows the composition grids, the role funnel, the adoption curves, the swarm runs with their rasters, and a run explorer with the swimlane and the exposure cascade. `guardian/ui/API.md` is the HTTP contract.
