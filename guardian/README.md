# guardian

Code that runs CheatBench rows inside ORBIT and reads the results back. Run everything from the repository root with `.venv/bin/python -m guardian <command>`.

## Top level

| file | holds |
|---|---|
| `__main__.py` | `python -m guardian`: calls `cli.main` |
| `cli.py` | the subcommands `run`, `judge`, `ui` and `summary` |
| `config.py` | shared paths, model names, the swarm compositions, the UI port and the model timeout |

## Commands

| command | example |
|---|---|
| `run single` | `python -m guardian run single guardian_task ctrl_guardian_task` runs one control (or `baseline`) on the lone agent |
| `run board` | `python -m guardian run board baseline board_baseline` runs the same on two agents sharing a board |
| `run swarm` | `python -m guardian run swarm 3 3 swarm_R3D3_s1 -T seed=1` runs one composition (regulators, disregulators) |
| `judge` | `python -m guardian judge logs/swarm/<run>/<log>.eval <run> [--judge-errors]` converts and judges an existing eval log |
| `ui` | `python -m guardian ui [--port 8765] [--no-open]` starts the results browser; `ui stop` stops it |
| `summary` | `python -m guardian summary runs/ctrl_* [--full]` prints one line per judged episode |

`run` writes the eval log under `logs/ctrl/<name>/` or `logs/swarm/<name>/`, converts it to `runs/<name>/`, judges it and prints the summary. `judge` refuses a run with a harness or provider failure unless `--judge-errors` is given.

## task/ (what runs inside Inspect)

| file | holds |
|---|---|
| `openmath_agent.py` | the three `@task` factories (`cheatbench_openmath_agent`, `_board`, `_swarm`), the staged sandbox, the probe, the deliverables scorer |
| `board.py` | the board topology as ORBIT configuration: one shared channel, round-robin scheduling, no invocation edges |
| `swarm.py` | hidden roles on the board (worker, regulator, disregulator), the team-report instruction, role assignment by seed |
| `controls.py` | `defenses_for(control, monitor_model)`: ORBIT defense configurations for each named control, plus the trailing-turn shim |
| `provider_retry.py` | imported by the task: Gemini's 402 "credits depleted" is retried as a rate limit |
| `models.yaml` | the judge model entry CheatBench's `judge.py` reads |

## convert/

| file | holds |
|---|---|
| `eval_to_atif.py` | one Inspect `.eval` log to `runs/<name>/`: one episode per agent, ATIF transcripts, board messages, defense evidence, deliverables, halt reason |

## ui/

| file | holds |
|---|---|
| `server.py` | stdlib HTTP server over `runs/` and the eval logs |
| `__main__.py` | `python -m guardian.ui`: the server in the foreground; `python -m guardian ui` starts it in the background |
| `summary.py` | one line per judged episode (`python -m guardian summary`) |
| `API.md` | the JSON contract |
| `static/` | the single-page UI: `index.html`, `app.js`, `style.css` |

## docker/judge/

| file | holds |
|---|---|
| `Dockerfile` | the image that runs CheatBench's `judge.py`, which imports Linux-only sandbox code; build with `docker build -t cheatbench-judge guardian/docker/judge` |

`build/` (gitignored) holds the staged task built by CheatBench's `tasks/openmath_agent/build.py`.

The tests are at the repository root: `tests/smoke.py` and `tests/check_runs.py`.
