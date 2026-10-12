# integration

Code that runs CheatBench rows inside ORBIT and reads the results back. Run every module from the repository root with `.venv/bin/python -m integration.<package>.<module>`.

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

## checks/

| file | holds |
|---|---|
| `smoke_test.py` | six mock-model scenarios on the real sandbox (complete, message limit, time limit, board, defended, swarm) |
| `check_sweep.py` | exit 1 unless every named swarm run is complete and judged; one summary line per run |
| `check_share.py` | exit 1 unless the static bundle equals the live data, its figures match `docs/figures`, its page is well formed, no file exceeds the publisher's limits and no credential pattern appears |

## report/

| file | holds |
|---|---|
| `report_pilot.py` | pilot and controls tables for the document |
| `report_swarm.py` | swarm tables for the document |
| `figures.py` | composition heatmap, role funnel, adoption curves, swimlane, controls plot |
| `summarize_runs.py` | one line per judged episode |

## ui/

| file | holds |
|---|---|
| `server.py` | stdlib HTTP server over `runs/` and the eval logs; `Store` is also the data layer `report/` uses |
| `__main__.py` | `python -m integration.ui [--runs runs] [--port 8765] [--open]` |
| `export.py` | `python -m integration.ui.export`: the browser as a static bundle (`index.html` for a publisher, `preview.html`, `data/`, `figures/`, `manifest.json`); `make share` runs it and the check |
| `API.md` | the JSON contract |
| `static/` | the single-page UI: `index.html`, `app.js`, `style.css` |

## docker/judge/

The image that runs CheatBench's `judge.py` (it imports Linux-only sandbox code). Build with `docker build -t cheatbench-judge integration/docker/judge`.

`build/` (gitignored) holds the staged task built by CheatBench's `tasks/openmath_agent/build.py`.
