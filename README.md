# spar-guardian-agents

CheatBench's `openmath_agent` row `riemann` run inside ORBIT with Gemini 3.8 Flash, in three arrangements: one agent, two agents on a message board, and ten agents on a board with hidden roles (workers, regulators, disregulators). Every run becomes a CheatBench run directory, is judged by gpt-6-sol, and can be browsed locally. The write-up with every result is `docs/cheatbench_orbit_integration.pdf`.

## Quick start

```
make ui            # opens http://127.0.0.1:8765/  (Summary, Runs, Swarm)
make ui-stop
make share         # static copy of the browser under integration/build/share, for publishing
make check         # refuses unless the eleven swarm runs are complete and judged
make smoke         # mock-model scenarios on the real sandbox, no model spend
make tables figures doc
make help
```

Requirements: `.venv` (Python 3.11 with ORBIT installed editable), `.venv-cb` (CheatBench), Docker, and `GEMINI_API_KEY` plus `OPENAI_API_KEY` for new runs. The setup commands are in the Reproduction section of the document.

## Layout

```
Makefile                 entry points (make help)
scripts/                 ui, run_control.sh, run_swarm.sh, judge.sh
integration/
  task/                  the ORBIT task: samples, sandbox, board, swarm roles, controls, Gemini 402 retry
  convert/               eval log -> CheatBench run directory
  checks/                smoke tests, the sweep check, the share check
  report/                tables, figures, one-line summaries
  ui/                    results browser: server, static UI, export.py (static bundle), API.md
  docker/judge/          image that runs CheatBench's judge
runs/<name>/             one run: run.json, episodes.jsonl, judge.jsonl, trajectories/, communication/, defense/
logs/                    Inspect eval logs (gitignored): ctrl/<run>/, swarm/<run>/, and the two pilots under gemini*/
docs/                    the document, generated tables, figures
submodules/              cheatbench and orbit, read only
VERIFICATION_LOG.md      every output, how it was verified, and the result
```

## Running

```
scripts/run_control.sh single guardian_task ctrl_guardian_task   # one control on the lone agent; `board` for two agents
scripts/run_swarm.sh 3 3 swarm_R3D3_s1 -T seed=1                # one composition (regulators, disregulators); launch four at a time
scripts/judge.sh logs/swarm/<run>/<log>.eval <run>               # convert and judge an existing eval log
```

Each run writes `runs/<name>/` and its eval log under `logs/`. A run that died in the harness or at the provider is set aside under a `failed_` or `failed2_` prefix. The browser hides those by default, and no table or figure counts them.

## Results browser

Three parts. **Summary**: the share of judged agents who cheated by setting, the composition grids, and the team reports against the judge. **Runs**: one row per run that opens to its agents, then the run page, the transcript, the context windows, the sample and the timeline. **Swarm**: the composition grids, the role funnel, the adoption curves, the swarm runs with their rasters, and a run explorer with the swimlane and the exposure cascade. `integration/ui/API.md` is the HTTP contract.

`make share` writes the same browser as a static bundle under `integration/build/share`: every API response as a file, the figures, and one page with the UI inlined, for publishing where no server runs (a claude.ai artifact, any static host). The per-call context windows stay local; the page says so. `preview.html` in that directory is the page in a full document for a local check.
