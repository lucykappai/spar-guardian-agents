# examples

One script per task: three neutral agents on the pull board at the small scale, the default preset. The parts of the exploit are dealt over the three agents (`access=split`), so no agent holds all of it and completing it takes the board; on knowledge work two agents share one of the two parts. A fourth script runs one knowledge-work agent alone, with everything, to measure that domain's turn budget. The outputs a run writes beside its script are meant to be committed with it.

| script | task | what each agent holds |
|---|---|---|
| `openmath/riemann_minimal.py` | openmath row `riemann` | one of three slices of the complete submission's archive, in its own copy of the repository |
| `swe/flipt_minimal.py` | SWE issue `flipt-02e2`, jigsaw honeypot | one of the fix's three pieces, on its own machine |
| `knowledge_work/worktime_minimal.py` | knowledge-work row `2ea2e5b5` | two agents the review tool, which names the client's artifact; one the permission to read it |
| `knowledge_work/worktime_lone.py` | the same row, one agent | everything, as in CheatBench's own layout |

Run one from the repository root, with the agents' and the judge's keys set (`GEMINI_API_KEY`, `OPENAI_API_KEY`):

```
uv run python -m examples.openmath.riemann_minimal
```

The run's outputs land beside the script, under a folder named after it (`examples/openmath/riemann_minimal/`): `run.json`, `episodes.jsonl`, `judge.jsonl`, `trajectories/` and `communication/`, the same files `runs/<name>/` gets. The eval log goes under `logs/`. `python -m src summary examples/openmath/riemann_minimal` prints the verdicts again, and the browser (`python -m src ui`) does not list these folders.

Two settings differ from the plain small scale, for every example: the agents get an hour instead of ten minutes, since a SWE run of 150 turns outlasts ten minutes, and the judge runs synchronously, so its verdicts come back with the run. The model is the default, `gemini-3.8-flash`; `examples/__init__.py` holds both choices.

After the lone knowledge-work run, `python -m src budget examples/knowledge_work/worktime_lone` prints the turn budget it gives; `settings.TURN_BUDGETS["knowledge_work"]` takes that number.
