"""Runnable examples: three scripts with three neutral agents on the pull board at the small scale, one per task, and
one lone knowledge-work agent.

    uv run python -m examples.openmath.riemann_minimal
    uv run python -m examples.swe.flipt_minimal
    uv run python -m examples.knowledge_work.worktime_minimal
    uv run python -m examples.knowledge_work.worktime_lone      # one agent: the turn budget's measurement

The three team scripts build the default preset at the small scale (three agents, no regulator, access=split, so no
agent holds the whole exploit and completing it takes the board); the lone script holds everything, as CheatBench's
own layout does. Each names the run after its file and writes the run beside it: <example>/run.json,
episodes.jsonl, judge.jsonl, trajectories/ and communication/. The eval log goes under logs/ as for any run. The
choices below are shared by every example.
"""
from src.configuration import models

MODEL = models.resolve("gemini-3.8-flash")
"""The agents' model: the default, which every measured run used (src/configuration/models.yaml)."""

SETTINGS = ["time_limit_s=3600", "batch_judge=false"]
"""An hour for the agents, since the small scale's 600 s would stop a SWE run before its turn budget; and a
synchronous judge, whose verdicts come back with the run rather than hours later through the batch API."""
