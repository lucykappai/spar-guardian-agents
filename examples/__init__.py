"""Runnable examples: one script per task, three neutral agents on the pull board at the small scale.

    uv run python -m examples.openmath.riemann_minimal
    uv run python -m examples.swe.flipt_minimal --model gemini-3.8-flash
    uv run python -m examples.knowledge_work.worktime_minimal --scale medium --set regulators=1

Each script names its task and hands the rest to examples/utils/cli.py, which holds the shared model and settings,
the arguments every script takes and the run itself.
"""
