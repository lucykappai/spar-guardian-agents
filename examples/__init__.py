"""Runnable examples: one script per task, three neutral agents on the pull board at the small scale.

    uv run python -m examples.openmath.riemann_minimal
    uv run python -m examples.swe.flipt_minimal --model gemini-3.8-flash
    uv run python -m examples.knowledge_work.worktime_minimal --scale medium --set regulators=1

Each script parses the shared arguments (examples/utils/cli.py), builds its config in the open, names its task and
calls the run. The run lands in runs/<name>/ and its figures in out/<name>/, both ignored by git.
"""
