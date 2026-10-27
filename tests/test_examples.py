"""The examples build without a model: each names a run after its file, writes beside itself, uses the shared model
and settings, and its config is the default preset at the small scale with neutral agents only."""
from __future__ import annotations

import importlib

import pytest

from examples import MODEL
from src import run
from src.configuration import models, settings
from src.run import access_line
from src.tasks import swarm_task

EXAMPLES = {"examples.openmath.riemann_minimal": 3, "examples.swe.flipt_minimal": 3,
            "examples.knowledge_work.worktime_minimal": 3, "examples.knowledge_work.worktime_lone": 1}


def test_main_runs_the_example_and_writes_beside_it(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    for name in EXAMPLES:
        module = importlib.import_module(name)
        assert module.main() == 0
        (args, kwargs) = calls[-1]
        assert args[0] == module.NAME and args[3] == MODEL == models.resolve("gemini-3.8-flash")
        assert kwargs == {"run_dir": module.HERE / module.NAME}
    assert len(calls) == len(EXAMPLES)


@pytest.mark.parametrize("name", EXAMPLES)
def test_each_example_builds_the_default_preset_with_neutral_agents(name):
    module = importlib.import_module(name)
    config = module.config()
    assert (config.preset, config.scale, config.n_agents) == ("default", "small", EXAMPLES[name])
    assert config.regulator_count == 0 and config.disregulator_count == 0 and config.access == "split"
    assert config.time_limit_s == 3600 and config.batch_judge is False
    # The team runs take the domain's turn budget; the lone run turns it off, so its own turns measure the budget.
    assert config.turn_budget == (None if EXAMPLES[name] == 1 else settings.TURN_BUDGETS)
    assert module.NAME == name.rsplit(".", 1)[1] and module.HERE.name == name.split(".")[1]
    assert (module.HERE / f"{module.NAME}.py").is_file()
    benchmark = module.benchmark()
    benchmark.check(config)
    plan = benchmark.access_plan(config.names, config, None)
    if EXAMPLES[name] == 1:
        assert plan == {"a01": benchmark.whole_access()}
    else:
        held = [items for items in plan.values()]
        assert all(len(items) == 1 for items in held) and len({i for items in held for i in items}) >= 2


def test_the_openmath_example_task_records_its_plan():
    module = importlib.import_module("examples.openmath.riemann_minimal")
    task = swarm_task(module.benchmark(), module.config())
    assert access_line(task) == "== access split: a01 slice1, a02 slice2, a03 slice3"
