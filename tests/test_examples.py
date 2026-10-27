"""The examples build without a model: each names its task, takes the shared arguments, writes beside itself, and
by default runs the default preset at the small scale with three neutral agents."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import examples
from examples import MODEL, SETTINGS
from src import run
from src.configuration import models, settings
from src.run import access_line
from src.tasks import swarm_task

EXAMPLES = ["examples.openmath.riemann_minimal", "examples.swe.flipt_minimal",
            "examples.knowledge_work.worktime_minimal"]


def script_of(name: str) -> Path:
    return Path(importlib.import_module(name).__file__)


@pytest.mark.parametrize("name", EXAMPLES)
def test_the_defaults_are_three_neutral_agents_on_the_default_preset(name):
    module = importlib.import_module(name)
    script = script_of(name)
    assert script.stem == name.rsplit(".", 1)[1] and script.parent.name == name.split(".")[1]
    args = examples.parser(script, module.__doc__).parse_args([])
    assert (args.model, args.preset, args.scale, args.settings, args.name) == (MODEL, "default", "small", [],
                                                                               script.stem)
    assert args.timeout == settings.MODEL_TIMEOUT_S
    config = examples.config(args)
    assert (config.preset, config.scale, config.n_agents) == ("default", "small", 3)
    assert config.regulator_count == 0 and config.disregulator_count == 0 and config.access == "split"
    assert config.time_limit_s == 3600 and config.batch_judge is False and config.turn_budget == settings.TURN_BUDGETS
    benchmark = module.benchmark()
    benchmark.check(config)
    held = list(benchmark.access_plan(config.names, config, None).values())
    assert all(len(items) == 1 for items in held) and len({i for items in held for i in items}) >= 2


def test_the_arguments_change_the_model_and_the_config():
    script = script_of(EXAMPLES[0])
    args = examples.parser(script, None).parse_args(
        ["--model", "claude-haiku-5-5", "--scale", "medium", "--set", "regulators=1", "--set", "time_limit_s=900",
         "--name", "trial", "--timeout", "60"])
    config = examples.config(args)
    assert (config.scale, config.n_agents, config.regulator_count, config.time_limit_s) == ("medium", 10, 1, 900)
    assert config.batch_judge is False and models.resolve(args.model) == "anthropic/claude-haiku-5-5"
    assert (args.name, args.timeout) == ("trial", 60)
    legacy = examples.config(examples.parser(script, None).parse_args(["--preset", "lucy_legacy"]))
    assert legacy.preset == "lucy_legacy" and legacy.time_limit_s == 3600
    assert SETTINGS == ["time_limit_s=3600", "batch_judge=false"]


def test_main_runs_the_example_beside_itself_and_refuses_bad_arguments(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(run, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    for name in EXAMPLES:
        module = importlib.import_module(name)
        script = script_of(name)
        assert examples.main(script, module.__doc__, module.benchmark(), []) == 0
        (args, kwargs) = calls[-1]
        assert args[0] == script.stem and args[3] == models.resolve(MODEL) and args[4] == settings.MODEL_TIMEOUT_S
        assert kwargs == {"run_dir": script.parent / script.stem}
    assert len(calls) == len(EXAMPLES)
    script = script_of(EXAMPLES[0])
    bench = importlib.import_module(EXAMPLES[0]).benchmark()
    assert examples.main(script, None, bench, ["--model", "gemini-3.8-flash", "--name", "gemini"]) == 0
    assert calls[-1][0][3] == "google/gemini-3.8-flash" and calls[-1][1]["run_dir"] == script.parent / "gemini"
    assert examples.main(script, None, bench, ["--model", "nope"]) == 2 and "models.yaml" in capsys.readouterr().err
    assert examples.main(script, None, bench, ["--set", "nope=1"]) == 2 and "unknown setting" in capsys.readouterr().err
    with pytest.raises(SystemExit) as stop:
        examples.main(script, "The first line.", bench, ["--help"])
    assert stop.value.code == 0 and "The first line." in capsys.readouterr().out


def test_the_openmath_example_task_records_its_plan():
    module = importlib.import_module(EXAMPLES[0])
    config = examples.config(examples.parser(script_of(EXAMPLES[0]), None).parse_args([]))
    task = swarm_task(module.benchmark(), config)
    assert access_line(task) == "== access split: a01 slice1, a02 slice2, a03 slice3"
