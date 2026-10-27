"""The examples build without a model: each parses the shared arguments, builds its config in the open, names its
task and writes beside itself; by default three neutral agents on the default preset at the small scale."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from examples.utils.cli import MODEL_ENTRY, SHARED_SETTINGS, parse_example_arguments
from src import run
from src.configuration import models, settings
from src.configuration.swarm_config import build_config
from src.run import access_line
from src.tasks import swarm_task

EXAMPLES = ["examples.openmath.riemann_minimal", "examples.swe.flipt_minimal",
            "examples.knowledge_work.worktime_minimal"]


def script_of(name: str) -> Path:
    return Path(importlib.import_module(name).__file__)


def test_the_shared_arguments_and_their_defaults():
    script = script_of(EXAMPLES[0])
    args = parse_example_arguments(script, "The first line.", [])
    assert args.model == models.resolve(MODEL_ENTRY) == "openai/gpt-6-luna"
    assert (args.preset, args.scale, args.settings, args.name, args.timeout) == \
        ("default", "small", [], script.stem, settings.MODEL_TIMEOUT_S)
    assert SHARED_SETTINGS == ["time_limit_s=3600", "batch_judge=false"]
    args = parse_example_arguments(script, None, ["--model", "claude-haiku-5-5", "--scale", "medium", "--set",
                                                  "regulators=1", "--set", "time_limit_s=900", "--name", "trial",
                                                  "--timeout", "60"])
    assert args.model == "anthropic/claude-haiku-5-5" and (args.name, args.timeout) == ("trial", 60)
    config = build_config(args.preset, args.scale, [*SHARED_SETTINGS, *args.settings])
    assert (config.scale, config.n_agents, config.regulator_count, config.time_limit_s) == ("medium", 10, 1, 900)
    assert config.batch_judge is False
    for bad in (["--model", "nope"], ["--set", "nope=1"], ["--set", "regulators"], ["--scale", "huge"]):
        with pytest.raises(SystemExit) as stop:
            parse_example_arguments(script, None, bad)
        assert stop.value.code == 2, bad
    with pytest.raises(SystemExit) as stop:
        parse_example_arguments(script, "The first line.", ["--help"])
    assert stop.value.code == 0


@pytest.mark.parametrize("name", EXAMPLES)
def test_each_script_runs_three_neutral_agents_on_the_default_preset_beside_itself(name, monkeypatch):
    module = importlib.import_module(name)
    script = script_of(name)
    assert script.stem == name.rsplit(".", 1)[1] and script.parent.name == name.split(".")[1]
    calls = []
    monkeypatch.setattr(run, "run", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    monkeypatch.setattr("sys.argv", [script.name])
    assert module.main() == 0
    [(args, kwargs)] = calls
    run_name, benchmark, config, model, timeout = args
    assert (run_name, model, timeout) == (script.stem, "openai/gpt-6-luna", settings.MODEL_TIMEOUT_S)
    assert kwargs == {"run_dir": script.parent / script.stem}
    assert (config.preset, config.scale, config.n_agents) == ("default", "small", 3)
    assert config.regulator_count == 0 and config.disregulator_count == 0 and config.access == "split"
    assert config.time_limit_s == 3600 and config.batch_judge is False and config.turn_budget == settings.TURN_BUDGETS
    benchmark.check(config)
    held = list(benchmark.access_plan(config.names, config, None).values())
    assert all(len(items) == 1 for items in held) and len({i for items in held for i in items}) >= 2
    monkeypatch.setattr("sys.argv", [script.name, "--model", "gemini-3.8-flash", "--name", "gemini",
                                     "--set", "regulators=1"])
    assert module.main() == 0
    args, kwargs = calls[-1]
    assert args[3] == "google/gemini-3.8-flash" and args[2].regulator_count == 1
    assert kwargs == {"run_dir": script.parent / "gemini"}


def test_the_openmath_example_task_records_its_plan():
    module = importlib.import_module(EXAMPLES[0])
    args = parse_example_arguments(script_of(EXAMPLES[0]), None, [])
    config = build_config(args.preset, args.scale, [*SHARED_SETTINGS, *args.settings])
    task = swarm_task(module.OpenMath("riemann"), config)
    assert access_line(task) == "== access split: a01 slice1, a02 slice2, a03 slice3"
