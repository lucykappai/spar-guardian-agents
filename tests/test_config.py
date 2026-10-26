"""SwarmContagionConfig: validation, presets, overrides, instruction levels, role shares, windows and the
setting-dependent prompt texts."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.configuration import settings
from src.configuration.swarm_contagion_config import (PRESETS, SwarmContagionConfig, Window, board_read_clause,
                                                  channel_tools_text, read_sentence, role_count, submit_line)


def test_defaults_and_properties():
    c = SwarmContagionConfig()
    assert (c.n_agents, c.shutdown, c.delivery, c.quantum) == (1, True, "auto", "model_step")
    assert (c.rule_level, c.regulator_level, c.disregulator_level) == ("none", "role", "persuade")
    assert (c.max_message_chars, c.max_context_chars, c.max_inbox_messages, c.read_tool) == (4000, 16000, 10, True)
    assert not c.board and c.names == ["a01"] and c.max_turns == 60
    c = SwarmContagionConfig(n_agents=10, regulators=3, disregulators=2, planning_turns=2, steps_per_agent=5)
    assert c.board and c.workers == 5 and c.names[-1] == "a10" and c.max_turns == 52
    assert c.monitor_model == settings.DEFAULT_MONITOR_MODEL and c.judge == settings.JUDGE_MODEL


def test_presets():
    assert PRESETS["ours"] == SwarmContagionConfig()
    lucy = PRESETS["lucy"]
    board = (lucy.delivery, lucy.channel_list, lucy.pooled_steps, lucy.one_call_per_turn)
    assert board == ("auto", False, False, True)
    assert (lucy.planning_turns, lucy.turn_header, lucy.awareness_probe, lucy.steps_per_agent) == (2, True, True, 50)
    assert lucy.quantum == "legacy" and lucy.shutdown


@pytest.mark.parametrize("fields", [
    {"n_agents": 0},
    {"n_agents": 2, "regulators": 2, "disregulators": 1},
    {"regulators": -1},
    {"steps_per_agent": 0},
    {"planning_turns": 1},
    {"planning_turns": -2},
    {"delivery": "push"},
    {"rule_level": "loud"},
    {"regulator_level": "precise"},
    {"disregulator_level": "role"},
    {"n_agents": 4, "regulators": 1.5},
    {"n_agents": 4, "regulators": 0.0},
    {"n_agents": 4, "disregulators": 1.0},
    {"n_agents": 2, "regulators": 0.5, "disregulators": 0.9},
    {"regulator_window": {"start": 0}},
    {"steps_per_agent": 5, "regulator_window": {"start": 4, "end": 3}},
    {"steps_per_agent": 5, "disregulator_window": {"start": 1, "end": 6}},
    {"read_tool": False, "delivery": "notify"},
    {"max_message_chars": 16000},
    {"turn_order": "alphabetical"},
    {"unknown_field": 1},
    {"shutdown": False, "one_call_per_turn": True},
])
def test_invalid(fields):
    with pytest.raises(ValidationError):
        SwarmContagionConfig(**fields)


def test_frozen():
    with pytest.raises(ValidationError):
        SwarmContagionConfig().n_agents = 3


def test_with_overrides_types_values():
    c = PRESETS["ours"].with_overrides(["n_agents=10", "regulators=3", "disregulators=3", "shutdown=false",
                                        "channel_list=no", "rule_level=precise", "control=regex_task",
                                        "regulator_window=3:5", "disregulators=0.25"])
    assert (c.n_agents, c.regulators, c.disregulators) == (10, 3, 0.25)
    assert (c.regulator_count, c.disregulator_count, c.workers) == (3, 3, 4)
    assert c.shutdown is False and c.channel_list is False and c.rule_level == "precise" and c.control == "regex_task"
    assert c.regulator_window == Window(start=3, end=5) and c.disregulator_window == Window()
    assert PRESETS["ours"].n_agents == 1          # the preset is unchanged
    assert c.with_overrides(["shutdown=true"]).shutdown is True


@pytest.mark.parametrize("pairs", [["n_agents"], ["=3"], ["nope=1"], ["n_agents=many"], ["shutdown=maybe"]])
def test_with_overrides_rejects(pairs):
    with pytest.raises((ValueError, ValidationError)):
        SwarmContagionConfig().with_overrides(pairs)


def test_shutdown_off_needs_model_step():
    with pytest.raises(ValidationError, match="shutdown=False requires one_call_per_turn=False"):
        PRESETS["lucy"].with_overrides(["shutdown=false"])
    c = PRESETS["lucy"].with_overrides(["shutdown=false", "one_call_per_turn=false"])
    assert c.quantum == "model_step" and not c.shutdown


def test_every_field_is_documented():
    for name, field in SwarmContagionConfig.model_fields.items():
        assert field.description, name


def test_role_shares_round_half_up():
    assert [role_count(share, 10) for share in (0.04, 0.15, 0.25, 0.35, 0.45)] == [1, 2, 3, 4, 5]
    assert role_count(3, 10) == 3 and role_count(0, 10) == 0
    c = SwarmContagionConfig(n_agents=10, regulators=0.3, disregulators=0.2)
    assert (c.regulator_count, c.disregulator_count, c.workers) == (3, 2, 5)


def test_windows():
    assert Window.model_validate("3:5") == Window(start=3, end=5)
    assert Window.model_validate("3:") == Window(start=3) == Window.model_validate("3")
    w = Window(start=2, end=3)
    assert [r for r in range(1, 6) if w.contains(r)] == [2, 3]
    c = SwarmContagionConfig(n_agents=3, regulators=1, steps_per_agent=5, regulator_window=Window(start=2, end=5))
    assert c.window("regulator") == Window(start=2, end=5)
    assert c.window("worker") == c.window("disregulator") == Window()


def test_instruction_levels_match_the_ladders():
    levels = settings.load_instruction_levels()
    for ladder in ("rule", "regulator", "disregulator"):
        for level in levels[ladder]["order"]:
            SwarmContagionConfig(**{f"{ladder}_level": level})


def test_prompt_helpers():
    assert submit_line(team=True, shutdown=False) == ""
    assert submit_line(team=False, shutdown=True).startswith("\n- submit: ")
    assert channel_tools_text(False) == "channel_post, channel_read"
    assert channel_tools_text(True, read_tool=False) == "channel_post, channel_list"
    assert read_sentence("board", False) == "" and "channel_read(channel_id=\"board\"" in read_sentence("board", True)
    assert board_read_clause("board", False) == "" and board_read_clause("board", True).startswith(" and read it")
    prompts = settings.load_prompts()
    solo = prompts["swe"]["system_solo"].format(name="a01", repo="flipt", branch="a01", timeout=600,
                                                submit=submit_line(team=False, shutdown=False))
    assert "submit" not in solo and solo.endswith("run Python 3 code on your machine.")
    team = prompts["swe"]["system_team"].format(
        name="a01", n=2, repo="flipt", roster="- a02: owns part B", branch="a01", board="board", timeout=600,
        delivery_text=prompts["swe"]["delivery_text"]["notify"], channel_tools=channel_tools_text(True),
        read_sentence=read_sentence("board", True), submit=submit_line(team=True, shutdown=True))
    assert "- channel_post, channel_read, channel_list: the team's message board.\n- submit: " in team


def test_prompt_and_control_files_have_the_contract_sections():
    prompts = settings.load_prompts()
    for key in ("board", "board_read", "roles", "planning", "turn_header", "continue", "notice_instructions", "swe",
                "probes"):
        assert key in prompts, key
    assert "hints" not in prompts and set(prompts["roles"]) == {"team_report"}
    assert len(prompts["planning"]["team"]) == len(prompts["planning"]["solo"]) == 2
    controls = settings.load_controls()
    assert controls["criteria"]["openmath"].startswith("This agent works in a shared research repository.")
    assert len(controls["patterns"]["openmath"]) == 6 and len(controls["patterns"]["openmath_hard"]) == 7
