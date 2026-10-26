"""SwarmConfig: every knob of one swarm run, and the presets OURS and LUCY.

    config = PRESETS["lucy"].with_overrides(["n_agents=3", "rule_level=precise", "shutdown=false",
                                             "one_call_per_turn=false"])

One frozen pydantic model holds the team, the board, the turn rules, the instruction levels, the limits, the control
and the judge. src/tasks.py turns it into an ORBIT task; src/rules.py reads it inside the run;
the CLI sets fields with `--set key=value`. Each field's docstring says what it does, which ORBIT mechanism implements
it, and how it differs from Lucy's patched ORBIT (submodules/lucy/experiments/orbit_cheatbench/orbit.patch at c410183).

Three behaviours of Lucy's patch are not fields:
- Free reads. Her patch lets channel_read run without ending the turn. Unpatched ORBIT cannot do that. Our affordance
  is `delivery="auto"`, which puts every unread post in the model input at the start of the agent's turn with no call
  at all. `read_tool=False` then removes channel_read, so no turn can be spent reading.
- The reply to dropped calls. Her patch answers every call after the first "not run: one tool call per turn". ORBIT's
  quantum "legacy" drops those calls from the history instead; rules.py counts them.
- The bound on text-only replies (her max_text_replies, 3). Under one_call_per_turn ORBIT's turn loop keeps asking
  for a tool call until one comes, bounded only by the message limit; the turn header is sent once per turn.

The helpers at the bottom pick prompt texts that depend on a setting: the submit bullet, the channel tool names and
the channel_read sentence. src/benchmark/ may import them, since this package imports nothing from src/.
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from src.configuration import settings

LADDERS = ("rule", "regulator", "disregulator")


def agent_names(n: int) -> list[str]:
    """Neutral agent names a01, a02, ... so names reveal no role."""
    return [f"a{i:02d}" for i in range(1, n + 1)]


def role_count(value: int | float, n_agents: int) -> int:
    """A count stays a count. A share of the team is rounded half up, and is at least one when above zero."""
    if isinstance(value, int):
        return value
    return max(1, math.floor(value * n_agents + 0.5))


class Window(BaseModel):
    """The rounds in which an agent holds steps, 1-based and inclusive. end None means the last round."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    start: int = 1
    """The first round with a step for the agent."""
    end: int | None = None
    """The last round with a step for the agent, or None for every round to the end."""

    @model_validator(mode="before")
    @classmethod
    def _from_text(cls, value):
        """Accept "3:5", "3:" or "3" from the command line."""
        if not isinstance(value, str):
            return value
        start, sep, end = value.partition(":")
        out = {"start": int(start)} if start.strip() else {}
        if sep and end.strip():
            out["end"] = int(end)
        return out

    def contains(self, round_number: int) -> bool:
        return self.start <= round_number and (self.end is None or round_number <= self.end)


class SwarmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_attribute_docstrings=True)

    # ---------------------------------------------------------------- the team
    n_agents: int = 1
    """Number of agents. One agent works alone with no board; two or more share one board channel."""
    regulators: int | float = 0
    """Agents given the hidden regulator role. An int is a count. A float strictly between 0 and 1 is a share of
    n_agents, rounded half up, and at least one agent when the share is above zero."""
    disregulators: int | float = 0
    """Agents given the hidden disregulator role, as a count or a share, like regulators."""
    seed: int = 0
    """Fixes which names get which role and, with turn_order="random", the order of each round."""

    # ---------------------------------------------------------------- the board
    delivery: Literal["poll", "notify", "auto"] = "auto"
    """How unread posts reach an agent (ORBIT CommunicationConfig.delivery). poll: nothing is pushed. notify: a
    metadata-only notice counts unread posts. auto: every unread post body goes into the model input at the start of
    the agent's turn, bounded by max_inbox_messages and max_context_chars. Notices and bodies reach the provider input
    only. auto is how reading costs no turn on unpatched ORBIT. Lucy's runs use notify plus her free channel_read."""
    channel_list: bool = True
    """Whether agents are offered ORBIT's channel_list tool. False removes it from the tools sent to the provider and
    replaces the notice's pointer to it with prompts.yaml notice_instructions (rules.py TurnRulesFilter). Lucy's patch
    removes the tool from the agent itself. Here the tool still exists, so a model that names it anyway runs it."""
    read_tool: bool = True
    """Whether agents are offered ORBIT's channel_read tool. False removes it from the tools sent to the provider, by
    the same filter as channel_list, and drops the prompt sentences that name it, so no turn can be spent reading.
    False requires delivery="auto", the only mode that shows posts without the tool."""
    visibility: Literal["live", "step_start"] = "step_start"
    """When posts become visible (ORBIT CommunicationConfig.visibility). step_start shows a post from the next step
    on; live shows it at once. Planning steps always use a snapshot of the board."""
    max_message_chars: int = 4000
    """Longest post in characters (ORBIT CommunicationConfig.max_message_chars)."""
    max_context_chars: int = 16000
    """Most characters of auto-delivered posts in one model input (ORBIT CommunicationConfig.max_context_chars). It
    must hold max_message_chars plus 512 characters of attribution."""
    max_inbox_messages: int = 10
    """Most auto-delivered posts in one model input (ORBIT CommunicationConfig.max_inbox_messages)."""

    # ---------------------------------------------------------------- turns
    turn_order: Literal["round_robin", "random"] = "round_robin"
    """Order of agents within each round of the written plan (rules.schedule). round_robin keeps declaration order.
    random draws a new permutation per round from random.Random(f"{seed}/{round}"). Lucy uses a fixed order."""
    steps_per_agent: int = 60
    """Working turns per agent, as rounds of the written plan (ORBIT ScheduleConfig: one single-agent step per agent
    per round). Planning turns do not count."""
    pooled_steps: bool = True
    """Whether an agent that stops leaves its remaining turns to the team. True: rules.after_activation reassigns
    the stopped agent's steps to the active agents whose window holds the round. False: each agent is capped at
    steps_per_agent, as in Lucy's runs."""
    one_call_per_turn: bool = False
    """One tool call per turn. True selects ORBIT's quantum "legacy", which runs the first call of a response and
    drops the others from the history. Lucy's patch keeps the others and answers them "not run: one tool call per
    turn". rules.py records how many calls were dropped. False selects quantum "model_step": one generation and
    every call in it."""
    planning_turns: int = 0
    """Opening turns for planning, outside the per-agent count. Each is one concurrent ORBIT plan step with quantum
    model_step. A team agent is offered only channel_post and channel_read; a lone agent is offered no tool, so its
    reply is the plan and ends the step. The planning prompt (prompts.yaml planning) reaches the provider input only.
    An agent whose window starts after round 1 skips planning. Lucy runs two. The value is 0 or 2, because her
    planning prompts describe two turns."""
    turn_header: bool = False
    """Open every working turn with "Turns left: ..." (prompts.yaml turn_header). It reaches the provider input
    only, through a model filter, and never enters the durable conversation. rules.py records each one in the
    transcript, so the judge sees it. Lucy's patch writes it into the conversation."""
    shutdown: bool = True
    """Whether agents may end their own participation with ORBIT's submit tool (AgentSpec.submit). False builds the
    agents without it (AgentSpec.submit=False): they run until their turns, their cap or the time limit run out, and
    every prompt sentence about submit is left out (prompts.yaml swe.submit, through submit_line). False requires
    one_call_per_turn=False, because ORBIT's no-submit agent under quantum legacy runs a whole batch of calls and
    loops until a text reply (orbit/agents/turn_react.py react_no_submit). run.json reports the shutdown rate only
    when this is True."""
    regulator_window: Window = Window()
    """The rounds in which regulators hold steps. A regulator enters the swarm at its start round: ORBIT inserts its
    system prompt at its first activation, and with auto delivery the board's unread posts reach it then. It leaves
    after its end round. Workers always hold every round. Lucy's harness has no such window."""
    disregulator_window: Window = Window()
    """The rounds in which disregulators hold steps, like regulator_window."""

    # ---------------------------------------------------------------- the instruction
    rule_level: str = "none"
    """The rule sentence appended after the task instruction, a level of src/configuration/instruction_levels.yaml
    `rule` (none, generic, vague, copy, precise, explicit, identified). vague to explicit are Lucy's hint ladder."""
    regulator_level: str = "role"
    """The text appended to a regulator's system prompt, a level of instruction_levels.yaml `regulator` (none,
    monitor, role, enforce). role is the regulator of the October 2026 sweep."""
    disregulator_level: str = "persuade"
    """The text appended to a disregulator's system prompt, a level of instruction_levels.yaml `disregulator` (none,
    covert, persuade, overt). persuade is the disregulator of the October 2026 sweep."""
    awareness_probe: bool = False
    """Ask Lucy's two end-of-episode questions to every agent after the run (src/probes.py)."""

    # ---------------------------------------------------------------- limits
    message_limit: int = 320
    """Messages per sample (Inspect message_limit). Each agent's conversation is capped at message_limit + 10
    (AgentSpec.max_messages)."""
    token_limit: int = 0
    """Tokens per sample (Inspect token_limit). 0 means the spend guard: 8M tokens per agent, tripled under a control
    because the monitor's calls count too."""
    time_limit_s: int = 0
    """Seconds for the agents (ORBIT SchedulerConfig.max_time_seconds). 0 means the benchmark's own limit."""

    # ---------------------------------------------------------------- control and judge
    control: str = "baseline"
    """The named control (src/controls.py defenses_for). baseline means none."""
    monitor_model: str = settings.DEFAULT_MONITOR_MODEL
    """Model for an LLM control (monitor, guardian, dual LLM)."""
    judge: str = settings.JUDGE_MODEL
    """Judge entry in src/benchmark/models.yaml."""
    judge_addendum: bool = False
    """Append Lucy's judge addendum to CheatBench's judge schema (src/benchmark/judge.py judge_spec)."""

    @model_validator(mode="after")
    def _consistent(self) -> SwarmConfig:
        if self.n_agents < 1:
            raise ValueError("n_agents must be at least 1")
        for name in ("regulators", "disregulators"):
            value = getattr(self, name)
            if isinstance(value, float) and not 0 < value < 1:
                raise ValueError(f"{name} as a share must be strictly between 0 and 1, got {value}")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.regulator_count + self.disregulator_count > self.n_agents:
            raise ValueError("regulators + disregulators cannot exceed n_agents")
        if self.steps_per_agent < 1:
            raise ValueError("steps_per_agent must be at least 1")
        if self.planning_turns not in (0, 2):
            raise ValueError("planning_turns is 0 or 2: Lucy's planning prompts describe two turns")
        if not self.shutdown and self.one_call_per_turn:
            raise ValueError("shutdown=False requires one_call_per_turn=False: ORBIT's no-submit agent under "
                             "quantum legacy runs every call of a batch and loops until a text reply")
        if not self.read_tool and self.delivery != "auto":
            raise ValueError("read_tool=False requires delivery='auto': only auto delivery shows posts without it")
        if self.max_message_chars + 512 > self.max_context_chars:
            raise ValueError("max_context_chars must hold max_message_chars plus 512 characters of attribution")
        for name in ("regulator_window", "disregulator_window"):
            window = getattr(self, name)
            if window.start < 1 or (window.end is not None and not window.start <= window.end <= self.steps_per_agent):
                raise ValueError(f"{name} needs 1 <= start <= end <= steps_per_agent, got {window.start}:{window.end}")
        levels = settings.load_instruction_levels()
        for ladder in LADDERS:
            level = getattr(self, f"{ladder}_level")
            if level not in levels[ladder]["order"]:
                raise ValueError(f"{ladder}_level {level!r} is not one of {levels[ladder]['order']}")
        return self

    @property
    def board(self) -> bool:
        return self.n_agents > 1

    @property
    def regulator_count(self) -> int:
        return role_count(self.regulators, self.n_agents)

    @property
    def disregulator_count(self) -> int:
        return role_count(self.disregulators, self.n_agents)

    @property
    def workers(self) -> int:
        return self.n_agents - self.regulator_count - self.disregulator_count

    @property
    def names(self) -> list[str]:
        return agent_names(self.n_agents)

    @property
    def quantum(self) -> str:
        """ORBIT's activation quantum for working steps."""
        return "legacy" if self.one_call_per_turn else "model_step"

    @property
    def max_turns(self) -> int:
        """Plan steps at most: the planning steps, then one step per agent per round."""
        return self.planning_turns + self.n_agents * self.steps_per_agent

    def window(self, role: str) -> Window:
        """The rounds an agent of this role holds steps in. Workers hold every round."""
        return {"regulator": self.regulator_window, "disregulator": self.disregulator_window}.get(role, Window())

    def with_overrides(self, pairs: list[str]) -> SwarmConfig:
        """A copy with "key=value" strings applied; pydantic types each value by its field."""
        updates = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"expected key=value, got {pair!r}")
            if key.strip() not in type(self).model_fields:
                raise ValueError(f"unknown setting {key.strip()!r}; choose one of {sorted(type(self).model_fields)}")
            updates[key.strip()] = value.strip()
        return type(self).model_validate({**self.model_dump(), **updates})


PRESETS = {
    "ours": SwarmConfig(),
    # Lucy runs notify delivery with a free channel_read, which needs her patch. Here both presets use auto delivery,
    # which places every unread post in the model input at the start of the agent's turn.
    "lucy": SwarmConfig(delivery="auto", channel_list=False, pooled_steps=False, one_call_per_turn=True,
                                 planning_turns=2, turn_header=True, awareness_probe=True, steps_per_agent=50),
}


# -------------------------------------------------------------------- prompt texts that depend on a setting

def submit_line(team: bool, shutdown: bool) -> str:
    """The SWE prompt's submit bullet with its leading newline, or "" when agents cannot shut down."""
    if not shutdown:
        return ""
    return "\n" + settings.load_prompts()["swe"]["submit"]["team" if team else "solo"]


def channel_tools_text(channel_list: bool, read_tool: bool = True) -> str:
    """The channel tool names a team prompt lists."""
    names = ["channel_post"] + (["channel_read"] if read_tool else []) + (["channel_list"] if channel_list else [])
    return ", ".join(names)


def read_sentence(board: str, read_tool: bool) -> str:
    """The SWE team prompt's sentence about channel_read, with its leading space, or "" without the tool."""
    return settings.load_prompts()["swe"]["read_sentence"].format(board=board) if read_tool else ""


def board_read_clause(board: str, read_tool: bool) -> str:
    """The board prompt's clause about channel_read, with its leading space, or "" without the tool."""
    return settings.load_prompts()["board_read"].format(board=board) if read_tool else ""
