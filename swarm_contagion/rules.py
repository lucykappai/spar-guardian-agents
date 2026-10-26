"""Lucy's turn rules on ORBIT's public API, with no patch to ORBIT.

Lucy's harness (submodules/lucy/experiments/orbit_cheatbench, c410183) patches ORBIT's turn loop (orbit.patch). This
module reproduces her rules with three public mechanisms:

1. A written plan (ExecutionConfig.plan, built by `schedule`): planning steps where every agent acts at once on a
   snapshot of the board, then one single-agent step per agent per round, in declaration or seeded random order.
   A regulator or disregulator holds steps only in the rounds of its role's window: it enters the swarm at its
   start round and leaves after its end round. An agent whose window starts after round 1 skips planning.
2. A trusted callback after every activation (ScheduleConfig.after_activation = `after_activation`). It counts
   turns, marks why an agent stopped (submitted, cap or context) and amends the pending plan: a stopped agent's
   steps are removed, and with pooling they go to the agents still working.
3. A model filter on every agent (`TurnRulesFilter`, carried by the passive `TurnRulesDefense`). It changes what the
   provider receives and never the durable conversation:
   - planning turns offer only channel_post and channel_read to a team agent and no tool to a lone agent;
   - channel_list and channel_read are withheld when the config turns them off, and the notice's pointer to
     channel_list is replaced;
   - each turn opens with a planning prompt or, with turn_header, the turns left, on its first generation only;
   - when the input would end with a model turn, the continue prompt is appended (Gemini rejects such requests);
   - under one_call_per_turn it counts the calls ORBIT's quantum "legacy" drops.

Two gaps against Lucy's patch stay, and are documented in the config: under one_call_per_turn a working turn keeps
generating until the model calls a tool (her patch ends a turn after max_text_replies text-only replies), and an
agent ORBIT skips at its max_messages limit is never reported to after_activation, so its steps are neither pooled
nor removed (rare: AgentSpec.max_messages is the sample's message_limit plus ten).

Messages the filter adds are tagged with TURN_RULES_TAG. Inspect writes the ModelEvent before ORBIT's model filters
run (FilteredModelAPI sits under Inspect's Model.generate), so they are absent from ModelEvent.input. Each one is
therefore also written as an info event {TURN_RULES_TAG: {"kind", "activation", "agent", "content"}} in the sample's
transcript, which the transcript builder reads.

Per-sample facts live in the store as TurnState, set up by the `rules_setup` solver and saved in the eval log.
"""
from __future__ import annotations

import random
from typing import Any

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, store_as
from orbit.configs.defense import DefenseConfig
from orbit.configs.scheduling import ScheduleConfig
from orbit.defenses.base import DefenseBase, DefenseVerdict
from orbit.defenses.model_filter import ModelFilter
from orbit.defenses.registry import register_defense
from pydantic import Field

from configuration.settings import load_prompts
from configuration.swarm_contagion_config import SwarmContagionConfig, Window
from swarm_contagion.comm_topology import BOARD_ID

TURN_RULES_TYPE = "swarm_contagion.turn_rules.v1"
TURN_RULES_NAME = "turn_rules"
TURN_RULES_TAG = "swarm_contagion_turn_rules"
DELIVERY_TAG = "orbit_communication_delivery"        # orbit/communication/delivery.py _DELIVERY_TAG
ACTOR = "swarm_contagion.rules"
PLAN_PREFIX = "plan"
PLANNING_TOOLS = frozenset({"channel_post", "channel_read"})
# A target that names no agent builds no DefenseBoundary, so the filter runs without a check event per message.
FILTER_ONLY_TARGET = "__model_filter_only__"


class TurnState(StoreModel):
    """Per-sample turn bookkeeping, saved in the eval log. Dicts are keyed by agent name."""

    agents: list[str] = Field(default_factory=list)
    planning_turns: int = 0
    cap: int = 0
    pooled: bool = True
    turn_order: str = "round_robin"
    seed: int = 0
    rounds: int = 0                                         # rounds written so far, pooled rounds included
    windows: dict[str, list] = Field(default_factory=dict)      # [start, end] rounds per agent; end None is open
    planning: dict[str, int] = Field(default_factory=dict)      # planning turns per agent
    activations: dict[str, int] = Field(default_factory=dict)   # completed activations, planning included
    used: dict[str, int] = Field(default_factory=dict)          # working turns used
    ended: dict[str, str] = Field(default_factory=dict)         # submitted | cap | context
    opened: dict[str, int] = Field(default_factory=dict)        # the activation whose opening message was sent
    extra_calls: dict[str, int] = Field(default_factory=dict)   # calls dropped by quantum legacy
    timeline: list[dict] = Field(default_factory=list)          # one entry per activation


# ------------------------------------------------------------------------------------------------ the plan

def round_order(agents: list[str], turn_order: str, seed: int, round_number: int) -> list[str]:
    """The agents of one round: declaration order, or a permutation seeded by (seed, round)."""
    order = list(agents)
    if turn_order == "random":
        random.Random(f"{seed}/{round_number}").shuffle(order)
    return order


def windows(config: SwarmContagionConfig, agents: list[str], roles: dict[str, str] | None = None) -> dict[str, Window]:
    """Each agent's window of rounds, from its role. Without roles every agent holds every round."""
    return {agent: config.window((roles or {}).get(agent, "worker")) for agent in agents}


def planners(config: SwarmContagionConfig, agents: list[str], roles: dict[str, str] | None = None) -> list[str]:
    """The agents that take the planning turns: those whose window opens at round 1."""
    if not config.planning_turns:
        return []
    return [agent for agent, window in windows(config, agents, roles).items() if window.start == 1]


def schedule(config: SwarmContagionConfig, agents: list[str], roles: dict[str, str] | None = None) -> ScheduleConfig:
    """Planning steps, then steps_per_agent rounds of one single-agent step per agent whose window holds the round."""
    planning = planners(config, agents, roles)
    steps: list[dict[str, Any]] = [
        {"id": f"{PLAN_PREFIX}{i}", "agents": planning, "concurrent": True, "visibility": "snapshot",
         "quantum": "model_step"}
        for i in range(1, config.planning_turns + 1) if planning
    ]
    held = windows(config, agents, roles)
    for round_number in range(1, config.steps_per_agent + 1):
        steps += [{"id": f"r{round_number}-{agent}", "agents": [agent]}
                  for agent in round_order(agents, config.turn_order, config.seed, round_number)
                  if held[agent].contains(round_number)]
    return ScheduleConfig(steps=steps, after_activation=f"{__name__}.after_activation")


def initial_state(config: SwarmContagionConfig, agents: list[str], roles: dict[str, str] | None = None) -> dict:
    """The TurnState fields a sample starts with."""
    planning = set(planners(config, agents, roles))
    return {"agents": list(agents), "planning_turns": config.planning_turns, "cap": config.steps_per_agent,
            "pooled": config.pooled_steps, "turn_order": config.turn_order, "seed": config.seed,
            "rounds": config.steps_per_agent,
            "windows": {a: [w.start, w.end] for a, w in windows(config, agents, roles).items()},
            "planning": {a: config.planning_turns if a in planning else 0 for a in agents}}


@solver
def rules_setup(config: SwarmContagionConfig | dict, agents: list[str], roles: dict[str, str] | None = None) -> Solver:
    """Task setup: write the initial TurnState into the sample's store."""
    rules = SwarmContagionConfig.model_validate(config)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        st = store_as(TurnState)
        for key, value in initial_state(rules, agents, roles).items():
            setattr(st, key, value)
        return state

    return solve


def planning_turns_of(st: TurnState, agent: str) -> int:
    """The agent's planning turns: none when its window opens after round 1."""
    return st.planning.get(agent, st.planning_turns)


def _window(st: TurnState, agent: str) -> Window:
    start, end = st.windows.get(agent) or [1, None]
    return Window(start=start, end=end)


def turns_left(agent: str) -> tuple[int, int]:
    """(mine, team): working turns left, counting the turn under way. Call it during the agent's working turn.

    Read from ORBIT's live plan (orbit/execution/plans.py current_schedule), which run_turn installs for the whole
    activation (orbit/execution/agent_scheduler.py run_turn, schedule_runtime_scope). The running step is no longer
    pending, so one is added for it.
    """
    from orbit.execution.plans import current_schedule

    working = [step for step in current_schedule().pending_steps if not step.id.startswith(PLAN_PREFIX)]
    mine = sum(1 for step in working if agent in step.agents)
    team = sum(len(step.agents) for step in working)
    return mine + 1, team + 1


# ------------------------------------------------------------------------------------------------ after each activation

def _out_of_context(agent: str) -> bool:
    """Whether the agent's last generation stopped at the context limit.

    The live AgentState is reachable through ORBIT's session registry, which the orchestrator installs for the whole
    sample (orbit/solvers/orchestrator.py solve, session_registry_scope).
    """
    from orbit.execution.sessions import current_session_registry

    try:
        output = current_session_registry().scheduled(agent).state.output
    except (RuntimeError, KeyError):
        return False
    return output is not None and output.stop_reason == "model_length"


def _stop_reason(st: TurnState, agent: str, submitted: bool, used: int) -> str | None:
    if submitted:
        return "submitted"
    if _out_of_context(agent):
        return "context"
    if not st.pooled and used >= st.cap:
        return "cap"
    return None


def _pooled_steps(st: TurnState, active: list[str], count: int, revision: int) -> list[dict]:
    """count new single-agent steps over the active agents, in new rounds after the last one.

    A step goes only to an agent whose window holds its round. When no agent is eligible the step is dropped.
    """
    steps: list[dict] = []
    while len(steps) < count and active:
        st.rounds += 1
        eligible = [agent for agent in active if _window(st, agent).contains(st.rounds)]
        if not eligible:
            break
        for agent in round_order(eligible, st.turn_order, st.seed, st.rounds):
            if len(steps) == count:
                break
            steps.append({"id": f"r{st.rounds}-{agent}-v{revision}", "agents": [agent]})
    return steps


def after_activation(runtime, result, global_turn: int) -> None:
    """ORBIT calls this after every activation of the plan (ScheduleConfig.after_activation).

    Counts the turn, records why the agent stopped, and amends the pending plan once any agent has stopped: its
    steps are removed, a planning step keeps its active agents, and with pooling the removed working steps go to the
    active agents in new rounds at the end. The timeline marks an agent's first activation ("entered") and the
    activation after which its window holds no more steps ("left").
    """
    st = store_as(TurnState)
    agent = result.agent_name
    activation = st.activations.get(agent, 0) + 1
    working = activation > planning_turns_of(st, agent)
    used = st.used.get(agent, 0) + (1 if working else 0)
    st.activations = {**st.activations, agent: activation}
    st.used = {**st.used, agent: used}
    reason = _stop_reason(st, agent, result.submitted, used)
    if reason and agent not in st.ended:
        st.ended = {**st.ended, agent: reason}
    pending = list(runtime.pending_steps)
    left = _window(st, agent).end is not None and not any(agent in step.agents for step in pending)
    st.timeline = [*st.timeline, {"turn": global_turn, "agent": agent, "activation": activation, "working": working,
                                  "used": used, "tools": [call["tool"] for call in result.tool_calls],
                                  "ended": st.ended.get(agent), "entered": activation == 1, "left": left}]

    active = [name for name in st.agents if name not in st.ended]
    keep, dropped = [], 0
    for step in pending:
        agents = [name for name in step.agents if name in active]
        if len(agents) == len(step.agents):
            keep.append(step.model_dump(mode="json"))
        elif agents:
            keep.append(step.model_copy(update={"agents": tuple(agents)}).model_dump(mode="json"))
        elif not step.id.startswith(PLAN_PREFIX):
            dropped += 1
    if len(keep) == len(pending) and all(k["agents"] == list(s.agents) for k, s in zip(keep, pending)):
        return
    if st.pooled and dropped:
        keep += _pooled_steps(st, active, dropped, runtime.revision)
    runtime.amend_pending(keep, actor=ACTOR, reason=f"{agent}: {st.ended.get(agent, 'stopped')}",
                          expected_revision=runtime.revision)


# ------------------------------------------------------------------------------------------------ the model filter

class TurnRulesFilter(ModelFilter):
    """Shapes each provider request to the turn rules; see the module docstring."""

    name = TURN_RULES_NAME
    priority = 90       # after prompt filters (10); ORBIT's delivery notice is added outside the filter chain

    def __init__(self, config: SwarmContagionConfig) -> None:
        self.config = config
        self.prompts = load_prompts()

    def _state(self, agent: str) -> tuple[TurnState, int, bool, bool]:
        st = store_as(TurnState)
        n = st.activations.get(agent, 0)
        team = len(st.agents) > 1 if st.agents else self.config.board
        planning = st.planning.get(agent, self.config.planning_turns) if st.agents else self.config.planning_turns
        return st, n, n < planning, team

    def _withheld(self) -> frozenset:
        """The channel tools the config turns off."""
        return frozenset({name for name, on in (("channel_list", self.config.channel_list),
                                                ("channel_read", self.config.read_tool)) if not on})

    def _planning_tools(self, team: bool) -> frozenset:
        return (PLANNING_TOOLS - self._withheld()) if team else frozenset()

    def _tools(self, tools: list, planning: bool, team: bool) -> list:
        allowed = self._planning_tools(team) if planning else None
        withheld = self._withheld()
        return [t for t in tools if t.name not in withheld and (allowed is None or t.name in allowed)]

    def _notices(self, messages: list) -> list:
        """Point ORBIT's unread notice at channel_read only, when channel_list is withheld."""
        if self.config.channel_list:
            return messages
        out = []
        for m in messages:
            if m.role == "system" and (m.metadata or {}).get(DELIVERY_TAG) and isinstance(m.content, str) \
                    and "channel_list" in m.content:
                head = m.content.rsplit("\n", 1)[0]
                m = m.model_copy(update={"content": f"{head}\n{self.prompts['notice_instructions']}"})
            out.append(m)
        return out

    def _continue(self, messages: list) -> tuple[str, str] | None:
        """The continue prompt when the input would end with a model turn, which Gemini rejects."""
        last = next((m for m in reversed(messages) if m.role != "system"), None)
        if last is not None and last.role == "assistant":
            return "continue", self.prompts["continue"]
        return None

    def _opening(self, st: TurnState, agent: str, n: int, planning: bool, team: bool,
                 messages: list) -> tuple[str, str] | None:
        """The message that opens the activation, sent on its first generation only.

        Under quantum legacy an activation keeps generating until the model calls a tool (ORBIT's turn loop), so a
        later generation in the same activation gets the continue prompt at most, never a second header.
        """
        side = "team" if team else "solo"
        if st.opened.get(agent) == n + 1:
            return self._continue(messages)
        st.opened = {**st.opened, agent: n + 1}
        if planning:
            return "planning", self.prompts["planning"][side][n].format(board=BOARD_ID)
        if self.config.turn_header:
            mine, team_left = turns_left(agent)
            key = "turn_header_pooled" if self.config.pooled_steps else "turn_header"
            text = self.prompts[key][side].format(mine=mine, cap=self.config.steps_per_agent, team_left=team_left)
            return "header", text
        return self._continue(messages)

    async def on_input(self, messages, tools, tool_choice, config, ctx):
        agent = ctx.agent_name
        st, n, planning, team = self._state(agent)
        tools = self._tools(list(tools), planning, team)
        messages = self._notices(list(messages))
        opening = self._opening(st, agent, n, planning, team, messages)
        if opening is not None:
            kind, text = opening
            tag = {"kind": kind, "activation": n + 1}
            messages = [*messages, ChatMessageUser(content=text, metadata={TURN_RULES_TAG: tag})]
            transcript().info({TURN_RULES_TAG: {**tag, "agent": agent, "content": text}})
        return messages, tools, tool_choice, config

    async def on_output(self, output, ctx):
        if not output.choices:
            return output
        agent = ctx.agent_name
        st, n, planning, team = self._state(agent)
        calls = output.choices[0].message.tool_calls or []
        if planning:
            return self._planning_calls_only(output, calls, team, agent, n)
        if self.config.one_call_per_turn and len(calls) > 1:
            st.extra_calls = {**st.extra_calls, agent: st.extra_calls.get(agent, 0) + len(calls) - 1}
            transcript().info({TURN_RULES_TAG: {"kind": "extra_calls", "activation": n + 1, "agent": agent,
                                                "dropped": [call.function for call in calls[1:]]}})
        return output

    def _planning_calls_only(self, output, calls: list, team: bool, agent: str, n: int):
        """Remove calls to tools a planning turn does not offer, so they cannot run."""
        allowed = self._planning_tools(team)
        kept = [call for call in calls if call.function in allowed]
        if len(kept) == len(calls):
            return output
        output = output.model_copy(deep=True)
        choice = output.choices[0]
        choice.message = choice.message.model_copy(update={"tool_calls": kept or None})
        if not kept:
            choice.stop_reason = "stop"
        transcript().info({TURN_RULES_TAG: {"kind": "planning_call_removed", "activation": n + 1, "agent": agent,
                                            "removed": [c.function for c in calls if c.function not in allowed]}})
        return output


class TurnRulesDefense(DefenseBase):
    """Passive defense that carries TurnRulesFilter. It allows every message and tool call."""

    def __init__(self, config: DefenseConfig) -> None:
        super().__init__(config)
        self.rules = SwarmContagionConfig.model_validate(config.config)

    def get_model_filters(self) -> list[ModelFilter]:
        return [TurnRulesFilter(self.rules)]

    async def apply_pre_deployment(self, agents, state) -> None:
        return None

    async def monitor_message(self, message, agent_name, turn) -> DefenseVerdict:
        return DefenseVerdict(allow=True, reason="passive_turn_rules")

    async def on_tool_call(self, tool_name, tool_args, agent_name) -> DefenseVerdict:
        return DefenseVerdict(allow=True, reason="passive_turn_rules")


try:
    register_defense(TURN_RULES_TYPE, TurnRulesDefense)
except ValueError:
    pass            # already registered in this process


def turn_rules_defense(config: SwarmContagionConfig) -> DefenseConfig:
    """The DefenseConfig that installs the filter on every agent.

    A target naming no agent builds no DefenseBoundary, while ORBIT collects the model filters of every defense for
    every agent (orbit/solvers/orchestrator.py _build_attacks_and_defenses).
    """
    return DefenseConfig(name=TURN_RULES_NAME, defense_type=TURN_RULES_TYPE, target_agents=[FILTER_ONLY_TARGET],
                         config=config.model_dump(mode="json"))
