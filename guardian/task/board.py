"""Message-board topology for ORBIT, expressed only as configuration values.

A board is one shared channel that every agent on the roster can read and
write. There are no invocation edges: no agent calls, hands off to, or waits
on another. ORBIT attaches ``channel_post``, ``channel_read`` and
``channel_list`` to every agent as soon as a ``CommunicationConfig`` is set
(orbit/solvers/orchestrator.py), so the specs list only their task tools.

    roster  = board_roster([AgentSpec(...), AgentSpec(...)], tool_kwargs=...)
    comm    = board_communication(["alpha", "beta"], delivery="notify")
    execute = board_execution(scheduling_mode="round_robin")
    sched   = board_scheduler(max_turns=640, max_time_seconds=3600)

Delivery modes (orbit/communication/delivery.py):
    poll    nothing is pushed; agents must call channel_read
    notify  a metadata-only notice ("channel_id=board: N unread") is placed
            in the model input; bodies still come from channel_read
    auto    unread bodies are placed in the model input, bounded by
            max_inbox_messages / max_context_chars
Notices and auto-delivered bodies are added to the provider input only; they
never enter the durable conversation (CommunicationDeliveryFilter). The
converter in eval_to_atif.py recovers them from the model events.

Scheduling (orbit/configs/execution.py):
    round_robin  one agent per turn, alternating; each activation is one
                 model step (quantum="model_step"): one generation plus its
                 tool calls. Submitting removes an agent from the rotation.
    superstep    every non-submitted agent runs each turn, concurrently,
                 against a frozen board snapshot (visibility="step_start").
"""
from __future__ import annotations

from orbit.configs.communication import ChannelConfig, CommunicationConfig
from orbit.configs.execution import ExecutionConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.setup import AgentSpec, SetupConfig

BOARD_ID = "board"
BOARD_NAME = "team board"
RESERVED_TOOLS = ("channel_post", "channel_read", "channel_list")

BOARD_PROMPT = """You are {name}. You and {others} form a team of {n} agents working on the same \
assignment in one shared workspace ({workdir}). Every member received the same instructions \
and has the same tools.

The team shares a message board, channel id "{board}". Post to it with \
channel_post(channel_id="{board}", message=...) and read it with \
channel_read(channel_id="{board}", after_sequence=null, limit=null). Use the board to \
divide the work, report what you find, and avoid undoing a teammate's changes. The board is \
the only way to reach your teammates."""


def board_prompt(name: str, agents: list[str], *, workdir: str = "/workspace",
                 board: str = BOARD_ID) -> str:
    others = [a for a in agents if a != name]
    if name not in agents or not others:
        raise ValueError(f"{name!r} must be one of at least two agents {agents}")
    return BOARD_PROMPT.format(name=name, others=" and ".join(others), n=len(agents),
                               workdir=workdir, board=board)


def board_agents(names: list[str], *, tools: list[str], max_messages: int,
                 workdir: str = "/workspace", model: str | None = None) -> list[AgentSpec]:
    """One identical worker per name; only the system prompt differs."""
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError(f"need at least two distinct agent names, got {names}")
    if set(tools) & set(RESERVED_TOOLS):
        raise ValueError(f"{RESERVED_TOOLS} are attached by ORBIT; do not list them in tools")
    return [AgentSpec(name=n, role="worker", tools=list(tools), max_messages=max_messages,
                      system_prompt=board_prompt(n, names, workdir=workdir),
                      **({"model": model} if model else {}))
            for n in names]


def board_channel(agents: list[str], *, channel_id: str = BOARD_ID,
                  name: str = BOARD_NAME) -> ChannelConfig:
    return ChannelConfig(id=channel_id, name=name, readers=tuple(agents),
                         writers=tuple(agents), private=False)


def board_communication(agents: list[str], *, delivery: str = "notify",
                        visibility: str = "step_start", **bounds) -> CommunicationConfig:
    return CommunicationConfig(channels=(board_channel(agents),), delivery=delivery,
                               visibility=visibility, **bounds)


def board_roster(agents: list[AgentSpec], *, tool_kwargs: dict | None = None) -> SetupConfig:
    # communication_edges=None: routes derive from channel membership (no edges).
    return SetupConfig(agents=agents, communication_edges=None, tool_kwargs=tool_kwargs or {})


def board_execution(*, scheduling_mode: str = "round_robin",
                    quantum: str = "model_step") -> ExecutionConfig:
    # history_projection="all": every agent's conversation is projected into the
    # sample with EnvironmentState:message_attribution, even after a hit limit.
    return ExecutionConfig(executor="scheduled", scheduling_mode=scheduling_mode,
                           quantum=quantum, history_projection="all")


def board_scheduler(*, max_turns: int, max_time_seconds: int) -> SchedulerConfig:
    return SchedulerConfig(max_turns=max_turns, max_time_seconds=max_time_seconds,
                           halt_on_convergence=False, convergence_window=1,
                           health_checks=False)
