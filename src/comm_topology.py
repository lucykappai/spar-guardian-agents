"""The message board as ORBIT configuration values.

A board is one shared channel that every agent on the roster can read and write. There are no invocation edges: no
agent calls, hands off to, or waits on another. ORBIT attaches channel_post, channel_read and channel_list to every
agent as soon as a CommunicationConfig is set (orbit/solvers/orchestrator.py _build_executor), so the specs list only
their task tools. The board prompt comes from src/configuration/prompts.yaml (`board`).

    prompt  = board_prompt("a01", names, workdir="/workspace")
    comm    = board_communication(names, delivery="auto", max_message_chars=4000)
    execute = board_execution(quantum=config.quantum, plan=rules.schedule(config, names))
    sched   = board_scheduler(max_turns=config.max_turns + 5, max_time_seconds=3600)

Delivery modes (orbit/communication/delivery.py):
    poll    nothing is pushed; agents must call channel_read
    notify  a metadata-only notice ("channel_id=board: N unread") is placed in the model input
    auto    unread bodies are placed in the model input, bounded by max_inbox_messages and max_context_chars
Notices and bodies reach the provider input only (CommunicationDeliveryModel). The transcript builder in
src/benchmark/transcript.py recovers them from the model events.

Turns follow a written plan (src/rules.py schedule). ORBIT's scheduling_mode does not apply once a plan
is set.
"""
from __future__ import annotations

from orbit.configs.communication import ChannelConfig, CommunicationConfig
from orbit.configs.execution import AgentGroup, ExecutionConfig
from orbit.configs.scheduler import SchedulerConfig
from orbit.configs.scheduling import ScheduleConfig

from src.configuration.settings import BOARD_ID, load_prompts
from src.configuration.swarm_config import board_read_clause

BOARD_NAME = "team board"


def board_prompt(name: str, agents: list[str], *, workdir: str = "/workspace", board: str = BOARD_ID,
                 read_tool: bool = True) -> str:
    """The board system prompt for one agent of a team of two or more. Without read_tool it names no channel_read."""
    others = [a for a in agents if a != name]
    if name not in agents or not others:
        raise ValueError(f"{name!r} must be one of at least two agents {agents}")
    return load_prompts()["board"].format(name=name, others=" and ".join(others), n=len(agents),
                                          workdir=workdir, board=board, read=board_read_clause(board, read_tool))


def board_channel(agents: list[str], *, channel_id: str = BOARD_ID, name: str = BOARD_NAME) -> ChannelConfig:
    return ChannelConfig(id=channel_id, name=name, readers=tuple(agents), writers=tuple(agents), private=False)


def board_communication(agents: list[str], *, delivery: str = "notify", visibility: str = "step_start",
                        **bounds) -> CommunicationConfig:
    """The board as ORBIT's CommunicationConfig; bounds pass through (max_message_chars, max_context_chars, ...)."""
    return CommunicationConfig(channels=(board_channel(agents),), delivery=delivery, visibility=visibility, **bounds)


def board_execution(*, quantum: str = "model_step", plan: ScheduleConfig | None = None,
                    agent_groups: list[AgentGroup] | None = None,
                    initial_context: str = "auto") -> ExecutionConfig:
    """Scheduled execution under a written plan.

    history_projection="all" projects every agent's conversation into the sample with
    EnvironmentState:message_attribution, even after a hit limit.
    """
    return ExecutionConfig(executor="scheduled", quantum=quantum, plan=plan, history_projection="all",
                           agent_groups=list(agent_groups or []), initial_context=initial_context)


def board_scheduler(*, max_turns: int, max_time_seconds: int) -> SchedulerConfig:
    return SchedulerConfig(max_turns=max_turns, max_time_seconds=max_time_seconds, halt_on_convergence=False,
                           convergence_window=1, health_checks=False)
