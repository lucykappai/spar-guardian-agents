"""The pull board: board_status, board_read, board_post and team_submit, as Inspect tools over ORBIT's store.

Nothing is pushed to an agent under the pull board. It checks, reads and posts through three tools, which tasks.py
binds to every agent of a board with board_tools on (ORBIT tool_bindings, one ToolFactory per name, the way SWE binds
shell and run_python). ORBIT's channel_post, channel_read and channel_list stay attached, and rules.TurnRulesFilter
withholds them from the provider.

The tools keep no store of their own. They read and write ORBIT's communication store (orbit/communication/):
    board_post    send_message, as channel_post does, so exports, the transcript and the browser see a normal post
    board_read    read_messages for the selection, record_read for the read event and its receipt, which calls
                  delivery.acknowledge_read and advances the agent's read cursor to the newest post returned
    board_status  read_messages for the counts and delivery.unread_messages for "new": posts by others above the
                  agent's read cursor
read_messages honours a concurrent round's snapshot, so an agent sees the board as of its step's start. A post's
number (#17) is its position on the board, oldest first; it does not change when later posts arrive.

team_submit is bound to neutral agents only (not to a regulator or a disregulator), and only when the mode has a team
to submit for (SwarmConfig.has_team_submit). The first call records rules.TeamSubmission and an info event tagged
rules.TEAM_SUBMIT_TAG; rules.after_activation then lets the round finish and drops every later step.

Each tool also counts the action in rules.TurnState (checks, reads, posts, read_senders, discovery_round).
Descriptions come from src/configuration/prompts.yaml `tools`, argument descriptions from `tool_args`, and the result
lines from `board_output`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from inspect_ai.log import transcript
from inspect_ai.tool import ToolDef, ToolError
from inspect_ai.util import store_as
from orbit.communication.delivery import unread_messages
from orbit.communication.errors import ChannelCapacityError
from orbit.communication.runtime import read_messages, record_read, send_message
from orbit.communication.state import CommunicationState
from orbit.tools import ToolFactory

from src.configuration.settings import BOARD_ID, load_prompts
from src.configuration.swarm_config import BOARD_TOOLS, SwarmConfig
from src.swarm.rules import (
    TEAM_SUBMIT,
    TEAM_SUBMIT_TAG,
    TeamSubmission,
    TurnState,
    current_round,
    record_board_action,
)

__all__ = ["BOARD_TOOLS", "TEAM_SUBMIT", "board_status", "board_read", "board_post", "team_submit",
           "board_bindings", "agent_tool_names"]

SUBMITTED = "Submitted for the team. The run ends after this round."
ALREADY_SUBMITTED = "{agent} already submitted the team's work in round {round}. The run ends after that round."


def _board(agent: str, board: str) -> list:
    """Every post on the board the agent may see, oldest first (ORBIT read_messages, snapshot included)."""
    try:
        return read_messages(board, agent_id=agent)
    except PermissionError as error:
        raise ToolError(str(error)) from None


def _numbers(board: str) -> dict[str, int]:
    """Each post's position on the board, from ORBIT's canonical store."""
    posts = [m for m in store_as(CommunicationState).messages if m.channel_id == board]
    return {m.id: i for i, m in enumerate(posts, 1)}


def _render(message, number: int) -> str:
    return load_prompts()["board_output"]["post"].format(sender=message.sender, sequence=number,
                                                         text=message.content)


def board_status(agent: str, board: str = BOARD_ID) -> ToolDef:
    """Counts of posts by sender, and how many are new since the agent's last read."""
    prompts = load_prompts()
    out = prompts["board_output"]

    async def execute() -> str:
        posts = _board(agent, board)
        new = [m for m in unread_messages(agent) if m.channel_id == board]
        record_board_action(agent, "check")
        lines = []
        for sender in sorted({m.sender for m in posts}):
            total = sum(1 for m in posts if m.sender == sender)
            fresh = sum(1 for m in new if m.sender == sender)
            lines.append(out["sender"].format(sender=sender, total=total, new=fresh))
        lines.append(out["total"].format(total=len(posts), new=len(new)))
        return "\n".join(lines)

    return ToolDef(execute, name="board_status", description=prompts["tools"]["board_status"])


def board_read(agent: str, read_page: int, board: str = BOARD_ID) -> ToolDef:
    """The selected posts, oldest first, at most read_page in all; marks them read."""
    prompts = load_prompts()
    args = prompts["tool_args"]["board_read"]

    async def execute(latest: int | None = None, from_agent: str | None = None,
                      latest_from: int | None = None) -> str:
        for name, value in (("latest", latest), ("latest_from", latest_from)):
            if value is not None and value < 1:
                raise ToolError(f"{name} must be at least 1")
        if latest_from is not None and from_agent is None:
            raise ToolError("latest_from needs from_agent")
        posts = _board(agent, board)
        if latest is None and from_agent is None:
            latest = read_page
        chosen: dict[str, object] = {}
        if latest is not None:
            chosen.update({m.id: m for m in posts[-latest:]})
        if from_agent is not None:
            theirs = [m for m in posts if m.sender == from_agent]
            chosen.update({m.id: m for m in theirs[-(latest_from or read_page):]})
        selected = sorted(chosen.values(), key=lambda m: m.sequence)[-read_page:]
        numbers = _numbers(board)
        rendered = {m.id: _render(m, numbers[m.id]) for m in selected}
        if rendered:
            record_read(board, agent_id=agent, rendered_messages=rendered)
        record_board_action(agent, "read", [m.sender for m in selected])
        return "\n".join(rendered.values())

    return ToolDef(execute, name="board_read", description=prompts["tools"]["board_read"],
                   parameters={"latest": args["latest"], "from_agent": args["from_agent"],
                               "latest_from": args["latest_from"]})


def board_post(agent: str, board: str = BOARD_ID) -> ToolDef:
    """Post through ORBIT's send_message, as channel_post does."""
    prompts = load_prompts()

    async def execute(message: str) -> str:
        if not message:
            raise ToolError("A nonempty message is required")
        try:
            sent = send_message(board, agent_id=agent, content=message)
        except (ChannelCapacityError, PermissionError) as error:
            raise ToolError(str(error)) from None
        record_board_action(agent, "post")
        return prompts["board_output"]["posted"].format(sequence=_numbers(board)[sent.id])

    return ToolDef(execute, name="board_post", description=prompts["tools"]["board_post"],
                   parameters={"message": prompts["tool_args"]["board_post"]["message"]})


def team_submit(agent: str) -> ToolDef:
    """Record the team's submission; the run ends after the round under way (rules.after_activation)."""
    prompts = load_prompts()

    async def execute(answer: str = "") -> str:
        submission = store_as(TeamSubmission)
        round_number = current_round()
        if submission.agent is not None:
            submission.refused = [*submission.refused, {"agent": agent, "round": round_number}]
            return ALREADY_SUBMITTED.format(agent=submission.agent, round=submission.round)
        submission.agent, submission.round, submission.answer = agent, round_number, answer
        submission.time = datetime.now(timezone.utc).isoformat()
        record = {"agent": agent, "round": round_number, "answer": answer, "time": submission.time}
        store_as(TurnState).team_submission = record
        transcript().info({TEAM_SUBMIT_TAG: record})
        return SUBMITTED

    return ToolDef(execute, name=TEAM_SUBMIT, description=prompts["tools"]["team_submit"],
                   parameters={"answer": prompts["tool_args"]["team_submit"]["answer"]})


# ------------------------------------------------------------------------------------------------ binding

def board_bindings(config: SwarmConfig, board: str = BOARD_ID) -> dict[str, ToolFactory]:
    """ORBIT tool_bindings for the board tools and team_submit, built per agent. Empty without a pull board.

    A binding is built only for the agents whose AgentSpec.tools names it (agent_tool_names)."""
    if not config.board or not config.board_tools:
        return {}
    return {"board_status": ToolFactory(lambda ctx: board_status(ctx.agent_name, board)),
            "board_read": ToolFactory(lambda ctx: board_read(ctx.agent_name, config.read_page, board)),
            "board_post": ToolFactory(lambda ctx: board_post(ctx.agent_name, board)),
            TEAM_SUBMIT: ToolFactory(lambda ctx: team_submit(ctx.agent_name))}


def agent_tool_names(config: SwarmConfig, names: list[str], domain: str,
                     roles: dict[str, str] | None = None) -> dict[str, list[str]]:
    """The board tool names each agent gets: the three board tools on a pull board, and team_submit for a neutral
    agent (no role, or the worker role) when the mode has a team to submit for."""
    if not config.board or not config.board_tools:
        return {n: [] for n in names}
    submit = config.has_team_submit(domain)
    return {n: [*BOARD_TOOLS, *([TEAM_SUBMIT] if submit and (roles or {}).get(n, "worker") == "worker" else [])]
            for n in names}
