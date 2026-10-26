"""What a benchmark is: the protocol every CheatBench task we run implements.

A benchmark gives one Inspect sample, the tools its agents use, the setup that runs before the agents,
the scorers that grade what they left behind, and the ways our harness departs from CheatBench's own runner.
The ORBIT configuration (roster, board, schedule) lives in swarm_contagion/, never here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver
from orbit.tools import ToolFactory

# ORBIT appends its submit instructions to the system prompt only when the agent has the submit tool.
SUBMIT_PROMPT_DEVIATION = "orbit_assistant_and_submit_prompt_appended_to_system_prompt"
NO_SUBMIT_DEVIATIONS = [
    "orbit_assistant_prompt_appended_to_system_prompt",
    "agents_have_no_submit_tool_and_run_until_their_turn_cap_or_the_time_limit",
]


@dataclass(frozen=True)
class AgentTools:
    """The tools every agent gets, in the three shapes ORBIT takes them."""

    names: list[str]
    """AgentSpec.tools for every agent."""
    bindings: dict[str, ToolFactory] = field(default_factory=dict)
    """ORBIT tool_bindings: tools built per agent. Empty for ORBIT's built-in tools."""
    tool_kwargs: dict[str, dict] = field(default_factory=dict)
    """SetupConfig.tool_kwargs for ORBIT's built-in bash and python tools."""


class Benchmark(ABC):
    """One CheatBench task as an Inspect sample plus its tools, setup and scorers."""

    name: str
    """CheatBench's task name, such as "openmath_agent" or "software_engineering"."""
    task_root: Path
    """The CheatBench task folder that holds judge_schema.py."""
    workdir: str
    """The agents' working directory in the sandbox."""

    @abstractmethod
    def time_limit_s(self) -> int:
        """The task's own time limit in seconds."""

    @abstractmethod
    def sample(self, agents: list[str], *, instruction_suffix: str = "") -> Sample:
        """One sample for this roster.

        sample.input is the instruction plus the suffix. metadata["cheatbench"] carries the task, the variant,
        the instruction, the prompt file and the timeout. It may also carry "first_message" {agent: text} when
        agents start from different messages, and "judge_prompt" {agent: text} when the judge should see more
        than the instruction as an agent's rules.
        """

    @abstractmethod
    def tools(self, agents: list[str]) -> AgentTools:
        """The tools for this roster."""

    @abstractmethod
    def setup(self) -> Solver | None:
        """The Inspect task_setup solver that runs before the agents, or None."""

    @abstractmethod
    def scorers(self) -> list[Scorer]:
        """The scorers that grade the outcome. The judge scorer is added by the task factory."""

    @abstractmethod
    def deviations(self, n_agents: int) -> list[str]:
        """How a run with this many agents departs from CheatBench's own runner, as short labels."""

    def run_deviations(self, n_agents: int, *, shutdown: bool = True) -> list[str]:
        """deviations() for a run whose agents do or do not have ORBIT's submit tool.

        Without the submit tool ORBIT appends no submit instructions, and an agent cannot end its own episode.
        """
        labels = self.deviations(n_agents)
        if shutdown or SUBMIT_PROMPT_DEVIATION not in labels:
            return labels
        at = labels.index(SUBMIT_PROMPT_DEVIATION)
        return labels[:at] + NO_SUBMIT_DEVIATIONS + labels[at + 1:]
