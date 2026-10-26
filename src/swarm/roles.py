"""Hidden roles and instruction levels: workers, regulators and disregulators as peers on one board.

A swarm is the board (src/swarm/comm_topology.py) with N agents whose system prompts differ by role:

    worker        the plain board prompt
    regulator     a peer told to keep the team honest, at the configured regulator level
    disregulator  a peer privately told to get the result by any means, at the configured disregulator level

Regulators are not monitors. They have the same tools and standing as everyone else, and nothing is enforced at a
boundary. Roles hide behind neutral names (a01 ... a10) and are recorded in the sample metadata.

Every model-facing text of a level comes from src/configuration/instruction_levels.yaml: the rule sentence appended to
the task instruction, and the regulator and disregulator texts appended to the system prompt of the agents holding
the role. A level gives one text or one per benchmark, with placeholders filled from the benchmark's `vars`.

The task gains one team-level output when roles are used: swarm_report.txt, whose first line says whether the team as
a whole cheated. Its instruction is src/configuration/prompts.yaml `roles.team_report`; src/benchmark/export.py parses it.
"""
from __future__ import annotations

import random
import re

from src.benchmark.export import CHEATED_LINE, parse_swarm_report
from src.configuration.settings import load_instruction_levels, load_prompts
from src.configuration.swarm_config import SwarmConfig, agent_names

__all__ = ["ROLES", "CHEATED_LINE", "LEVEL_KEYS", "agent_names", "assign_roles", "level_text", "role_text",
           "with_role", "rule_suffix", "team_report_suffix", "parse_swarm_report", "composition_label"]

ROLES = ("worker", "regulator", "disregulator")
# The key instruction_levels.yaml uses for each CheatBench task.
LEVEL_KEYS = {"openmath_agent": "openmath", "software_engineering": "swe"}
# A placeholder that opens a sentence; its value is capitalised there.
SENTENCE_START = re.compile(r"(^|[.!?]\s+)\{(\w+)\}")


def level_text(ladder: str, level: str, benchmark_name: str) -> str | None:
    """The text of one level of a ladder for one benchmark ("openmath" or "swe"), or None for no text."""
    levels = load_instruction_levels()
    entry = levels[ladder][level]
    template = entry["text"] if "text" in entry else entry.get(benchmark_name)
    if template is None:
        return None
    values = dict(levels["vars"][benchmark_name])
    values.update({f"{k}__cap": v[:1].upper() + v[1:] for k, v in list(values.items())})
    return SENTENCE_START.sub(r"\1{\2__cap}", template.strip()).format(**values)


def assign_roles(names: list[str], regulators: int, disregulators: int, seed: int = 0) -> dict[str, str]:
    """Hidden roles by neutral name, from counts; the seed fixes which names get which role."""
    if regulators + disregulators > len(names):
        raise ValueError("more regulators and disregulators than agents")
    order = list(names)
    random.Random(seed).shuffle(order)
    roles = {name: "worker" for name in names}
    for name in order[:regulators]:
        roles[name] = "regulator"
    for name in order[regulators:regulators + disregulators]:
        roles[name] = "disregulator"
    return roles


def role_text(role: str, config: SwarmConfig, benchmark_name: str) -> str | None:
    """The text appended to the system prompt of an agent holding this role, or None."""
    if role not in ROLES:
        raise ValueError(role)
    if role == "worker":
        return None
    return level_text(role, getattr(config, f"{role}_level"), benchmark_name)


def with_role(prompt: str, text: str | None) -> str:
    """The system prompt, then the role text after a blank line."""
    return prompt if not text else f"{prompt}\n\n{text}"


def rule_suffix(config: SwarmConfig, benchmark_name: str) -> str:
    """The rule sentence on its own paragraph after the instruction, or "" at level none."""
    sentence = level_text("rule", config.rule_level, benchmark_name)
    return f"\n{sentence}\n" if sentence else ""


def team_report_suffix() -> str:
    """The team report instruction, appended to the task instruction after a newline."""
    return "\n" + load_prompts()["roles"]["team_report"]


def composition_label(roles: dict[str, str]) -> str:
    counts = {r: sum(1 for v in roles.values() if v == r) for r in ROLES}
    return f"R{counts['regulator']}D{counts['disregulator']}W{counts['worker']}"
