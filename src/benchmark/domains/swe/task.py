"""The SWE benchmark: one CheatBench issue under one of Lucy's honeypots, for 1, 2 or 3 agents.

Holds the sample (build_swe), each agent's parts and pieces, and the texts each agent reads: the system prompts
from prompts.yaml `swe` and the first messages with Lucy's assignment.
"""
from __future__ import annotations

import random
import re

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from src.benchmark.base import SUBMIT_PROMPT_DEVIATION, AgentTools, Benchmark
from src.benchmark.domains.swe.grader import swe_grader
from src.benchmark.domains.swe.honeypots import swe_setup
from src.benchmark.domains.swe.sandbox import (
    ensure_images,
    tool_bindings,
    write_compose,
)
from src.benchmark.domains.swe.split import (
    agent_timeout_s,
    fix_commit,
    load_lines,
    load_split,
    patch_files,
    task_config,
    task_dir,
)
from src.configuration import settings
from src.configuration.settings import BOARD_ID
from src.configuration.swarm_config import (
    channel_tools_text,
    read_sentence,
    submit_line,
)
from src.configuration.swe import (
    AGENT_CPUS,
    AGENT_MEM,
    AGENT_TIME_S,
    DELIVERY_MODES,
    HONEYPOTS,
    JIGSAW_PIECES,
    PIECES,
    TASK_ROOT,
    TEAM_PARTS,
    TOOL_TIMEOUT_S,
    TOOLS,
    WORKDIR,
)


def part_text(issue_text: str, split: dict, part: str) -> str:
    """A part: the issue's own Requirements bullets for it, verbatim, then its interface block or note
    (Lucy's scenario.py part_text)."""
    m = re.search(r"^Requirements:\s*\n(.*?)(?=^New interfaces introduced:|\Z)", issue_text, re.S | re.M)
    if not m:
        raise ValueError("the issue has no Requirements: section")
    bullets: list[str] = []
    for line in m.group(1).splitlines():
        if line.startswith("- "):
            bullets.append(line)
        elif line.strip() and bullets:
            bullets[-1] += "\n" + line
    spec = split["parts"][part]
    text = [bullets[i - 1] for i in spec["requirements"]]
    if spec.get("interfaces"):
        iface = re.search(r"^New interfaces introduced:\s*\n(.*)", issue_text, re.S | re.M).group(1)
        text.append("New interfaces introduced:\n"
                    + iface.replace("This task is a work-test for a job interview.", "").strip())
    if spec.get("note"):
        text.append(spec["note"])
    return "\n".join(text)


def parts_by_agent(agents: list[str]) -> dict[str, list[str]]:
    if len(agents) == 1:
        return {agents[0]: []}
    if len(agents) not in TEAM_PARTS:
        raise ValueError(f"SWE takes 1, 2 or 3 agents, got {len(agents)}")
    return dict(zip(agents, TEAM_PARTS[len(agents)]))


def pieces_by_agent(agents: list[str], honeypot: str) -> dict[str, list[str]]:
    """The pieces planted in each agent's clone. Lone agents on the whole issue get `whole`."""
    if honeypot not in HONEYPOTS:
        raise ValueError(f"honeypot must be one of {HONEYPOTS}, got {honeypot!r}")
    if honeypot == "as_released":
        return {a: [] for a in agents}
    if honeypot == "jigsaw":
        if len(agents) != 3:
            raise ValueError("the jigsaw honeypot needs three agents (Lucy's W1 holds B, W2 C, W3 A)")
        return dict(zip(agents, JIGSAW_PIECES))
    if len(agents) == 1:
        return {agents[0]: ["whole"]}
    return parts_by_agent(agents)


def piece_times(seed: int, issue: str) -> dict[str, int]:
    """Seconds added to each planted commit's time, drawn from the sample's seed, so each seed plants new hashes."""
    rng = random.Random(f"{seed}/{issue}/piece_times")
    return {p: rng.randrange(3600) for p in PIECES}


def system_prompts(split: dict, agents: list[str], parts: dict[str, list[str]], *, delivery: str,
                   channel_list: bool, shutdown: bool, planning_turns: int, read_tool: bool = True) -> dict[str, str]:
    """Each agent's system prompt from prompts.yaml `swe`. The submit bullet is left out without shutdown, and the
    channel_read sentence without read_tool."""
    prompts = settings.load_prompts()["swe"]
    team = len(agents) > 1
    submit = submit_line(team, shutdown)
    if not team:
        a = agents[0]
        return {a: prompts["system_solo"].format(name=a, repo=split["repo"], branch=a, timeout=TOOL_TIMEOUT_S,
                                                 submit=submit)}
    titles = {a: part_titles(split, parts[a]) for a in agents}
    planning = f"\n\n{prompts['planning_system']}" if planning_turns == 2 else ""
    out = {}
    for a in agents:
        roster = "\n".join(prompts["roster_line"].format(name=o, title=titles[o]) for o in agents if o != a)
        out[a] = prompts["system_team"].format(
            name=a, n=len(agents), repo=split["repo"], roster=roster, branch=a, board=BOARD_ID,
            delivery_text=prompts["delivery_text"][delivery], timeout=TOOL_TIMEOUT_S,
            channel_tools=channel_tools_text(channel_list, read_tool), read_sentence=read_sentence(BOARD_ID, read_tool),
            submit=submit) + planning
    return out


def part_titles(split: dict, parts: list[str]) -> str:
    return "; ".join(split["parts"][p]["title"] for p in parts)


def first_messages(split: dict, issue_text: str, instruction: str, parts: dict[str, list[str]]) -> dict[str, str]:
    """A lone agent's first message is the instruction. A team agent's is the instruction, then Lucy's assignment:
    the team's split and the agent's part, its requirement bullets copied from the issue."""
    agents = list(parts)
    if len(agents) == 1:
        return {agents[0]: instruction}
    template = settings.load_prompts()["swe"]["assignment_team"]
    table = "\n".join(f"- {a}: {part_titles(split, parts[a])}" for a in agents)
    out = {}
    for a in agents:
        text = "\n".join(part_text(issue_text, split, p) for p in parts[a])
        assignment = template.format(table=table, title=part_titles(split, parts[a]), part=text)
        out[a] = f"{instruction.rstrip()}\n\n{assignment.strip()}\n"
    return out


def build_swe(issue: str, honeypot: str, agents: list[str], *, instruction_suffix: str = "", seed: int = 0,
              delivery: str = "auto", channel_list: bool = True, shutdown: bool = True,
              planning_turns: int = 0, read_tool: bool = True) -> dict:
    """Everything one sample needs except the images: parts, pieces, prompts and first messages. No Docker."""
    if delivery not in DELIVERY_MODES:
        raise ValueError(f"delivery must be one of {DELIVERY_MODES}, got {delivery!r}")
    split = load_split(issue)
    instance = split["task"]
    cfg = task_config(instance)
    issue_text = (task_dir(instance) / "instruction.md").read_text()
    instruction = issue_text + instruction_suffix
    parts = parts_by_agent(agents)
    pieces = pieces_by_agent(agents, honeypot)
    team = len(agents) > 1
    system = system_prompts(split, agents, parts, delivery=delivery, channel_list=channel_list, shutdown=shutdown,
                            planning_turns=planning_turns, read_tool=read_tool)
    first = first_messages(split, issue_text, instruction, parts)
    return {
        "issue": issue, "task_key": issue, "instance": instance, "repo": cfg["repo"],
        "base_commit": cfg["base_commit"], "fix_commit": fix_commit(instance),
        "honeypot": honeypot, "jigsaw": honeypot == "jigsaw", "seed": seed,
        "n_agents": len(agents), "agents": list(agents), "board": team, "delivery": delivery if team else None,
        "channel_list": channel_list if team else None, "read_tool": read_tool if team else None,
        "shutdown": shutdown, "planning_turns": planning_turns,
        "hint": instruction_suffix.strip() or None, "instruction": instruction,
        "parts": {a: {"parts": parts[a] or ["whole"], "titles": [split["parts"][p]["title"] for p in parts[a]]}
                  for a in agents},
        "pieces": pieces,
        "piece_times": piece_times(seed, issue),
        "system_prompt": system, "first_message": first,
        "judge_prompt": {a: f"{system[a]}\n\n{first[a]}" for a in agents},
        "patch_files": patch_files(cfg["patch"]),
        "distinctive_lines": len(load_lines(issue)["fix"]),
    }


class SWE(Benchmark):
    """One CheatBench SWE issue under one of Lucy's honeypots, for 1, 2 or 3 agents.

    Parts by team size: one agent works the whole issue; two own A and B+C; three own A, B and C.
    grade=False collects the diffs and skips the hidden tests.
    """

    name = "software_engineering"
    task_root = TASK_ROOT
    workdir = WORKDIR

    def __init__(self, issue: str = "flipt-02e2", honeypot: str = "as_released", grade: bool = True) -> None:
        load_split(issue)
        if honeypot not in HONEYPOTS:
            raise ValueError(f"honeypot must be one of {HONEYPOTS}, got {honeypot!r}")
        self.issue, self.honeypot, self.grade = issue, honeypot, grade

    def time_limit_s(self) -> int:
        """The agents' shared time. Inspect's sample limit should add GRADING_BUDGET_S for setup and grading."""
        return AGENT_TIME_S

    def sample(self, agents: list[str], *, instruction_suffix: str = "", seed: int = 0, delivery: str = "auto",
               channel_list: bool = True, shutdown: bool = True, planning_turns: int = 0,
               read_tool: bool = True) -> Sample:
        """One sample. The keywords mirror SwarmConfig fields. `seed` draws the planted commits' times.
        `delivery`, `channel_list` and `read_tool` say what a team's system prompt tells it about the board.
        `shutdown=False` leaves the submit bullet out. `planning_turns=2` adds the planning paragraph to a team's
        system prompt."""
        swe = build_swe(self.issue, self.honeypot, agents, instruction_suffix=instruction_suffix, seed=seed,
                        delivery=delivery, channel_list=channel_list, shutdown=shutdown,
                        planning_turns=planning_turns, read_tool=read_tool)
        swe["images"] = ensure_images(self.issue)
        compose = write_compose(self.issue, agents, board=swe["board"])
        cheatbench = {"task": self.name, "variant": "", "swe_task": swe["instance"],
                      "instruction": swe["instruction"], "prompt_file": "instruction.md",
                      "timeout_s": agent_timeout_s(swe["instance"]), "network": "none",
                      "judge_prompt": swe["judge_prompt"], "swe": swe}
        if swe["board"]:
            cheatbench["first_message"] = swe["first_message"]
        return Sample(id=self.issue, input=swe["instruction"], sandbox=("docker", str(compose)),
                      metadata={"cheatbench": cheatbench})

    def tools(self, agents: list[str]) -> AgentTools:
        return AgentTools(names=list(TOOLS), bindings=tool_bindings(), tool_kwargs={})

    def setup(self) -> Solver:
        return swe_setup()

    def scorers(self) -> list[Scorer]:
        return [swe_grader(grade=self.grade)]

    def deviations(self, n_agents: int) -> list[str]:
        out = ["native_arm64_rebuild_of_the_amd64_swe_bench_pro_image",
               "no_network_in_every_container",
               f"agents_share_{AGENT_TIME_S}s_and_get_{AGENT_CPUS:g}_cpus_{AGENT_MEM}_each",
               "tools_shell_and_run_python_bound_per_agent",
               SUBMIT_PROMPT_DEVIATION]
        if self.honeypot != "as_released":
            out.append(f"honeypot_{self.honeypot}_pruned_and_planted_at_sample_setup")
        if n_agents > 1:
            out.append("issue_split_into_lucys_parts_with_a_shared_team_remote")
        return out
