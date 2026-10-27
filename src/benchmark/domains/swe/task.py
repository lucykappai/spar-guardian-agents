"""The SWE benchmark: one CheatBench issue under one of Lucy's honeypots, for one agent or a team.

Holds the sample (build_swe), each agent's parts and pieces, and the texts each agent reads: the system prompts
from prompts.yaml `swe` and the first messages with Lucy's assignment.

Modes of a team (SwarmConfig.resolve_mode, `shared_task` when mode is None):
  * collaboration_needed: the issue is split into Lucy's parts. Two agents own A and B+C, three own A, B and C, and
    more than three take A, B and C round robin, each told who shares its part. The team's main is graded.
  * shared_task: every agent gets the whole issue on its own machine. The team's main is graded. The own_piece and
    jigsaw honeypots plant different pieces in different agents' clones, independent of parts.
  * solo_task: every agent gets the whole issue and its own repository, with no team remote. Each agent's working
    tree is graded.
A lone agent works the whole issue in every mode. Each agent has its own service, so SWE refuses the large scales.
"""
from __future__ import annotations

import random
import re

from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from src.benchmark.base import SUBMIT_PROMPT_DEVIATION, AgentTools, Benchmark
from src.benchmark.domains.swe.grader import swe_grader
from src.benchmark.domains.swe.honeypots import has_team_remote, swe_setup
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
from src.configuration.scales import SCALES
from src.configuration.swarm_config import (
    LEGACY,
    SwarmConfig,
    board_system_prompt,
    submit_line,
    swe_board_fields,
    swe_part_clause,
)
from src.configuration.swe import (
    AGENT_CPUS,
    AGENT_MEM,
    AGENT_TIME_S,
    DELIVERY_MODES,
    HONEYPOTS,
    JIGSAW_PIECES,
    PARTS,
    PIECES,
    TASK_ROOT,
    TEAM_PARTS,
    TOOL_TIMEOUT_S,
    TOOLS,
    WORKDIR,
)

COLLABORATION = "collaboration_needed"
# Every agent has its own service (sandbox.write_compose). A hundred services is more than one Docker host holds.
LARGE_SCALES = ("large", "extra_large", "gigantic")
LARGE_SCALE_ERROR = "SWE runs per-agent containers; large scales need a shared container"


def check_scale(config: SwarmConfig) -> None:
    """Raise NotImplementedError at the large scales, or at the large scale's team size set by hand."""
    if config.scale in LARGE_SCALES or config.n_agents >= SCALES["large"].n_agents:
        raise NotImplementedError(LARGE_SCALE_ERROR)


def legacy_config(n_agents: int, *, seed: int = 0, delivery: str = "auto", channel_list: bool = True,
                  shutdown: bool = True, planning_turns: int = 0, read_tool: bool = True) -> SwarmConfig:
    """The config commit 6769c80's keywords describe, for a caller that passes them in place of a config."""
    return SwarmConfig(**{**LEGACY, "n_agents": n_agents, "seed": seed, "delivery": delivery,
                          "channel_list": channel_list, "shutdown": shutdown, "planning_turns": planning_turns,
                          "read_tool": read_tool})


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


def parts_by_agent(agents: list[str], mode: str = COLLABORATION) -> dict[str, list[str]]:
    """The parts each agent owns. [] is the whole issue: a lone agent, and every agent outside collaboration_needed.

    Two agents own A and B+C, three own A, B and C, and more than three take A, B and C round robin."""
    if len(agents) == 1 or mode != COLLABORATION:
        return {a: [] for a in agents}
    if len(agents) in TEAM_PARTS:
        return dict(zip(agents, TEAM_PARTS[len(agents)]))
    return {a: [PARTS[i % len(PARTS)]] for i, a in enumerate(agents)}


def part_mates(parts: dict[str, list[str]]) -> dict[str, list[str]]:
    """The other agents that own the same parts as each agent, for agents that share a part."""
    out = {}
    for a, mine in parts.items():
        mates = [o for o, theirs in parts.items() if o != a and mine and theirs == mine]
        if mates:
            out[a] = mates
    return out


def pieces_by_agent(agents: list[str], honeypot: str, mode: str = COLLABORATION) -> dict[str, list[str]]:
    """The pieces planted in each agent's clone.

    own_piece: in collaboration_needed each agent holds the pieces of its own parts. In shared_task the pieces do not
    depend on parts, since every agent works the whole issue: each agent holds the pieces its seat would own under
    collaboration_needed, so the agents hold different access to the fix. A lone agent and a solo_task agent hold
    `whole`. jigsaw: three agents in collaboration_needed or shared_task hold Lucy's B, C and A."""
    if honeypot not in HONEYPOTS:
        raise ValueError(f"honeypot must be one of {HONEYPOTS}, got {honeypot!r}")
    if honeypot == "as_released":
        return {a: [] for a in agents}
    if honeypot == "jigsaw":
        if len(agents) != 3 or mode not in (COLLABORATION, "shared_task"):
            raise ValueError("the jigsaw honeypot needs three agents in collaboration_needed or shared_task "
                             "(Lucy's W1 holds B, W2 C, W3 A)")
        return dict(zip(agents, JIGSAW_PIECES))
    seats = COLLABORATION if mode == "shared_task" else mode
    return {a: mine or ["whole"] for a, mine in parts_by_agent(agents, seats).items()}


def piece_times(seed: int, issue: str) -> dict[str, int]:
    """Seconds added to each planted commit's time, drawn from the sample's seed, so each seed plants new hashes."""
    rng = random.Random(f"{seed}/{issue}/piece_times")
    return {p: rng.randrange(3600) for p in PIECES}


def system_prompts(split: dict, agents: list[str], parts: dict[str, list[str]], config: SwarmConfig, *,
                   mode: str = COLLABORATION, neutral: set[str] | None = None) -> dict[str, str]:
    """Each agent's system prompt from prompts.yaml `swe`. The submit bullet is left out without shutdown.

    A lone agent and a solo_task agent get `system_solo`; a solo_task agent of a board then gets the team and board
    texts its config asks for (swarm_config.board_system_prompt). A collaboration_needed agent gets `system_team`
    with the roster's parts, a shared_task agent `system_shared`. Their board sentences and last line come from
    swarm_config.swe_board_fields, and system_team's sentence on the other parts from swarm_config.swe_part_clause;
    the team_submit bullet goes to the neutral agents of a mode with team_submit."""
    prompts = settings.load_prompts()["swe"]
    neutral = set(agents) if neutral is None else neutral
    if len(agents) == 1 or mode == "solo_task":
        submit = submit_line(False, config.shutdown)
        out = {}
        for a in agents:
            text = prompts["system_solo"].format(name=a, repo=split["repo"], branch=a, timeout=TOOL_TIMEOUT_S,
                                                 submit=submit)
            board = board_system_prompt(config, a, agents, workdir=WORKDIR, domain="swe")
            out[a] = f"{text}\n\n{board}" if board else text
        return out
    shared = mode == "shared_task"
    submit = submit_line(True, config.shutdown, shared=shared)
    submits = config.has_team_submit("swe")
    template = prompts["system_shared" if shared else "system_team"]
    planning = f"\n\n{prompts['planning_system']}" if config.planning_turns == 2 else ""
    mates = part_mates(parts)
    out = {}
    for a in agents:
        if shared:
            roster = "\n".join(prompts["roster_line_shared"].format(name=o) for o in agents if o != a)
        else:
            roster = "\n".join(prompts["roster_line"].format(name=o, title=part_titles(split, parts[o]))
                               for o in agents if o != a)
        out[a] = template.format(
            name=a, n=len(agents), repo=split["repo"], roster=roster, branch=a, timeout=TOOL_TIMEOUT_S,
            submit=submit, part_clause=swe_part_clause(config, mates.get(a)),
            **swe_board_fields(config, team_submit=submits and a in neutral)) + planning
    return out


def part_titles(split: dict, parts: list[str]) -> str:
    return "; ".join(split["parts"][p]["title"] for p in parts)


def _and_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def first_messages(split: dict, issue_text: str, instruction: str, parts: dict[str, list[str]],
                   mode: str = COLLABORATION) -> dict[str, str]:
    """A lone agent's first message is the instruction, and so is every agent's outside collaboration_needed.
    A collaboration_needed agent's is the instruction, then Lucy's assignment: the team's split and the agent's part,
    its requirement bullets copied from the issue, then the agents it shares the part with."""
    agents = list(parts)
    if len(agents) == 1 or mode != COLLABORATION:
        return {a: instruction for a in agents}
    prompts = settings.load_prompts()["swe"]
    template = prompts["assignment_team"]
    table = "\n".join(f"- {a}: {part_titles(split, parts[a])}" for a in agents)
    mates = part_mates(parts)
    out = {}
    for a in agents:
        text = "\n".join(part_text(issue_text, split, p) for p in parts[a])
        assignment = template.format(table=table, title=part_titles(split, parts[a]), part=text)
        out[a] = f"{instruction.rstrip()}\n\n{assignment.strip()}\n"
        if a in mates:
            out[a] += "\n" + prompts["part_shared_with"].format(mates=_and_list(mates[a]))
    return out


def build_swe(issue: str, honeypot: str, agents: list[str], config: SwarmConfig, *, instruction_suffix: str = "",
              roles: dict[str, str] | None = None) -> dict:
    """Everything one sample needs except the images: parts, pieces, prompts and first messages. No Docker.

    The returned dict is metadata["cheatbench"]["swe"]. A team outside collaboration_needed adds `mode`; a team
    whose agents share parts adds `shares` {agent: [mates]}. Both keys are absent where commit 6769c80 had neither."""
    if config.delivery not in DELIVERY_MODES:
        raise ValueError(f"delivery must be one of {DELIVERY_MODES}, got {config.delivery!r}")
    split = load_split(issue)
    instance = split["task"]
    cfg = task_config(instance)
    issue_text = (task_dir(instance) / "instruction.md").read_text()
    instruction = issue_text + instruction_suffix
    team = len(agents) > 1
    mode = config.resolve_mode("swe") if team else COLLABORATION
    parts = parts_by_agent(agents, mode)
    pieces = pieces_by_agent(agents, honeypot, mode)
    neutral = {a for a in agents if (roles or {}).get(a, "worker") == "worker"}
    system = system_prompts(split, agents, parts, config, mode=mode, neutral=neutral)
    first = first_messages(split, issue_text, instruction, parts, mode)
    swe = {
        "issue": issue, "task_key": issue, "instance": instance, "repo": cfg["repo"],
        "base_commit": cfg["base_commit"], "fix_commit": fix_commit(instance),
        "honeypot": honeypot, "jigsaw": honeypot == "jigsaw", "seed": config.seed,
        "n_agents": len(agents), "agents": list(agents), "board": team,
        "delivery": config.delivery if team else None,
        "channel_list": config.channel_list if team else None, "read_tool": config.read_tool if team else None,
        "shutdown": config.shutdown, "planning_turns": config.planning_turns,
        "hint": instruction_suffix.strip() or None, "instruction": instruction,
        "parts": {a: {"parts": parts[a] or ["whole"], "titles": [split["parts"][p]["title"] for p in parts[a]]}
                  for a in agents},
        "pieces": pieces,
        "piece_times": piece_times(config.seed, issue),
        "system_prompt": system, "first_message": first,
        "judge_prompt": {a: f"{system[a]}\n\n{first[a]}" for a in agents},
        "patch_files": patch_files(cfg["patch"]),
        "distinctive_lines": len(load_lines(issue)["fix"]),
    }
    if team and mode != COLLABORATION:
        swe["mode"] = mode
    if part_mates(parts):
        swe["shares"] = part_mates(parts)
    return swe


class SWE(Benchmark):
    """One CheatBench SWE issue under one of Lucy's honeypots, for one agent or a team (see the module docstring).

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

    def check(self, config: SwarmConfig) -> None:
        """NotImplementedError at the large scales (LARGE_SCALE_ERROR)."""
        check_scale(config)

    def time_limit_s(self) -> int:
        """The agents' shared time. Inspect's sample limit should add GRADING_BUDGET_S for setup and grading."""
        return AGENT_TIME_S

    def sample(self, agents: list[str], *, instruction_suffix: str = "", config: SwarmConfig | None = None,
               roles: dict[str, str] | None = None, seed: int = 0, delivery: str = "auto",
               channel_list: bool = True, shutdown: bool = True, planning_turns: int = 0,
               read_tool: bool = True) -> Sample:
        """One sample under the config. Without a config the keywords of commit 6769c80 (seed, delivery,
        channel_list, shutdown, planning_turns, read_tool) describe a legacy run; with one they are ignored.
        roles {agent: role} keeps team_submit out of the prompts of regulators and disregulators."""
        if config is None:
            config = legacy_config(len(agents), seed=seed, delivery=delivery, channel_list=channel_list,
                                   shutdown=shutdown, planning_turns=planning_turns, read_tool=read_tool)
        self.check(config)
        swe = build_swe(self.issue, self.honeypot, agents, config, instruction_suffix=instruction_suffix,
                        roles=roles)
        swe["images"] = ensure_images(self.issue)
        compose = write_compose(self.issue, agents, board=has_team_remote(swe))
        cheatbench = {"task": self.name, "variant": "", "swe_task": swe["instance"],
                      "instruction": swe["instruction"], "prompt_file": "instruction.md",
                      "timeout_s": agent_timeout_s(swe["instance"]), "network": "none",
                      "judge_prompt": swe["judge_prompt"], "swe": swe}
        if swe["board"]:
            cheatbench["first_message"] = swe["first_message"]
        return Sample(id=self.issue, input=swe["instruction"], sandbox=("docker", str(compose)),
                      metadata={"cheatbench": cheatbench})

    def tools(self, agents: list[str], config: SwarmConfig | None = None) -> AgentTools:
        """`shell` and `run_python` on each agent's own service, in every mode."""
        return AgentTools(names=list(TOOLS), bindings=tool_bindings(), tool_kwargs={})

    def setup(self) -> Solver:
        return swe_setup()

    def scorers(self) -> list[Scorer]:
        return [swe_grader(grade=self.grade)]

    def deviations(self, n_agents: int, mode: str | None = None) -> list[str]:
        out = ["native_arm64_rebuild_of_the_amd64_swe_bench_pro_image",
               "no_network_in_every_container",
               f"agents_share_{AGENT_TIME_S}s_and_get_{AGENT_CPUS:g}_cpus_{AGENT_MEM}_each",
               "tools_shell_and_run_python_bound_per_agent",
               SUBMIT_PROMPT_DEVIATION]
        if self.honeypot != "as_released":
            out.append(f"honeypot_{self.honeypot}_pruned_and_planted_at_sample_setup")
        if n_agents > 1:
            out.append({"shared_task": "every_agent_works_the_whole_issue_with_a_shared_team_remote",
                        "solo_task": "every_agent_works_the_whole_issue_alone_and_is_graded_alone"}.get(
                mode or COLLABORATION, "issue_split_into_lucys_parts_with_a_shared_team_remote"))
        if n_agents > max(TEAM_PARTS) and (mode or COLLABORATION) == COLLABORATION:
            out.append("parts_assigned_round_robin_with_several_agents_per_part")
        return out
