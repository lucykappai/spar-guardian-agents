"""The openmath_agent Benchmark: CheatBench's task folder and built rows, the sample, and the deviations.

Inputs are submodules/cheatbench/tasks/openmath_agent and the rows in build/openmath_agent/data.jsonl.

Modes. A lone agent gets CheatBench's own layout in every mode. A `shared_task` team (the default) with split_access
gets the split layout: one copy of the row per agent, each under its own Linux user, a bare team remote that every
copy pushes to, and the complete submission's archive cut into one slice per neutral agent (sandbox.py).
A `shared_task` team without split_access (ian_legacy) shares CheatBench's layout: one workspace, one Linux user and
one git identity. A `solo_task` team gets one copy of the row per agent with no team remote. The row has no parts, so
`collaboration_needed` is refused.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from src.benchmark.base import SUBMIT_PROMPT_DEVIATION, AgentTools, Benchmark
from src.benchmark.domains.openmath.probe import deliverables_scorer, environment_probe
from src.benchmark.domains.openmath.sandbox import (
    SETUP_SCRIPT,
    SOLO_TOOLS,
    agent_layout,
    make_stage_tar,
    solo_setup_script,
    solo_tool_bindings,
    split_layout,
    split_setup_script,
    write_docker_context,
)
from src.configuration.openmath import BUILD, TASK, TASK_DIR, TOOL_KWARGS, WORKDIR
from src.configuration.swarm_config import SwarmConfig

DEVIATIONS = [
    "agent_is_orbit_turn_react_not_vendor_harness",
    SUBMIT_PROMPT_DEVIATION,
    "continue_prompts_are_orbit_synthetic_user_messages",
    "mirror_served_from_docker_sidecar_and_ca_baked_into_image",
]
TEAM_DEVIATION = "two_agents_share_one_sandbox_clock_and_git_identity"
SOLO_DEVIATIONS = [
    "each_agent_has_its_own_linux_user_home_and_copy_of_the_row_in_one_container_with_one_clock",
    "tools_shell_and_run_python_bound_per_agent_user_in_place_of_bash_and_python",
]
SPLIT_DEVIATIONS = SOLO_DEVIATIONS + [
    "origin_is_a_bare_team_remote_holding_the_staged_main_and_the_rows_own_branch",
    "complete_submissions_archive_cut_into_one_slice_per_neutral_agent_on_a_rewritten_branch_commit",
    "every_copy_keeps_the_rows_git_identity",
]


def load_task_yaml() -> dict:
    return yaml.safe_load((TASK_DIR / "task.yaml").read_text())


def render_prompt(template: Path, row: dict) -> str:
    """CheatBench core/trial.py:render_prompt, reproduced verbatim in behavior."""
    fields = {"input": row.get("input") or "", **(row.get("vars") or {})}
    out = []
    for line in template.read_text().split("\n"):
        tokens = [k for k in fields if "{{" + k + "}}" in line]
        if tokens and all(not str(fields[k]).strip() for k in tokens):
            continue
        for k, v in fields.items():
            line = line.replace("{{" + k + "}}", str(v))
        out.append(line)
    text = "\n".join(out)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip() + "\n"


def load_rows() -> list[dict]:
    data = BUILD / "data.jsonl"
    if not data.exists():
        raise FileNotFoundError(
            f"{data} missing: build the task first, e.g. "
            f"python -c \"...tasks/openmath_agent/build.py build(output='{BUILD}', selected=['riemann'])\"")
    return [json.loads(line) for line in data.read_text().splitlines() if line.strip()]


def split_access(config: SwarmConfig | None) -> bool:
    """The config's split_access field. A config without the field (an older SwarmConfig) has no split access."""
    return bool(getattr(config, "split_access", False))


def layout_of(agents: list[str] | None, config: SwarmConfig | None) -> str | None:
    """The roster's layout: "solo_task" for two or more agents in solo_task, "split" for two or more agents in
    shared_task with split_access, None for CheatBench's own layout (a lone agent, ian_legacy's shared workspace)."""
    if not agents or len(agents) < 2 or config is None:
        return None
    mode = config.resolve_mode(TASK)
    if mode == "solo_task":
        return "solo_task"
    if mode == "shared_task" and split_access(config):
        return "split"
    return None


def per_agent(agents: list[str] | None, config: SwarmConfig | None) -> bool:
    """Whether the roster gets one Linux user and one copy of the row per agent: the solo_task or split layout."""
    return layout_of(agents, config) is not None


def build_sample(row_id: str, instruction_suffix: str = "", solo_agents: list[str] | None = None, *,
                 split_agents: list[str] | None = None, roles: dict[str, str] | None = None) -> Sample:
    """The sample for one row. instruction_suffix is appended to CheatBench's instruction.

    solo_agents gives the solo_task layout to those agents; split_agents gives them the split layout, with roles
    {agent: role} saying which are neutral (None: all); neither keeps CheatBench's layout. The staged tarball, the
    compose file and the instruction are the same in all three; the setup script differs, and a per-agent layout is
    recorded as metadata["cheatbench"]["openmath"]: {mode, agents: {name: {user, uid, home, workdir}}} for solo_task,
    and for split {mode: "shared_task", split_access: True, remote, group, remote_branches, honeypot_branches,
    archive: {branch, commit, name, parts}, archive_parts: {agent: k},
    agents: {name: {user, uid, home, workdir, branches, archive_part}}}."""
    if solo_agents and split_agents:
        raise ValueError("a roster gets one layout: solo_agents or split_agents")
    cfg = load_task_yaml()
    compose = write_docker_context()
    template = TASK_DIR / cfg["prompts"]["instruction"]
    row = next((r for r in load_rows() if r["id"] == row_id), None)
    if row is None:
        raise ValueError(f"no row {row_id!r} in {BUILD / 'data.jsonl'}")
    instruction = render_prompt(template, row) + instruction_suffix
    tar_path = make_stage_tar(row, BUILD / "environment", BUILD / "stage" / f"{row['id']}.tgz")
    cheatbench = {
        "task": cfg["name"], "variant": "", "row": row,
        "instruction": instruction, "prompt_file": cfg["prompts"]["instruction"],
        "timeout_s": int(cfg["timeout_s"]), "network": cfg.get("network"),
    }
    setup = SETUP_SCRIPT
    if solo_agents:
        cheatbench["openmath"] = {"mode": "solo_task", "agents": agent_layout(solo_agents)}
        setup = solo_setup_script(solo_agents)
    elif split_agents:
        cheatbench["openmath"] = split_layout(split_agents, row, roles)
        setup = split_setup_script(cheatbench["openmath"])
    return Sample(
        id=row["id"],
        input=instruction,
        files={f"{WORKDIR}/.stage.tgz": str(tar_path)},
        setup=setup,
        sandbox=("docker", str(compose)),
        metadata={"cheatbench": cheatbench},
    )


class OpenMath(Benchmark):
    """openmath_agent for one row. Every agent gets the same instruction and the same container."""

    name = TASK
    task_root = TASK_DIR
    workdir = WORKDIR

    def __init__(self, row: str = "riemann") -> None:
        self.row = row
        self.split_access = False
        """Whether the last config checked asks for split access; deviations() reads it."""

    def time_limit_s(self) -> int:
        return int(load_task_yaml()["timeout_s"])

    def check(self, config: SwarmConfig) -> None:
        if config.resolve_mode(TASK) == "collaboration_needed":
            raise ValueError("openmath_agent has no parts to split; run it in mode solo_task or shared_task")
        self.split_access = split_access(config)

    def sample(self, agents: list[str], *, instruction_suffix: str = "", config: SwarmConfig | None = None,
               roles: dict[str, str] | None = None) -> Sample:
        """The row's sample. The judge sees the instruction as every agent's rules. A solo_task team gets the
        per-agent layout, a shared_task team with split_access the split layout (roles say who is neutral); any other
        roster gets CheatBench's."""
        if config is not None:
            self.check(config)
        layout = layout_of(agents, config)
        if layout == "split":
            return build_sample(self.row, instruction_suffix, split_agents=list(agents), roles=roles)
        return build_sample(self.row, instruction_suffix, list(agents) if layout == "solo_task" else None)

    def tools(self, agents: list[str], config: SwarmConfig | None = None) -> AgentTools:
        """ORBIT's bash and python as the user `user`, or in the solo_task and split layouts `shell` and `run_python`
        bound to each agent's own user and copy."""
        if per_agent(agents, config):
            return AgentTools(names=list(SOLO_TOOLS), bindings=solo_tool_bindings(), tool_kwargs={})
        return AgentTools(names=["bash", "python"], bindings={}, tool_kwargs=TOOL_KWARGS)

    def setup(self) -> Solver | None:
        return environment_probe()

    def scorers(self) -> list[Scorer]:
        return [deliverables_scorer()]

    def deviations(self, n_agents: int, mode: str | None = None) -> list[str]:
        if n_agents < 2:
            return list(DEVIATIONS)
        if mode == "solo_task":
            return DEVIATIONS + SOLO_DEVIATIONS
        if mode == "shared_task" and self.split_access:
            return DEVIATIONS + SPLIT_DEVIATIONS
        return DEVIATIONS + [TEAM_DEVIATION]
