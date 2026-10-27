"""The openmath_agent Benchmark: CheatBench's task folder and built rows, the sample, and the deviations.

Inputs are submodules/cheatbench/tasks/openmath_agent and the rows in build/openmath_agent/data.jsonl.

Modes. A lone agent gets CheatBench's own layout in every mode. A `shared_task` team shares that layout: one
workspace, one Linux user and one git identity. A `solo_task` team (the default) gets one copy of the row per agent in
the one container, each under its own Linux user (sandbox.py). The row has no parts, so `collaboration_needed` is
refused.
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


def per_agent(agents: list[str] | None, config: SwarmConfig | None) -> bool:
    """Whether the roster gets the solo_task layout: two or more agents in solo_task."""
    return bool(agents) and len(agents) > 1 and config is not None and config.resolve_mode(TASK) == "solo_task"


def build_sample(row_id: str, instruction_suffix: str = "", solo_agents: list[str] | None = None) -> Sample:
    """The sample for one row. instruction_suffix is appended to CheatBench's instruction.

    solo_agents gives the solo_task layout to those agents; None keeps CheatBench's layout. The staged tarball, the
    compose file and the instruction are the same in both; the setup script differs, and the solo layout is recorded
    as metadata["cheatbench"]["openmath"] {mode, agents: {name: {user, uid, home, workdir}}}."""
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
    if solo_agents:
        cheatbench["openmath"] = {"mode": "solo_task", "agents": agent_layout(solo_agents)}
    return Sample(
        id=row["id"],
        input=instruction,
        files={f"{WORKDIR}/.stage.tgz": str(tar_path)},
        setup=solo_setup_script(solo_agents) if solo_agents else SETUP_SCRIPT,
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

    def time_limit_s(self) -> int:
        return int(load_task_yaml()["timeout_s"])

    def check(self, config: SwarmConfig) -> None:
        if config.resolve_mode(TASK) == "collaboration_needed":
            raise ValueError("openmath_agent has no parts to split; run it in mode solo_task or shared_task")

    def sample(self, agents: list[str], *, instruction_suffix: str = "", config: SwarmConfig | None = None,
               roles: dict[str, str] | None = None) -> Sample:
        """The row's sample. The judge sees the instruction as every agent's rules. A solo_task team gets the
        per-agent layout; any other roster gets CheatBench's."""
        if config is not None:
            self.check(config)
        return build_sample(self.row, instruction_suffix, list(agents) if per_agent(agents, config) else None)

    def tools(self, agents: list[str], config: SwarmConfig | None = None) -> AgentTools:
        """ORBIT's bash and python as the user `user`, or in the solo_task layout `shell` and `run_python` bound to
        each agent's own user and copy."""
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
        return DEVIATIONS + (SOLO_DEVIATIONS if mode == "solo_task" else [TEAM_DEVIATION])
