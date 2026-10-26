"""The openmath_agent Benchmark: CheatBench's task folder and built rows, the sample, and the deviations.

Inputs are submodules/cheatbench/tasks/openmath_agent and the rows in build/openmath_agent/data.jsonl.
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
    make_stage_tar,
    write_docker_context,
)
from src.configuration.openmath import BUILD, TASK, TASK_DIR, TOOL_KWARGS, WORKDIR

DEVIATIONS = [
    "agent_is_orbit_turn_react_not_vendor_harness",
    SUBMIT_PROMPT_DEVIATION,
    "continue_prompts_are_orbit_synthetic_user_messages",
    "mirror_served_from_docker_sidecar_and_ca_baked_into_image",
]
TEAM_DEVIATION = "two_agents_share_one_sandbox_clock_and_git_identity"


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


def build_sample(row_id: str, instruction_suffix: str = "") -> Sample:
    """The sample for one row. instruction_suffix is appended to CheatBench's instruction."""
    cfg = load_task_yaml()
    compose = write_docker_context()
    template = TASK_DIR / cfg["prompts"]["instruction"]
    row = next((r for r in load_rows() if r["id"] == row_id), None)
    if row is None:
        raise ValueError(f"no row {row_id!r} in {BUILD / 'data.jsonl'}")
    instruction = render_prompt(template, row) + instruction_suffix
    tar_path = make_stage_tar(row, BUILD / "environment", BUILD / "stage" / f"{row['id']}.tgz")
    return Sample(
        id=row["id"],
        input=instruction,
        files={f"{WORKDIR}/.stage.tgz": str(tar_path)},
        setup=SETUP_SCRIPT,
        sandbox=("docker", str(compose)),
        metadata={"cheatbench": {
            "task": cfg["name"], "variant": "", "row": row,
            "instruction": instruction, "prompt_file": cfg["prompts"]["instruction"],
            "timeout_s": int(cfg["timeout_s"]), "network": cfg.get("network"),
        }},
    )


class OpenMath(Benchmark):
    """openmath_agent for one row. Every agent gets the same instruction and the same sandbox."""

    name = TASK
    task_root = TASK_DIR
    workdir = WORKDIR

    def __init__(self, row: str = "riemann") -> None:
        self.row = row

    def time_limit_s(self) -> int:
        return int(load_task_yaml()["timeout_s"])

    def sample(self, agents: list[str], *, instruction_suffix: str = "") -> Sample:
        """The row's sample. The roster does not change it: the judge sees the instruction as every agent's rules."""
        return build_sample(self.row, instruction_suffix)

    def tools(self, agents: list[str]) -> AgentTools:
        return AgentTools(names=["bash", "python"], bindings={}, tool_kwargs=TOOL_KWARGS)

    def setup(self) -> Solver | None:
        return environment_probe()

    def scorers(self) -> list[Scorer]:
        return [deliverables_scorer()]

    def deviations(self, n_agents: int) -> list[str]:
        return DEVIATIONS + ([TEAM_DEVIATION] if n_agents > 1 else [])
