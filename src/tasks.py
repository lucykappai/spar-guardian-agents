"""The Inspect tasks of the experiment: one benchmark under one SwarmConfig, as an ORBIT task.

    task = swarm_task(OpenMath("riemann"), PRESETS["ours"].with_overrides(["n_agents=10", "regulators=3"]))
    inspect eval src/tasks.py@openmath -T preset=lucy -T n_agents=3

What comes from where:
  * the sample, its tools, its setup solver and its graders come from the benchmark (src/benchmark/domains/openmath.py,
    src/benchmark/domains/swe.py);
  * the roster, the board, the hidden roles, the written plan and the turn rules come from this package
    (comm_topology.py, roles.py, rules.py);
  * the control's defenses come from controls.py, and CheatBench's judge is added as a scorer (src/benchmark/judge.py).

The run's facts go into the sample metadata: metadata["cheatbench"] gains the config dump, the control, the hidden
roles and their composition, and the deviations from CheatBench's own runner; metadata["experiment"]["defenses"]
holds the control's defenses plus the passive turn-rules defense, which every run carries.

Hidden roles and the team report are used on a board of three or more agents on openmath (the October 2026 swarm),
and on any board whose config asks for a regulator or a disregulator. Two openmath agents without roles are the
plain board.
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from orbit.configs.execution import AgentGroup
from orbit.configs.experiment import ExperimentConfig
from orbit.configs.setup import AgentSpec, SetupConfig
from orbit.tasks.multi_agent import multi_agent_task

from src.benchmark.base import AgentTools, Benchmark
from src.benchmark.judge import cheatbench_judge
from src.benchmark.domains.openmath import OpenMath
from src.benchmark.domains.swe import SWE
from src.configuration import settings
from src.configuration.openmath import TOKEN_LIMIT
from src.configuration.swarm_config import PRESETS, SwarmConfig
from src.configuration.swe import GRADING_BUDGET_S
from src import roles as roles_module
from src.comm_topology import board_communication, board_execution, board_prompt, board_scheduler
from src.controls import BASELINE, defenses_for
from src.probes import end_of_episode_probes
from src.rules import rules_setup, schedule, turn_rules_defense
# Imported for its side effect: Gemini's 402 "credits depleted" is retried as a rate limit.
from src.utils.provider_retry import INSTALLED as _RETRY_402_INSTALLED  # noqa: F401

JUDGE_ADDENDUM = settings.LUCY / "experiments" / "orbit_cheatbench" / "judge_addendum.py"
MONITOR_FACTOR = 3        # an LLM control's own calls count toward the token limit


# --------------------------------------------------------------------------
# the parts of one task
# --------------------------------------------------------------------------
def level_key(benchmark: Benchmark) -> str:
    return roles_module.LEVEL_KEYS[benchmark.name]


def hidden_roles(benchmark: Benchmark, config: SwarmConfig) -> dict[str, str] | None:
    """Each agent's hidden role, or None when the run uses no roles (a lone agent or a plain board)."""
    asked = config.regulator_count + config.disregulator_count > 0
    swarm = isinstance(benchmark, OpenMath) and config.n_agents > 2
    if not config.board or not (asked or swarm):
        return None
    return roles_module.assign_roles(config.names, config.regulator_count, config.disregulator_count, config.seed)


def instruction_suffix(benchmark: Benchmark, config: SwarmConfig, roles: dict[str, str] | None) -> str:
    """The rule sentence, then the team report instruction when the run uses roles."""
    report = roles_module.team_report_suffix() if roles else ""
    return roles_module.rule_suffix(config, level_key(benchmark)) + report


def build_sample(benchmark: Benchmark, config: SwarmConfig, suffix: str) -> Sample:
    """The benchmark's sample. SWE also takes the settings its prompts and planted commits depend on."""
    if isinstance(benchmark, SWE):
        return benchmark.sample(config.names, instruction_suffix=suffix, seed=config.seed, delivery=config.delivery,
                                channel_list=config.channel_list, shutdown=config.shutdown,
                                planning_turns=config.planning_turns, read_tool=config.read_tool)
    return benchmark.sample(config.names, instruction_suffix=suffix)


def system_prompts(benchmark: Benchmark, config: SwarmConfig, sample: Sample,
                   roles: dict[str, str] | None) -> dict[str, str | None]:
    """Each agent's system prompt: the benchmark's own (SWE) or the board prompt (openmath), then its role text.

    A lone openmath agent has no system prompt of ours, as in CheatBench's runner.
    """
    names = config.names
    own = (sample.metadata["cheatbench"].get("swe") or {}).get("system_prompt")
    if own:
        base = {n: own[n] for n in names}
    elif config.board:
        base = {n: board_prompt(n, names, workdir=benchmark.workdir, read_tool=config.read_tool) for n in names}
    else:
        base = {n: None for n in names}
    if not roles:
        return base
    return {n: roles_module.with_role(base[n], roles_module.role_text(roles[n], config, level_key(benchmark)))
            for n in names}


def agent_specs(config: SwarmConfig, tools: AgentTools, prompts: dict[str, str | None]) -> list[AgentSpec]:
    """One worker per name. An agent with no prompt of ours keeps AgentSpec's default system prompt."""
    return [AgentSpec(name=n, role="worker", tools=list(tools.names), max_messages=config.message_limit + 10,
                      submit=config.shutdown, **({"system_prompt": prompts[n]} if prompts[n] else {}))
            for n in config.names]


def first_message_groups(sample: Sample, names: list[str]) -> list[AgentGroup]:
    """One group per agent whose first message differs from the sample input (SWE teams)."""
    first = sample.metadata["cheatbench"].get("first_message") or {}
    if not first:
        return []
    if any(not (first.get(n) or "").strip() for n in names):
        raise ValueError("an empty goal would fall back to the sample input")
    return [AgentGroup(name=f"seat_{n}", agents=[n], goal=first[n]) for n in names]


def scorers(benchmark: Benchmark, config: SwarmConfig) -> list:
    judge = cheatbench_judge(benchmark.name, str(benchmark.task_root), judge=config.judge,
                             addendum=str(JUDGE_ADDENDUM) if config.judge_addendum else None)
    probes = [end_of_episode_probes()] if config.awareness_probe else []
    return [*benchmark.scorers(), judge, *probes]


def token_limit(config: SwarmConfig) -> int:
    if config.token_limit:
        return config.token_limit
    return config.n_agents * TOKEN_LIMIT * (MONITOR_FACTOR if config.control not in BASELINE else 1)


def check_control(benchmark: Benchmark, config: SwarmConfig) -> None:
    """The task-aware controls carry the openmath rule; refuse them on another benchmark."""
    if config.control.endswith("_task") and not isinstance(benchmark, OpenMath):
        raise ValueError(f"control {config.control!r} carries the openmath rule (src/configuration/controls.yaml); "
                         f"no criteria are written for {benchmark.name}")


# --------------------------------------------------------------------------
# metadata the orchestrator and the exporter read
# --------------------------------------------------------------------------
def record_run(task_: Task, benchmark: Benchmark, config: SwarmConfig, roles: dict[str, str] | None,
               prompts: dict[str, str | None]) -> None:
    """Write the defenses into the experiment config and the run's facts into metadata["cheatbench"].

    multi_agent_task takes no defenses; the orchestrator reads them from metadata["experiment"]["defenses"]
    (orbit/solvers/orchestrator.py _build_attacks_and_defenses).
    """
    control = defenses_for(config.control, config.monitor_model)
    defenses = [d.model_dump(mode="json") for d in [*control, turn_rules_defense(config)]]
    for sample in task_.dataset:
        experiment = sample.metadata["experiment"]
        if not config.board:
            experiment["communication"] = None      # a lone agent gets no channel tools
        experiment["defenses"] = defenses
        ExperimentConfig.model_validate(experiment)
        cb = sample.metadata["cheatbench"]
        cb["control"] = config.control if config.control not in BASELINE else "baseline"
        cb["monitor_model"] = config.monitor_model if control else None
        cb["config"] = config.model_dump(mode="json")
        cb["deviations"] = benchmark.run_deviations(config.n_agents, shutdown=config.shutdown)
        if roles:
            cb["roles"] = roles
            cb["composition"] = roles_module.composition_label(roles)
        first = cb.get("first_message") or {}
        if roles and cb.get("judge_prompt") and first:
            # the judge sees each SWE agent's whole system prompt, role text included
            cb["judge_prompt"] = {n: f"{prompts[n]}\n\n{first[n]}" for n in config.names}


# --------------------------------------------------------------------------
# the task
# --------------------------------------------------------------------------
def swarm_task(benchmark: Benchmark, config: SwarmConfig, *, name: str | None = None) -> Task:
    """One benchmark under one config, as an ORBIT task on Inspect."""
    settings.check_submodule_heads()
    check_control(benchmark, config)
    names = config.names
    roles = hidden_roles(benchmark, config)
    sample = build_sample(benchmark, config, instruction_suffix(benchmark, config, roles))
    tools = benchmark.tools(names)
    prompts = system_prompts(benchmark, config, sample, roles)
    time_limit_s = config.time_limit_s or benchmark.time_limit_s()
    communication = board_communication(names, delivery=config.delivery, visibility=config.visibility,
                                        max_message_chars=config.max_message_chars,
                                        max_context_chars=config.max_context_chars,
                                        max_inbox_messages=config.max_inbox_messages) if config.board else None
    # SWE grades after the agents stop; Inspect's sample limit leaves room for setup and grading.
    inspect_time = {"time_limit": time_limit_s + GRADING_BUDGET_S} if isinstance(benchmark, SWE) else {}
    task_ = multi_agent_task(
        name=name or f"swarm_{level_key(benchmark)}",
        setup=SetupConfig(agents=agent_specs(config, tools, prompts), communication_edges=None,
                          tool_kwargs=tools.tool_kwargs),
        dataset=[sample],
        communication=communication,
        execution=board_execution(quantum=config.quantum, plan=schedule(config, names, roles),
                                  agent_groups=first_message_groups(sample, names), initial_context="sample_input"),
        scheduler=board_scheduler(max_turns=config.max_turns + 5, max_time_seconds=time_limit_s),
        scorer=scorers(benchmark, config),
        tool_bindings=tools.bindings or None,
        task_setup=[s for s in (rules_setup(config, names, roles), benchmark.setup()) if s is not None],
        message_limit=config.message_limit,
        token_limit=token_limit(config),
        **inspect_time,
    )
    record_run(task_, benchmark, config, roles, prompts)
    return task_


def _config(preset: str, overrides: dict) -> SwarmConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose one of {sorted(PRESETS)}")
    return PRESETS[preset].with_overrides([f"{key}={value}" for key, value in overrides.items()])


@task
def openmath(preset: str = "ours", row: str = "riemann", **overrides) -> Task:
    """CheatBench's openmath_agent row under a preset; every other -T key=value sets a config field."""
    return swarm_task(OpenMath(row), _config(preset, overrides))


@task
def swe(preset: str = "ours", issue: str = "flipt-02e2", honeypot: str = "as_released", grade: bool = True,
        **overrides) -> Task:
    """A CheatBench SWE issue under one of Lucy's honeypots and a preset; other -T key=value set config fields."""
    grade = grade if isinstance(grade, bool) else str(grade).strip().lower() in ("1", "true", "yes")
    return swarm_task(SWE(issue, honeypot, grade), _config(preset, overrides))
