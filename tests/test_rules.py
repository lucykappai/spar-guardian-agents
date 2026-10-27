"""Offline tests of src/swarm/rules.py on Inspect's mock provider: no Docker, no credentials, no paid calls.

Each test builds a small ORBIT task (agents a01..a0n on one board, each with a private `work` tool, as ORBIT's
examples/turn_rules_lucy.py does), runs it under a SwarmConfig with the turn-rules defense, and reads what
the provider received (the mock sees the input after every model filter) and what the store recorded.
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("PYTHON_DOTENV_DISABLED", "1")

from inspect_ai import Task  # noqa: E402
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import (  # noqa: E402
    ChatCompletionChoice,
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Target, scorer  # noqa: E402
from inspect_ai.tool import Tool, ToolCall, tool  # noqa: E402
from inspect_ai.util import StoreModel, store_as  # noqa: E402
from orbit import AgentSpec, SetupConfig, ToolFactory, multi_agent_task  # noqa: E402
from orbit.configs.experiment import ExperimentConfig  # noqa: E402
from pydantic import Field  # noqa: E402

from src import run as run_module  # noqa: E402
from src.benchmark.transcript import SampleView, episode_transcript  # noqa: E402
from src.configuration.settings import load_prompts  # noqa: E402
from src.configuration.swarm_config import LEGACY, PRESETS, SwarmConfig, Window  # noqa: E402
from src.swarm import board, rules  # noqa: E402
from src.swarm.comm_topology import (  # noqa: E402
    BOARD_ID,
    board_communication,
    board_execution,
    board_scheduler,
)
from src.swarm.probes import end_of_episode_probes  # noqa: E402

MODEL = "mockllm/turn-rules"
PROMPTS = load_prompts()


class WorkLog(StoreModel):
    executed: list[dict] = Field(default_factory=list)


@tool
def work(owner: str) -> Tool:
    async def execute(item: str) -> str:
        """Do one unit of work on your part.

        Args:
            item: What to work on.
        """
        log = store_as(WorkLog)
        log.executed = [*log.executed, {"agent": owner, "item": item}]
        return f"{owner} worked on {item}"

    return execute


@scorer(metrics=[])
def nothing():
    async def score(state, target: Target) -> Score:
        return Score(value=0)

    return score


# ------------------------------------------------------------------------------------------------ scripted model

def calls(*requests: tuple[str, dict]) -> ModelOutput:
    tool_calls = [ToolCall(id=f"call-{uuid.uuid4()}", function=name, arguments=args) for name, args in requests]
    return ModelOutput(model=MODEL, choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="", model=MODEL, source="generate", tool_calls=tool_calls),
        stop_reason="tool_calls")])


def post(text: str = "plan") -> ModelOutput:
    return calls(("channel_post", {"channel_id": BOARD_ID, "message": text}))


def do(*items: str) -> ModelOutput:
    return calls(*(("work", {"item": item}) for item in items))


def submit() -> ModelOutput:
    return calls(("submit", {"answer": "done"}))


def text(content: str = "thinking") -> ModelOutput:
    return ModelOutput.from_content(MODEL, content)


def out_of_context() -> ModelOutput:
    return ModelOutput(model=MODEL, choices=[ChatCompletionChoice(
        message=ChatMessageAssistant(content="...", model=MODEL, source="generate"), stop_reason="model_length")])


PROBE_ANSWER = "CODING_EVAL: 70\nSOCIAL_EVAL: 20\nREASON: it looked like a work test."


def scripted_model(scripts: dict[str, list[ModelOutput]], recorder: list[dict]):
    """Each agent follows its script, one output per provider call; probe questions get fixed answers."""
    count: dict[str, int] = {}

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        last = messages[-1]
        if last.role == "user" and last.text in (PROMPTS["probes"]["reflection"], PROMPTS["probes"]["awareness"]):
            return text("It went fine." if last.text == PROMPTS["probes"]["reflection"] else PROBE_ANSWER)
        system = "\n".join(m.text for m in messages if m.role == "system")
        agent = next(name for name in scripts if f"You are {name}," in system)
        k = count.get(agent, 0)
        count[agent] = k + 1
        recorder.append({
            "agent": agent, "call": k, "tools": sorted(t.name for t in tools), "system": messages[0].text,
            "last": last.text, "tag": (last.metadata or {}).get(rules.TURN_RULES_TAG),
            "notices": [m.text for m in messages if m.role == "system" and (m.metadata or {}).get(rules.DELIVERY_TAG)],
            "added": [((m.metadata or {})[rules.TURN_RULES_TAG]["kind"], m.text) for m in messages
                      if (m.metadata or {}).get(rules.TURN_RULES_TAG)],
            "batch": config.batch,
        })
        script = scripts[agent]
        return script[min(k, len(script) - 1)]

    model = get_model(MODEL, custom_outputs=outputs, memoize=False)

    async def count_text_tokens(text: str) -> int:   # avoids the mock provider's tokenizer download
        return max(1, (len(text) + 3) // 4)

    model.api.count_text_tokens = count_text_tokens
    return model


# ------------------------------------------------------------------------------------------------ the task

def build_task(config: SwarmConfig, *, scorer_=None, roles: dict[str, str] | None = None,
               domain: str | None = None) -> Task:
    """The test task; on a pull board each agent also gets the board tools and, if neutral, team_submit, bound as
    src/tasks.py binds them."""
    names = config.names
    extra = board.agent_tool_names(config, names, "openmath", roles)
    agents = [AgentSpec(name=n, role="engineer", tools=["work", *extra[n]], max_messages=10_000,
                        submit=config.shutdown,
                        system_prompt=f"You are {n}, an engineer. Call work to make progress.") for n in names]
    task = multi_agent_task(
        name="turn_rules_test",
        setup=SetupConfig(agents=agents),
        tool_bindings={"work": ToolFactory(lambda ctx: work(ctx.agent_name)), **board.board_bindings(config)},
        communication=board_communication(names, delivery=config.delivery, visibility=config.visibility)
        if config.board else None,
        execution=board_execution(quantum=config.quantum, plan=rules.schedule(config, names, roles),
                                  initial_context="sample_input"),
        scheduler=board_scheduler(max_turns=config.max_turns + 5, max_time_seconds=120),
        dataset=[Sample(id="issue", input="Fix the issue together.")],
        scorer=scorer_ or nothing(),
        task_setup=[rules.rules_setup(config, names, roles, domain)],
        token_limit=1_000_000,
    )
    for sample in task.dataset:
        if not config.board:
            sample.metadata["experiment"]["communication"] = None   # a lone agent gets no channel tools
        sample.metadata["experiment"]["defenses"] = [rules.turn_rules_defense(config).model_dump(mode="json")]
        ExperimentConfig.model_validate(sample.metadata["experiment"])
    return task


def run(config: SwarmConfig, scripts: dict, tmp_path, *, scorer_=None, roles=None, with_log=False, domain=None,
        **generate):
    recorder: list[dict] = []
    [log] = inspect_eval(build_task(config, scorer_=scorer_, roles=roles, domain=domain),
                         model=scripted_model(scripts, recorder),
                         log_dir=str(tmp_path), display="none", log_realtime=False, max_retries=0, **generate)
    assert log.status == "success", log.error
    log = read_eval_log(log.location, resolve_attachments=True)
    [sample] = log.samples
    assert sample.error is None and sample.limit is None, (sample.error, sample.limit)
    return (sample, recorder, log) if with_log else (sample, recorder)


def by_agent(recorder: list[dict], agent: str) -> list[dict]:
    return [r for r in recorder if r["agent"] == agent]


def store(sample, field: str):
    return sample.store.get(f"TurnState:{field}")


def rule_infos(sample) -> list[dict]:
    return [e.data[rules.TURN_RULES_TAG] for e in sample.events
            if e.event == "info" and isinstance(e.data, dict) and rules.TURN_RULES_TAG in e.data]


# ------------------------------------------------------------------------------------------------ Lucy's rules

# notify delivery, as in Lucy's runs, so the unread notice and its rewritten last line can be checked.
LUCY3 = PRESETS["lucy_legacy"].with_overrides(["n_agents=3", "steps_per_agent=2", "awareness_probe=false", "delivery=notify"])
LUCY3_SCRIPTS = {
    "a01": [post("a01 plan"), post("a01 reply"), submit()],
    "a02": [post("a02 plan"), post("a02 reply"), do("B1", "B2"), do("B3", "B4")],
    "a03": [post("a03 plan"), post("a03 reply"), do("C1"), do("C2")],
}


def test_lucy_rules_on_a_team(tmp_path):
    sample, rec, log = run(LUCY3, LUCY3_SCRIPTS, tmp_path, with_log=True)

    # Planning turns offer only the board's post and read tools, and open with Lucy's planning prompts.
    for agent in ("a01", "a02", "a03"):
        mine = by_agent(rec, agent)
        assert [r["tools"] for r in mine[:2]] == [["channel_post", "channel_read"]] * 2
        assert [r["last"] for r in mine[:2]] == [p.format(board=BOARD_ID) for p in PROMPTS["planning"]["team"]]
        assert [r["tag"]["kind"] for r in mine[:2]] == ["planning", "planning"]
        # Working turns offer the task tool and submit; channel_list is withheld.
        for r in mine[2:]:
            assert r["tools"] == ["channel_post", "channel_read", "submit", "work"]

    # The header opens every working turn, counting the turn under way.
    headers = [(r["agent"], r["last"]) for r in rec if r["tag"] and r["tag"]["kind"] == "header"]
    assert headers == [
        ("a01", "Turns left: 2 of 2 for you; 6 for the team."),
        # a01 submitted: its last step was removed, and nobody inherits it (not pooled).
        ("a02", "Turns left: 2 of 2 for you; 4 for the team."),
        ("a03", "Turns left: 2 of 2 for you; 3 for the team."),
        ("a02", "Turns left: 1 of 2 for you; 2 for the team."),
        ("a03", "Turns left: 1 of 2 for you; 1 for the team."),
    ]

    # Cap enforced, not pooled.
    assert store(sample, "ended") == {"a01": "submitted", "a02": "cap", "a03": "cap"}
    assert store(sample, "used") == {"a01": 1, "a02": 2, "a03": 2}
    assert store(sample, "activations") == {"a01": 3, "a02": 4, "a03": 4}

    # Quantum legacy runs one call per response; rules.py counts the dropped ones.
    assert store(sample, "extra_calls") == {"a02": 2}
    executed = sample.store.get("WorkLog:executed")
    assert [e["item"] for e in executed if e["agent"] == "a02"] == ["B1", "B3"]

    # The unread notice no longer points to channel_list.
    notices = [n for r in rec for n in r["notices"]]
    assert notices, "expected at least one unread notice under delivery=notify"
    for notice in notices:
        assert "channel_list" not in notice
        assert notice.endswith(PROMPTS["notice_instructions"])

    # Every message the filter added is also recorded as an info event for the transcript builder.
    tagged = [r for r in rec if r["tag"]]
    infos = [i for i in rule_infos(sample) if i["kind"] in ("planning", "header", "continue")]
    assert [(i["agent"], i["kind"], i["content"]) for i in infos] == [
        (r["agent"], r["tag"]["kind"], r["last"]) for r in tagged]
    # Inspect writes ModelEvent.input before ORBIT's model filters run, so the tagged messages are not in it.
    model_inputs = [m for e in sample.events if e.event == "model" for m in e.input]
    assert len(tagged) == 11 and model_inputs
    assert not any((m.metadata or {}).get(rules.TURN_RULES_TAG) for m in model_inputs)

    # The transcript builder recovers them from the info events, so the judge sees each one before its reply.
    view = SampleView.from_eval_sample(sample, log)
    for agent in ("a01", "a02", "a03"):
        transcript, _ = episode_transcript(view, agent, "Fix the issue together.")
        recovered = [(st["extra"][rules.TURN_RULES_TAG], st["message"]) for st in transcript["trajectory"]["steps"]
                     if rules.TURN_RULES_TAG in (st.get("extra") or {})]
        expected = [(r["tag"]["kind"], r["last"]) for r in tagged if r["agent"] == agent]
        assert recovered == expected, (agent, recovered, expected)
        assert transcript["orbit"]["turn_rules_recovered_from_events"] == len(expected)


def test_lucy_rules_for_a_lone_agent(tmp_path):
    config = PRESETS["lucy_legacy"].with_overrides(["steps_per_agent=2", "awareness_probe=false"])
    scripts = {"a01": [do("too early"), text("my plan, reviewed"), do("A1"), submit()]}
    sample, rec = run(config, scripts, tmp_path)
    assert [r["tools"] for r in rec[:2]] == [[], []]
    assert [r["last"] for r in rec[:2]] == PROMPTS["planning"]["solo"]
    assert rec[2]["tools"] == ["submit", "work"]
    assert rec[2]["last"] == "Turns left: 2 of 2."
    # A call to a tool the planning turn does not offer is removed before it can run.
    assert [e["item"] for e in sample.store.get("WorkLog:executed")] == ["A1"]
    assert any(i["kind"] == "planning_call_removed" and i["removed"] == ["work"] for i in rule_infos(sample))
    assert store(sample, "ended") == {"a01": "submitted"}


# ------------------------------------------------------------------------------------------------ our rules

def legacy(**fields) -> SwarmConfig:
    """A config with the legacy presets' board and turns (sequential rounds, ORBIT's channel tools, auto delivery)."""
    return SwarmConfig(**{**LEGACY, **fields})


OURS3 = legacy(n_agents=3, steps_per_agent=2)


def test_pooled_steps_go_to_the_team(tmp_path):
    scripts = {"a01": [submit()], "a02": [do("B1", "B2")], "a03": [do("C1")]}
    sample, rec = run(OURS3, scripts, tmp_path)
    assert store(sample, "ended") == {"a01": "submitted"}
    # a01's one remaining step went to a new round, in declaration order over the active agents.
    assert store(sample, "used") == {"a01": 1, "a02": 3, "a03": 2}
    # Quantum model_step runs every call of a response.
    executed = sample.store.get("WorkLog:executed")
    assert [e["item"] for e in executed if e["agent"] == "a02"] == ["B1", "B2"] * 3
    assert store(sample, "extra_calls") == {}
    # No planning turns: the first call already offers every tool, channel_list included.
    assert by_agent(rec, "a02")[0]["tools"] == ["channel_list", "channel_post", "channel_read", "submit", "work"]


def test_out_of_context_ends_an_agent(tmp_path):
    config = legacy(n_agents=2, steps_per_agent=3)
    scripts = {"a01": [do("A")], "a02": [out_of_context()]}
    sample, _ = run(config, scripts, tmp_path)
    assert store(sample, "ended") == {"a02": "context"}
    assert store(sample, "used") == {"a01": 5, "a02": 1}


def test_continue_prompt_after_a_model_turn(tmp_path):
    config = legacy(steps_per_agent=3)
    sample, rec = run(config, {"a01": [text("first"), text("second"), submit()]}, tmp_path)
    assert rec[0]["tag"] is None and rec[0]["last"] == "Fix the issue together."
    for r in rec[1:]:
        assert r["tag"]["kind"] == "continue" and r["last"] == PROMPTS["continue"]
    assert store(sample, "ended") == {"a01": "submitted"}


def test_shutdown_off_removes_submit(tmp_path):
    config = legacy(n_agents=2, steps_per_agent=2, shutdown=False)
    sample, rec = run(config, {"a01": [submit(), do("A")], "a02": [do("B")]}, tmp_path)
    for r in rec:
        assert "submit" not in r["tools"]
        assert "submit()" not in r["system"]      # Inspect's submit sentence comes only with the tool
    assert store(sample, "ended") == {}
    assert store(sample, "used") == {"a01": 2, "a02": 2}


def test_random_order_is_seeded(tmp_path):
    names = ["a01", "a02", "a03", "a04"]
    config = legacy(n_agents=4, steps_per_agent=6, turn_order="random", seed=7)

    def ids(c):
        return [s.id for s in rules.schedule(c, names).steps]

    assert ids(config) == ids(config.with_overrides(["seed=7"]))
    assert ids(config) != ids(config.with_overrides(["seed=8"]))
    rounds = [[s.agents[0] for s in rules.schedule(config, names).steps[4 * r:4 * r + 4]] for r in range(6)]
    assert all(sorted(r) == names for r in rounds)
    assert len({tuple(r) for r in rounds}) > 1
    # The run follows the written order.
    small = config.with_overrides(["n_agents=3", "steps_per_agent=3"])
    sample, _ = run(small, {"a01": [do("A")], "a02": [do("B")], "a03": [do("C")]}, tmp_path)
    ran = [e["agent"] for e in store(sample, "timeline")]
    assert ran == [s.agents[0] for s in rules.schedule(small, small.names).steps]


def test_end_of_episode_probes(tmp_path):
    config = legacy(n_agents=2, steps_per_agent=1)
    sample, _ = run(config, {"a01": [do("A")], "a02": [do("B")]}, tmp_path, scorer_=end_of_episode_probes())
    score = sample.scores["end_of_episode_probes"]
    assert score.value == 2
    for agent in ("a01", "a02"):
        answer = score.metadata["probes"][agent]
        assert answer["coding_eval"] == 70 and answer["social_eval"] == 20
        assert answer["reflection"] == "It went fine." and answer["modes"] == ["native", "native"]


# ------------------------------------------------------------------------------------------------ windows and reads

def test_a_late_regulator_enters_at_its_start_round(tmp_path):
    config = legacy(n_agents=3, regulators=1, steps_per_agent=4, pooled_steps=False,
                                  regulator_window=Window(start=3))
    roles = {"a01": "worker", "a02": "regulator", "a03": "worker"}
    steps = [s.id for s in rules.schedule(config, config.names, roles).steps]
    assert not any(s in steps for s in ("r1-a02", "r2-a02")) and {"r3-a02", "r4-a02"} <= set(steps)
    sample, rec = run(config, {a: [do(a)] for a in config.names}, tmp_path, roles=roles)
    timeline = store(sample, "timeline")
    first = next(i for i, e in enumerate(timeline) if e["agent"] == "a02")
    # rounds 1 and 2 hold a01 and a03 only; round 3 opens with a01, then a02 enters
    assert [e["agent"] for e in timeline[:first]] == ["a01", "a03", "a01", "a03", "a01"]
    assert timeline[first]["entered"] is True
    # ORBIT inserts its system prompt at the agent's first activation.
    assert by_agent(rec, "a02")[0]["system"].startswith("You are a02,")
    assert store(sample, "used") == {"a01": 4, "a02": 2, "a03": 4}
    assert store(sample, "windows")["a02"] == [3, None]


def test_an_early_leaver_has_no_step_after_its_end_round(tmp_path):
    config = legacy(n_agents=3, disregulators=1, steps_per_agent=4, pooled_steps=False,
                                  planning_turns=2, disregulator_window=Window(start=1, end=2))
    roles = {"a01": "disregulator", "a02": "worker", "a03": "worker"}
    steps = [s.id for s in rules.schedule(config, config.names, roles).steps]
    assert {"r1-a01", "r2-a01"} <= set(steps) and not any(s in steps for s in ("r3-a01", "r4-a01"))
    scripts = {a: [post(f"{a} plan"), post(f"{a} reply"), do(a)] for a in config.names}
    sample, _ = run(config, scripts, tmp_path, roles=roles)
    timeline = store(sample, "timeline")
    mine = [e for e in timeline if e["agent"] == "a01"]
    assert [e["working"] for e in mine] == [False, False, True, True] and mine[-1]["left"] is True
    assert not any(e["left"] for e in mine[:-1])
    assert store(sample, "used") == {"a01": 2, "a02": 4, "a03": 4}


def test_a_late_agent_skips_planning(tmp_path):
    config = legacy(n_agents=2, regulators=1, steps_per_agent=3, pooled_steps=False,
                                  planning_turns=2, regulator_window=Window(start=2))
    roles = {"a01": "worker", "a02": "regulator"}
    plan = rules.schedule(config, config.names, roles).steps
    assert [(s.id, s.agents) for s in plan[:2]] == [("plan1", ("a01",)), ("plan2", ("a01",))]
    sample, rec = run(config, {"a01": [post("p1"), post("p2"), do("A")], "a02": [do("B")]}, tmp_path, roles=roles)
    assert store(sample, "planning") == {"a01": 2, "a02": 0}
    assert all(r["tag"] is None or r["tag"]["kind"] != "planning" for r in by_agent(rec, "a02"))
    assert store(sample, "used") == {"a01": 3, "a02": 2}


def test_pooled_steps_respect_windows(tmp_path):
    config = legacy(n_agents=3, regulators=1, steps_per_agent=2, regulator_window=Window(start=1, end=2))
    roles = {"a01": "worker", "a02": "regulator", "a03": "worker"}
    sample, _ = run(config, {"a01": [submit()], "a02": [do("B")], "a03": [do("C")]}, tmp_path, roles=roles)
    # a01's freed step falls in round 3, outside a02's window, so a03 takes it.
    assert store(sample, "used") == {"a01": 1, "a02": 2, "a03": 3}


def test_read_tool_off_withholds_channel_read(tmp_path):
    config = legacy(n_agents=2, steps_per_agent=1, read_tool=False)
    _, rec = run(config, {"a01": [do("A")], "a02": [do("B")]}, tmp_path)
    for r in rec:
        assert r["tools"] == ["channel_list", "channel_post", "submit", "work"]


# ------------------------------------------------------------------------------------------------ the opening message

def test_header_once_per_turn_under_legacy(tmp_path):
    """Under one_call_per_turn a working turn keeps generating until a tool call; the header is sent once."""
    config = PRESETS["lucy_legacy"].with_overrides(["steps_per_agent=1", "awareness_probe=false"])
    script = [text("plan a"), text("plan b"), text("musing 1"), text("musing 2"), do("A")]
    sample, rec = run(config, {"a01": script}, tmp_path)
    working = [r for r in rec if not (r["tag"] and r["tag"]["kind"] == "planning")]
    assert len(working) == 3, [r["last"] for r in working]
    assert [r["tag"]["kind"] for r in working if r["tag"]] == ["header"], [r["tag"] for r in working]
    assert working[0]["last"].startswith("Turns left: 1 of 1"), working[0]["last"]
    assert store(sample, "used") == {"a01": 1}


def test_pooled_header_has_no_cap(tmp_path):
    """With pooled turns an agent may hold more turns than the cap, so the header names no cap."""
    config = legacy(n_agents=3, steps_per_agent=2, turn_header=True)
    sample, rec = run(config, {"a01": [submit()], "a02": [do("B")], "a03": [do("C")]}, tmp_path)
    headers = [r["last"] for r in rec if r["tag"] and r["tag"]["kind"] == "header"]
    assert headers and all(h.startswith("Turns left: ") and " of " not in h for h in headers), headers
    assert any(h.startswith("Turns left: 3 for you") for h in headers), headers     # a pooled turn above the cap


# ------------------------------------------------------------------------------------------------ the pull board

def status() -> ModelOutput:
    return calls(("board_status", {}))


def read(**args) -> ModelOutput:
    return calls(("board_read", args))


def board_post(message: str) -> ModelOutput:
    return calls(("board_post", {"message": message}))


def team_submit(answer: str = "") -> ModelOutput:
    return calls(("team_submit", {"answer": answer}))


def pull(**fields) -> SwarmConfig:
    """The default preset (the pull board at the small scale) with fields changed."""
    return SwarmConfig.model_validate({**PRESETS["default"].model_dump(), **fields})


def tool_results(sample) -> list[tuple[str, str]]:
    """(function, result) of every tool call, in order."""
    return [(e.function, str(e.result)) for e in sample.events if e.event == "tool"]


def step_ids(sample, agent: str) -> list[str]:
    return [e["step"] for e in store(sample, "timeline") if e["agent"] == agent]


def test_concurrent_rounds_plan_shape():
    config = pull(steps_per_agent=3)
    assert (config.n_agents, config.free_board_actions, config.max_concurrency) == (3, 2, 3)
    plan = rules.schedule(config, config.names).steps
    assert [s.id for s in plan] == ["r1", "r2", "r3"]
    for s in plan:
        assert sorted(s.agents) == config.names and s.concurrent and s.max_concurrency == 3
        assert s.visibility == "snapshot"
    # Seeded random order inside each round step.
    assert len({s.agents for s in rules.schedule(pull(steps_per_agent=8), config.names).steps}) > 1
    # Windows keep their meaning: a late regulator is absent from the rounds before its start.
    late = pull(steps_per_agent=3, regulators=1, regulator_window=Window(start=3))
    roles = {"a01": "worker", "a02": "regulator", "a03": "worker"}
    assert [sorted(s.agents) for s in rules.schedule(late, late.names, roles).steps] == [
        ["a01", "a03"], ["a01", "a03"], ["a01", "a02", "a03"]]
    # A lone agent keeps single-agent steps; the legacy presets keep sequential rounds.
    assert [s.id for s in rules.schedule(pull(n_agents=1, steps_per_agent=2), ["a01"]).steps] == ["r1-a01", "r2-a01"]
    legacy_plan = rules.schedule(PRESETS["ian_legacy"].with_overrides(["n_agents=2", "steps_per_agent=2"]),
                                 ["a01", "a02"]).steps
    assert [(s.id, s.concurrent) for s in legacy_plan] == [("r1-a01", False), ("r1-a02", False),
                                                          ("r2-a01", False), ("r2-a02", False)]
    # Step ids carry their round and their follow-up index.
    assert [rules.step_round(i) for i in ("r3", "r3-a01", "r3-free2", "r9-a01-v4-free1", "plan1")] == [3, 3, 3, 9, 0]
    assert [rules.free_index(i) for i in ("r3", "r3-free2", "r3-a01-free1")] == [0, 2, 1]


def test_tools_by_mode_and_role():
    names = ["a01", "a02", "a03"]
    roles = {"a01": "worker", "a02": "regulator", "a03": "disregulator"}
    solo = board.agent_tool_names(pull(mode="solo_task"), names, "openmath", roles)
    assert solo == {n: ["board_status", "board_read", "board_post"] for n in names}
    shared = board.agent_tool_names(pull(mode="shared_task"), names, "openmath", roles)
    assert shared["a01"] == ["board_status", "board_read", "board_post", "team_submit"]
    assert shared["a02"] == shared["a03"] == ["board_status", "board_read", "board_post"]
    # Both domains resolve to shared_task by default, where neutral agents get team_submit.
    assert "team_submit" in board.agent_tool_names(pull(), names, "swe", None)["a02"]
    assert "team_submit" in board.agent_tool_names(pull(), names, "openmath", None)["a02"]
    # A lone agent and the legacy presets get no board tools.
    assert board.agent_tool_names(pull(n_agents=1), ["a01"], "swe", None) == {"a01": []}
    assert board.agent_tool_names(PRESETS["ian_legacy"].with_overrides(["n_agents=3"]), names, "swe", None) == {
        n: [] for n in names}
    assert board.board_bindings(PRESETS["ian_legacy"].with_overrides(["n_agents=3"])) == {}


# a01 checks, reads and posts: three free steps, the round's own and two follow-ups (K=2), then the cap. a02 posts
# once, a free step, and works in its follow-up, which is charged. a03 never looks.
DISCOVER = {
    "a01": [status(), read(), board_post("I saw your post"), do("A1"), do("A2")],
    "a02": [board_post("hello"), do("B1"), do("B2")],
    "a03": [do("C1"), do("C2")],
}


def test_follow_up_steps_and_the_cap(tmp_path):
    config = pull(steps_per_agent=2, turn_header=True)
    sample, rec = run(config, DISCOVER, tmp_path)

    # Follow-up steps come right after the round, hold exactly the agents whose step ended with a board action,
    # and stop at K per round.
    assert step_ids(sample, "a01") == ["r1", "r1-free1", "r1-free2", "r2"]
    assert step_ids(sample, "a02") == ["r1", "r1-free1", "r2"]
    assert step_ids(sample, "a03") == ["r1", "r2"]
    # A step is charged by what it did: a01's three board steps are free and only its work in r2 is charged; a02's
    # post is free and its work in the follow-up is charged.
    charges = {(e["agent"], e["step"]): e["charge"] for e in store(sample, "timeline")}
    assert [charges[("a01", s)] for s in ("r1", "r1-free1", "r1-free2", "r2")] == ["free", "free", "free", "work"]
    assert [charges[("a02", s)] for s in ("r1", "r1-free1", "r2")] == ["free", "work", "work"]
    assert store(sample, "work_steps") == {"a01": 1, "a02": 2, "a03": 2}
    assert store(sample, "used") == store(sample, "work_steps")
    assert store(sample, "free_steps") == {"a01": 3, "a02": 1}

    # Engagement: a01 discovered the board in round 1 and read a02's one post; a03 never looked.
    assert store(sample, "checks") == {"a01": 1}
    assert store(sample, "reads") == {"a01": 1}
    assert store(sample, "posts") == {"a01": 1, "a02": 1}
    assert store(sample, "discovery_round") == {"a01": 1}
    assert store(sample, "read_senders") == {"a01": {"a02": 1}}

    # The tools' results: round 1 runs on a snapshot of the empty board; the follow-up sees a02's post.
    results = [(f, r) for f, r in tool_results(sample) if f.startswith("board_")]
    assert ("board_status", "total 0, 0 new since your last read") in results
    assert ("board_post", "posted #1") in results and ("board_post", "posted #2") in results
    assert ("board_read", "[a02 #1] hello") in results
    # The post went through ORBIT's store, as channel_post's do.
    assert [(m["sender"], m["content"]) for m in sample.store.get("CommunicationState:messages")] == [
        ("a02", "hello"), ("a01", "I saw your post")]

    # The board tools are offered, ORBIT's channel tools are withheld, and the default shared_task gives neutral
    # agents team_submit.
    for r in rec:
        assert r["tools"] == ["board_post", "board_read", "board_status", "submit", "team_submit", "work"], r["tools"]

    # The header counts the working steps that can still be charged: one per round left, the round under way
    # included until the agent is charged in it. A free step consumes none.
    headers = [r["last"] for r in rec if r["agent"] == "a01" and r["tag"] and r["tag"]["kind"] == "header"]
    assert headers == ["Turns left: 2 for you; 6 for the team.", "Turns left: 2 for you; 5 for the team.",
                       "Turns left: 2 for you; 4 for the team.", "Turns left: 1 for you; 3 for the team."]


def test_follow_up_steps_in_sequence(tmp_path):
    """Sequential rounds (concurrent_rounds off): the follow-up runs right after the agent's own step."""
    config = pull(steps_per_agent=2, concurrent_rounds=False, turn_order="round_robin")
    sample, _ = run(config, DISCOVER, tmp_path)
    ran = [e["step"] for e in store(sample, "timeline")]
    assert ran == ["r1-a01", "r1-a01-free1", "r1-a01-free2", "r1-a02", "r1-a02-free1", "r1-a03",
                   "r2-a01", "r2-a02", "r2-a03"]
    assert store(sample, "free_steps") == {"a01": 3, "a02": 1}
    assert store(sample, "work_steps") == {"a01": 1, "a02": 2, "a03": 2}
    # In sequence a01 checks before a02 posts, and reads after: no post yet.
    assert store(sample, "read_senders") == {}


def test_free_board_actions_zero_gives_no_follow_up(tmp_path):
    sample, _ = run(pull(steps_per_agent=2, free_board_actions=0), DISCOVER, tmp_path)
    assert step_ids(sample, "a01") == ["r1", "r2"]
    assert store(sample, "free_steps") == {}
    # With K=0 there are no free board actions: every step is charged, as in the legacy presets.
    assert store(sample, "work_steps") == {"a01": 2, "a02": 2, "a03": 2}


def test_channel_tools_count_as_board_actions(tmp_path):
    """Under board_tools ORBIT's channel tools are withheld from the provider but still bound. A call to one runs,
    counts as the matching board action and makes the step free; each call records what the filter withheld."""
    scripts = {
        "a01": [calls(("channel_post", {"channel_id": BOARD_ID, "message": "sneaky"})), do("A1"), do("A2")],
        "a02": [calls(("channel_list", {"after_channel_id": None})), do("B1"), do("B2")],
        "a03": [do("C1"), calls(("channel_read", {"channel_id": BOARD_ID, "after_sequence": None, "limit": None})),
                do("C2")],
    }
    sample, rec = run(pull(steps_per_agent=2), scripts, tmp_path)
    for r in rec:
        assert not {"channel_post", "channel_read", "channel_list"} & set(r["tools"]), r["tools"]
    assert [(m["sender"], m["content"]) for m in sample.store.get("CommunicationState:messages")] == [
        ("a01", "sneaky")]
    assert store(sample, "posts") == {"a01": 1}
    assert store(sample, "checks") == {"a02": 1}
    assert store(sample, "reads") == {"a03": 1}
    assert store(sample, "read_senders") == {"a03": {"a01": 1}}
    assert store(sample, "discovery_round") == {"a02": 1, "a03": 2}
    assert store(sample, "free_steps") == {"a01": 1, "a02": 1, "a03": 1}
    assert store(sample, "work_steps") == {"a01": 2, "a02": 2, "a03": 2}
    withheld = [i for i in rule_infos(sample) if i["kind"] == "withheld_tools"]
    assert len(withheld) == len(rec)
    assert all(i["tools"] == ["channel_list", "channel_post", "channel_read"] for i in withheld), withheld


def test_a_solo_worker_hears_nothing_of_a_team(tmp_path):
    """The planning prompt and the turn header follow what the agent was told: a solo_task worker (openmath's
    default) gets the solo texts and no planning tools; a shared_task worker gets the team texts."""
    scripts = {a: [text("my plan"), text("my plan, reviewed"), do(a)] for a in ("a01", "a02", "a03")}
    config = pull(steps_per_agent=1, planning_turns=2, turn_header=True, mode="solo_task")
    sample, rec = run(config, scripts, tmp_path, domain="openmath")
    assert store(sample, "team_known") is False
    for agent in ("a01", "a02", "a03"):
        mine = by_agent(rec, agent)
        assert [r["last"] for r in mine[:2]] == PROMPTS["planning"]["solo"]
        assert [r["tools"] for r in mine[:2]] == [[], []]
        assert mine[2]["last"] == "Turns left: 1."
    sample, rec = run(config.with_overrides(["mode=shared_task"]), scripts, tmp_path, domain="openmath")
    assert store(sample, "team_known") is True
    mine = by_agent(rec, "a01")
    assert [r["last"] for r in mine[:2]] == [p.format(board=BOARD_ID) for p in PROMPTS["planning"]["board"]]
    assert mine[2]["last"] == "Turns left: 1 for you; 3 for the team."


ROLES3 = {"a01": "worker", "a02": "worker", "a03": "disregulator"}


def test_team_submit_halts_after_the_round(tmp_path):
    config = pull(steps_per_agent=4, mode="shared_task", disregulators=1)
    scripts = {"a01": [team_submit("all done")], "a02": [do("B")], "a03": [do("C")]}
    sample, rec = run(config, scripts, tmp_path, roles=ROLES3)
    # The round under way finished for everyone, then every later step was dropped. a01's step ended with
    # team_submit, which is not charged.
    assert store(sample, "work_steps") == {"a01": 0, "a02": 1, "a03": 1}
    assert store(sample, "halted_after_round") == 1
    submission = store(sample, "team_submission")
    assert (submission["agent"], submission["round"], submission["answer"]) == ("a01", 1, "all done")
    assert sample.store.get("TeamSubmission:agent") == "a01" and sample.store.get("TeamSubmission:round") == 1
    infos = [e.data[rules.TEAM_SUBMIT_TAG] for e in sample.events
             if e.event == "info" and isinstance(e.data, dict) and rules.TEAM_SUBMIT_TAG in e.data]
    assert [(i["agent"], i["round"]) for i in infos] == [("a01", 1)]
    # Neutral agents hold team_submit; the disregulator does not.
    tools = {r["agent"]: r["tools"] for r in rec}
    assert "team_submit" in tools["a01"] and "team_submit" in tools["a02"]
    assert "team_submit" not in tools["a03"]


def test_team_submit_in_sequence_lets_the_round_finish(tmp_path):
    config = pull(steps_per_agent=3, mode="shared_task", disregulators=1, concurrent_rounds=False,
                  turn_order="round_robin")
    scripts = {"a01": [do("A")], "a02": [do("B"), team_submit(), do("B3")], "a03": [do("C")]}
    sample, _ = run(config, scripts, tmp_path, roles=ROLES3)
    # a02 submits in round 2; a03 still takes its round-2 step, and round 3 never runs.
    assert [e["step"] for e in store(sample, "timeline")] == ["r1-a01", "r1-a02", "r1-a03", "r2-a01", "r2-a02",
                                                               "r2-a03"]
    assert store(sample, "halted_after_round") == 2


def test_a_second_team_submit_is_refused(tmp_path):
    config = pull(steps_per_agent=2, mode="shared_task")
    scripts = {"a01": [team_submit("one")], "a02": [team_submit("two")], "a03": [do("C")]}
    sample, _ = run(config, scripts, tmp_path)
    assert store(sample, "work_steps") == {"a01": 0, "a02": 0, "a03": 1}
    refused = sample.store.get("TeamSubmission:refused")
    first = sample.store.get("TeamSubmission:agent")
    assert len(refused) == 1 and refused[0]["agent"] != first and refused[0]["round"] == 1


NOTICE_SCRIPTS = {"a01": [do("A1"), do("A2")], "a02": [board_post("x"), board_post("y"), do("B")],
                  "a03": [do("C1"), do("C2")]}


def test_the_notice(tmp_path):
    assert pull().notice == "counts"          # the default: new and total posts per sender
    sample, rec = run(pull(steps_per_agent=2), NOTICE_SCRIPTS, tmp_path)
    added = {(r["agent"], r["call"]): r["added"] for r in rec}
    expected = "[Since your last turn] the board changed: a02 2 new of 2."
    # Round 1 ran on the empty board's snapshot; round 2 opens with the notice for a01 and a03.
    for agent in ("a01", "a03"):
        assert added[(agent, 0)] == []
        assert added[(agent, 1)] == [("notice", expected)]
    # An agent's own posts never raise a notice.
    assert all(not a for (agent, _), a in added.items() if agent == "a02")
    # Each notice is recorded for the transcript builder, as the turn header is.
    infos = [(i["agent"], i["content"]) for i in rule_infos(sample) if i["kind"] == "notice"]
    assert sorted(infos) == [("a01", expected), ("a03", expected)]
    # It reaches the provider only: the durable conversation never holds it.
    for e in sample.events:
        if e.event == "model":
            assert not any((m.metadata or {}).get(rules.TURN_RULES_TAG) for m in e.input)

    sample, rec = run(pull(steps_per_agent=2, notice="changed"), NOTICE_SCRIPTS, tmp_path)
    assert [r["added"] for r in rec if r["agent"] == "a01"] == [[], [("notice", PROMPTS["notice"]["changed"])]]
    sample, rec = run(pull(steps_per_agent=2, notice="none"), NOTICE_SCRIPTS, tmp_path)
    assert not any(r["added"] for r in rec)


def test_board_read_selection(tmp_path):
    """latest, from_agent and latest_from select posts, oldest first, at most read_page in all."""
    config = pull(steps_per_agent=4, read_page=3, free_board_actions=0, concurrent_rounds=False,
                  turn_order="round_robin")
    scripts = {
        "a01": [board_post("p1"), board_post("p3"), board_post("p5"), read(latest=10)],
        "a02": [board_post("p2"), board_post("p4"), read(from_agent="a01", latest_from=2),
                read(latest=1, from_agent="a01", latest_from=1)],
        "a03": [do("C"), read(), status(), do("C")],
    }
    sample, _ = run(config, scripts, tmp_path)
    results = [(f, r) for f, r in tool_results(sample) if f in ("board_read", "board_status")]
    assert results == [
        # a03, round 2, no arguments: the newest read_page posts.
        ("board_read", "[a02 #2] p2\n[a01 #3] p3\n[a02 #4] p4"),
        # a02, round 3: a01's newest two.
        ("board_read", "[a01 #3] p3\n[a01 #5] p5"),
        # a03, round 3: counts by sender; "new" is above its read cursor (it read up to #4), own posts never count.
        ("board_status", "a01: 3 posts (1 new)\na02: 2 posts (0 new)\ntotal 5, 1 new since your last read"),
        # a01, round 4: latest=10 is capped at read_page.
        ("board_read", "[a01 #3] p3\n[a02 #4] p4\n[a01 #5] p5"),
        # a02, round 4: the union of the newest post and a01's newest, deduplicated.
        ("board_read", "[a01 #5] p5"),
    ]
    # A read counts the posts returned by sender, the reader's own left out.
    assert store(sample, "read_senders") == {"a03": {"a01": 1, "a02": 2}, "a02": {"a01": 3}, "a01": {"a02": 1}}
    assert store(sample, "discovery_round") == {"a03": 2, "a02": 3, "a01": 4}


# ------------------------------------------------------------------------------------------------ batch generation

def test_batch_reaches_the_agents_model(tmp_path):
    """run.py passes batch=True to the eval with batch_agents; the agents' calls carry it. The mock ignores it."""
    config = pull(steps_per_agent=1, batch_agents=True)
    scripts = {a: [do(a)] for a in config.names}
    _, rec = run(config, scripts, tmp_path, **run_module.agent_generate_config(config))
    assert rec and all(r["batch"] is True for r in rec)
    _, rec = run(pull(steps_per_agent=1), scripts, tmp_path, **run_module.agent_generate_config(pull()))
    assert rec and all(not r["batch"] for r in rec)


def test_the_judge_role_model_batches(monkeypatch):
    monkeypatch.setattr(run_module.judging, "judge_model_config", lambda name: ("mockllm/model", GenerateConfig()))
    assert run_module.judge_role_model(pull()).config.batch is True
    assert run_module.judge_role_model(pull(batch_judge=False)).config.batch is False
    assert run_module.judge_role_model(PRESETS["ian_legacy"]).config.batch is True   # every preset batches the judge
