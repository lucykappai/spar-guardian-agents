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
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Target, scorer  # noqa: E402
from inspect_ai.tool import Tool, ToolCall, tool  # noqa: E402
from inspect_ai.util import StoreModel, store_as  # noqa: E402
from orbit import AgentSpec, SetupConfig, ToolFactory, multi_agent_task  # noqa: E402
from orbit.configs.experiment import ExperimentConfig  # noqa: E402
from pydantic import Field  # noqa: E402

from src.benchmark.transcript import SampleView, episode_transcript  # noqa: E402
from src.configuration.settings import load_prompts  # noqa: E402
from src.configuration.swarm_config import PRESETS, SwarmConfig, Window  # noqa: E402
from src.swarm import rules  # noqa: E402
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
        })
        script = scripts[agent]
        return script[min(k, len(script) - 1)]

    model = get_model(MODEL, custom_outputs=outputs, memoize=False)

    async def count_text_tokens(text: str) -> int:   # avoids the mock provider's tokenizer download
        return max(1, (len(text) + 3) // 4)

    model.api.count_text_tokens = count_text_tokens
    return model


# ------------------------------------------------------------------------------------------------ the task

def build_task(config: SwarmConfig, *, scorer_=None, roles: dict[str, str] | None = None) -> Task:
    names = config.names
    agents = [AgentSpec(name=n, role="engineer", tools=["work"], max_messages=10_000, submit=config.shutdown,
                        system_prompt=f"You are {n}, an engineer. Call work to make progress.") for n in names]
    task = multi_agent_task(
        name="turn_rules_test",
        setup=SetupConfig(agents=agents),
        tool_bindings={"work": ToolFactory(lambda ctx: work(ctx.agent_name))},
        communication=board_communication(names, delivery=config.delivery, visibility=config.visibility)
        if config.board else None,
        execution=board_execution(quantum=config.quantum, plan=rules.schedule(config, names, roles),
                                  initial_context="sample_input"),
        scheduler=board_scheduler(max_turns=config.max_turns + 5, max_time_seconds=120),
        dataset=[Sample(id="issue", input="Fix the issue together.")],
        scorer=scorer_ or nothing(),
        task_setup=[rules.rules_setup(config, names, roles)],
        token_limit=1_000_000,
    )
    for sample in task.dataset:
        if not config.board:
            sample.metadata["experiment"]["communication"] = None   # a lone agent gets no channel tools
        sample.metadata["experiment"]["defenses"] = [rules.turn_rules_defense(config).model_dump(mode="json")]
        ExperimentConfig.model_validate(sample.metadata["experiment"])
    return task


def run(config: SwarmConfig, scripts: dict, tmp_path, *, scorer_=None, roles=None, with_log=False):
    recorder: list[dict] = []
    [log] = inspect_eval(build_task(config, scorer_=scorer_, roles=roles), model=scripted_model(scripts, recorder),
                         log_dir=str(tmp_path), display="none", log_realtime=False, max_retries=0)
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
LUCY3 = PRESETS["lucy"].with_overrides(["n_agents=3", "steps_per_agent=2", "awareness_probe=false", "delivery=notify"])
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
    config = PRESETS["lucy"].with_overrides(["steps_per_agent=2", "awareness_probe=false"])
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

OURS3 = SwarmConfig(n_agents=3, steps_per_agent=2)


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
    config = SwarmConfig(n_agents=2, steps_per_agent=3)
    scripts = {"a01": [do("A")], "a02": [out_of_context()]}
    sample, _ = run(config, scripts, tmp_path)
    assert store(sample, "ended") == {"a02": "context"}
    assert store(sample, "used") == {"a01": 5, "a02": 1}


def test_continue_prompt_after_a_model_turn(tmp_path):
    config = SwarmConfig(steps_per_agent=3)
    sample, rec = run(config, {"a01": [text("first"), text("second"), submit()]}, tmp_path)
    assert rec[0]["tag"] is None and rec[0]["last"] == "Fix the issue together."
    for r in rec[1:]:
        assert r["tag"]["kind"] == "continue" and r["last"] == PROMPTS["continue"]
    assert store(sample, "ended") == {"a01": "submitted"}


def test_shutdown_off_removes_submit(tmp_path):
    config = SwarmConfig(n_agents=2, steps_per_agent=2, shutdown=False)
    sample, rec = run(config, {"a01": [submit(), do("A")], "a02": [do("B")]}, tmp_path)
    for r in rec:
        assert "submit" not in r["tools"]
        assert "submit()" not in r["system"]      # Inspect's submit sentence comes only with the tool
    assert store(sample, "ended") == {}
    assert store(sample, "used") == {"a01": 2, "a02": 2}


def test_random_order_is_seeded(tmp_path):
    names = ["a01", "a02", "a03", "a04"]
    config = SwarmConfig(n_agents=4, steps_per_agent=6, turn_order="random", seed=7)

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
    config = SwarmConfig(n_agents=2, steps_per_agent=1)
    sample, _ = run(config, {"a01": [do("A")], "a02": [do("B")]}, tmp_path, scorer_=end_of_episode_probes())
    score = sample.scores["end_of_episode_probes"]
    assert score.value == 2
    for agent in ("a01", "a02"):
        answer = score.metadata["probes"][agent]
        assert answer["coding_eval"] == 70 and answer["social_eval"] == 20
        assert answer["reflection"] == "It went fine." and answer["modes"] == ["native", "native"]


# ------------------------------------------------------------------------------------------------ windows and reads

def test_a_late_regulator_enters_at_its_start_round(tmp_path):
    config = SwarmConfig(n_agents=3, regulators=1, steps_per_agent=4, pooled_steps=False,
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
    config = SwarmConfig(n_agents=3, disregulators=1, steps_per_agent=4, pooled_steps=False,
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
    config = SwarmConfig(n_agents=2, regulators=1, steps_per_agent=3, pooled_steps=False,
                                  planning_turns=2, regulator_window=Window(start=2))
    roles = {"a01": "worker", "a02": "regulator"}
    plan = rules.schedule(config, config.names, roles).steps
    assert [(s.id, s.agents) for s in plan[:2]] == [("plan1", ("a01",)), ("plan2", ("a01",))]
    sample, rec = run(config, {"a01": [post("p1"), post("p2"), do("A")], "a02": [do("B")]}, tmp_path, roles=roles)
    assert store(sample, "planning") == {"a01": 2, "a02": 0}
    assert all(r["tag"] is None or r["tag"]["kind"] != "planning" for r in by_agent(rec, "a02"))
    assert store(sample, "used") == {"a01": 3, "a02": 2}


def test_pooled_steps_respect_windows(tmp_path):
    config = SwarmConfig(n_agents=3, regulators=1, steps_per_agent=2, regulator_window=Window(start=1, end=2))
    roles = {"a01": "worker", "a02": "regulator", "a03": "worker"}
    sample, _ = run(config, {"a01": [submit()], "a02": [do("B")], "a03": [do("C")]}, tmp_path, roles=roles)
    # a01's freed step falls in round 3, outside a02's window, so a03 takes it.
    assert store(sample, "used") == {"a01": 1, "a02": 2, "a03": 3}


def test_read_tool_off_withholds_channel_read(tmp_path):
    config = SwarmConfig(n_agents=2, steps_per_agent=1, read_tool=False)
    _, rec = run(config, {"a01": [do("A")], "a02": [do("B")]}, tmp_path)
    for r in rec:
        assert r["tools"] == ["channel_list", "channel_post", "submit", "work"]


# ------------------------------------------------------------------------------------------------ the opening message

def test_header_once_per_turn_under_legacy(tmp_path):
    """Under one_call_per_turn a working turn keeps generating until a tool call; the header is sent once."""
    config = PRESETS["lucy"].with_overrides(["steps_per_agent=1", "awareness_probe=false"])
    script = [text("plan a"), text("plan b"), text("musing 1"), text("musing 2"), do("A")]
    sample, rec = run(config, {"a01": script}, tmp_path)
    working = [r for r in rec if not (r["tag"] and r["tag"]["kind"] == "planning")]
    assert len(working) == 3, [r["last"] for r in working]
    assert [r["tag"]["kind"] for r in working if r["tag"]] == ["header"], [r["tag"] for r in working]
    assert working[0]["last"].startswith("Turns left: 1 of 1"), working[0]["last"]
    assert store(sample, "used") == {"a01": 1}


def test_pooled_header_has_no_cap(tmp_path):
    """With pooled turns an agent may hold more turns than the cap, so the header names no cap."""
    config = SwarmConfig(n_agents=3, steps_per_agent=2, turn_header=True)
    sample, rec = run(config, {"a01": [submit()], "a02": [do("B")], "a03": [do("C")]}, tmp_path)
    headers = [r["last"] for r in rec if r["tag"] and r["tag"]["kind"] == "header"]
    assert headers and all(h.startswith("Turns left: ") and " of " not in h for h in headers), headers
    assert any(h.startswith("Turns left: 3 for you") for h in headers), headers     # a pooled turn above the cap
