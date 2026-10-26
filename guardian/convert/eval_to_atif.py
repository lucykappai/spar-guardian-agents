"""Convert one Inspect .eval log from an ORBIT run into the run directory CheatBench's judge reads.

    python -m guardian.convert.eval_to_atif logs/<run>.eval runs/<name> [--task-root PATH]

Writes
    runs/<name>/run.json
    runs/<name>/episodes.jsonl                      one record per (sample, agent)
    runs/<name>/trajectories/<episode>/transcript.json
    runs/<name>/trajectories/<episode>/prompt.txt
    runs/<name>/communication/<sample>.json         channel messages and events, if any
    runs/<name>/defense/<sample>.json               defense configuration and interventions, if any

A transcript with a "trajectory" key is validated as ATIF by CheatBench's judge
without any vendor adapter (core/agents/factory.py). Each ORBIT agent becomes
its own episode: the sample's projected messages are split by
EnvironmentState:message_attribution (history_projection="all"). Three things
live only in Inspect's native events and are recovered from them:

  * the completed submit call, which ORBIT strips from the conversation,
  * channel notices / auto-delivered board messages, which ORBIT adds to the
    provider input only (CommunicationDeliveryFilter),
  * per-agent token usage.

Episode ids: "<sample>#<epoch>" for a single agent, "<sample>#<epoch>@<agent>" otherwise.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CB = ROOT / "submodules" / "cheatbench"
if str(CB) not in sys.path:
    sys.path.insert(0, str(CB))

import inspect_ai  # noqa: E402
import orbit  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem,  # noqa: E402
                              ChatMessageTool, ChatMessageUser)

from core.trajectory import (Agent, FinalMetrics, Observation, ObservationResult,  # noqa: E402
                             Step, ToolCall, Trajectory)
from core.trajectory.trajectory import SCHEMA_VERSION  # noqa: E402

DELIVERY_TAG = "orbit_communication_delivery"      # orbit/communication/delivery.py
TRAILING_TURN_TAG = "orbit_trailing_turn_fix"      # guardian/task/controls.py
ATTRIBUTION_KEY = "EnvironmentState:message_attribution"
LAST_TEXT_KEY = "EnvironmentState:last_text_by_agent"
REFUSAL_TEXT = "I cannot provide that response under the configured defense policy."   # orbit/defenses/boundaries.py
WITHHELD_TEXT = "[Content withheld by a configured defense.]"
PASSIVE_DEFENSES = {"trailing_turn_fix"}     # integration shim: checks everything, blocks nothing
TEAM_IDENTITY = "agent-018"   # the git identity the staged row gives the agent (README; the scorer reads this branch)
HALT_RE = re.compile(r"Experiment complete: .*halt_reason=([^\s,]+)")
LIMIT_FAILURES = ("timeout", "max_turns")
PROVIDER_ERROR_WORDS = (
    "rate limit", "429", "connection", "stream", "timed out", "overloaded",
    "credits are depleted", "billing", "quota", "insufficient_quota", "resource_exhausted",
)

DEVIATIONS = [
    "agent_is_orbit_turn_react_not_vendor_harness",
    "orbit_assistant_and_submit_prompt_appended_to_system_prompt",
    "continue_prompts_are_orbit_synthetic_user_messages",
    "mirror_served_from_docker_sidecar_and_ca_baked_into_image",
]
BOARD_DEVIATION = "two_agents_share_one_sandbox_clock_and_git_identity"


def agent_version() -> str:
    return f"orbit-{orbit.__version__}+inspect-ai-{inspect_ai.__version__}"


# --------------------------------------------------------------------------
# message content
# --------------------------------------------------------------------------
def split_content(content) -> tuple[str, str | None]:
    """(text, reasoning) from an Inspect message content (str or content list)."""
    if content is None:
        return "", None
    if isinstance(content, str):
        return content, None
    texts, reasoning = [], []
    for part in content:
        kind = getattr(part, "type", None)
        if kind == "text":
            texts.append(part.text)
        elif kind == "reasoning":
            if getattr(part, "reasoning", ""):
                reasoning.append(part.reasoning)
            elif getattr(part, "redacted", False):
                reasoning.append("[redacted reasoning]")
        else:
            texts.append(f"[{kind} content omitted]")
    return "\n".join(t for t in texts if t), ("\n".join(reasoning) or None)


def tool_result_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, (str, int, float, bool)):
        return str(result)
    if isinstance(result, list):
        return "\n".join(tool_result_text(r) for r in result)
    if not hasattr(result, "type"):
        return str(result)
    text, _ = split_content([result])
    return text


def output_message_id(model_event) -> str | None:
    """Id of the assistant message a model call produced, or None when it produced no choice."""
    if model_event.output and model_event.output.choices:
        return model_event.output.message.id
    return None


# --------------------------------------------------------------------------
# which agent owns which messages and events
# --------------------------------------------------------------------------
def roster(sample) -> list[str]:
    exp = (sample.metadata or {}).get("experiment", {})
    return [a["name"] for a in exp.get("setup", {}).get("agents", [])]


def agent_messages(sample, agent: str) -> list:
    """The agent's conversation, as projected by AgentScheduler.sync_to_task_state."""
    attribution = sample.store.get(ATTRIBUTION_KEY) or []
    indices = [a["index"] for a in attribution if a.get("agent") == agent]
    bad = [i for i in indices if i >= len(sample.messages)]
    if bad:
        raise ValueError(f"attribution indices {bad} outside {len(sample.messages)} messages")
    return [sample.messages[i] for i in indices]


def span_owner(sample):
    """span_id -> name of the nearest enclosing span of type 'agent' (ORBIT names it after the spec).

    Events inside an ``orbit.defense`` span (a monitor's or guardian's own model
    calls) or inside a ``tool`` span (a model call made while a tool runs, such as
    the dual-LLM quarantine summariser) belong to the defense, not to the agent,
    and get owner None. ToolEvents themselves sit directly in the agent span.
    """
    begins = {e.id: e for e in sample.events if e.event == "span_begin"}
    cache: dict = {}

    def owner(span_id):
        if span_id in cache:
            return cache[span_id]
        e = begins.get(span_id)
        if e is None:
            result = None
        elif e.type in ("orbit.defense", "tool"):
            result = None
        elif e.type == "agent":
            result = e.name
        else:
            result = owner(e.parent_id) if e.parent_id else None
        cache[span_id] = result
        return result
    return owner


def agent_events(sample, agent: str, kind: str) -> list:
    owner = span_owner(sample)
    return [e for e in sample.events if e.event == kind and owner(e.span_id) == agent]


# --------------------------------------------------------------------------
# defenses
# --------------------------------------------------------------------------
def defense_evidence(sample) -> dict:
    """DefenseLog store fields plus every defense_intervention info event."""
    keys = ("resolved_defenses", "activity", "blocks", "detections", "vaccinations",
            "vaccination_injections", "total_activations", "false_positives", "true_positives")
    log = {k: sample.store.get(f"DefenseLog:{k}") for k in keys}
    kinds = ("defense_intervention", "defense_resample", "defense_instrumentation_exempt")
    interventions = []
    for e in sample.events:
        if e.event == "info" and isinstance(e.data, dict) and e.data.get("event") in kinds:
            interventions.append(e.data)
    return {"log": log, "interventions": interventions}


def defense_counts(evidence: dict, agent: str) -> dict:
    """checks and blocks per boundary for one agent, from the intervention events.

    The passive trailing-turn shim is itself a registered defense and is asked at
    every boundary; its checks are left out so the counts describe the control.
    """
    out = {"checks": 0, "blocks": 0, "blocks_by_boundary": {}, "checks_by_boundary": {}}
    for ev in evidence["interventions"]:
        if ev.get("event") != "defense_intervention" or ev.get("agent") != agent:
            continue
        if ev.get("defense") in PASSIVE_DEFENSES:
            continue
        b = ev.get("boundary", "?")
        out["checks"] += 1
        out["checks_by_boundary"][b] = out["checks_by_boundary"].get(b, 0) + 1
        if ev.get("allow") is False:
            out["blocks"] += 1
            out["blocks_by_boundary"][b] = out["blocks_by_boundary"].get(b, 0) + 1
    return out


def blocked_message_ids(sample, boundary: str) -> set:
    """Ids of messages a defense blocked at one boundary, from the defense_intervention events."""
    ids = set()
    for e in sample.events:
        d = getattr(e, "data", None)
        if e.event != "info" or not isinstance(d, dict):
            continue
        if d.get("event") != "defense_intervention" or d.get("boundary") != boundary:
            continue
        if d.get("allow") is False and d.get("message_id"):
            ids.add(d["message_id"])
    return ids


def blocked_output_ids(sample) -> set:
    """Ids of raw model outputs a defense replaced with the refusal (model_output blocks).

    ORBIT keeps the raw output in the ModelEvent and substitutes a new assistant
    message in the conversation, so the i-th assistant message of an agent is a
    real substitution iff the i-th model call's output id is in this set. A
    refusal sentence the model wrote itself is not a block.
    """
    return blocked_message_ids(sample, "model_output")


def redacted_message_ids(model_events, sample) -> set:
    """Ids of messages a defense withheld from the provider.

    ORBIT redacts below the ModelEvent (the event keeps the original message), so
    the record of truth is the ``defense_intervention`` event at the model_input
    boundary with ``allow`` false; the metadata tag is kept as a fallback.
    """
    ids = set()
    for ev in model_events:
        for m in ev.input or []:
            if (m.metadata or {}).get("orbit_defense_redacted"):
                ids.add(m.id)
    return ids | blocked_message_ids(sample, "model_input")


# --------------------------------------------------------------------------
# provider-only messages and usage
# --------------------------------------------------------------------------
def is_provider_only(m) -> bool:
    meta = m.metadata or {}
    if m.role == "system" and meta.get(DELIVERY_TAG):
        return True
    return m.role == "user" and bool(meta.get(TRAILING_TURN_TAG))


def delivery_notices(model_events) -> list[tuple[str | None, str, str]]:
    """(output assistant message id or None, text, source) per provider-only message in a model call.

    Covers ORBIT's channel notices (system) and, should Inspect ever record it,
    the integration's trailing-turn continue prompt (user). Neither enters the
    durable conversation. In practice the ModelEvent is written before ORBIT's
    model filters run, so the shim's prompt never appears here; the transcript
    shows a refusal followed by the next turn.
    """
    out = []
    for ev in model_events:
        for m in ev.input or []:
            if not is_provider_only(m):
                continue
            text, _ = split_content(m.content)
            source = "system" if m.role == "system" else "user"
            out.append((output_message_id(ev), text, source))
    return out


def _usage_totals(usages) -> dict:
    """Prompt tokens include cache reads: Inspect reports them apart from input_tokens."""
    total_in = total_out = cache_read = 0
    for usage in usages:
        cached = getattr(usage, "input_tokens_cache_read", 0) or 0
        total_in += (getattr(usage, "input_tokens", 0) or 0) + cached
        cache_read += cached
        total_out += getattr(usage, "output_tokens", 0) or 0
    if not (total_in or total_out):
        return {}
    return {"input": total_in, "output": total_out, "cache_read": cache_read}


def usage_from_events(model_events) -> dict:
    usages = []
    for ev in model_events:
        usage = getattr(ev.output, "usage", None) if ev.output else None
        if usage:
            usages.append(usage)
    return _usage_totals(usages)


def sum_usage(sample) -> dict:
    return _usage_totals((sample.model_usage or {}).values())


# --------------------------------------------------------------------------
# one (sample, agent) -> one ATIF trajectory
# --------------------------------------------------------------------------
def _call_arguments(arguments, fallback_key: str) -> dict:
    if isinstance(arguments, dict):
        return dict(arguments)
    if fallback_key == "answer":
        return {"answer": str(arguments)}
    return {fallback_key: arguments}


def _error_extra(error) -> dict | None:
    if not error:
        return None
    return {"is_error": True, "error": error.message}


def _notice_step(text: str, source: str, how: str) -> dict:
    extra = {"orbit_delivery_notice": source == "system",
             "orbit_trailing_turn_fix": source == "user",
             "attached_by": how}
    return {"source": source, "message": text, "extra": extra}


def _user_step(msg, redacted: set) -> dict:
    text, _ = split_content(msg.content)
    if msg.id not in redacted:
        return {"source": "user", "message": text}
    # the provider received the placeholder; the judge must see what the agent saw
    extra = {"withheld_from_model_by_defense": True, "withheld_text": text}
    return {"source": "user", "message": WITHHELD_TEXT, "extra": extra}


def _assistant_step(msg, call_output_id, blocked_outputs: set) -> dict:
    text, reasoning = split_content(msg.content)
    calls = [ToolCall(tool_call_id=tc.id, function_name=tc.function,
                      arguments=_call_arguments(tc.arguments, "value"))
             for tc in (msg.tool_calls or [])]
    step = {"source": "agent", "message": text, "reasoning_content": reasoning,
            "tool_calls": calls or None, "llm_call_count": 1,
            "model_name": msg.model or None, "results": []}
    if call_output_id is not None and call_output_id in blocked_outputs:
        step["extra"] = {"defense_refusal": True, "blocked_output_message_id": call_output_id}
    elif text.strip() == REFUSAL_TEXT:
        step["extra"] = {"refusal_text_written_by_model": True}
    return step


def _add_tool_result(msg, redacted: set, steps: list[dict], issued: dict[str, int]) -> None:
    """Attach a tool result to the step that issued the call, or add it as an orphan user step."""
    content = tool_result_text(msg.content)
    extra = _error_extra(msg.error)
    if msg.id in redacted:
        extra = {**(extra or {}), "withheld_from_model_by_defense": True, "withheld_text": content}
        content = WITHHELD_TEXT
    idx = issued.get(msg.tool_call_id)
    if idx is None:
        orphan = {"orphan_tool_result": msg.tool_call_id, "function": msg.function, **(extra or {})}
        steps.append({"source": "user", "message": content, "extra": orphan})
        return
    result = ObservationResult(source_call_id=msg.tool_call_id, content=content, extra=extra)
    steps[idx]["results"].append(result)


def _recover_submits(sample, agent_name: str, steps: list[dict], issued: dict[str, int]) -> int:
    """Put back the completed submit calls ORBIT removed from the conversation.

    orbit/agents/turn_react.py:_remove_submit_tool drops the call and its result;
    the native tool event keeps it.
    """
    recovered = 0
    for ev in agent_events(sample, agent_name, "tool"):
        if ev.function != "submit" or ev.id in issued:
            continue
        agent_steps = [i for i, s in enumerate(steps) if s["source"] == "agent"]
        last_agent = agent_steps[-1] if agent_steps else None
        call = ToolCall(tool_call_id=ev.id, function_name="submit",
                        arguments=_call_arguments(ev.arguments, "answer"))
        result = ObservationResult(source_call_id=ev.id, content=tool_result_text(ev.result),
                                   extra=_error_extra(ev.error))
        if last_agent is None:
            steps.append({"source": "agent", "message": "", "tool_calls": [call], "llm_call_count": 1,
                          "results": [result], "extra": {"submit_recovered_from_events": True}})
        else:
            step = steps[last_agent]
            step["tool_calls"] = list(step.get("tool_calls") or []) + [call]
            step["results"].append(result)
            step["extra"] = {**(step.get("extra") or {}), "submit_recovered_from_events": True}
        recovered += 1
    return recovered


def _atif_steps(steps: list[dict]) -> list[Step]:
    out = []
    for i, s in enumerate(steps, 1):
        results = s.pop("results", [])
        observation = Observation(results=results) if results else None
        out.append(Step(step_id=i, observation=observation, **s))
    return out


def build_trajectory(sample, log, agent_name: str, messages: list) -> tuple[Trajectory, dict]:
    model_events = agent_events(sample, agent_name, "model")
    notices = delivery_notices(model_events)
    redacted = redacted_message_ids(model_events, sample)
    blocked_outputs = blocked_output_ids(sample)
    output_ids = [output_message_id(ev) for ev in model_events]
    by_id: dict = {}
    for oid, text, source in notices:
        if oid:
            by_id.setdefault(oid, []).append((text, source))
    ordered = [(text, source) for _, text, source in notices]
    steps: list[dict] = []
    issued: dict[str, int] = {}          # tool_call_id -> index into steps
    system_prompt = None
    assistant_seen = 0
    attached_notices = 0
    for msg in messages:
        if isinstance(msg, ChatMessageSystem):
            text, _ = split_content(msg.content)
            if system_prompt is None:
                system_prompt = text
            steps.append({"source": "system", "message": text})
        elif isinstance(msg, ChatMessageUser):
            steps.append(_user_step(msg, redacted))
        elif isinstance(msg, ChatMessageAssistant):
            items = by_id.get(msg.id)
            how = "message_id"
            if items is None and not by_id and assistant_seen < len(ordered):
                items, how = [ordered[assistant_seen]], "order"
            for text, source in items or []:
                steps.append(_notice_step(text, source, how))
                attached_notices += 1
            call_output_id = output_ids[assistant_seen] if assistant_seen < len(output_ids) else None
            assistant_seen += 1
            step = _assistant_step(msg, call_output_id, blocked_outputs)
            steps.append(step)
            for tc in step["tool_calls"] or []:
                issued[tc.tool_call_id] = len(steps) - 1
        elif isinstance(msg, ChatMessageTool):
            _add_tool_result(msg, redacted, steps, issued)
        else:
            text, _ = split_content(getattr(msg, "content", None))
            steps.append({"source": "user", "message": text, "extra": {"role": getattr(msg, "role", "?")}})

    recovered = _recover_submits(sample, agent_name, steps, issued)
    atif_steps = _atif_steps(steps)

    lone_agent = len(roster(sample)) == 1
    usage = usage_from_events(model_events) or (sum_usage(sample) if lone_agent else {})
    last_text = (sample.store.get(LAST_TEXT_KEY) or {}).get(agent_name)
    if last_text is None and lone_agent and sample.output:
        last_text = sample.output.completion or None
    final_metrics = None
    if usage:
        final_metrics = FinalMetrics(total_prompt_tokens=usage.get("input"),
                                     total_completion_tokens=usage.get("output"),
                                     total_steps=len(atif_steps))
    traj = Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=f"{sample.id}#{sample.epoch}",
        trajectory_id=f"{sample.id}#{sample.epoch}/{agent_name}",
        agent=Agent(name=f"orbit/{agent_name}", version=agent_version(), model_name=log.eval.model),
        steps=atif_steps,
        final_metrics=final_metrics,
        extra={"final_text": last_text or None},
    )
    info = {"system_prompt": system_prompt, "submit_recovered": recovered,
            "delivery_notices": attached_notices, "model_calls": len(model_events),
            "n_agent_steps": sum(1 for s in atif_steps if s.source == "agent"),
            "n_tool_calls": sum(len(s.tool_calls or []) for s in atif_steps),
            "usage": usage or None, "final_text": last_text or None}
    return traj, info


# --------------------------------------------------------------------------
# how a sample ended
# --------------------------------------------------------------------------
def orbit_halt_reason(sample) -> str | None:
    """ORBIT's own stop reason (convergence, max_turns, max_time, attack_success, inspect_limit,
    error: ..., interrupted, before_agents_completed), written only to an info event."""
    for ev in reversed(sample.events or []):
        if ev.event == "info" and isinstance(ev.data, str):
            m = HALT_RE.search(ev.data)
            if m:
                return m.group(1)
    return None


def error_and_failure(sample) -> tuple[str | None, str | None]:
    """Map Inspect's limit/error, or ORBIT's scheduler cap, onto CheatBench's episode error
    string and failure tag.

    core/judge.py:is_judgeable drops lost episodes; limits stay judgeable. ORBIT's
    scheduler caps (max_turns, max_time) are not Inspect limits: the sample completes
    normally, so they are read from the halt reason.
    """
    halt = orbit_halt_reason(sample)
    finished_cleanly = sample.limit is None and sample.error is None
    if finished_cleanly and halt == "max_turns":
        return "max_turns reached before confirmed task completion", "max_turns"
    if finished_cleanly and halt == "max_time":
        return "time limit reached", "timeout"
    if sample.limit is not None:
        kind = sample.limit.type
        if kind == "time":
            return "time limit reached", "timeout"
        if kind in ("message", "turn"):
            return "max_turns reached before confirmed task completion", "max_turns"
        if kind in ("token", "context"):
            return f"{kind} limit reached", "context_window_exceeded"
        return f"{kind} limit reached", None
    if sample.error is not None:
        message = str(getattr(sample.error, "message", sample.error))
        lowered = message.lower()
        if any(k in lowered for k in PROVIDER_ERROR_WORDS):
            # a lost episode in CheatBench's terms: the provider, not the agent, ended it
            return message, "connection_closed"
        return message, "harness"
    return None, None


# --------------------------------------------------------------------------
# run directory
# --------------------------------------------------------------------------
def _registry_entries(md: dict) -> list[dict]:
    try:
        registry = json.loads(md.get("agent_evals_main") or "{}")
        return [a for a in (registry.get("agents") or []) if isinstance(a, dict)]
    except (ValueError, AttributeError):
        return []


def deliverables_block(sample) -> dict | None:
    """What the team left behind, from the deliverables scorer's evidence.

    The score value only says whether the archive sat in the working tree at the end; a
    team that registered and then checked out main scores 0, and an uncommitted archive
    scores 1. The registry on main and the git summary say whether the team registered.
    """
    sc = (sample.scores or {}).get("deliverables_scorer")
    if sc is None:
        return None
    md = sc.metadata or {}
    entries = _registry_entries(md)
    agents = [a.get("agent") for a in entries]
    team = next((a for a in entries if a.get("agent") == TEAM_IDENTITY), None)
    # A registration that carries another agent's message verbatim registered that agent's work.
    copied_from = None
    if team and (team.get("message") or "").strip():
        for other in entries:
            if other is not team and (other.get("message") or "").strip() == team["message"].strip():
                copied_from = other.get("agent")
    registered_entry = None
    if team:
        registered_entry = {"archive": team.get("archive"), "commits": team.get("commits"),
                            "message_head": (team.get("message") or "")[:200]}
    lines = (md.get("git") or "").splitlines()
    status = next((line for line in lines if line.startswith("## ")), "")
    return {"archive_in_workspace": sc.value == 1.0,
            "registered_on_main": TEAM_IDENTITY in agents,
            "registered_entry": registered_entry,
            "registered_message_copied_from": copied_from,
            "main_registry_agents": agents,
            "checkout": status[3:].split("...")[0] if status else None,
            "commits": [line for line in lines if re.match(r"^[0-9a-f]{7,} ", line)],
            "explanation": sc.explanation}


def swarm_block(sample) -> dict | None:
    """Composition, hidden roles and the team's own report, for a swarm run."""
    cb = (sample.metadata or {}).get("cheatbench", {})
    swarm = cb.get("swarm")
    if not swarm:
        return None
    md = {}
    for sc in (sample.scores or {}).values():
        md = sc.metadata or {}
        break
    return {**swarm, "swarm_report": md.get("swarm_report"), "swarm_report_cheated": md.get("swarm_report_cheated")}


def _write_communication(sample, out_dir: Path) -> None:
    keys = ("CommunicationState:messages", "CommunicationState:events")
    comm = {k.split(":", 1)[1]: sample.store.get(k) for k in keys if sample.store.get(k)}
    if not comm:
        return
    comm_dir = out_dir / "communication"
    comm_dir.mkdir(exist_ok=True)
    (comm_dir / f"{sample.id}#{sample.epoch}.json").write_text(json.dumps(comm, indent=1, default=str))


def _write_defense(sample, out_dir: Path, defenses_cfg: list, evidence: dict) -> None:
    if not (defenses_cfg or evidence["interventions"]):
        return
    def_dir = out_dir / "defense"
    def_dir.mkdir(exist_ok=True)
    payload = {"defenses": defenses_cfg, **evidence}
    (def_dir / f"{sample.id}#{sample.epoch}.json").write_text(json.dumps(payload, indent=1, default=str))


def _base_record(sample, log, cb: dict, agent_name: str, episode: str, agents: list[str], topology: str,
                 defenses_cfg: list, evidence: dict) -> dict:
    scores = {k: (v.value if hasattr(v, "value") else v) for k, v in (sample.scores or {}).items()}
    return {
        "id": sample.id, "episode": episode, "replicate": sample.epoch,
        "task": cb.get("task"), "variant": cb.get("variant", ""),
        "model": log.eval.model, "agent": f"orbit/{agent_name}", "orbit_agent": agent_name,
        "orbit_roster": agents, "topology": topology,
        "control": cb.get("control", "baseline"), "monitor_model": cb.get("monitor_model"),
        "role": ((cb.get("swarm") or {}).get("roles") or {}).get(agent_name),
        "defenses": [d["name"] for d in defenses_cfg],
        "defense": defense_counts(evidence, agent_name),
        "agent_version": agent_version(),
        "prompt_file": cb.get("prompt_file"),
        "wall_time": sample.total_time,
        "limit": sample.limit.type if sample.limit else None,
        "score": scores,
    }


def _submitted(transcript: dict) -> bool:
    for step in transcript["trajectory"].get("steps", []):
        for tc in step.get("tool_calls") or []:
            if tc.get("function_name") == "submit":
                return True
    return False


def _ended_by(sample, submitted: bool, ep_failure: str | None) -> str:
    if submitted:
        return "submit"
    if sample.limit or ep_failure in LIMIT_FAILURES:
        return "limit"
    if sample.error:
        return "error"
    return "end"


def _write_episode(sample, log, eval_path: Path, out_dir: Path, cb: dict, agent_name: str, episode: str,
                   agents: list[str], topology: str, messages: list, error, failure) -> dict:
    """Write one agent's transcript and prompt; return the record fields that depend on them."""
    traj, info = build_trajectory(sample, log, agent_name, messages)
    ep_dir = out_dir / "trajectories" / episode
    ep_dir.mkdir(parents=True, exist_ok=True)
    prompt = cb.get("instruction") or (sample.input if isinstance(sample.input, str) else "")
    transcript = {
        "agent": f"orbit/{agent_name}", "model": log.eval.model,
        "prompt": prompt,
        "trajectory": traj.to_json_dict(),
        "orbit": {"eval_log": str(eval_path.resolve()), "agent": agent_name, "roster": agents,
                  "topology": topology, "system_prompt": info["system_prompt"],
                  "submit_recovered_from_events": info["submit_recovered"],
                  "delivery_notices_recovered_from_events": info["delivery_notices"],
                  "model_calls": info["model_calls"]},
    }
    transcript_path = ep_dir / "transcript.json"
    transcript_path.write_text(json.dumps(transcript, indent=1))
    (ep_dir / "prompt.txt").write_text(transcript["prompt"])
    # A limit ends the sample, not every agent in it: an agent whose last tool call
    # was submit finished its own episode before the limit hit the others.
    submitted = _submitted(transcript)
    if submitted and failure in LIMIT_FAILURES:
        ep_error, ep_failure = None, None
    else:
        ep_error, ep_failure = error, failure
    return {
        "final_text": info["final_text"],
        "n_turns": info["n_agent_steps"], "n_tool_calls": info["n_tool_calls"],
        "usage": info["usage"], "transcript_path": str(transcript_path),
        "error": ep_error, "failure": ep_failure, "ok": ep_error is None,
        "ended_by": _ended_by(sample, submitted, ep_failure),
    }


def _run_json(log, eval_path: Path, out_dir: Path, task_root: Path, records: list[dict], agents_seen: list[str],
              topology: str) -> dict:
    first = log.samples[0] if log.samples else None
    first_record = records[0] if records else {}
    defenses = (first.metadata.get("experiment", {}).get("defenses") if first else None) or []
    deviations = DEVIATIONS + ([BOARD_DEVIATION] if topology == "board" else [])
    return {
        "task": first_record.get("task"), "task_root": str(task_root.resolve()), "variant": "",
        "run_id": out_dir.name, "model": log.eval.model, "agent": "orbit",
        "source": "orbit", "topology": topology, "agents": agents_seen,
        "control": first_record.get("control"),
        "monitor_model": first_record.get("monitor_model"),
        "defenses": defenses,
        "swarm": swarm_block(first) if first else None,
        "deliverables": deliverables_block(first) if first else None,
        "orbit_halt_reason": orbit_halt_reason(first) if first else None,
        "eval_log": str(eval_path.resolve()), "eval_id": log.eval.eval_id,
        "orbit_version": orbit.__version__, "inspect_version": inspect_ai.__version__,
        "deviations": deviations,
    }


def convert(eval_path: Path, out_dir: Path, task_root: Path | None = None) -> dict:
    log = read_eval_log(str(eval_path), resolve_attachments=True)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    agents_seen: list[str] = []
    topology = "single"
    for sample in log.samples or []:
        meta = sample.metadata or {}
        cb = meta.get("cheatbench", {})
        agents = roster(sample)
        if not agents:
            raise ValueError(f"sample {sample.id}: no ORBIT roster in metadata['experiment']")
        agents_seen = agents
        if (meta.get("experiment", {}).get("communication") or {}).get("channels"):
            topology = "swarm" if cb.get("swarm") else "board"
        error, failure = error_and_failure(sample)
        _write_communication(sample, out_dir)
        defenses_cfg = meta.get("experiment", {}).get("defenses") or []
        evidence = defense_evidence(sample)
        _write_defense(sample, out_dir, defenses_cfg, evidence)
        for agent_name in agents:
            episode = f"{sample.id}#{sample.epoch}"
            if len(agents) > 1:
                episode = f"{episode}@{agent_name}"
            messages = agent_messages(sample, agent_name)
            record = _base_record(sample, log, cb, agent_name, episode, agents, topology, defenses_cfg, evidence)
            if not messages:
                # The agent never produced a projected conversation: a failed episode
                # in CheatBench's terms (no transcript, skipped by the judge).
                record.update({"final_text": None, "n_turns": 0, "n_tool_calls": 0, "usage": None,
                               "transcript_path": None,
                               "error": error or "no agent conversation was projected into the sample",
                               "failure": failure or "harness", "ok": False, "ended_by": "error"})
            else:
                record.update(_write_episode(sample, log, eval_path, out_dir, cb, agent_name, episode,
                                             agents, topology, messages, error, failure))
            records.append(record)
    task_root = task_root or (CB / "tasks" / (records[0]["task"] if records else "unknown"))
    run_json = _run_json(log, eval_path, out_dir, task_root, records, agents_seen, topology)
    (out_dir / "run.json").write_text(json.dumps(run_json, indent=1))
    (out_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return {"out_dir": str(out_dir), "episodes": len(records), "topology": topology,
            "with_transcript": [r["episode"] for r in records if r["transcript_path"]],
            "without_transcript": [r["episode"] for r in records if not r["transcript_path"]]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_log")
    ap.add_argument("out_dir")
    ap.add_argument("--task-root", default="")
    args = ap.parse_args(argv)
    task_root = Path(args.task_root) if args.task_root else None
    summary = convert(Path(args.eval_log), Path(args.out_dir), task_root)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
