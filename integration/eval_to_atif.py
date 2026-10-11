"""Seam C: an Inspect .eval log from an ORBIT run -> the run directory CheatBench's judge reads.

    python -m integration.eval_to_atif logs/<run>.eval runs/<name> [--task-root PATH]

Writes
    runs/<name>/run.json
    runs/<name>/episodes.jsonl                      one record per (sample, agent)
    runs/<name>/trajectories/<episode>/transcript.json
    runs/<name>/trajectories/<episode>/prompt.txt
    runs/<name>/communication/<sample>.json         channel messages and events, if any

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
ATTRIBUTION_KEY = "EnvironmentState:message_attribution"
LAST_TEXT_KEY = "EnvironmentState:last_text_by_agent"

DEVIATIONS = [
    "agent_is_orbit_turn_react_not_vendor_harness",
    "orbit_assistant_and_submit_prompt_appended_to_system_prompt",
    "continue_prompts_are_orbit_synthetic_user_messages",
    "mirror_served_from_docker_sidecar_and_ca_baked_into_image",
]


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
    text, _ = split_content([result]) if hasattr(result, "type") else (str(result), None)
    return text


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
    calls) belong to the defense, not to the agent, and get owner None.
    """
    begins = {e.id: e for e in sample.events if e.event == "span_begin"}
    cache: dict = {}

    def owner(span_id):
        if span_id in cache:
            return cache[span_id]
        e = begins.get(span_id)
        if e is None:
            result = None
        elif e.type == "orbit.defense":
            result = None
        elif e.type == "agent":
            result = e.name
        else:
            result = owner(e.parent_id) if e.parent_id else None
        cache[span_id] = result
        return result
    return owner


def defense_evidence(sample) -> dict:
    """DefenseLog store fields plus every defense_intervention info event."""
    keys = ("resolved_defenses", "activity", "blocks", "detections", "vaccinations",
            "vaccination_injections", "total_activations", "false_positives", "true_positives")
    log = {k: sample.store.get(f"DefenseLog:{k}") for k in keys}
    interventions = []
    for e in sample.events:
        if e.event == "info" and isinstance(e.data, dict) and e.data.get("event") in (
                "defense_intervention", "defense_resample", "defense_instrumentation_exempt"):
            interventions.append(e.data)
    return {"log": log, "interventions": interventions}


def defense_counts(evidence: dict, agent: str) -> dict:
    """checks and blocks per boundary for one agent, from the intervention events."""
    out = {"checks": 0, "blocks": 0, "blocks_by_boundary": {}, "checks_by_boundary": {}}
    for ev in evidence["interventions"]:
        if ev.get("event") != "defense_intervention" or ev.get("agent") != agent:
            continue
        b = ev.get("boundary", "?")
        out["checks"] += 1
        out["checks_by_boundary"][b] = out["checks_by_boundary"].get(b, 0) + 1
        if ev.get("allow") is False:
            out["blocks"] += 1
            out["blocks_by_boundary"][b] = out["blocks_by_boundary"].get(b, 0) + 1
    return out


def redacted_message_ids(model_events) -> set:
    """Ids of messages a defense withheld from the provider (orbit_defense_redacted)."""
    ids = set()
    for ev in model_events:
        for m in ev.input or []:
            if (m.metadata or {}).get("orbit_defense_redacted"):
                ids.add(m.id)
    return ids


def agent_events(sample, agent: str, kind: str) -> list:
    owner = span_owner(sample)
    return [e for e in sample.events if e.event == kind and owner(e.span_id) == agent]


TRAILING_TURN_TAG = "orbit_trailing_turn_fix"   # integration/controls.py


def delivery_notices(model_events) -> list[tuple[str | None, str, str]]:
    """(output assistant message id or None, text, source) per provider-only message in a model call.

    Covers ORBIT's channel notices (system) and the integration's trailing-turn
    continue prompt (user); neither enters the durable conversation.
    """
    out = []
    for ev in model_events:
        for m in ev.input or []:
            meta = m.metadata or {}
            if (m.role == "system" and meta.get(DELIVERY_TAG)) or (m.role == "user" and meta.get(TRAILING_TURN_TAG)):
                text, _ = split_content(m.content)
                out_id = ev.output.message.id if (ev.output and ev.output.choices) else None
                out.append((out_id, text, "system" if m.role == "system" else "user"))
    return out


def usage_from_events(model_events) -> dict:
    """Prompt tokens include cache reads: Inspect reports them apart from input_tokens."""
    total_in = total_out = cache_read = 0
    for ev in model_events:
        usage = getattr(ev.output, "usage", None) if ev.output else None
        if usage:
            total_in += (usage.input_tokens or 0) + (usage.input_tokens_cache_read or 0)
            cache_read += usage.input_tokens_cache_read or 0
            total_out += usage.output_tokens or 0
    return {"input": total_in, "output": total_out, "cache_read": cache_read} if (total_in or total_out) else {}


# --------------------------------------------------------------------------
# one (sample, agent) -> one ATIF trajectory
# --------------------------------------------------------------------------
def build_trajectory(sample, log, agent_name: str, messages: list) -> tuple[Trajectory, dict]:
    model_events = agent_events(sample, agent_name, "model")
    notices = delivery_notices(model_events)
    redacted = redacted_message_ids(model_events)
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
            system_prompt = text if system_prompt is None else system_prompt
            steps.append({"source": "system", "message": text})
        elif isinstance(msg, ChatMessageUser):
            text, _ = split_content(msg.content)
            steps.append({"source": "user", "message": text,
                          **({"extra": {"withheld_from_model_by_defense": True}} if msg.id in redacted else {})})
        elif isinstance(msg, ChatMessageAssistant):
            items = by_id.get(msg.id)
            how = "message_id"
            if items is None and not by_id and assistant_seen < len(ordered):
                items, how = [ordered[assistant_seen]], "order"
            for text, source in items or []:
                steps.append({"source": source, "message": text,
                              "extra": {"orbit_delivery_notice": source == "system",
                                        "orbit_trailing_turn_fix": source == "user", "attached_by": how}})
                attached_notices += 1
            assistant_seen += 1
            text, reasoning = split_content(msg.content)
            calls = [ToolCall(tool_call_id=tc.id, function_name=tc.function,
                              arguments=dict(tc.arguments) if isinstance(tc.arguments, dict) else {"value": tc.arguments})
                     for tc in (msg.tool_calls or [])]
            step = {"source": "agent", "message": text, "reasoning_content": reasoning,
                    "tool_calls": calls or None, "llm_call_count": 1,
                    "model_name": msg.model or None, "results": []}
            steps.append(step)
            for tc in calls:
                issued[tc.tool_call_id] = len(steps) - 1
        elif isinstance(msg, ChatMessageTool):
            content = tool_result_text(msg.content)
            extra = {"is_error": True, "error": msg.error.message} if msg.error else None
            if msg.id in redacted:
                extra = {**(extra or {}), "withheld_from_model_by_defense": True}
            idx = issued.get(msg.tool_call_id)
            if idx is None:
                steps.append({"source": "user", "message": content,
                              "extra": {"orphan_tool_result": msg.tool_call_id, "function": msg.function,
                                        **(extra or {})}})
            else:
                steps[idx]["results"].append(ObservationResult(
                    source_call_id=msg.tool_call_id, content=content, extra=extra))
        else:
            text, _ = split_content(getattr(msg, "content", None))
            steps.append({"source": "user", "message": text, "extra": {"role": getattr(msg, "role", "?")}})

    # ORBIT removes the completed submit call and its result from the conversation
    # (orbit/agents/turn_react.py:_remove_submit_tool); the native tool event keeps it.
    recovered = 0
    for ev in agent_events(sample, agent_name, "tool"):
        if ev.function != "submit" or ev.id in issued:
            continue
        last_agent = next((i for i in range(len(steps) - 1, -1, -1) if steps[i]["source"] == "agent"), None)
        call = ToolCall(tool_call_id=ev.id, function_name="submit",
                        arguments=dict(ev.arguments) if isinstance(ev.arguments, dict) else {"answer": str(ev.arguments)})
        result = ObservationResult(source_call_id=ev.id, content=tool_result_text(ev.result),
                                   extra={"is_error": True, "error": ev.error.message} if ev.error else None)
        if last_agent is None:
            steps.append({"source": "agent", "message": "", "tool_calls": [call], "llm_call_count": 1,
                          "results": [result], "extra": {"submit_recovered_from_events": True}})
        else:
            step = steps[last_agent]
            step["tool_calls"] = list(step.get("tool_calls") or []) + [call]
            step["results"].append(result)
            step["extra"] = {**(step.get("extra") or {}), "submit_recovered_from_events": True}
        recovered += 1

    atif_steps = []
    for i, s in enumerate(steps, 1):
        results = s.pop("results", [])
        observation = Observation(results=results) if results else None
        atif_steps.append(Step(step_id=i, observation=observation, **s))

    usage = usage_from_events(model_events) or (sum_usage(sample) if len(roster(sample)) == 1 else {})
    last_text = (sample.store.get(LAST_TEXT_KEY) or {}).get(agent_name)
    if last_text is None and len(roster(sample)) == 1 and sample.output:
        last_text = sample.output.completion or None
    traj = Trajectory(
        schema_version=SCHEMA_VERSION,
        session_id=f"{sample.id}#{sample.epoch}",
        trajectory_id=f"{sample.id}#{sample.epoch}/{agent_name}",
        agent=Agent(name=f"orbit/{agent_name}",
                    version=f"orbit-{orbit.__version__}+inspect-ai-{inspect_ai.__version__}",
                    model_name=log.eval.model),
        steps=atif_steps,
        final_metrics=FinalMetrics(total_prompt_tokens=usage.get("input"), total_completion_tokens=usage.get("output"),
                                   total_steps=len(atif_steps)) if usage else None,
        extra={"final_text": last_text or None},
    )
    info = {"system_prompt": system_prompt, "submit_recovered": recovered,
            "delivery_notices": attached_notices, "model_calls": len(model_events),
            "n_agent_steps": sum(1 for s in atif_steps if s.source == "agent"),
            "n_tool_calls": sum(len(s.tool_calls or []) for s in atif_steps),
            "usage": usage or None, "final_text": last_text or None}
    return traj, info


def sum_usage(sample) -> dict:
    total_in = total_out = cache_read = 0
    for usage in (sample.model_usage or {}).values():
        total_in += (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "input_tokens_cache_read", 0) or 0)
        cache_read += getattr(usage, "input_tokens_cache_read", 0) or 0
        total_out += getattr(usage, "output_tokens", 0) or 0
    return {"input": total_in, "output": total_out, "cache_read": cache_read} if (total_in or total_out) else {}


def error_and_failure(sample) -> tuple[str | None, str | None]:
    """Map Inspect's limit/error onto CheatBench's episode error string and failure tag.

    core/judge.py:is_judgeable drops lost episodes; limits stay judgeable.
    """
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
        if any(k in lowered for k in ("rate limit", "429", "connection", "stream", "timed out", "overloaded")):
            return message, "connection_closed"
        return message, "harness"
    return None, None


# --------------------------------------------------------------------------
# run directory
# --------------------------------------------------------------------------
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
            topology = "board"
        error, failure = error_and_failure(sample)
        comm = {k.split(":", 1)[1]: sample.store.get(k) for k in ("CommunicationState:messages", "CommunicationState:events")
                if sample.store.get(k)}
        if comm:
            comm_dir = out_dir / "communication"
            comm_dir.mkdir(exist_ok=True)
            (comm_dir / f"{sample.id}#{sample.epoch}.json").write_text(json.dumps(comm, indent=1, default=str))
        defenses_cfg = meta.get("experiment", {}).get("defenses") or []
        evidence = defense_evidence(sample)
        if defenses_cfg or evidence["interventions"]:
            def_dir = out_dir / "defense"
            def_dir.mkdir(exist_ok=True)
            (def_dir / f"{sample.id}#{sample.epoch}.json").write_text(json.dumps(
                {"defenses": defenses_cfg, **evidence}, indent=1, default=str))
        for agent_name in agents:
            episode = f"{sample.id}#{sample.epoch}" if len(agents) == 1 else f"{sample.id}#{sample.epoch}@{agent_name}"
            messages = agent_messages(sample, agent_name)
            record = {
                "id": sample.id, "episode": episode, "replicate": sample.epoch,
                "task": cb.get("task"), "variant": cb.get("variant", ""),
                "model": log.eval.model, "agent": f"orbit/{agent_name}", "orbit_agent": agent_name,
                "orbit_roster": agents, "topology": topology,
                "control": cb.get("control", "baseline"), "monitor_model": cb.get("monitor_model"),
                "defenses": [d["name"] for d in defenses_cfg],
                "defense": defense_counts(evidence, agent_name),
                "agent_version": f"orbit-{orbit.__version__}+inspect-ai-{inspect_ai.__version__}",
                "prompt_file": cb.get("prompt_file"),
                "wall_time": sample.total_time,
                "limit": sample.limit.type if sample.limit else None,
                "score": {k: (v.value if hasattr(v, "value") else v) for k, v in (sample.scores or {}).items()},
            }
            if not messages:
                # The agent never produced a projected conversation: a failed episode
                # in CheatBench's terms (no transcript, skipped by the judge).
                record.update({"final_text": None, "n_turns": 0, "n_tool_calls": 0, "usage": None,
                               "transcript_path": None,
                               "error": error or "no agent conversation was projected into the sample",
                               "failure": failure or "harness", "ok": False})
                records.append(record)
                continue
            traj, info = build_trajectory(sample, log, agent_name, messages)
            ep_dir = out_dir / "trajectories" / episode
            ep_dir.mkdir(parents=True, exist_ok=True)
            transcript = {
                "agent": f"orbit/{agent_name}", "model": log.eval.model,
                "prompt": cb.get("instruction") or (sample.input if isinstance(sample.input, str) else ""),
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
            record.update({
                "final_text": info["final_text"],
                "n_turns": info["n_agent_steps"], "n_tool_calls": info["n_tool_calls"],
                "usage": info["usage"], "transcript_path": str(transcript_path),
                "error": error, "failure": failure, "ok": error is None,
            })
            records.append(record)
    task_root = task_root or (CB / "tasks" / (records[0]["task"] if records else "unknown"))
    run_json = {
        "task": records[0]["task"] if records else None, "task_root": str(task_root.resolve()), "variant": "",
        "run_id": out_dir.name, "model": log.eval.model, "agent": "orbit",
        "source": "orbit", "topology": topology, "agents": agents_seen,
        "control": records[0]["control"] if records else None,
        "monitor_model": records[0]["monitor_model"] if records else None,
        "defenses": (log.samples[0].metadata.get("experiment", {}).get("defenses") if log.samples else None) or [],
        "eval_log": str(eval_path.resolve()), "eval_id": log.eval.eval_id,
        "orbit_version": orbit.__version__, "inspect_version": inspect_ai.__version__,
        "deviations": DEVIATIONS + (["two_agents_share_one_sandbox_clock_and_git_identity"] if topology == "board" else []),
    }
    (out_dir / "run.json").write_text(json.dumps(run_json, indent=1))
    (out_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return {"out_dir": str(out_dir), "episodes": len(records), "topology": topology,
            "with_transcript": [r["episode"] for r in records if r["transcript_path"]],
            "without_transcript": [r["episode"] for r in records if not r["transcript_path"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_log")
    ap.add_argument("out_dir")
    ap.add_argument("--task-root", default="")
    args = ap.parse_args()
    summary = convert(Path(args.eval_log), Path(args.out_dir), Path(args.task_root) if args.task_root else None)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
