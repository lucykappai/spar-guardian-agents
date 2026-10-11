"""Mock-model smoke tests for the ORBIT <-> CheatBench seams. No model spend.

    python -m integration.smoke_test [--only complete,message_limit,time_limit,board]

Every scenario starts the real sandbox (CheatBench's image plus the article
mirror), drives ORBIT with Inspect's mockllm scripted to call tools, converts
the log with eval_to_atif, and reads the run directory back the way
CheatBench's judge does (core.judge.is_judgeable, core.agents.trajectory_from_transcript,
core.render.render_trajectory). Each check is an assertion: a failing seam
exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "submodules" / "cheatbench"
if str(CB) not in sys.path:
    sys.path.insert(0, str(CB))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from core.agents import trajectory_from_transcript  # noqa: E402
from core.judge import is_judgeable  # noqa: E402
from core.render import render_trajectory  # noqa: E402

from integration.eval_to_atif import ATTRIBUTION_KEY, convert  # noqa: E402
from integration.openmath_agent_task import (cheatbench_openmath_agent,  # noqa: E402
                                             cheatbench_openmath_agent_board)

MODEL = "mockllm/model"
LOGS = ROOT / "logs" / "smoke"
RUNS = ROOT / "runs" / "smoke"


def call(name: str, args: dict, call_id: str) -> ModelOutput:
    return ModelOutput.for_tool_call(MODEL, name, args, tool_call_id=call_id)


def run_scenario(name: str, task, outputs) -> tuple:
    log_dir = LOGS / name
    shutil.rmtree(log_dir, ignore_errors=True)
    shutil.rmtree(RUNS / name, ignore_errors=True)
    model = get_model(MODEL, custom_outputs=outputs)
    [log] = inspect_eval(task, model=model, log_dir=str(log_dir), display="plain", log_level="warning")
    log = read_eval_log(log.location, resolve_attachments=True)
    summary = convert(Path(log.location), RUNS / name)
    records = [json.loads(line) for line in (RUNS / name / "episodes.jsonl").read_text().splitlines()]
    return log, summary, records


def load_transcript(record: dict) -> dict:
    assert record["transcript_path"], f"{record['episode']}: no transcript"
    transcript = json.loads(Path(record["transcript_path"]).read_text())
    traj = trajectory_from_transcript(transcript)          # the judge's reader; raises on invalid ATIF
    rendered = render_trajectory(traj)                      # the judge's view; raises on unrenderable steps
    assert rendered.strip(), "rendered transcript is empty"
    return transcript


def tool_names_offered(sample) -> set[str]:
    """Tool names in the schemas sent with the first model call."""
    ev = next(e for e in sample.events if e.event == "model")
    return {t.name for t in ev.tools}


def steps_of(transcript: dict) -> list[dict]:
    return transcript["trajectory"]["steps"]


def tool_calls_of(transcript: dict) -> list[tuple[str, dict]]:
    return [(tc["function_name"], tc["arguments"]) for s in steps_of(transcript) for tc in (s.get("tool_calls") or [])]


def results_text(transcript: dict) -> str:
    return "\n".join(r["content"] for s in steps_of(transcript)
                     for r in ((s.get("observation") or {}).get("results") or []))


# --------------------------------------------------------------------------
def scenario_complete() -> dict:
    outputs = [
        call("bash", {"command": "id -un; pwd; git -C /workspace branch -a"}, "c1"),
        call("python", {"code": "print(open('/workspace/Problem.md').read())"}, "c2"),
        call("bash", {"command": "git -C /workspace log --oneline -3 agent-014"}, "c3"),
        call("submit", {"answer": "smoke: done"}, "c4"),
    ]
    log, summary, records = run_scenario("complete", cheatbench_openmath_agent(time_limit_s=600), outputs)
    sample = log.samples[0]
    assert log.status == "success", log.status
    assert sample.limit is None, sample.limit
    attribution = sample.store.get(ATTRIBUTION_KEY) or []
    assert {a["agent"] for a in attribution} == {"solver"}, attribution
    [rec] = records
    assert rec["failure"] is None and rec["ok"] is True, rec
    assert is_judgeable(rec), rec
    t = load_transcript(rec)
    calls = tool_calls_of(t)
    assert [c[0] for c in calls] == ["bash", "python", "bash", "submit"], calls
    out = results_text(t)
    for needle in ("user\n/workspace", "agent-014", "Riemann", "ce9c639 Complete Riemann submission"):
        assert needle in out, f"missing {needle!r} in tool results:\n{out[:2000]}"
    assert t["orbit"]["submit_recovered_from_events"] == 1, t["orbit"]
    assert steps_of(t)[0]["source"] == "system" and "submit" in steps_of(t)[0]["message"], "system prompt not first"
    assert rec["usage"]["input"] > 0, rec["usage"]
    offered = tool_names_offered(sample)
    assert offered == {"bash", "python", "submit"}, f"single agent sees extra tools: {offered}"
    return {"steps": len(steps_of(t)), "tool_calls": [c[0] for c in calls], "score": rec["score"],
            "tools_offered": sorted(offered)}


def scenario_message_limit() -> dict:
    log, summary, records = run_scenario("message_limit", cheatbench_openmath_agent(message_limit=6, time_limit_s=600), None)
    sample = log.samples[0]
    assert sample.limit is not None and sample.limit.type == "message", sample.limit
    [rec] = records
    assert rec["failure"] == "max_turns" and rec["ok"] is False, rec
    assert is_judgeable(rec), rec
    t = load_transcript(rec)
    agent_steps = [s for s in steps_of(t) if s["source"] == "agent"]
    assert len(agent_steps) >= 2, f"partial conversation lost: {len(agent_steps)} agent steps"
    return {"steps": len(steps_of(t)), "agent_steps": len(agent_steps), "limit": rec["limit"]}


def scenario_time_limit() -> dict:
    outputs = [call("bash", {"command": "echo started; sleep 120; echo finished"}, "t1")]
    log, summary, records = run_scenario("time_limit", cheatbench_openmath_agent(time_limit_s=15), outputs)
    sample = log.samples[0]
    assert sample.limit is not None and sample.limit.type == "time", sample.limit
    [rec] = records
    assert rec["failure"] == "timeout", rec
    assert is_judgeable(rec), rec
    t = load_transcript(rec)
    calls = tool_calls_of(t)
    assert calls and calls[0][0] == "bash" and "sleep 120" in calls[0][1]["command"], calls
    return {"steps": len(steps_of(t)), "tool_calls": [c[0] for c in calls], "limit": rec["limit"]}


def scenario_board() -> dict:
    script = {
        "alpha": [
            call("channel_post", {"channel_id": "board", "message": "alpha: I will read README.md first"}, "a1"),
            call("bash", {"command": "id -un; git -C /workspace branch --show-current"}, "a2"),
            call("channel_read", {"channel_id": "board", "after_sequence": None, "limit": None}, "a3"),
            call("submit", {"answer": "alpha done"}, "a4"),
        ],
        "beta": [
            call("channel_read", {"channel_id": "board", "after_sequence": None, "limit": None}, "b1"),
            call("channel_post", {"channel_id": "board", "message": "beta: I will check the git branches"}, "b2"),
            call("submit", {"answer": "beta done"}, "b3"),
        ],
    }
    seen: dict[str, int] = {}

    def outputs(messages, tools, tool_choice, config) -> ModelOutput:
        system = next((m.text for m in messages if m.role == "system" and m.text.startswith("You are ")), "")
        who = system.split()[2].rstrip(".") if system else "?"
        assert who in script, f"cannot tell which agent is asking from system prompt {system[:60]!r}"
        step = sum(1 for m in messages if m.role == "assistant")
        seen[who] = step + 1
        plan = script[who]
        return plan[min(step, len(plan) - 1)]

    task = cheatbench_openmath_agent_board(time_limit_s=600, delivery="notify")
    log, summary, records = run_scenario("board", task, outputs)
    sample = log.samples[0]
    assert log.status == "success", log.status
    assert sample.limit is None, sample.limit
    assert summary["topology"] == "board", summary
    assert sorted(r["orbit_agent"] for r in records) == ["alpha", "beta"], records
    spans = {e.name for e in sample.events if e.event == "span_begin" and e.type == "agent"}
    assert spans == {"alpha", "beta"}, spans
    attribution = sample.store.get(ATTRIBUTION_KEY) or []
    assert {a["agent"] for a in attribution} == {"alpha", "beta"}, attribution
    out = {}
    for rec in records:
        assert rec["failure"] is None and rec["ok"] is True, rec
        assert is_judgeable(rec), rec
        t = load_transcript(rec)
        calls = tool_calls_of(t)
        names = [c[0] for c in calls]
        expected = [c.message.tool_calls[0].function for c in script[rec["orbit_agent"]]]
        assert names == expected, (rec["orbit_agent"], names, expected)
        assert t["orbit"]["submit_recovered_from_events"] == 1, t["orbit"]
        assert "You are " + rec["orbit_agent"] in steps_of(t)[0]["message"], "wrong system prompt"
        out[rec["orbit_agent"]] = {"steps": len(steps_of(t)), "tool_calls": names,
                                   "notices": t["orbit"]["delivery_notices_recovered_from_events"],
                                   "results": results_text(t)}
    # alpha posts on turn 1; beta's first model call (turn 2) must carry the notify notice,
    # and beta's channel_read must return alpha's post.
    offered = tool_names_offered(sample)
    assert offered == {"bash", "python", "submit", "channel_post", "channel_read", "channel_list"}, offered
    assert out["beta"]["notices"] >= 1, out["beta"]
    assert "alpha: I will read README.md first" in out["beta"]["results"], out["beta"]["results"][:1500]
    assert "beta: I will check the git branches" in out["alpha"]["results"], out["alpha"]["results"][:1500]
    assert "user\n" in out["alpha"]["results"], out["alpha"]["results"][:500]
    board = json.loads((RUNS / "board" / "communication" / "riemann#1.json").read_text())
    posts = [(m["sender"], m["content"]) for m in board["messages"]]
    assert posts == [("alpha", "alpha: I will read README.md first"), ("beta", "beta: I will check the git branches")], posts
    for v in out.values():
        v.pop("results")
    return {"agents": out, "board_posts": len(posts)}


SCENARIOS = {"complete": scenario_complete, "message_limit": scenario_message_limit,
             "time_limit": scenario_time_limit, "board": scenario_board}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated scenario names")
    args = ap.parse_args()
    names = [n for n in args.only.split(",") if n] or list(SCENARIOS)
    results, failed = {}, []
    for name in names:
        try:
            results[name] = {"status": "PASS", **SCENARIOS[name]()}
        except Exception as ex:  # noqa: BLE001 - report every scenario, then fail
            import traceback
            results[name] = {"status": "FAIL", "error": f"{type(ex).__name__}: {ex}",
                             "traceback": traceback.format_exc()[-3000:]}
            failed.append(name)
        print(f"[{results[name]['status']}] {name}: {json.dumps({k: v for k, v in results[name].items() if k != 'traceback'})}",
              flush=True)
    (RUNS / "smoke_results.json").parent.mkdir(parents=True, exist_ok=True)
    (RUNS / "smoke_results.json").write_text(json.dumps(results, indent=1))
    print("FAILED:" if failed else "ALL PASS:", failed or names)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
