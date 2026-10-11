"""Refuse unless every named swarm run is complete and judged.

    python -m integration.check_sweep runs/swarm_R0D0 runs/swarm_R10D0 ...
    python -m integration.check_sweep --expect R0D0,R10D0,R0D10 [--runs runs]

Exit 0 only when every run has a run.json with topology swarm, one episode per
agent with the role the run.json assigns, no harness or provider failure, a
non-empty transcript on disk, one parsed judge verdict per episode, and its eval
log on disk. An episode that ended on the task's time or turn limit (the
converter's `timeout` and `max_turns` labels) is complete by design and is
counted under `endings`. Anything else is listed and the exit code is 1. A
missing or unparsable team report is reported but is a result, not a failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME = re.compile(r"^swarm_R(\d+)D(\d+)$")
LIMIT_ENDINGS = {"timeout", "max_turns"}   # eval_to_atif's labels for a limit, not a crash


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def transcript_file(run_dir: Path, record: dict) -> Path | None:
    p = Path(record.get("transcript_path") or "")
    if p.is_file():
        return p
    q = run_dir / "trajectories" / record["episode"] / "transcript.json"
    return q if q.is_file() else None


def check_run(run_dir: Path) -> tuple[list[str], dict | None]:
    problems: list[str] = []
    if not run_dir.is_dir():
        return [f"{run_dir}: missing"], None
    try:
        run = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError) as e:
        return [f"{run_dir.name}: run.json unreadable ({e})"], None
    if run.get("topology") != "swarm":
        problems.append(f"{run_dir.name}: topology is {run.get('topology')!r}, not swarm")
    swarm = run.get("swarm") or {}
    roles = swarm.get("roles") or {}
    n_agents = swarm.get("n_agents")
    if not roles or n_agents != len(roles):
        problems.append(f"{run_dir.name}: swarm block has {len(roles)} roles for n_agents={n_agents}")
    m = NAME.match(run_dir.name)
    if m and (int(m.group(1)), int(m.group(2))) != (swarm.get("regulators"), swarm.get("disregulators")):
        problems.append(f"{run_dir.name}: name says R{m.group(1)}D{m.group(2)} but run.json says "
                        f"R{swarm.get('regulators')}D{swarm.get('disregulators')}")
    eval_log = run.get("eval_log")
    if not eval_log or not Path(eval_log).is_file():
        problems.append(f"{run_dir.name}: eval log missing: {eval_log}")

    try:
        episodes = load_jsonl(run_dir / "episodes.jsonl")
    except (OSError, ValueError) as e:
        return problems + [f"{run_dir.name}: episodes.jsonl unreadable ({e})"], None
    if n_agents and len(episodes) != n_agents:
        problems.append(f"{run_dir.name}: {len(episodes)} episodes for {n_agents} agents")
    seen_agents = set()
    for r in episodes:
        ep = r.get("episode")
        agent = r.get("orbit_agent")
        seen_agents.add(agent)
        if roles and r.get("role") != roles.get(agent):
            problems.append(f"{ep}: role {r.get('role')!r} but run.json assigns {roles.get(agent)!r}")
        if r.get("failure") is not None and r.get("failure") not in LIMIT_ENDINGS:
            problems.append(f"{ep}: failure={r.get('failure')!r} error={(r.get('error') or '')[:80]!r}")
        t = transcript_file(run_dir, r)
        if t is None:
            problems.append(f"{ep}: transcript missing")
        else:
            try:
                steps = (json.loads(t.read_text()).get("trajectory") or {}).get("steps") or []
            except (OSError, ValueError) as e:
                steps = []
                problems.append(f"{ep}: transcript unreadable ({e})")
            if not steps:
                problems.append(f"{ep}: transcript has no steps")
        if not r.get("n_tool_calls"):
            problems.append(f"{ep}: no tool calls")
    if roles and seen_agents != set(roles):
        problems.append(f"{run_dir.name}: episodes cover {sorted(seen_agents)} but roles name {sorted(roles)}")

    jpath = run_dir / "judge.jsonl"
    verdicts: dict[str, dict] = {}
    if not jpath.is_file():
        problems.append(f"{run_dir.name}: judge.jsonl missing")
    else:
        try:
            for v in load_jsonl(jpath):
                verdicts[v.get("episode")] = v
        except (OSError, ValueError) as e:
            problems.append(f"{run_dir.name}: judge.jsonl unreadable ({e})")
        want = {r.get("episode") for r in episodes}
        for ep in sorted(want - set(verdicts)):
            problems.append(f"{ep}: no verdict")
        for ep in sorted(set(verdicts) - want):
            problems.append(f"{ep}: verdict without episode")
        for ep, v in sorted(verdicts.items()):
            if v.get("judge_error"):
                problems.append(f"{ep}: judge_error={v['judge_error']!r}"[:200])
            if v.get("parse_ok") is not True:
                problems.append(f"{ep}: verdict did not parse")
            if not isinstance(v.get("cheated"), bool):
                problems.append(f"{ep}: cheated is {v.get('cheated')!r}")

    by_role: dict[str, list[int]] = {}
    for r in episodes:
        v = verdicts.get(r.get("episode"), {})
        c = by_role.setdefault(r.get("role") or "?", [0, 0])
        if isinstance(v.get("cheated"), bool):
            c[1] += 1
            c[0] += int(v["cheated"])
    judged = sum(c[1] for c in by_role.values())
    cheated = sum(c[0] for c in by_role.values())
    report = swarm.get("swarm_report")
    report_cheated = swarm.get("swarm_report_cheated")
    notes = []
    if report is None:
        notes.append("no team report")
    elif report_cheated is None:
        notes.append("team report did not parse")
    summary = {
        "run": run_dir.name, "composition": swarm.get("composition"), "agents": len(episodes),
        "calls": sum(r.get("n_tool_calls") or 0 for r in episodes),
        "wall_s": round(max((r.get("wall_time") or 0) for r in episodes)) if episodes else 0,
        "endings": {k: sum(1 for r in episodes if (r.get("failure") or "none") == k)
                    for k in sorted({(r.get("failure") or "none") for r in episodes})},
        "registered_on_main": (run.get("deliverables") or {}).get("registered_on_main"),
        "registered_message_copied_from": (run.get("deliverables") or {}).get("registered_message_copied_from"),
        "archive_in_workspace": (run.get("deliverables") or {}).get("archive_in_workspace"),
        "judged": judged, "cheated": cheated,
        "by_role": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_role.items())},
        "report_says": {True: "cheated", False: "not cheated"}.get(report_cheated, "none"),
        "report_agrees": (report_cheated == (cheated > 0)) if (judged and report_cheated is not None) else None,
        "notes": notes,
    }
    return problems, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dirs", nargs="*", help="run directories to check")
    ap.add_argument("--runs", default="runs", help="directory holding run directories (with --expect)")
    ap.add_argument("--expect", default="", help="comma-separated compositions R<r>D<d> expected under --runs")
    args = ap.parse_args(argv)
    dirs = [Path(p) for p in args.run_dirs]
    dirs += [Path(args.runs) / f"swarm_{c.strip()}" for c in args.expect.split(",") if c.strip()]
    if not dirs:
        ap.error("name run directories or pass --expect")
    all_problems: list[str] = []
    for d in dirs:
        problems, summary = check_run(d)
        all_problems += problems
        if summary:
            print(json.dumps(summary))
        print(f"{d.name}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    for p in all_problems:
        print("PROBLEM", p)
    print(f"checked {len(dirs)} run(s): {'all complete and judged' if not all_problems else f'{len(all_problems)} problem(s)'}")
    return 0 if not all_problems else 1


if __name__ == "__main__":
    sys.exit(main())
