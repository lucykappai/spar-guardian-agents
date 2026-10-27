"""Refuse unless every named swarm run is complete and judged.

    .venv/bin/python tests/check_runs.py
    .venv/bin/python tests/check_runs.py --expect R0D0,R10D0,R0D10 [--runs runs]
    .venv/bin/python tests/check_runs.py runs/swarm_R0D0 runs/swarm_R10D0 ...

With no run directories and no --expect, the expected compositions are
src.configuration.settings.SWARM_COMPOSITIONS. Run it from the repository root: eval log
paths in run.json are relative to it.

Exit 0 only when every run has a run.json with topology swarm, one episode per
agent with the role the run.json assigns, no harness or provider failure, a
non-empty transcript on disk, one parsed judge verdict per episode, and its eval
log on disk. An episode that ended on the task's time or turn limit (the
converter's `timeout` and `max_turns` labels) is complete by design and is
counted under `endings`. Anything else is listed and the exit code is 1. A
missing or unparsable team report is reported but is a result, not a failure; a
run with a mode (the pull board) was never asked for one and gets no note.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^swarm_R(\d+)D(\d+)$")
LIMIT_ENDINGS = {"timeout", "max_turns"}


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def transcript_file(run_dir: Path, record: dict) -> Path | None:
    recorded = Path(record.get("transcript_path") or "")
    if recorded.is_file():
        return recorded
    fallback = run_dir / "trajectories" / record["episode"] / "transcript.json"
    if fallback.is_file():
        return fallback
    return None


def check_run_json(run_dir: Path, run: dict) -> list[str]:
    name = run_dir.name
    problems = []
    if run.get("topology") != "swarm":
        problems.append(f"{name}: topology is {run.get('topology')!r}, not swarm")
    swarm = run.get("swarm") or {}
    roles = swarm.get("roles") or {}
    n_agents = swarm.get("n_agents")
    if not roles or n_agents != len(roles):
        problems.append(f"{name}: swarm block has {len(roles)} roles for n_agents={n_agents}")
    match = NAME.match(name)
    if match:
        named = (int(match.group(1)), int(match.group(2)))
        recorded = (swarm.get("regulators"), swarm.get("disregulators"))
        if named != recorded:
            problems.append(f"{name}: name says R{match.group(1)}D{match.group(2)} but run.json says "
                            f"R{swarm.get('regulators')}D{swarm.get('disregulators')}")
    eval_log = run.get("eval_log")
    if not eval_log or not Path(eval_log).is_file():
        problems.append(f"{name}: eval log missing: {eval_log}")
    return problems


def check_transcript(run_dir: Path, record: dict) -> list[str]:
    episode = record.get("episode")
    path = transcript_file(run_dir, record)
    if path is None:
        return [f"{episode}: transcript missing"]
    problems = []
    try:
        trajectory = json.loads(path.read_text()).get("trajectory") or {}
        steps = trajectory.get("steps") or []
    except (OSError, ValueError) as e:
        steps = []
        problems.append(f"{episode}: transcript unreadable ({e})")
    if not steps:
        problems.append(f"{episode}: transcript has no steps")
    return problems


def check_episode(run_dir: Path, record: dict, roles: dict) -> list[str]:
    episode = record.get("episode")
    agent = record.get("orbit_agent")
    failure = record.get("failure")
    problems = []
    if roles and record.get("role") != roles.get(agent):
        problems.append(f"{episode}: role {record.get('role')!r} but run.json assigns {roles.get(agent)!r}")
    if failure is not None and failure not in LIMIT_ENDINGS:
        error = (record.get("error") or "")[:80]
        problems.append(f"{episode}: failure={failure!r} error={error!r}")
    problems += check_transcript(run_dir, record)
    if not record.get("n_tool_calls"):
        problems.append(f"{episode}: no tool calls")
    return problems


def check_episodes(run_dir: Path, episodes: list[dict], swarm: dict) -> list[str]:
    name = run_dir.name
    roles = swarm.get("roles") or {}
    n_agents = swarm.get("n_agents")
    problems = []
    if n_agents and len(episodes) != n_agents:
        problems.append(f"{name}: {len(episodes)} episodes for {n_agents} agents")
    for record in episodes:
        problems += check_episode(run_dir, record, roles)
    seen_agents = {record.get("orbit_agent") for record in episodes}
    if roles and seen_agents != set(roles):
        problems.append(f"{name}: episodes cover {sorted(seen_agents)} but roles name {sorted(roles)}")
    return problems


def check_verdict(episode: str, verdict: dict) -> list[str]:
    problems = []
    if verdict.get("judge_error"):
        problems.append(f"{episode}: judge_error={verdict['judge_error']!r}"[:200])
    if verdict.get("parse_ok") is not True:
        problems.append(f"{episode}: verdict did not parse")
    if not isinstance(verdict.get("cheated"), bool):
        problems.append(f"{episode}: cheated is {verdict.get('cheated')!r}")
    return problems


def check_verdicts(run_dir: Path, episodes: list[dict]) -> tuple[list[str], dict[str, dict]]:
    name = run_dir.name
    path = run_dir / "judge.jsonl"
    verdicts: dict[str, dict] = {}
    if not path.is_file():
        return [f"{name}: judge.jsonl missing"], verdicts
    problems = []
    try:
        for verdict in load_jsonl(path):
            verdicts[verdict.get("episode")] = verdict
    except (OSError, ValueError) as e:
        problems.append(f"{name}: judge.jsonl unreadable ({e})")
    wanted = {record.get("episode") for record in episodes}
    for episode in sorted(wanted - set(verdicts)):
        problems.append(f"{episode}: no verdict")
    for episode in sorted(set(verdicts) - wanted):
        problems.append(f"{episode}: verdict without episode")
    for episode, verdict in sorted(verdicts.items()):
        problems += check_verdict(episode, verdict)
    return problems, verdicts


def count_by_role(episodes: list[dict], verdicts: dict[str, dict]) -> dict[str, list[int]]:
    """Map each role to [cheated, judged]."""
    by_role: dict[str, list[int]] = {}
    for record in episodes:
        verdict = verdicts.get(record.get("episode"), {})
        counts = by_role.setdefault(record.get("role") or "?", [0, 0])
        if isinstance(verdict.get("cheated"), bool):
            counts[1] += 1
            counts[0] += int(verdict["cheated"])
    return by_role


def count_endings(episodes: list[dict]) -> dict[str, int]:
    endings = [record.get("failure") or "none" for record in episodes]
    return {ending: endings.count(ending) for ending in sorted(set(endings))}


def longest_wall_time(episodes: list[dict]) -> int:
    if not episodes:
        return 0
    return round(max((record.get("wall_time") or 0) for record in episodes))


def report_notes(swarm: dict, run: dict) -> list[str]:
    """Notes on the team report. Runs with a mode (run.json "mode") were never asked for one."""
    if swarm.get("swarm_report") is None:
        return [] if run.get("mode") is not None else ["no team report"]
    if swarm.get("swarm_report_cheated") is None:
        return ["team report did not parse"]
    return []


def summarize(run_dir: Path, run: dict, episodes: list[dict], verdicts: dict[str, dict]) -> dict:
    swarm = run.get("swarm") or {}
    deliverables = run.get("deliverables") or {}
    by_role = count_by_role(episodes, verdicts)
    judged = sum(counts[1] for counts in by_role.values())
    cheated = sum(counts[0] for counts in by_role.values())
    report_cheated = swarm.get("swarm_report_cheated")
    report_agrees = None
    if judged and report_cheated is not None:
        report_agrees = report_cheated == (cheated > 0)
    return {
        "run": run_dir.name,
        "mode": run.get("mode"), "scale": run.get("scale"), "preset": run.get("preset"),
        "agents_checked_board": (run.get("engagement") or {}).get("agents_checked"),
        "composition": swarm.get("composition"),
        "agents": len(episodes),
        "calls": sum(record.get("n_tool_calls") or 0 for record in episodes),
        "wall_s": longest_wall_time(episodes),
        "endings": count_endings(episodes),
        "registered_on_main": deliverables.get("registered_on_main"),
        "registered_message_copied_from": deliverables.get("registered_message_copied_from"),
        "archive_in_workspace": deliverables.get("archive_in_workspace"),
        "judged": judged,
        "cheated": cheated,
        "by_role": {role: f"{counts[0]}/{counts[1]}" for role, counts in sorted(by_role.items())},
        "report_says": {True: "cheated", False: "not cheated"}.get(report_cheated, "none"),
        "report_agrees": report_agrees,
        "notes": report_notes(swarm, run),
    }


def check_run(run_dir: Path) -> tuple[list[str], dict | None]:
    if not run_dir.is_dir():
        return [f"{run_dir}: missing"], None
    try:
        run = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError) as e:
        return [f"{run_dir.name}: run.json unreadable ({e})"], None
    problems = check_run_json(run_dir, run)
    try:
        episodes = load_jsonl(run_dir / "episodes.jsonl")
    except (OSError, ValueError) as e:
        return problems + [f"{run_dir.name}: episodes.jsonl unreadable ({e})"], None
    problems += check_episodes(run_dir, episodes, run.get("swarm") or {})
    verdict_problems, verdicts = check_verdicts(run_dir, episodes)
    problems += verdict_problems
    return problems, summarize(run_dir, run, episodes, verdicts)


def default_compositions() -> list[str]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from src.configuration.settings import SWARM_COMPOSITIONS
    except ImportError as e:
        raise SystemExit(f"check_runs: cannot read src.configuration.settings.SWARM_COMPOSITIONS ({e}); pass --expect") from e
    return list(SWARM_COMPOSITIONS)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="*", help="run directories to check")
    parser.add_argument("--runs", default="runs", help="directory holding run directories (with --expect)")
    parser.add_argument("--expect", default=None,
                        help="comma-separated compositions R<r>D<d> expected under --runs "
                             "(default: src.configuration.settings.SWARM_COMPOSITIONS when no run directories are named)")
    return parser.parse_args(argv)


def run_dirs_to_check(args: argparse.Namespace) -> list[Path]:
    dirs = [Path(p) for p in args.run_dirs]
    if args.expect is not None:
        compositions = [c.strip() for c in args.expect.split(",") if c.strip()]
    elif not dirs:
        compositions = default_compositions()
    else:
        compositions = []
    dirs += [Path(args.runs) / f"swarm_{c}" for c in compositions]
    return dirs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dirs = run_dirs_to_check(args)
    if not dirs:
        raise SystemExit("check_runs: name run directories or pass --expect")
    all_problems: list[str] = []
    for run_dir in dirs:
        problems, summary = check_run(run_dir)
        all_problems += problems
        if summary:
            print(json.dumps(summary))
        status = "OK" if not problems else f"{len(problems)} problem(s)"
        print(f"{run_dir.name}: {status}")
    for problem in all_problems:
        print("PROBLEM", problem)
    verdict = "all complete and judged" if not all_problems else f"{len(all_problems)} problem(s)"
    print(f"checked {len(dirs)} run(s): {verdict}")
    return 0 if not all_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
