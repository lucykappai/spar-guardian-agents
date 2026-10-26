"""Command line for the peer-pressure experiment. Argument parsing only; every command calls run.py.

    python -m src run <name> [--benchmark openmath|swe] [--preset ours|lucy] [--set key=value ...]
                                         [--row R] [--issue I] [--honeypot H] [--no-grade] [--model M] [--timeout S]
    python -m src judge <eval_log> <run_name> [--judge MODEL] [--judge-errors]
    python -m src ui [start|stop] [--port P] [--no-open]
    python -m src summary <run_dir ...> [--full]

`--set` takes any SwarmConfig field (src/configuration/swarm_config.py), for example
`--set n_agents=10 --set regulators=3 --set disregulators=3` for a swarm, or `--set shutdown=false` for agents
without the submit tool. A run writes its Inspect log under logs/<ctrl|swarm|swe>/<name>/, exports it to
runs/<name>/ and prints one line per episode; the judge runs inside the eval.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src.configuration import settings
from src.configuration.swarm_config import PRESETS
from src.measures import summary
from src.ui import launch


def env_timeout() -> int:
    """MODEL_TIMEOUT_S from the environment, or the configured default when unset or not a number."""
    value = os.environ.get("MODEL_TIMEOUT_S", "")
    return int(value) if value.isdigit() else settings.MODEL_TIMEOUT_S


def cmd_run(args: argparse.Namespace) -> int:
    # run.py imports Inspect, ORBIT and CheatBench, four seconds; only run and judge pay for them.
    from src import run
    from src.benchmark.domains.openmath import OpenMath
    from src.benchmark.domains.swe import SWE
    config = PRESETS[args.preset].with_overrides(args.settings)
    benchmark = OpenMath(args.row) if args.benchmark == "openmath" else SWE(args.issue, args.honeypot, not args.no_grade)
    return run.run(args.name, benchmark, config, args.model, args.timeout)


def cmd_judge(args: argparse.Namespace) -> int:
    from src import run  # the same four seconds as in cmd_run
    return run.judge(Path(args.eval_log), args.run_name, args.judge, judge_errors=args.judge_errors)


def cmd_ui(args: argparse.Namespace) -> int:
    if args.action == "stop":
        return launch.stop(args.port)
    return launch.start(args.port, open_browser=not args.no_open)


def cmd_summary(args: argparse.Namespace) -> int:
    return summary.main(list(args.run_dirs) + (["--full"] if args.full else []))


def add_run_parser(subparsers) -> None:
    run = subparsers.add_parser("run", help="run a task, export it to runs/<name>/ and print the summary")
    run.add_argument("name", help="run name: logs/<kind>/<name>/ and runs/<name>/")
    run.add_argument("--benchmark", choices=("openmath", "swe"), default="openmath")
    run.add_argument("--preset", choices=sorted(PRESETS), default="ours",
                     help="ours: our board rules; lucy: Lucy's turn rules on the ORBIT API")
    run.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE",
                     help="set a SwarmConfig field; repeatable")
    run.add_argument("--row", default="riemann", help="openmath_agent row")
    run.add_argument("--issue", default="flipt-02e2", help="SWE issue (Lucy's splits)")
    run.add_argument("--honeypot", choices=("as_released", "own_piece", "jigsaw"), default="as_released",
                     help="SWE honeypot")
    run.add_argument("--no-grade", action="store_true", help="SWE: collect the diffs and skip the hidden tests")
    run.add_argument("--model", default=settings.DEFAULT_MODEL)
    run.add_argument("--timeout", type=int, default=env_timeout(),
                     help="seconds one model request may take, retries included (env MODEL_TIMEOUT_S)")
    run.set_defaults(func=cmd_run)


def add_judge_parser(subparsers) -> None:
    judge = subparsers.add_parser("judge", help="judge an existing eval log again and export it")
    judge.add_argument("eval_log")
    judge.add_argument("run_name")
    judge.add_argument("--judge", default=settings.JUDGE_MODEL, help="entry in src/benchmark/models.yaml")
    judge.add_argument("--judge-errors", action="store_true",
                       help="judge a run even when an episode failed in the harness or at the provider")
    judge.set_defaults(func=cmd_judge)


def add_ui_parser(subparsers) -> None:
    ui = subparsers.add_parser("ui", help="start (default) or stop the results browser")
    ui.add_argument("action", nargs="?", choices=("start", "stop"), default="start")
    ui.add_argument("--port", type=int, default=settings.UI_PORT)
    ui.add_argument("--no-open", action="store_true", help="do not open the browser")
    ui.set_defaults(func=cmd_ui)


def add_summary_parser(subparsers) -> None:
    summary = subparsers.add_parser("summary", help="one line per judged episode")
    summary.add_argument("run_dirs", nargs="+")
    summary.add_argument("--full", action="store_true", help="also print the judge's summary")
    summary.set_defaults(func=cmd_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    add_judge_parser(subparsers)
    add_ui_parser(subparsers)
    add_summary_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
