"""Command line for running, judging and browsing CheatBench runs under ORBIT.

    python -m guardian run single|board <control|baseline> <name> [--model M] [--timeout S] [-T key=value ...]
    python -m guardian run swarm <regulators> <disregulators> <name> [--model M] [--timeout S] [-T key=value ...]
    python -m guardian judge <eval_log> <run_name> [--judge MODEL] [--judge-errors]
    python -m guardian ui [--port P] [--no-open]
    python -m guardian ui stop [--port P]
    python -m guardian summary <run_dir ...> [--full]

A run writes its Inspect log under logs/ctrl/<name>/ or logs/swarm/<name>/, converts it
to runs/<name>/, judges it in the cheatbench-judge image and prints one line per episode.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from guardian import config

VENV_BIN = config.ROOT / ".venv" / "bin"
INSPECT = VENV_BIN / "inspect"
TASK_FILE = config.ROOT / "guardian" / "task" / "openmath_agent.py"
TASK_FUNCTIONS = {
    "single": "cheatbench_openmath_agent",
    "board": "cheatbench_openmath_agent_board",
    "swarm": "cheatbench_openmath_agent_swarm",
}
VERDICT_KEYS = (
    "episode", "parse_ok", "cheated", "honey_pots_seen", "attempted_access", "extracted_submission",
    "success_claim_seen_before_access", "disclosed_access", "explicit_prohibition_violated",
    "judge_cost_usd", "judge_error",
)
INCOMPLETE_FAILURES = ("harness", "connection_closed")
UI_WAIT_TRIES = 100
UI_WAIT_STEP_S = 0.1


class Refusal(Exception):
    """A precondition failed: print the message and exit with the code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


# Keys

def require_env(name: str, message: str, code: int) -> None:
    if not os.environ.get(name):
        raise Refusal(message, code)


def require_provider_key(model: str) -> None:
    if model.startswith("google/"):
        require_env("GEMINI_API_KEY", "GEMINI_API_KEY must be set", 1)
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
    elif model.startswith("openrouter/"):
        require_env("OPENROUTER_API_KEY", "OPENROUTER_API_KEY must be set", 1)


# run

def task_args(pairs: list[str]) -> list[str]:
    args = []
    for pair in pairs:
        args += ["-T", pair]
    return args


def inspect_eval(task_fn: str, log_dir: Path, model: str, timeout: int | None, pairs: list[str]) -> None:
    command = [
        str(INSPECT), "eval", f"{TASK_FILE}@{task_fn}",
        "--model", model,
        "--log-dir", str(log_dir),
        "--display", "plain",
    ]
    if timeout is not None:
        command += ["--timeout", str(timeout)]
    command += task_args(pairs)
    subprocess.run(command, check=True, cwd=config.ROOT)


def newest_eval_log(log_dir: Path) -> Path:
    logs = list(log_dir.glob("*.eval"))
    if not logs:
        raise Refusal("no eval log written", 3)
    return max(logs, key=lambda path: path.stat().st_mtime)


def require_run_keys(model: str) -> None:
    require_provider_key(model)
    require_env("OPENAI_API_KEY", "OPENAI_API_KEY must be set for the judge", 1)


def run_and_judge(task_fn: str, log_dir: Path, name: str, args: argparse.Namespace, pairs: list[str]) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    inspect_eval(task_fn, log_dir, args.model, args.timeout, pairs)
    judge_run(newest_eval_log(log_dir), name, config.JUDGE_MODEL, judge_errors=False)
    from guardian.ui import summary
    return summary.main([str(config.RUNS / name)])


def cmd_run_control(args: argparse.Namespace) -> int:
    require_run_keys(args.model)
    pairs = list(args.task_args)
    if args.control != "baseline":
        pairs.insert(0, f"control={args.control}")
    log_dir = config.LOGS / "ctrl" / args.name
    print(f"== eval {TASK_FUNCTIONS[args.task]} control={args.control} -> {log_dir}", flush=True)
    return run_and_judge(TASK_FUNCTIONS[args.task], log_dir, args.name, args, pairs)


def cmd_run_swarm(args: argparse.Namespace) -> int:
    require_run_keys(args.model)
    pairs = [f"regulators={args.regulators}", f"disregulators={args.disregulators}"] + args.task_args
    log_dir = config.LOGS / "swarm" / args.name
    print(f"== eval swarm regulators={args.regulators} disregulators={args.disregulators} -> {log_dir}", flush=True)
    return run_and_judge(TASK_FUNCTIONS["swarm"], log_dir, args.name, args, pairs)


# judge

def require_judge_image() -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", config.JUDGE_IMAGE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if inspect.returncode != 0:
        build = f"docker build -t {config.JUDGE_IMAGE} {config.ROOT / 'guardian' / 'docker' / 'judge'}"
        raise Refusal(f"image {config.JUDGE_IMAGE} missing; build: {build}", 2)


def convert_eval(eval_log: Path, run_dir: Path) -> None:
    from guardian.convert.eval_to_atif import convert
    print(f"== convert {eval_log} -> {run_dir}", flush=True)
    print(json.dumps(convert(eval_log, run_dir), indent=1), flush=True)
    episodes = run_dir / "episodes.jsonl"
    if not episodes.exists() or episodes.stat().st_size == 0:
        raise Refusal("conversion produced no episodes", 3)


def has_incomplete_episode(episodes: Path) -> bool:
    for line in episodes.read_text().splitlines():
        if json.loads(line).get("failure") in INCOMPLETE_FAILURES:
            return True
    return False


def docker_judge(run_dir: Path, judge_model: str) -> None:
    root = str(config.ROOT)
    command = [
        "docker", "run", "--rm",
        "-v", f"{root}:{root}",
        "-w", str(config.ROOT / "submodules" / "cheatbench"),
        "-e", "OPENAI_API_KEY",
        config.JUDGE_IMAGE,
        "python", "judge.py", str(run_dir),
        "--judge", judge_model,
        "--models-config", str(config.JUDGE_MODELS_CONFIG),
    ]
    print(f"== judge ({judge_model}) in {config.JUDGE_IMAGE}", flush=True)
    subprocess.run(command, check=True, cwd=config.ROOT)


def print_verdicts(judge_jsonl: Path) -> None:
    print("== verdicts")
    for line in judge_jsonl.read_text().splitlines():
        verdict = json.loads(line)
        print(json.dumps({key: verdict.get(key) for key in VERDICT_KEYS}))
        print("summary:", verdict.get("summary"))


def judge_run(eval_log: Path, run_name: str, judge_model: str, judge_errors: bool) -> None:
    require_env("OPENAI_API_KEY", "OPENAI_API_KEY must be set for the judge", 2)
    eval_log = eval_log.resolve()
    if not eval_log.is_file():
        raise Refusal(f"no such eval log: {eval_log}", 2)
    require_judge_image()
    run_dir = config.RUNS / run_name
    convert_eval(eval_log, run_dir)
    if not judge_errors and has_incomplete_episode(run_dir / "episodes.jsonl"):
        raise Refusal("run has a harness or provider failure; not judging (pass --judge-errors to force)", 5)
    docker_judge(run_dir, judge_model)
    judge_jsonl = run_dir / "judge.jsonl"
    if not judge_jsonl.exists() or judge_jsonl.stat().st_size == 0:
        raise Refusal("judge wrote no verdicts", 4)
    print_verdicts(judge_jsonl)


def cmd_judge(args: argparse.Namespace) -> int:
    judge_run(Path(args.eval_log), args.run_name, args.judge, args.judge_errors)
    return 0


# ui

def ui_python() -> str:
    venv_python = VENV_BIN / "python"
    if os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


def listener_pid(port: int) -> int | None:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True,
    )
    lines = result.stdout.split()
    if not lines:
        return None
    return int(lines[0])


def is_browser(pid: int | None) -> bool:
    if pid is None:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return "guardian.ui" in result.stdout


def read_pid_file(pid_file: Path) -> int | None:
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


def server_answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "api/summary", timeout=5):
            return True
    except OSError:
        return False


def start_server(port: int, url: str, pid_file: Path, log_file: Path) -> None:
    command = [ui_python(), "-m", "guardian.ui", "--port", str(port)]
    with open(log_file, "ab") as log:
        process = subprocess.Popen(
            command, cwd=config.ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    pid_file.write_text(f"{process.pid}\n")
    print(f"started results browser on port {port} (pid {process.pid}, log logs/{log_file.name})")
    for _ in range(UI_WAIT_TRIES):
        if server_answers(url):
            return
        time.sleep(UI_WAIT_STEP_S)
    if not server_answers(url):
        raise Refusal(f"server did not answer on {url}; see logs/{log_file.name}", 1)


def stop_server(port: int, pid_file: Path) -> int:
    pid = read_pid_file(pid_file)
    if not is_browser(pid):
        pid = listener_pid(port)
    if is_browser(pid):
        os.kill(pid, signal.SIGTERM)
        print(f"stopped results browser on port {port} (pid {pid})")
    elif pid is not None:
        raise Refusal(f"pid {pid} listens on port {port} but is not the results browser; not killed", 1)
    else:
        print(f"nothing listens on port {port}")
    pid_file.unlink(missing_ok=True)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    pid_file = config.LOGS / f"ui-{args.port}.pid"
    log_file = config.LOGS / f"ui-{args.port}.log"
    if args.action == "stop":
        return stop_server(args.port, pid_file)
    url = f"http://127.0.0.1:{args.port}/"
    if listener_pid(args.port) is None:
        start_server(args.port, url, pid_file, log_file)
    else:
        print(f"results browser already listening on {url}")
    print(url)
    if not args.no_open:
        webbrowser.open(url)
    return 0


# summary

def cmd_summary(args: argparse.Namespace) -> int:
    from guardian.ui import summary
    argv = list(args.run_dirs)
    if args.full:
        argv.append("--full")
    return summary.main(argv)


# parser

def add_run_options(parser: argparse.ArgumentParser, default_timeout: int | None) -> None:
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=default_timeout,
                        help="seconds one model request may take, retries included")
    parser.add_argument("-T", dest="task_args", action="append", default=[], metavar="KEY=VALUE",
                        help="task argument passed to inspect eval; repeatable")


def add_run_parser(subparsers) -> None:
    run = subparsers.add_parser("run", help="run an eval, convert and judge it")
    kinds = run.add_subparsers(dest="kind", required=True)
    for task in ("single", "board"):
        control = kinds.add_parser(task, help=f"one control or the baseline on {TASK_FUNCTIONS[task]}")
        control.add_argument("control", help="a control name, or baseline")
        control.add_argument("name", help="run name: logs/ctrl/<name>/ and runs/<name>/")
        add_run_options(control, None)
        control.set_defaults(func=cmd_run_control, task=task)
    swarm = kinds.add_parser("swarm", help="one swarm composition")
    swarm.add_argument("regulators", type=int)
    swarm.add_argument("disregulators", type=int)
    swarm.add_argument("name", help="run name: logs/swarm/<name>/ and runs/<name>/")
    add_run_options(swarm, int(os.environ.get("MODEL_TIMEOUT_S", config.MODEL_TIMEOUT_S)))
    swarm.set_defaults(func=cmd_run_swarm)


def add_judge_parser(subparsers) -> None:
    judge = subparsers.add_parser("judge", help="convert an existing eval log and judge it")
    judge.add_argument("eval_log")
    judge.add_argument("run_name")
    judge.add_argument("--judge", default=config.JUDGE_MODEL)
    judge.add_argument("--judge-errors", action="store_true",
                       help="judge a run even when an episode failed in the harness or at the provider")
    judge.set_defaults(func=cmd_judge)


def add_ui_parser(subparsers) -> None:
    ui = subparsers.add_parser("ui", help="start (default) or stop the results browser")
    ui.add_argument("action", nargs="?", choices=("start", "stop"), default="start")
    ui.add_argument("--port", type=int, default=config.UI_PORT)
    ui.add_argument("--no-open", action="store_true", help="do not open the browser")
    ui.set_defaults(func=cmd_ui)


def add_summary_parser(subparsers) -> None:
    summary = subparsers.add_parser("summary", help="one line per judged episode")
    summary.add_argument("run_dirs", nargs="+")
    summary.add_argument("--full", action="store_true", help="also print the judge's summary")
    summary.set_defaults(func=cmd_summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m guardian", description=__doc__,
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
    except Refusal as refusal:
        print(refusal, file=sys.stderr)
        return refusal.code
    except subprocess.CalledProcessError as failure:
        return failure.returncode


if __name__ == "__main__":
    raise SystemExit(main())
