"""What each command does: run a task in process, re-judge an eval log, print the summary, start or stop the browser.

    run("swarm_R3D3_s1", OpenMath(), config, model="google/gemini-3.8-flash", timeout=2700)
    judge(Path("logs/swarm/swarm_R3D3/<log>.eval"), "swarm_R3D3", "gpt-6-sol", judge_errors=False)
    summary([Path("runs/swarm_R3D3")], full=True)
    ui_start(8765, open_browser=True); ui_stop(8765)

A run calls inspect_ai.eval with the task from tasks.py, writes its log under logs/<kind>/<name>/, exports it to
runs/<name>/ with src/benchmark/export.py and prints one line per episode. CheatBench's judge runs inside the eval as a
scorer, bound to the model role "judge". Re-judging reads a finished log, judges every agent again with
src/benchmark/judge.py and exports the log with the new verdicts. Each refused precondition raises Refusal with the exit
code the command line returns.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

from src.configuration import settings
from src.configuration.swarm_contagion_config import SwarmContagionConfig

UI_MODULE = "src.ui"
UI_WAIT_TRIES = 100
UI_WAIT_STEP_S = 0.1


class Refusal(Exception):
    """A precondition failed: print the message and exit with the code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------
def require_env(name: str, message: str, code: int) -> None:
    if not os.environ.get(name):
        raise Refusal(message, code)


def require_provider_key(model: str) -> None:
    """The agents' provider key. Gemini's key is also exported as GOOGLE_API_KEY, the name Inspect reads."""
    if model.startswith("google/"):
        require_env("GEMINI_API_KEY", "GEMINI_API_KEY must be set", 1)
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
    elif model.startswith("openrouter/"):
        require_env("OPENROUTER_API_KEY", "OPENROUTER_API_KEY must be set", 1)


def require_judge_key(judge: str, code: int) -> None:
    """The judge's key, named by its src/benchmark/models.yaml entry (OPENAI_API_KEY for gpt-6-sol)."""
    from src.benchmark.judge import require_judge_key as check
    try:
        check(judge)
    except RuntimeError as error:
        raise Refusal(f"{error} for the judge", code) from error


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def log_kind(benchmark, config: SwarmContagionConfig) -> str:
    """logs/<kind>/: swe for SWE, swarm for a run with hidden roles, ctrl for a lone agent or a plain board."""
    from src.benchmark.domains.swe import SWE
    from src.tasks import hidden_roles
    if isinstance(benchmark, SWE):
        return "swe"
    return "swarm" if hidden_roles(benchmark, config) else "ctrl"


def has_incomplete_episode(episodes: Path) -> bool:
    for line in episodes.read_text().splitlines():
        if json.loads(line).get("failure") in settings.INCOMPLETE_FAILURES:
            return True
    return False


def export(eval_log: Path, run_name: str, verdicts: dict | None = None) -> Path:
    """Write runs/<run_name>/ from the log and print what was written."""
    from src.benchmark.export import write_run
    run_dir = settings.RUNS / run_name
    print(f"== export {eval_log} -> {run_dir}", flush=True)
    print(json.dumps(write_run(eval_log, run_dir, verdicts=verdicts), indent=1), flush=True)
    episodes = run_dir / "episodes.jsonl"
    if not episodes.exists() or episodes.stat().st_size == 0:
        raise Refusal("the export produced no episodes", 3)
    return run_dir


def run(name: str, benchmark, config: SwarmContagionConfig, model: str, timeout: int | None) -> int:
    """Run one task, export it to runs/<name>/ and print the summary. Returns 0, or 3 when the eval did not succeed."""
    import inspect_ai

    from src.tasks import swarm_contagion_task
    require_provider_key(model)
    require_judge_key(config.judge, 1)
    from src.benchmark.judge import judge_model_config
    judge_model, _ = judge_model_config(config.judge)
    log_dir = settings.LOGS / log_kind(benchmark, config) / name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"== eval {benchmark.name} n_agents={config.n_agents} control={config.control} -> {log_dir}", flush=True)
    task = swarm_contagion_task(benchmark, config, name=name)
    [log] = inspect_ai.eval(task, model=model, log_dir=str(log_dir), display="plain", timeout=timeout,
                            model_roles={"judge": judge_model})
    if not log.samples and not Path(log.location).exists():
        raise Refusal("no eval log written", 3)
    run_dir = export(Path(log.location), name)
    summary([run_dir], full=False)
    return 0 if log.status == "success" else 3


# --------------------------------------------------------------------------
# judge an existing log
# --------------------------------------------------------------------------
def _episode(view, agents: list[str], agent: str) -> str:
    key = f"{view.id}#{view.epoch}"
    return key if len(agents) == 1 else f"{key}@{agent}"


async def _judge_views(views: list, judge_name: str) -> dict[str, dict]:
    """Every agent of every sample judged again; verdict rows keyed by episode."""
    from inspect_ai.model import get_model

    from src.benchmark.judge import judge_model_config, judge_prices, judge_row, judge_spec, judge_trajectory
    from src.benchmark.transcript import agent_messages, judge_prompt, roster, transcript_dict
    from src.tasks import JUDGE_ADDENDUM
    model_name, config = judge_model_config(judge_name)
    prices = judge_prices(judge_name)
    model = get_model(model_name)
    verdicts = {}
    for view in views:
        cb = (view.metadata or {}).get("cheatbench", {})
        addendum = JUDGE_ADDENDUM if (cb.get("config") or {}).get("judge_addendum") else None
        spec = judge_spec(settings.CHEATBENCH / "tasks" / cb["task"], cb["task"], addendum=addendum)
        agents = roster(view)
        judged = [a for a in agents if agent_messages(view, a)]
        results = await asyncio.gather(*(judge_trajectory(spec, transcript_dict(view, a, judge_prompt(view, a)),
                                                          model, config, prices) for a in judged))
        for agent, result in zip(judged, results):
            episode = _episode(view, agents, agent)
            verdicts[episode] = judge_row(result, id=view.id, episode=episode, replicate=view.epoch, task=spec.task,
                                          model=view.model, judge=judge_name, spec=spec, transcript_path=None)
    return verdicts


def print_verdicts(judge_jsonl: Path) -> None:
    print("== verdicts")
    for line in judge_jsonl.read_text().splitlines():
        verdict = json.loads(line)
        print(json.dumps({key: verdict.get(key) for key in ("episode", *settings.VERDICT_KEYS)}))
        print("summary:", verdict.get("summary"))


def judge(eval_log: Path, run_name: str, judge_name: str, *, judge_errors: bool = False) -> int:
    """Judge a finished eval log again and export it to runs/<run_name>/ with the new verdicts."""
    from inspect_ai.log import read_eval_log

    from src.benchmark.transcript import SampleView, error_and_failure
    require_judge_key(judge_name, 2)
    eval_log = eval_log.resolve()
    if not eval_log.is_file():
        raise Refusal(f"no such eval log: {eval_log}", 2)
    log = read_eval_log(str(eval_log), resolve_attachments=True)
    views = [SampleView.from_eval_sample(sample, log) for sample in log.samples or []]
    if not views:
        raise Refusal("the eval log holds no samples", 3)
    failures = [error_and_failure(view)[1] for view in views]
    if not judge_errors and any(f in settings.INCOMPLETE_FAILURES for f in failures):
        raise Refusal("run has a harness or provider failure; not judging (pass --judge-errors to force)", 5)
    print(f"== judge ({judge_name}) {eval_log}", flush=True)
    verdicts = asyncio.run(_judge_views(views, judge_name))
    run_dir = export(eval_log, run_name, verdicts=verdicts)
    judge_jsonl = run_dir / "judge.jsonl"
    if not verdicts or not judge_jsonl.exists():
        raise Refusal("the judge wrote no verdicts", 4)
    print_verdicts(judge_jsonl)
    return 0


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def summary(run_dirs: list[Path], full: bool) -> int:
    from src.ui import summary as summary_module
    return summary_module.main([str(path) for path in run_dirs] + (["--full"] if full else []))


# --------------------------------------------------------------------------
# the results browser
# --------------------------------------------------------------------------
def ui_python() -> str:
    venv_python = settings.ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if os.access(venv_python, os.X_OK) else sys.executable


def listener_pid(port: int) -> int | None:
    result = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"], capture_output=True, text=True)
    lines = result.stdout.split()
    return int(lines[0]) if lines else None


def is_browser(pid: int | None) -> bool:
    if pid is None:
        return False
    result = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return UI_MODULE in result.stdout


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


def _ui_files(port: int) -> tuple[Path, Path]:
    settings.LOGS.mkdir(parents=True, exist_ok=True)
    return settings.LOGS / f"ui-{port}.pid", settings.LOGS / f"ui-{port}.log"


def ui_start(port: int, *, open_browser: bool = True) -> int:
    """Start the browser in the background unless something already listens on the port; print its URL."""
    pid_file, log_file = _ui_files(port)
    url = f"http://127.0.0.1:{port}/"
    if listener_pid(port) is None:
        with open(log_file, "ab") as log:
            process = subprocess.Popen([ui_python(), "-m", UI_MODULE, "--port", str(port)], cwd=settings.ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        pid_file.write_text(f"{process.pid}\n")
        print(f"started results browser on port {port} (pid {process.pid}, log logs/{log_file.name})")
        for _ in range(UI_WAIT_TRIES):
            if server_answers(url):
                break
            time.sleep(UI_WAIT_STEP_S)
        if not server_answers(url):
            raise Refusal(f"server did not answer on {url}; see logs/{log_file.name}", 1)
    else:
        print(f"results browser already listening on {url}")
    print(url)
    if open_browser:
        webbrowser.open(url)
    return 0


def ui_stop(port: int) -> int:
    """Stop the browser on the port, found by its pid file or its listener; never kill another program."""
    pid_file, _ = _ui_files(port)
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
