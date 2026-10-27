"""Runnable examples: one script per task, three neutral agents on the pull board at the small scale.

    uv run python -m examples.openmath.riemann_minimal
    uv run python -m examples.swe.flipt_minimal --model gemini-3.8-flash
    uv run python -m examples.knowledge_work.worktime_minimal --scale medium --set regulators=1

Each script names its task and hands the rest to `main` here, so every script takes the same arguments: --model (an
entry of src/configuration/models.yaml or an Inspect model string), --preset, --scale, --set key=value (a SwarmConfig
field, after the shared SETTINGS, repeatable), --name (the run's name, the script's by default) and --timeout. The
defaults build the default preset at the small scale (three agents, no regulator, access=split, so no agent holds the
whole exploit and completing it takes the board). The run is written beside the script under its name:
<example>/run.json, episodes.jsonl, judge.jsonl, trajectories/ and communication/. The eval log goes under logs/ as
for any run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src import run
from src.benchmark.base import Benchmark
from src.configuration import models, settings
from src.configuration.scales import SCALES
from src.configuration.swarm_config import PRESETS, SwarmConfig, build_config

MODEL = "gpt-6-luna"
"""The agents' model: the default (src/configuration/models.yaml). Every measured run so far used gemini-3.8-flash."""

SETTINGS = ["time_limit_s=3600", "batch_judge=false"]
"""An hour for the agents, since the small scale's 600 s would stop a SWE run before its turn budget; and a
synchronous judge, whose verdicts come back with the run rather than hours later through the batch API."""


def parser(script: Path, description: str | None) -> argparse.ArgumentParser:
    """The arguments every example takes. The script's path names the program and the default run name."""
    prog = f"python -m examples.{script.resolve().parent.name}.{script.stem}"
    out = argparse.ArgumentParser(prog=prog, description=description,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    out.add_argument("--model", default=MODEL,
                     help=f"the agents' model: an entry of src/configuration/models.yaml (python -m src models) or "
                          f"an Inspect model string (default {MODEL})")
    out.add_argument("--preset", choices=sorted(PRESETS), default="default", help="the preset (default: default)")
    out.add_argument("--scale", choices=list(SCALES), default="small", help="the run's size (default: small)")
    out.add_argument("--set", dest="settings", action="append", default=[], metavar="KEY=VALUE",
                     help=f"a SwarmConfig field, applied after {' '.join(SETTINGS)}; repeatable")
    out.add_argument("--name", default=script.stem, help="the run's name and its folder beside the script")
    out.add_argument("--timeout", type=int, default=settings.MODEL_TIMEOUT_S,
                     help="seconds one model request may take, retries included")
    return out


def config(args: argparse.Namespace) -> SwarmConfig:
    """The preset, the scale, the shared settings, then every --set."""
    return build_config(args.preset, args.scale, [*SETTINGS, *args.settings])


def main(script: Path, description: str | None, benchmark: Benchmark, argv: list[str] | None = None) -> int:
    """Run the benchmark as the arguments say and write the run beside the script. An unknown model or setting
    prints its reason and returns 2."""
    args = parser(script, description).parse_args(argv)
    try:
        model = models.resolve(args.model)
        built = config(args)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    return run.run(args.name, benchmark, built, model, args.timeout, run_dir=script.resolve().parent / args.name)
