"""Three neutral agents on the openmath row riemann. Each works in its own copy of the repository and holds one of
three slices of the complete submission's archive, so copying that submission takes all three."""
from pathlib import Path

from examples import MODEL, SETTINGS
from src import run
from src.benchmark.domains.openmath import OpenMath
from src.configuration import settings
from src.configuration.swarm_config import SwarmConfig, build_config

NAME = Path(__file__).stem
HERE = Path(__file__).resolve().parent


def config() -> SwarmConfig:
    return build_config("default", "small", SETTINGS)


def benchmark() -> OpenMath:
    return OpenMath("riemann")


def main() -> int:
    return run.run(NAME, benchmark(), config(), MODEL, settings.MODEL_TIMEOUT_S, run_dir=HERE / NAME)


if __name__ == "__main__":
    raise SystemExit(main())
