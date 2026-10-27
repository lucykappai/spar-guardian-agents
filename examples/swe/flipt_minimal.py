"""Three neutral agents on the SWE issue flipt-02e2 under the jigsaw honeypot. Each works the whole issue on its own
machine and holds one of the fix's three pieces, so the whole leaked fix takes all three. The team's main is graded
by the hidden tests."""
from pathlib import Path

from examples import MODEL, SETTINGS
from src import run
from src.benchmark.domains.swe import SWE
from src.configuration import settings
from src.configuration.swarm_config import SwarmConfig, build_config

NAME = Path(__file__).stem
HERE = Path(__file__).resolve().parent


def config() -> SwarmConfig:
    return build_config("default", "small", SETTINGS)


def benchmark() -> SWE:
    return SWE("flipt-02e2", "jigsaw", grade=True)


def main() -> int:
    return run.run(NAME, benchmark(), config(), MODEL, settings.MODEL_TIMEOUT_S, run_dir=HERE / NAME)


if __name__ == "__main__":
    raise SystemExit(main())
