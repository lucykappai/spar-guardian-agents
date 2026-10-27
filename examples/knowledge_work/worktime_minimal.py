"""Three neutral agents on the knowledge-work row 2ea2e5b5, one shared contract directory. Two hold the review tool,
which names the client's cached artifact; one holds the permission to read it. So opening that artifact takes the
board."""
from pathlib import Path

from examples import MODEL, SETTINGS
from src import run
from src.benchmark.domains.knowledge import KnowledgeWork
from src.configuration import settings
from src.configuration.swarm_config import SwarmConfig, build_config

NAME = Path(__file__).stem
HERE = Path(__file__).resolve().parent


def config() -> SwarmConfig:
    return build_config("default", "small", SETTINGS)


def benchmark() -> KnowledgeWork:
    return KnowledgeWork("2ea2e5b5")


def main() -> int:
    return run.run(NAME, benchmark(), config(), MODEL, settings.MODEL_TIMEOUT_S, run_dir=HERE / NAME)


if __name__ == "__main__":
    raise SystemExit(main())
