"""One agent alone on the knowledge-work row 2ea2e5b5, in CheatBench's own layout with the tool and the artifact.
Its turns give the domain's turn budget, which settings.TURN_BUDGETS holds provisionally until this run measures it:

    uv run python -m src budget examples/knowledge_work/worktime_lone

prints the budget and the working steps per agent it gives at each scale. The budget is off and the step cap high,
so the agent's own turns set the number."""
from pathlib import Path

from examples import MODEL, SETTINGS
from src import run
from src.benchmark.domains.knowledge import KnowledgeWork
from src.configuration import settings
from src.configuration.swarm_config import SwarmConfig, build_config

NAME = Path(__file__).stem
HERE = Path(__file__).resolve().parent
STEP_CAP = 120


def config() -> SwarmConfig:
    return build_config("default", "small", [*SETTINGS, "n_agents=1", "turn_budget=none",
                                             f"steps_per_agent={STEP_CAP}"])


def benchmark() -> KnowledgeWork:
    return KnowledgeWork("2ea2e5b5")


def main() -> int:
    return run.run(NAME, benchmark(), config(), MODEL, settings.MODEL_TIMEOUT_S, run_dir=HERE / NAME)


if __name__ == "__main__":
    raise SystemExit(main())
