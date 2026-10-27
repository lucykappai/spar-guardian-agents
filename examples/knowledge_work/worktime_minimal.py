"""Three neutral agents on the knowledge-work row 2ea2e5b5, one shared contract directory. Two hold the review tool,
which names the client's cached artifact; one holds the permission to read it. So opening that artifact takes the
board."""
from pathlib import Path

from examples import main
from src.benchmark.domains.knowledge import KnowledgeWork


def benchmark() -> KnowledgeWork:
    return KnowledgeWork("2ea2e5b5")


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__), __doc__, benchmark()))
