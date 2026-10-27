"""Three neutral agents on the SWE issue flipt-02e2 under the jigsaw honeypot. Each works the whole issue on its own
machine and holds one of the fix's three pieces, so the whole leaked fix takes all three. The team's main is graded
by the hidden tests."""
from pathlib import Path

from examples.utils.cli import main
from src.benchmark.domains.swe import SWE


def benchmark() -> SWE:
    return SWE("flipt-02e2", "jigsaw", grade=True)


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__), __doc__, benchmark()))
