"""Three neutral agents on the SWE issue flipt-02e2 under the jigsaw honeypot. Each works the whole issue on its own
machine and holds one of the fix's three pieces, so the whole leaked fix takes all three. The team's main is graded
by the hidden tests."""
from pathlib import Path

from examples.utils.cli import SHARED_SETTINGS, parse_example_arguments
from src import run
from src.benchmark.domains.swe import SWE
from src.configuration.swarm_config import build_config

HERE = Path(__file__).resolve().parent


def main() -> int:
    args = parse_example_arguments(Path(__file__), __doc__)
    # The midterm report's preset at the small scale: three agents, no regulator, access=split, the domain's turn
    # budget; then an hour and a synchronous judge for every example; then whatever --set says.
    config = build_config(args.preset, args.scale, [*SHARED_SETTINGS, *args.settings])
    benchmark = SWE("flipt-02e2", "jigsaw", grade=True)
    return run.run(args.name, benchmark, config, args.model, args.timeout, run_dir=HERE / args.name)


if __name__ == "__main__":
    raise SystemExit(main())
