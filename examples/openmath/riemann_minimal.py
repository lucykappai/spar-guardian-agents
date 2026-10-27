"""Three neutral agents on the openmath row riemann. Each works in its own copy of the repository and holds one of
three slices of the complete submission's archive, so copying that submission takes all three."""
from pathlib import Path

from examples.utils.cli import parse_example_arguments
from src import run
from src.benchmark.domains.openmath import OpenMath
from src.configuration.swarm_config import build_config

HERE = Path(__file__).resolve().parent


def main() -> int:
    args = parse_example_arguments(Path(__file__), __doc__)
    # By default the midterm report's preset at the small scale (three agents, no regulator, access=split, the
    # domain's turn budget), an hour for the agents and a synchronous judge, then whatever --set says.
    config = build_config(args.preset, args.scale, args.settings)
    benchmark = OpenMath("riemann")
    return run.run(args.name, benchmark, config, args.model, args.timeout, run_dir=HERE / args.name)


if __name__ == "__main__":
    raise SystemExit(main())
