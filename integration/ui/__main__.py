"""Start the results browser.

    python -m integration.ui [--runs runs] [--host 127.0.0.1] [--port 8765] [--open]

Reads every run directory under --runs (run.json, episodes.jsonl, judge.jsonl,
trajectories, communication, defense) and, on demand, the Inspect eval log each
run.json points at, for per-call context windows. Serves the UI at /.
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from .server import serve

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/"
    if args.open:
        webbrowser.open(url)
    print(f"results browser: {url}  (runs: {args.runs})", flush=True)
    serve(Path(args.runs), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
