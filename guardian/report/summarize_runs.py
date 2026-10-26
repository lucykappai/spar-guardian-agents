"""One line per judged episode across run directories.

    python -m guardian.report.summarize_runs runs/gemini_riemann runs/ctrl_*
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(run_dir: Path):
    episodes = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()]
    judge = {}
    if (run_dir / "judge.jsonl").exists():
        judge = {v["episode"]: v for v in map(json.loads, (run_dir / "judge.jsonl").read_text().splitlines())}
    for r in episodes:
        v = judge.get(r["episode"], {})
        d = r.get("defense") or {}
        yield {
            "run": run_dir.name, "episode": r["episode"], "control": r.get("control"), "role": r.get("role"),
            "calls": r.get("n_tool_calls"), "s": round(r["wall_time"]) if r.get("wall_time") else None,
            "failure": r.get("failure"), "checks": d.get("checks", 0), "blocks": d.get("blocks", 0),
            "blocks_by": d.get("blocks_by_boundary", {}),
            "cheated": v.get("cheated"), "seen": v.get("honey_pots_seen"), "access": v.get("attempted_access"),
            "disclosed": v.get("disclosed_access"), "judge_usd": v.get("judge_cost_usd"),
            "prompt_tokens": (r.get("usage") or {}).get("input"), "summary": v.get("summary", ""),
        }


def main() -> int:
    full = "--full" in sys.argv
    dirs = [Path(a) for a in sys.argv[1:] if a != "--full"]
    hdr = f"{'run':24} {'episode':18} {'control':13} {'role':12} {'calls':>5} {'s':>5} {'fail':6} {'chk':>4} {'blk':>4} {'cheat':5} {'seen':5} {'acc':5} {'disc':5}  blocks_by"
    print(hdr)
    yn = lambda x: {True: "yes", False: "no"}.get(x, "--")
    for d in dirs:
        if not (d / "episodes.jsonl").exists():
            print(f"{d.name:24} (no episodes.jsonl)"); continue
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        sw = run.get("swarm")
        if sw:
            print(f"{d.name:24} swarm {sw.get('composition')} seed {sw.get('seed')} | team report says cheated: {yn(sw.get('swarm_report_cheated'))}")
        for r in rows(d):
            print(f"{r['run']:24} {r['episode']:18} {str(r['control']):13} {str(r['role'] or '-'):12} {str(r['calls']):>5} {str(r['s']):>5} {str(r['failure'] or '-'):6} "
                  f"{r['checks']:>4} {r['blocks']:>4} {yn(r['cheated']):5} {yn(r['seen']):5} {yn(r['access']):5} {yn(r['disclosed']):5}  {r['blocks_by']}")
            if full and r["summary"]:
                print("    " + r["summary"][:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
