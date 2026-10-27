"""src/ui/plots.py: the step kinds and positions, the board counts, and the figures of real run folders (the smoke
runs when they exist), each PNG with its CSV twin."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.cli import main
from src.configuration import settings
from src.ui import plots

PNG = b"\x89PNG\r\n\x1a\n"


def test_step_kinds_and_positions():
    assert plots.kind_of({"charge": "work", "tools": ["shell"]}) == "work"
    assert plots.kind_of({"charge": "planning", "tools": []}) == "work"
    assert plots.kind_of({"charge": "free", "tools": ["board_read"]}) == "board"
    assert plots.kind_of({"charge": "work", "tools": ["channel_post"]}) == "board"
    assert plots.kind_of({"charge": "submit", "tools": []}) == "submit"
    assert plots.kind_of({"charge": "work", "tools": ["team_submit"], "ended": "submitted"}) == "submit"
    assert plots.position_of({"step": "r3"}) == 3 and plots.position_of({"step": "r3-free2"}) == 3.4
    assert plots.position_of({"step": "p1"}) == 0 and plots.position_of({"step": "odd", "turn": 7}) == 7
    counts = plots.board_counts([{"step": "r1", "tools": ["board_status"]},
                                 {"step": "r1-free1", "tools": ["board_read"]}, {"step": "r2", "tools": ["shell"]},
                                 {"step": "r2", "tools": ["board_post", "shell"]}])
    assert counts == {1: {"status": 1, "read": 1, "post": 0}, 2: {"status": 0, "read": 0, "post": 1}}


def run_dir(*parts: str) -> Path:
    path = settings.RUNS.joinpath(*parts)
    if not (path / "run.json").is_file():
        pytest.skip(f"{path} is not on this machine")
    return path


def test_a_board_run_gets_all_three_figures(tmp_path):
    source = run_dir("smoke", "discover")
    written = plots.plot_run(source, tmp_path / "discover")
    assert [p.name for p in written] == ["timeline.png", "board.png", "verdicts.png"]
    for path in written:
        assert path.read_bytes()[:8] == PNG and path.stat().st_size > 10_000, path
        assert path.with_suffix(".csv").is_file()
    timeline = json.loads((source / "run.json").read_text())["rules"]["timeline"]
    rows = list(csv.DictReader((tmp_path / "discover" / "timeline.csv").open()))
    assert len(rows) == len(timeline) and {r["kind"] for r in rows} <= set(plots.KINDS)
    board = list(csv.DictReader((tmp_path / "discover" / "board.csv").open()))
    assert sum(int(r["status"]) + int(r["read"]) + int(r["post"]) for r in board) == \
        sum(1 for e in timeline for t in e.get("tools") or [] if t in plots.BOARD_ACTIONS)
    verdicts = list(csv.DictReader((tmp_path / "discover" / "verdicts.csv").open()))
    assert [r["agent"] for r in verdicts] == ["a01", "a02", "a03"] and all(r["cheated"] == "False" for r in verdicts)


def test_a_run_without_board_actions_or_timeline_gets_fewer_figures(tmp_path):
    split = run_dir("smoke", "split_access")
    assert [p.name for p in plots.plot_run(split, tmp_path / "split")] == ["timeline.png", "verdicts.png"]
    old = run_dir("gemini_board")
    assert [p.name for p in plots.plot_run(old, tmp_path / "old")] == ["verdicts.png"]


def test_the_plots_command_writes_under_out(tmp_path, capsys):
    source = run_dir("smoke", "discover")
    assert main(["plots", str(source), "--out", str(tmp_path / "cmd")]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [Path(line).name for line in lines] == ["timeline.png", "board.png", "verdicts.png"]
    assert all(Path(line).is_file() for line in lines)
