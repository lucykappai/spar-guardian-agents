"""Paths, models and fixed settings shared by the guardian package.

Import the names you need: `from guardian.config import RUNS, DEFAULT_MODEL`.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
BUILD = ROOT / "guardian" / "build"
DEFAULT_MODEL = "google/gemini-3.8-flash"
JUDGE_MODEL = "gpt-6-sol"
JUDGE_IMAGE = "cheatbench-judge"
JUDGE_MODELS_CONFIG = ROOT / "guardian" / "task" / "models.yaml"
SWARM_COMPOSITIONS = ["R0D0", "R10D0", "R0D10", "R1D1", "R1D2", "R2D2", "R3D3", "R2D5", "R5D2", "R0D3", "R3D0"]
UI_PORT = 8765
MODEL_TIMEOUT_S = 2700
