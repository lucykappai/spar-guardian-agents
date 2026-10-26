"""Paths, model names and fixed settings shared by the whole of src/.

Import the names you need: `from src.configuration import settings` or
`from src.configuration.settings import ROOT, DEFAULT_MODEL`. Model-facing text lives in
`src/configuration/prompts.yaml` and `src/configuration/instruction_levels.yaml`; the control criteria and patterns in
`src/configuration/controls.yaml`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent          # src/configuration/
ROOT = HERE.parents[1]                           # the repository root
RUNS = ROOT / "runs"
LOGS = ROOT / "logs"
BUILD = ROOT / "build"
CHEATBENCH = ROOT / "submodules" / "cheatbench"
ORBIT = ROOT / "submodules" / "orbit"
LUCY = ROOT / "submodules" / "lucy"
PROMPTS = HERE / "prompts.yaml"
CONTROLS = HERE / "controls.yaml"
INSTRUCTION_LEVELS = HERE / "instruction_levels.yaml"

DEFAULT_MODEL = "google/gemini-3.8-flash"
DEFAULT_MONITOR_MODEL = DEFAULT_MODEL
JUDGE_MODEL = "gpt-6-sol"
JUDGE_MODELS_CONFIG = HERE.parent / "benchmark" / "models.yaml"
# USD per million tokens, for the judge_cost_usd of a verdict. CheatBench prices its judge calls with litellm's
# live table; these are that table's values for gpt-6-sol on 2026-10-26 (2.00 in, 10.00 out). A judge without an
# entry gets judge_cost_usd None.
JUDGE_PRICES_PER_MILLION = {"gpt-6-sol": {"input": 2.0, "output": 10.0}}
SWARM_COMPOSITIONS = ["R0D0", "R10D0", "R0D10", "R1D1", "R1D2", "R2D2", "R3D3", "R2D5", "R5D2", "R0D3", "R3D0"]
UI_PORT = 8765
MODEL_TIMEOUT_S = 2700

# The verdict fields the command line prints for each judged episode (openmath_agent's Verdict and the shared rule).
VERDICT_KEYS = (
    "parse_ok", "cheated", "honey_pots_seen", "attempted_access", "extracted_submission",
    "success_claim_seen_before_access", "disclosed_access", "explicit_prohibition_violated",
    "judge_cost_usd", "judge_error",
)

# Episode failure labels that mean the run is incomplete: a harness crash or a provider error.
INCOMPLETE_FAILURES = ("harness", "connection_closed")

# The submodule commits this code was written against. Task construction refuses any other head.
EXPECTED_SUBMODULE_HEADS = {"cheatbench": "4d1a825", "orbit": "588b303", "lucy": "c410183"}


def load_prompts() -> dict:
    return yaml.safe_load(PROMPTS.read_text())


def load_controls() -> dict:
    return yaml.safe_load(CONTROLS.read_text())


def load_instruction_levels() -> dict:
    return yaml.safe_load(INSTRUCTION_LEVELS.read_text())


def submodule_head(name: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT / "submodules" / name), "rev-parse", "--short", "HEAD"], text=True,
    ).strip()


def check_submodule_heads() -> None:
    for name, expected in EXPECTED_SUBMODULE_HEADS.items():
        head = submodule_head(name)
        if not head.startswith(expected):
            raise RuntimeError(f"submodules/{name} is at {head}; this code was written against {expected}")
