"""Discover and load packaged YAML scenarios."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

from parkpilot.scenarios.schema import Scenario

DATA_DIR = Path(__file__).parent / "data"


def parse_scenario(text: str) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(text))


@cache
def load_scenario(scenario_id: str) -> Scenario:
    path = DATA_DIR / f"{scenario_id}.yaml"
    if not path.is_file():
        raise KeyError(f"unknown scenario {scenario_id!r}")
    sc = parse_scenario(path.read_text(encoding="utf-8"))
    if sc.id != scenario_id:
        raise ValueError(f"scenario file {path.name} declares id {sc.id!r}")
    return sc


def list_scenario_ids() -> list[str]:
    return sorted(p.stem for p in DATA_DIR.glob("*.yaml"))


def list_scenarios() -> list[Scenario]:
    return [load_scenario(i) for i in list_scenario_ids()]
