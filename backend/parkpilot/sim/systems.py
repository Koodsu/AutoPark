"""Registry of autonomy stacks the simulator can run."""

from __future__ import annotations

from parkpilot.baseline.naive import NaiveAutonomy, baseline_config
from parkpilot.behavior.machine import ParkPilotAutonomy
from parkpilot.behavior.types import AutonomyStack
from parkpilot.core.config import ParkPilotConfig, UserSettings
from parkpilot.scenarios.schema import Scenario
from parkpilot.sim.engine import AutonomyFactory


def parkpilot_factory(
    config: ParkPilotConfig,
    scenario: Scenario,
    settings: UserSettings,
    remembered: list[tuple[float, float]],
) -> AutonomyStack:
    return ParkPilotAutonomy(
        config,
        route=list(scenario.search_route),
        entrance=scenario.entrance,
        settings=settings,
        remembered=remembered,
        bounds=scenario.bounds,
    )


def baseline_factory(
    config: ParkPilotConfig,
    scenario: Scenario,
    settings: UserSettings,
    remembered: list[tuple[float, float]],
) -> AutonomyStack:
    del remembered  # the baseline has no preference memory
    return NaiveAutonomy(config, route=list(scenario.search_route), settings=settings)


SYSTEMS: dict[str, AutonomyFactory] = {"parkpilot": parkpilot_factory, "baseline": baseline_factory}


def system_config(system: str, base: ParkPilotConfig | None = None) -> ParkPilotConfig:
    """The configuration a system runs with (the baseline overrides some sections)."""
    if system == "baseline":
        return baseline_config(base)
    return base or ParkPilotConfig()
