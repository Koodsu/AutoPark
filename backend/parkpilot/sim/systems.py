"""Registry of autonomy stacks the simulator can run."""

from __future__ import annotations

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


SYSTEMS: dict[str, AutonomyFactory] = {"parkpilot": parkpilot_factory}


def system_config(system: str, base: ParkPilotConfig | None = None) -> ParkPilotConfig:
    """The configuration a system runs with (the baseline overrides some sections)."""
    return base or ParkPilotConfig()
