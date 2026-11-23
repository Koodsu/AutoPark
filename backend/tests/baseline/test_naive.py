from parkpilot.baseline.naive import baseline_config
from parkpilot.behavior.types import DriveState, OverrideSpot
from parkpilot.core.config import ParkPilotConfig
from parkpilot.scenarios.loader import load_scenario
from parkpilot.sim.engine import Simulation
from parkpilot.sim.systems import SYSTEMS, system_config
from parkpilot.world.model import SpotType


def test_baseline_config_switches_off_the_fixes() -> None:
    cfg = baseline_config(ParkPilotConfig())
    assert cfg.grid.memory is False
    assert cfg.safety.persistence_filter is False
    assert cfg.speed.constant_speed == 1.0
    assert system_config("parkpilot").grid.memory is True


def test_baseline_takes_first_spot_even_if_restricted() -> None:
    sim = Simulation(
        load_scenario("restricted_spots"), SYSTEMS["baseline"], system_config("baseline"), 0
    )
    sim.run()
    tel = sim.autonomy.telemetry
    assert tel.state is DriveState.PARKED
    assert tel.parked_pose is not None
    center = sim.vehicle.body_center(tel.parked_pose)
    spot = min(sim.world.spots, key=lambda s: s.pose.distance_to(center))
    assert spot.spot_type is not SpotType.STANDARD
    assert sim.autonomy.command(OverrideSpot("nope")).endswith("not known")


def test_baseline_cannot_park_without_lines() -> None:
    sim = Simulation(
        load_scenario("unmarked_lot"), SYSTEMS["baseline"], system_config("baseline"), 0
    )
    sim.run(max_ticks=800)
    assert sim.autonomy.telemetry.state is not DriveState.PARKED
    assert sim.autonomy.telemetry.candidates == []
