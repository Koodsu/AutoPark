import numpy as np
import pytest

from parkpilot.behavior.types import DriveState
from parkpilot.scenarios.loader import load_scenario
from parkpilot.sim.engine import Simulation
from parkpilot.sim.systems import SYSTEMS, system_config


def _sim(sid: str, seed: int = 0) -> Simulation:
    return Simulation(load_scenario(sid), SYSTEMS["parkpilot"], system_config("parkpilot"), seed)


@pytest.mark.parametrize("sid", ["charger_stall", "unmarked_lot"])
def test_parks_without_touching_anything(sid: str) -> None:
    sim = _sim(sid)
    log = sim.run()
    tel = sim.autonomy.telemetry
    assert tel.state is DriveState.PARKED, tel.reason
    assert not any(s.collision or s.curb_contact for _, s in log.truth)
    assert min(s.static_clearance for _, s in log.truth) > 0.1
    assert tel.parked_pose is not None
    if sid == "charger_stall":
        assert tel.goal is not None and tel.goal.port_distance is not None
        port = sim.vehicle.charge_port(tel.parked_pose)
        charger = min(sim.world.chargers, key=lambda c: np.hypot(*(np.array(c.position) - port)))
        assert np.hypot(*(np.array(charger.position) - port)) < charger.cable_reach


def test_deterministic_for_a_seed() -> None:
    a, b = _sim("unmarked_lot", 3), _sim("unmarked_lot", 3)
    a.run(max_ticks=250)
    b.run(max_ticks=250)
    pa, pb = a.state.pose, b.state.pose
    assert (pa.x, pa.y, pa.theta) == (pb.x, pb.y, pb.theta)
    c = _sim("unmarked_lot", 4)
    c.run(max_ticks=250)
    assert a.last_perception is not None and b.last_perception is not None
    assert c.last_perception is not None
    assert np.array_equal(a.last_perception.frame.ranges, b.last_perception.frame.ranges)
    assert not np.array_equal(a.last_perception.frame.ranges, c.last_perception.frame.ranges)


def test_abort_hands_back_with_reason() -> None:
    sim = _sim("open_lot_speed")
    sim.run(max_ticks=60)
    assert "aborting" in sim.abort()
    sim.run(max_ticks=400)
    tel = sim.autonomy.telemetry
    assert tel.state is DriveState.HANDBACK and tel.handback_reason == "aborted by the driver"
    assert sim.done
    assert abs(sim.state.speed) < 0.05
