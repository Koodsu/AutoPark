import math

import numpy as np

from parkpilot.behavior.driving import Driver, route_path
from parkpilot.behavior.machine import ParkPilotAutonomy, clearance_readouts
from parkpilot.behavior.types import Abort, DriveState, OverrideSpot
from parkpilot.core.config import ParkPilotConfig, UserSettings
from parkpilot.core.geometry import Pose
from parkpilot.spots.rank import RankedSpot
from parkpilot.vehicle.model import VehicleModel, VehicleState
from tests.helpers import survey

CFG = ParkPilotConfig()


def _autonomy() -> ParkPilotAutonomy:
    return ParkPilotAutonomy(
        CFG, route=[(4.0, 17.0), (58.0, 17.0)], entrance=(30.0, 33.0), settings=UserSettings()
    )


def test_route_path_is_forward_polyline() -> None:
    p = route_path(Pose(0, 0, 0), [(10.0, 0.0), (10.0, 10.0)])
    assert p.gear_changes == 0
    assert abs(p.length - 20.0) < 1e-6
    assert math.isclose(p.poses[-1][2], math.pi / 2)
    single = route_path(Pose(1, 1, 0.3), [])
    assert len(single.poses) == 1


def test_driver_holds_when_moving_wrong_way() -> None:
    veh = VehicleModel(CFG.vehicle)
    d = Driver(CFG.control, CFG.speed, veh)
    path = route_path(Pose(0, 0, 0), [(10.0, 0.0)])
    reverse = type(path)(path.poses, -np.ones(len(path.poses)))
    d.set_path(reverse, None)
    res = d.step(VehicleState(Pose(0, 0, 0), speed=1.0), 0.05, math.inf, False)
    assert res.control.accel < 0 and res.control.gear == 1
    d.clear()
    assert not d.active and d.remaining_poses().shape == (0, 3)


def test_spot_selection_and_override_rules() -> None:
    sc, world, out, _, detector = survey("restricted_spots")
    a = ParkPilotAutonomy(
        CFG, route=list(sc.search_route), entrance=sc.entrance, settings=sc.user, bounds=sc.bounds
    )
    a.detector = detector  # keeps the inspection history of the drive along the aisle
    state = VehicleState(out.ego)
    a.step(0.0, 0.05, state, out)
    ranked = a.telemetry.ranked
    assert ranked and ranked[0].eligible

    def truth(r: RankedSpot) -> str:
        return min(world.spots, key=lambda s: s.pose.distance_to(r.candidate.pose)).id

    assert truth(ranked[0]) in {"A14", "B9"}
    accessible = next(r for r in ranked if truth(r) == "A3")
    msg = a.command(OverrideSpot(accessible.candidate.id))
    assert msg.startswith("cannot park there") and "placard" in msg
    occupied = next(r for r in ranked if r.candidate.occupied)
    assert a.command(OverrideSpot(occupied.candidate.id)) == "that spot is occupied"
    assert "not known" in a.command(OverrideSpot("nope"))
    legal = next(r for r in ranked if r.eligible and r is not ranked[0])
    assert a.command(OverrideSpot(legal.candidate.id)).startswith("rerouting")
    assert a.telemetry.state is DriveState.SELECT
    assert a.telemetry.transitions[-1].reason.startswith("driver picked")
    assert a.command(Abort()).startswith("aborting")
    assert a.telemetry.state.value == "HANDBACK" and a.done
    assert a.telemetry.handback_reason == "aborted by the driver"
    assert a.command(OverrideSpot(legal.candidate.id)) == "control already handed back"


def test_clearance_readouts_sectors() -> None:
    _, _, out, _, _ = survey("crooked_neighbor")
    veh = VehicleModel(CFG.vehicle)
    # Inside the aisle the side rows are a couple of metres away, nothing right ahead.
    c = clearance_readouts(out, Pose(20.0, 17.0, 0.0), veh)
    assert set(c) == {"front", "rear", "left", "right"}
    assert 1.0 < c["left"] < 5.0 and 1.0 < c["right"] < 5.0


def test_no_perception_means_hold() -> None:
    a = _autonomy()
    ctrl = a.step(0.0, 0.05, VehicleState(Pose(4, 17, 0), speed=1.0), None)
    assert ctrl.accel < 0
    assert not a.done and a.state is DriveState.SEARCH
