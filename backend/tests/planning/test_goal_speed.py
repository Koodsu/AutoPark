import math

import numpy as np

from parkpilot.core.config import ParkPilotConfig, SpeedConfig
from parkpilot.core.geometry import Pose, polygon_distance
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.goal import GoalResult, compute_goal
from parkpilot.planning.speed import path_curvature, speed_profile
from parkpilot.spots.detect import SpotCandidate
from parkpilot.vehicle.model import VehicleModel
from parkpilot.world.model import ObstacleKind, World
from tests.helpers import survey

CFG = ParkPilotConfig()
VEH = VehicleModel(CFG.vehicle)


def _cand_near(cands: list[SpotCandidate], world: World, spot_id: str) -> SpotCandidate:
    spot = world.spot(spot_id)
    return min(cands, key=lambda c: c.pose.distance_to(spot.pose))


def _goal(
    sid: str, spot_id: str, reverse_in: bool, use_neighbors: bool = True
) -> tuple[World, SpotCandidate, GoalResult]:
    _, world, out, cands, _ = survey(sid)
    cand = _cand_near(cands, world, spot_id)
    checker = CollisionChecker(out.grid, VEH, CFG.planner.safety_margin)
    res = compute_goal(
        cand,
        reverse_in=reverse_in,
        checker=checker,
        vehicle=VEH,
        cfg=CFG.goal,
        ranking=CFG.ranking,
        obstacle_points=out.grid.occupied_points(),
        use_neighbors=use_neighbors,
    )
    return world, cand, res


def _side_clearances(world: World, center: Pose) -> tuple[float, float]:
    rear = VEH.rear_axle_from_center(center)
    body = VEH.body_polygon(rear)
    parked = [
        o for o in world.obstacles if o.kind in (ObstacleKind.PARKED_CAR, ObstacleKind.PARKED_TRUCK)
    ]
    d = sorted(polygon_distance(body, o.polygon) for o in parked)
    return d[0], d[1]


def test_centering_between_actual_neighbors() -> None:
    world, cand, res = _goal("crooked_neighbor", "A7", reverse_in=True)
    assert res.ok
    spot = world.spot("A7")
    assert res.center.x > spot.pose.x + 0.1  # moved away from the encroaching neighbor
    near, far = _side_clearances(world, res.center)
    line_center = Pose(spot.pose.x, spot.pose.y, res.center.theta)
    near_naive, _ = _side_clearances(world, line_center)
    assert near > near_naive + 0.1
    assert abs(near - far) < 0.2
    assert any("neighbors" in r for r in res.reasons)
    # Never crosses the painted lines with the mirrors.
    assert abs(res.lateral_offset) <= (cand.width - CFG.vehicle.mirror_width) / 2


def test_parallel_curb_gap() -> None:
    world, _, res = _goal("parallel_curb", "P4", reverse_in=False)
    assert res.ok
    body = VEH.body_polygon(res.pose)
    curb = next(o for o in world.obstacles if o.kind is ObstacleKind.CURB)
    gap = polygon_distance(body, curb.polygon)
    assert abs(gap - CFG.goal.curb_gap) < 0.2


def test_garage_goal_avoids_pillar_and_is_free() -> None:
    world, _, res = _goal("garage_pillars", "G10", reverse_in=True)
    assert res.ok
    for poly in VEH.footprint(res.pose):
        for ob in world.obstacles:
            assert polygon_distance(poly, ob.polygon) > 0.1


def test_speed_profile_shape() -> None:
    xs = np.linspace(0, 30, 151)
    poses = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    dirs = np.ones_like(xs)
    open_clear = np.full_like(xs, 3.0)
    v = speed_profile(poses, dirs, open_clear, SpeedConfig())
    assert v[-1] == 0.0
    assert math.isclose(v.max(), SpeedConfig().v_open, rel_tol=1e-6)
    tight = speed_profile(poses, dirs, np.full_like(xs, 0.1), SpeedConfig())
    assert tight.max() <= SpeedConfig().v_tight + 1e-9
    # Acceleration/deceleration limits hold: v^2 changes by at most 2*a*ds.
    dv2 = np.diff(v**2)
    ds = np.diff(xs)
    assert (np.abs(dv2) <= 2 * SpeedConfig().accel * ds + 1e-6).all()
    crawl = speed_profile(poses, dirs, open_clear, SpeedConfig(constant_speed=1.0))
    assert crawl.max() <= 1.0
    cusp_dirs = np.where(xs < 15, 1.0, -1.0)
    vc = speed_profile(poses, cusp_dirs, open_clear, SpeedConfig())
    assert vc[np.flatnonzero(np.diff(cusp_dirs))[0]] == 0.0
    assert vc[xs >= 15].max() <= SpeedConfig().v_reverse
    assert speed_profile(np.zeros((0, 3)), np.zeros(0), np.zeros(0), SpeedConfig()).size == 0


def test_curvature_of_arc() -> None:
    th = np.linspace(0, 1.0, 50)
    r = 5.0
    poses = np.stack([r * np.sin(th), r * (1 - np.cos(th)), th], axis=1)
    k = path_curvature(poses)
    assert np.allclose(k, 1 / r, rtol=1e-2)
    assert path_curvature(poses[:1]).tolist() == [0.0]


def test_charger_goal_puts_port_within_reach() -> None:
    from parkpilot.core.config import Unlock, UserSettings
    from parkpilot.spots.classify import classify
    from parkpilot.spots.rank import wants_reverse_in

    _, world, out, cands, _ = survey("charger_stall")
    cand = _cand_near(cands, world, "C2")
    settings = UserSettings(unlocks=frozenset({Unlock.CHARGING}))
    classify(cand, out.semantic, settings.unlocks, CFG.spots)
    assert cand.charger is not None and cand.legal
    reverse = wants_reverse_in(cand, settings, CFG.vehicle)
    assert reverse
    checker = CollisionChecker(out.grid, VEH, CFG.planner.safety_margin)
    res = compute_goal(
        cand,
        reverse_in=reverse,
        checker=checker,
        vehicle=VEH,
        cfg=CFG.goal,
        ranking=CFG.ranking,
        obstacle_points=out.grid.occupied_points(),
    )
    assert res.ok and res.port_distance is not None
    truth = world.charger(world.spot("C2").charger_id or "")
    port = VEH.charge_port(res.pose)
    assert float(np.hypot(*(port - np.array(truth.position)))) < truth.cable_reach
