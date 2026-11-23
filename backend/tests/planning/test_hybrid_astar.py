import math
from collections.abc import Sequence

import numpy as np

from parkpilot.core.config import GridConfig, ParkPilotConfig, PlannerConfig, VehicleConfig
from parkpilot.core.geometry import Pose, polygon_distance
from parkpilot.perception.grid import OccupancyGrid
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.hybrid_astar import HybridAStar, Path
from parkpilot.vehicle.model import VehicleModel
from tests.helpers import survey

VEH = VehicleModel(VehicleConfig())
CFG = ParkPilotConfig()


def _open_grid(obstacles: Sequence[tuple[float, float, float, float]] = ()) -> OccupancyGrid:
    g = OccupancyGrid(GridConfig(), (-20, -20, 20, 20), VEH.config, 1.0)
    g.observed[:] = True
    for x0, y0, x1, y1 in obstacles:
        r0, c0 = g.to_cells(np.array([x0, y0]))
        r1, c1 = g.to_cells(np.array([x1, y1]))
        g.logodds[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = 3.0
    g.version += 1
    return g


def test_straight_plan_in_open_space() -> None:
    planner = HybridAStar(PlannerConfig(), VEH)
    checker = CollisionChecker(_open_grid(), VEH, 0.15)
    res = planner.plan(Pose(-10, 0, 0), Pose(5, 0, 0), checker)
    assert res.success and res.path is not None
    assert res.path.gear_changes == 0
    assert abs(res.path.length - 15.0) < 0.5
    assert res.path.end.distance_to(Pose(5, 0, 0)) < 1e-6


def test_gear_change_penalty_reduces_shuffling() -> None:
    walls = [(-20, 4.5, 20, 5.0), (-20, -5.0, 20, -4.5)]  # a 9 m wide corridor
    checker = CollisionChecker(_open_grid(walls), VEH, 0.15)
    start, goal = Pose(0, -1.5, 0), Pose(0, 1.5, math.pi)  # turn around in the corridor
    cheap = HybridAStar(PlannerConfig(gear_change_cost=0.0, reverse_multiplier=1.0), VEH)
    costly = HybridAStar(PlannerConfig(gear_change_cost=30.0), VEH)
    a = cheap.plan(start, goal, checker)
    b = costly.plan(start, goal, checker)
    assert a.success and b.success and a.path and b.path
    assert b.path.gear_changes <= a.path.gear_changes


def test_goal_in_collision_is_rejected() -> None:
    checker = CollisionChecker(_open_grid([(4, -1, 6, 1)]), VEH, 0.15)
    res = HybridAStar(PlannerConfig(), VEH).plan(Pose(-10, 0, 0), Pose(4, 0, 0), checker)
    assert not res.success and "collision" in res.reason


def test_blocked_goal_exhausts_or_fails() -> None:
    box = [(2, -3, 12, -2.6), (2, 2.6, 12, 3), (11.6, -3, 12, 3), (2, -3, 2.4, 3)]
    checker = CollisionChecker(_open_grid(box), VEH, 0.15)
    res = HybridAStar(PlannerConfig(max_expansions=300), VEH).plan(
        Pose(-10, 0, 0), Pose(6, 0, 0), checker
    )
    assert not res.success
    assert res.reason in ("search budget exhausted", "no collision-free path")


def test_reverse_into_real_stall_is_collision_free() -> None:
    _, world, out, _, _ = survey("destination_entrance")
    spot = world.spot("A15")
    goal = VEH.rear_axle_from_center(Pose(spot.pose.x, spot.pose.y, spot.pose.theta + math.pi))
    checker = CollisionChecker(out.grid, VEH, CFG.planner.safety_margin)
    res = HybridAStar(CFG.planner, VEH).plan(Pose(40, 17, 0), goal, checker)
    assert res.success and res.path is not None
    assert res.path.gear_changes <= 2
    # Ground truth: never touches anything (mirrors included).
    for x, y, th in res.path.poses[::3]:
        for poly in VEH.footprint(Pose(x, y, th)):
            for ob in world.obstacles:
                assert polygon_distance(poly, ob.polygon) > 0.05


def test_path_segments_split_at_cusps() -> None:
    poses = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    p = Path(poses, np.array([1, 1, 1, -1, -1], dtype=float))
    segs = p.segments()
    assert [len(s.poses) for s in segs] == [3, 3]
    assert p.gear_changes == 1
    assert segs[1].directions[0] == -1
    assert Path(np.zeros((0, 3)), np.zeros(0)).segments() == []
    assert Path(np.zeros((1, 3)), np.ones(1)).length == 0.0
