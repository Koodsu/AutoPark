import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from parkpilot.core.config import GridConfig, SensorConfig, VehicleConfig
from parkpilot.core.geometry import Pose, points_polygon_distance
from parkpilot.perception.camera import CameraRig
from parkpilot.perception.grid import OccupancyGrid, edt_capped
from parkpilot.world.model import Obstacle, ObstacleKind, World, circle_polygon

SENSOR = SensorConfig()
VEH = VehicleConfig()
RIG = CameraRig(SENSOR, VEH)
BOUNDS = (-20.0, -20.0, 20.0, 20.0)


def _grid(memory: bool = True) -> OccupancyGrid:
    return OccupancyGrid(GridConfig(memory=memory), BOUNDS, VEH, SENSOR.low_blind_range)


def _bollard_world() -> tuple[World, Obstacle]:
    b = Obstacle("b", ObstacleKind.BOLLARD, circle_polygon((6.0, 0.0), 0.15), 0.9)
    return World(BOUNDS, (0, 19), "t", obstacles=[b]), b


def _drive_past(grid: OccupancyGrid, world: World, xs: list[float]) -> None:
    rng = np.random.default_rng(0)
    for i, x in enumerate(xs):
        grid.update(RIG.sense(0.1 * i, Pose(x, 0, 0), world, [], rng))


def _remembers(grid: OccupancyGrid, ob: Obstacle) -> bool:
    pts = grid.occupied_points()
    return bool((points_polygon_distance(pts, ob.polygon) < 0.3).any()) if len(pts) else False


def test_memory_keeps_low_obstacle_inside_blind_zone() -> None:
    world, bollard = _bollard_world()
    # Approach from far away until the bollard is right at the bumper (blind zone).
    xs = [float(x) for x in np.linspace(-6.0, 2.0, 25)]
    with_memory = _grid(True)
    _drive_past(with_memory, world, xs)
    assert _remembers(with_memory, bollard)
    without = _grid(False)
    _drive_past(without, world, xs)
    assert not _remembers(without, bollard)


def test_free_space_clears_and_hits_mark() -> None:
    world, bollard = _bollard_world()
    g = _grid()
    _drive_past(g, world, [-6.0, -6.0, -6.0])
    assert g.observed.sum() > 1000
    assert _remembers(g, bollard)
    probe = np.array([[2.0, 0.0]])
    assert not g.is_occupied_at(probe)[0]
    assert g.is_occupied_at(np.array([[500.0, 0.0]]))[0]  # outside counts as blocked
    assert g.probability().max() > 0.7
    assert g.version == 3


def test_distance_field_matches_brute_force() -> None:
    world, _bollard = _bollard_world()
    g = _grid()
    _drive_past(g, world, [-6.0] * 4)
    d = g.distance_at(np.array([[6.0, 2.0], [0.0, 0.0]]))
    assert abs(d[0] - 2.0) < 0.35
    assert math.isclose(d[1], g.cfg.distance_cap)
    assert g.distance_field() is g.distance_field()  # cached by version


@settings(max_examples=25, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 29), st.integers(0, 39)), min_size=1, max_size=12))
def test_edt_exact_against_brute_force(cells: list[tuple[int, int]]) -> None:
    occ = np.zeros((30, 40), dtype=bool)
    for r, c in cells:
        occ[r, c] = True
    res, cap = 0.2, 2.0
    d = edt_capped(occ, res, cap)
    rr, cc = np.mgrid[0:30, 0:40]
    pts = np.array(cells)
    brute = np.min(np.hypot(rr[..., None] - pts[:, 0], cc[..., None] - pts[:, 1]), axis=-1) * res
    np.testing.assert_allclose(d, np.minimum(brute, cap), atol=1e-9)


def test_decay_forgets_unobserved_cells() -> None:
    world, bollard = _bollard_world()
    g = OccupancyGrid(GridConfig(decay_per_s=5.0), BOUNDS, VEH, SENSOR.low_blind_range)
    rng = np.random.default_rng(0)
    for i in range(3):
        g.update(RIG.sense(0.1 * i, Pose(-6, 0, 0), world, [], rng))
    assert _remembers(g, bollard)
    # Look away for a long time: decayed memory drops below the occupied threshold.
    g.update(RIG.sense(10.0, Pose(-6, 0, math.pi), world, [], rng))
    assert not _remembers(g, bollard)
