import math

from hypothesis import given, settings
from hypothesis import strategies as st

from parkpilot.core.geometry import Pose
from parkpilot.core.units import angle_diff
from parkpilot.planning.reeds_shepp import (
    RSPath,
    all_paths,
    mod2pi,
    paths_between,
    sample,
    shortest_length,
)

coord = st.floats(-15, 15, allow_nan=False)
ang = st.floats(-math.pi, math.pi, allow_nan=False)
R = 4.2


def _end(start: Pose, path: RSPath, radius: float) -> Pose:
    poses, _ = sample(start, path, radius, 0.5)
    x, y, th = poses[-1]
    return Pose(float(x), float(y), float(th))


@settings(max_examples=300, deadline=None)
@given(coord, coord, ang)
def test_every_path_reaches_goal(x: float, y: float, phi: float) -> None:
    start = Pose(1.0, -2.0, 0.3)
    goal = start.compose(Pose(x, y, phi))
    paths = paths_between(start, goal, R)
    assert paths, "Reeds-Shepp is complete: some path always exists"
    for p in paths:
        end = _end(start, p, R)
        assert end.distance_to(goal) < 1e-6
        assert abs(angle_diff(end.theta, goal.theta)) < 1e-6


@settings(max_examples=200, deadline=None)
@given(coord, coord, ang)
def test_shortest_is_symmetric_and_bounded(x: float, y: float, phi: float) -> None:
    a, b = Pose(0, 0, 0), Pose(x, y, phi)
    lab = shortest_length(a, b, R)
    lba = shortest_length(b, a, R)
    assert math.isclose(lab, lba, rel_tol=1e-6, abs_tol=1e-6)
    assert lab >= math.hypot(x, y) - 1e-9


def test_straight_line_and_known_cases() -> None:
    start = Pose(0, 0, 0)
    best = paths_between(start, Pose(10, 0, 0), R)[0]
    assert math.isclose(best.length * R, 10.0)
    assert best.gear_changes == 0
    back = paths_between(start, Pose(-5, 0, 0), R)[0]
    assert math.isclose(back.length * R, 5.0)
    assert all(v <= 0 for _, v in back.segments())
    # A quarter turn left on the minimum radius.
    q = paths_between(start, Pose(R, R, math.pi / 2), R)[0]
    assert math.isclose(q.length * R, R * math.pi / 2, rel_tol=1e-6)


def test_parallel_park_needs_gear_change() -> None:
    # Sideways shift of 2.5 m with the same heading: classic parallel-parking geometry.
    paths = paths_between(Pose(0, 0, 0), Pose(-1.0, -2.5, 0.0), R)
    assert min(p.gear_changes for p in paths[:3]) >= 1
    words = {p.word for p in all_paths(0.0, -0.6, 0.0)}
    assert len(words) > 3


def test_sample_directions_and_mod2pi() -> None:
    poses, dirs = sample(Pose(0, 0, 0), RSPath("SS", (1.0, -0.5)), 2.0, 0.25)
    assert len(poses) == len(dirs)
    assert dirs[0] == 1.0 and dirs[-1] == -1.0
    assert math.isclose(poses[-1][0], 1.0)
    empty, d = sample(Pose(0, 0, 0), RSPath("S", (0.0,)), 2.0, 0.25)
    assert len(empty) == 1 and d[0] == 1.0
    assert math.isclose(mod2pi(3 * math.pi), math.pi) or math.isclose(mod2pi(3 * math.pi), -math.pi)
    assert math.isclose(mod2pi(-3.5 * math.pi), 0.5 * math.pi)
