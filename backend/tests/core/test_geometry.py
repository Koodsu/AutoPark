import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from parkpilot.core.geometry import (
    Pose,
    ensure_ccw,
    oriented_bounding_box,
    points_in_convex_polygon,
    points_polygon_distance,
    polygon_area,
    polygon_distance,
    polygon_edges,
    polygons_intersect,
    rect_polygon,
    segment_intersections,
)
from parkpilot.core.units import angle_diff, deg, kph, wrap_angle

coords = st.floats(min_value=-100, max_value=100, allow_nan=False)
angles = st.floats(min_value=-10, max_value=10, allow_nan=False)
poses = st.builds(Pose, coords, coords, angles)
sizes = st.floats(min_value=0.2, max_value=10)


def _close(a: Pose, b: Pose, tol: float = 1e-6) -> bool:
    return abs(a.x - b.x) < tol and abs(a.y - b.y) < tol and abs(angle_diff(a.theta, b.theta)) < tol


@given(poses, poses)
def test_compose_inverse_roundtrip(a: Pose, b: Pose) -> None:
    assert _close(a.compose(b).relative_to(a), b, 1e-6)
    assert _close(a.compose(a.inverse()), Pose(0, 0, 0), 1e-6)


@given(poses, st.lists(st.tuples(coords, coords), min_size=1, max_size=5))
def test_to_world_to_local_roundtrip(p: Pose, pts: list[tuple[float, float]]) -> None:
    arr = np.array(pts)
    np.testing.assert_allclose(p.to_local(p.to_world(arr)), arr, atol=1e-7)


@given(angles)
def test_wrap_angle_range(a: float) -> None:
    w = wrap_angle(a)
    assert -math.pi <= w < math.pi
    assert math.isclose(math.cos(w), math.cos(a), abs_tol=1e-9)


@given(poses, sizes, sizes)
def test_rect_area_and_ccw(p: Pose, length: float, width: float) -> None:
    poly = rect_polygon(p, length, width)
    assert math.isclose(polygon_area(poly), length * width, rel_tol=1e-6)
    assert points_in_convex_polygon(p.xy[None, :], poly)[0]


@given(poses, sizes, sizes, st.floats(min_value=0.01, max_value=5))
def test_separated_rects_do_not_intersect(p: Pose, length: float, width: float, gap: float) -> None:
    a = rect_polygon(p, length, width)
    shifted = p.compose(Pose(length + gap, 0.0, 0.0))
    b = rect_polygon(shifted, length, width)
    assert not polygons_intersect(a, b)
    assert math.isclose(polygon_distance(a, b), gap, rel_tol=1e-5, abs_tol=1e-6)
    assert polygons_intersect(a, b, margin=gap + 1e-6)


@given(poses, sizes, sizes)
def test_overlapping_rects_intersect(p: Pose, length: float, width: float) -> None:
    a = rect_polygon(p, length, width)
    b = rect_polygon(p.compose(Pose(length * 0.5, width * 0.25, 0.3)), length, width)
    assert polygons_intersect(a, b)
    assert polygon_distance(a, b) == 0.0


def test_point_polygon_distance() -> None:
    sq = rect_polygon(Pose(0, 0, 0), 2, 2)
    d = points_polygon_distance(np.array([[0.0, 0.0], [3.0, 0.0], [2.0, 2.0]]), sq)
    np.testing.assert_allclose(d, [0.0, 2.0, math.sqrt(2)], atol=1e-9)


def test_ensure_ccw_flips_clockwise() -> None:
    sq = rect_polygon(Pose(0, 0, 0), 2, 2)
    assert polygon_area(ensure_ccw(sq[::-1])) > 0


@given(poses, st.floats(min_value=1, max_value=8), st.floats(min_value=0.3, max_value=0.7))
def test_obb_recovers_rectangle(p: Pose, length: float, aspect: float) -> None:
    width = length * aspect
    rng = np.random.default_rng(0)
    local = rng.uniform([-length / 2, -width / 2], [length / 2, width / 2], size=(400, 2))
    local = np.vstack([local, [[-length / 2, -width / 2], [length / 2, width / 2]]])
    pose, ln, wd = oriented_bounding_box(p.to_world(local))
    assert ln >= wd
    assert abs(pose.x - p.x) < 0.2 * length and abs(pose.y - p.y) < 0.2 * length
    assert abs(math.sin(angle_diff(pose.theta, p.theta))) < 0.35


def test_obb_degenerate() -> None:
    pose, ln, wd = oriented_bounding_box(np.array([[1.0, 2.0]]))
    assert (pose.x, pose.y, ln, wd) == (1.0, 2.0, 0.0, 0.0)


def test_segment_intersections() -> None:
    a, b = polygon_edges([rect_polygon(Pose(5, 0, 0), 2, 2)])
    origins = np.zeros((2, 2))
    dirs = np.array([[1.0, 0.0], [-1.0, 0.0]])
    t = segment_intersections(origins, dirs, a, b)
    assert math.isclose(float(t[0].min()), 4.0)
    assert np.isinf(t[1]).all()
    empty = segment_intersections(origins, dirs, np.zeros((0, 2)), np.zeros((0, 2)))
    assert empty.shape == (2, 0)
    assert polygon_edges([])[0].shape == (0, 2)


def test_pose_helpers() -> None:
    p = Pose(1, 2, math.pi / 2)
    q = p.advanced(2)
    assert math.isclose(q.x, 1, abs_tol=1e-9) and math.isclose(q.y, 4)
    assert math.isclose(p.distance_to(q), 2)
    assert math.isclose(deg(180), math.pi)
    assert math.isclose(kph(36), 10)
