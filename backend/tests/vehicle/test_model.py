import math

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from parkpilot.core.config import VehicleConfig
from parkpilot.core.geometry import Pose, points_in_convex_polygon, polygon_area
from parkpilot.vehicle.model import VehicleModel, VehicleState, arc_poses

CFG = VehicleConfig()
MODEL = VehicleModel(CFG)


def test_turn_radius() -> None:
    assert 4.0 < MODEL.min_turn_radius < 4.5
    assert math.isclose(MODEL.max_curvature * MODEL.min_turn_radius, 1.0)


@given(
    st.floats(-0.3, 0.3),
    st.floats(-20, 20),
)
def test_arc_poses_constant_curvature(kappa: float, dist: float) -> None:
    start = Pose(1.0, -2.0, 0.4)
    end = arc_poses(start, kappa, np.array([dist]))[0]
    if abs(kappa) > 1e-6:
        # The arc centre is equidistant (radius 1/|kappa|) from start and end.
        cx = start.x - math.sin(start.theta) / kappa
        cy = start.y + math.cos(start.theta) / kappa
        r0 = math.hypot(start.x - cx, start.y - cy)
        r1 = math.hypot(end[0] - cx, end[1] - cy)
        assert math.isclose(r0, r1, rel_tol=1e-6, abs_tol=1e-6)
    else:
        assert math.isclose(math.hypot(end[0] - start.x, end[1] - start.y), abs(dist), abs_tol=1e-4)


def test_footprint_includes_mirrors() -> None:
    pose = Pose(0, 0, 0)
    polys = MODEL.footprint(pose)
    assert len(polys) == 3
    assert math.isclose(polygon_area(polys[0]), CFG.length * CFG.width)
    ys = np.concatenate([p[:, 1] for p in polys])
    assert math.isclose(ys.max() - ys.min(), CFG.mirror_width)
    # Body spans from rear bumper to front bumper around the rear axle reference.
    xs = polys[0][:, 0]
    assert math.isclose(xs.min(), -CFG.rear_overhang)
    assert math.isclose(xs.max(), CFG.length - CFG.rear_overhang)


def test_circles_cover_footprint() -> None:
    pose = Pose(3.0, 1.0, 0.7)
    centers = MODEL.circle_centers(np.array([pose.x, pose.y, pose.theta]))
    rng = np.random.default_rng(1)
    for poly in MODEL.footprint(pose):
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        pts = rng.uniform(lo, hi, size=(2000, 2))
        pts = pts[points_in_convex_polygon(pts, poly)]
        pts = np.vstack([pts, poly])
        d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
        assert (d <= MODEL.circle_radius + 1e-9).all()


def test_center_roundtrip_and_port() -> None:
    p = Pose(2, 3, 1.0)
    assert MODEL.rear_axle_from_center(MODEL.body_center(p)).distance_to(p) < 1e-9
    port = MODEL.charge_port(Pose(0, 0, 0))
    assert port[0] < 0 and port[1] > 0  # rear-left


def test_step_limits_and_braking() -> None:
    s = VehicleState(Pose(0, 0, 0))
    for _ in range(200):
        s = MODEL.step(s, 10.0, 1.0, 0.05)
    assert s.speed <= CFG.max_speed + 1e-9
    assert s.steer <= CFG.max_steer + 1e-9
    # Hard braking stops without reversing.
    for _ in range(100):
        s = MODEL.step(s, -20.0, 0.0, 0.05)
    assert s.speed == 0.0
    # Hard braking while reversing also stops instead of driving forward.
    back = VehicleState(Pose(0, 0, 0), speed=-1.0)
    for _ in range(20):
        back = MODEL.step(back, 20.0, 0.0, 0.05, gear=-1)
    assert back.speed == 0.0
    r = MODEL.step(VehicleState(Pose(0, 0, 0)), -1.0, 0.0, 0.05, gear=-1)
    assert r.speed < 0 and r.pose.x < 0


def test_steer_curvature_inverse() -> None:
    for steer in (-0.5, 0.0, 0.3):
        assert math.isclose(MODEL.steer_for_curvature(MODEL.curvature(steer)), steer)
    assert MODEL.steer_for_curvature(10.0) == CFG.max_steer
