import math

import numpy as np

from parkpilot.core.config import AgentConfig, SensorConfig, VehicleConfig
from parkpilot.core.geometry import Pose, points_polygon_distance
from parkpilot.perception.camera import CameraRig
from parkpilot.world.agents import AgentSet, AgentSpec
from parkpilot.world.layout import RowSpec, build_row, spot_features
from parkpilot.world.model import (
    Obstacle,
    ObstacleKind,
    Phantom,
    SpotType,
    World,
    box_obstacle,
    circle_polygon,
)

RIG = CameraRig(SensorConfig(), VehicleConfig())
EGO = Pose(0.0, 0.0, 0.0)


def _world(obstacles: list[Obstacle], lighting: float = 1.0) -> World:
    return World((-30, -30, 30, 30), (0, 29), "t", obstacles=obstacles, lighting=lighting)


def _wall_ahead(dist: float) -> Obstacle:
    return box_obstacle("w", ObstacleKind.WALL, Pose(dist + 0.1, 0, math.pi / 2), 10, 0.2, 2.0)


def test_wall_detected_at_correct_range() -> None:
    frame = RIG.sense(0.0, EGO, _world([_wall_ahead(10.0)]), [], np.random.default_rng(0))
    pts = frame.hit_points
    near_wall = pts[np.abs(pts[:, 1]) < 3]
    assert len(near_wall) > 20
    assert abs(float(np.median(near_wall[:, 0])) - 10.0) < 0.1
    assert frame.ranges.shape == frame.hits.shape


def test_low_bollard_disappears_in_near_field_but_post_does_not() -> None:
    bollard = Obstacle("b", ObstacleKind.BOLLARD, circle_polygon((4.6, 0.0), 0.15), 0.9)
    post = Obstacle("p", ObstacleKind.POST, circle_polygon((4.6, 0.0), 0.15), 1.5)
    rng = np.random.default_rng(1)
    # Front bumper is at x=3.8: the obstacle is 0.65 m away, inside the low blind range.
    seen_bollard = RIG.sense(0.0, EGO, _world([bollard]), [], rng).hit_points
    seen_post = RIG.sense(0.0, EGO, _world([post]), [], rng).hit_points
    assert not (points_polygon_distance(seen_bollard, bollard.polygon) < 0.2).any()
    assert (points_polygon_distance(seen_post, post.polygon) < 0.2).any()
    # From farther away the bollard is visible.
    far = RIG.sense(0.0, Pose(-4, 0, 0), _world([bollard]), [], rng).hit_points
    assert (points_polygon_distance(far, bollard.polygon) < 0.2).any()


def test_night_is_noisier_and_shorter_range() -> None:
    day = RIG.sense(0.0, EGO, _world([_wall_ahead(12.0)], 1.0), [], np.random.default_rng(2))
    night = RIG.sense(0.0, EGO, _world([_wall_ahead(12.0)], 0.0), [], np.random.default_rng(2))
    assert night.noise_sigma > 2 * day.noise_sigma
    assert night.max_ranges.max() < day.max_ranges.max()
    day_err = np.abs(day.hit_points[:, 0] - 12.0)[np.abs(day.hit_points[:, 1]) < 3]
    night_err = np.abs(night.hit_points[:, 0] - 12.0)[np.abs(night.hit_points[:, 1]) < 3]
    assert night_err.std() > day_err.std()


def test_phantoms_produce_flagged_false_hits() -> None:
    w = _world([])
    w.phantoms.append(Phantom((8.0, 0.0), 0.5, 1.0))
    frame = RIG.sense(0.0, EGO, w, [], np.random.default_rng(3))
    assert frame.truth_phantom.any()
    assert frame.hits[frame.truth_phantom].all()


def test_occlusion_and_agent_cues() -> None:
    wall = _wall_ahead(6.0)
    hidden = Obstacle("h", ObstacleKind.POST, circle_polygon((12.0, 0.0), 0.3), 1.5)
    frame = RIG.sense(0.0, EGO, _world([wall, hidden]), [], np.random.default_rng(4))
    assert not (points_polygon_distance(frame.hit_points, hidden.polygon) < 0.3).any()
    agents = AgentSet.from_specs(
        AgentConfig(), [AgentSpec("c", "car", ((8, 4), (8, 0)), 1.0, reversing=True)]
    )
    agents.step(0.0, 0.1, Pose(100, 100, 0), [])
    frame = RIG.sense(0.0, EGO, _world([]), agents.active(), np.random.default_rng(5))
    assert len(frame.cues) == 1 and frame.cues[0].reverse_lights
    assert abs(frame.cues[0].position[0] - 8.0) < 1.5


def test_lines_and_features_observed_when_visible() -> None:
    spots, lines = build_row(RowSpec("A", Pose(6, 0, 0), 2), {0: SpotType.ACCESSIBLE})
    w = World((-30, -30, 30, 30), (0, 29), "t", spots=spots, lines=lines)
    w.features.extend(spot_features(spots[0]))
    frame = RIG.sense(0.0, EGO, w, [], np.random.default_rng(6))
    assert {o.line_id for o in frame.lines} <= {ln.id for ln in lines}
    assert len(frame.lines) >= 2
    found = set()
    for k in range(10):
        f = RIG.sense(0.0, EGO, w, [], np.random.default_rng(10 + k))
        found |= {o.label for o in f.features}
    assert "accessible" in found
