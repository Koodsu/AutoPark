import math

import numpy as np
import pytest

from parkpilot.control.tracker import PathFollower
from parkpilot.core.config import ControlConfig, SpeedConfig, VehicleConfig
from parkpilot.core.geometry import Pose
from parkpilot.core.units import angle_diff
from parkpilot.planning.hybrid_astar import Path
from parkpilot.planning.reeds_shepp import paths_between, sample
from parkpilot.planning.speed import speed_profile
from parkpilot.vehicle.model import VehicleModel, VehicleState

VEH = VehicleModel(VehicleConfig())


def _drive(path: Path, start: Pose, dt: float = 0.05) -> tuple[VehicleState, float, float]:
    follower = PathFollower(ControlConfig(), SpeedConfig(), VEH)
    state = VehicleState(start)
    max_cte = 0.0
    peak_jerk = 0.0
    last_acc = 0.0
    for seg in path.segments():
        v = speed_profile(
            seg.poses,
            seg.directions,
            np.full(len(seg.poses), 2.0),
            SpeedConfig(),
            wheelbase=VEH.config.wheelbase,
            max_steer_rate=VEH.config.max_steer_rate,
        )
        follower.set_segment(seg.poses, int(seg.directions[0]), v)
        for _ in range(2000):
            cmd = follower.step(state, dt)
            if cmd.done:
                break
            prev_speed = state.speed
            state = VEH.step(state, cmd.accel, cmd.steer, dt, cmd.gear)
            max_cte = max(max_cte, abs(cmd.cross_track))
            if abs(prev_speed) > 0.1 and abs(state.speed) > 0.1:  # comfort while moving
                peak_jerk = max(peak_jerk, abs(state.accel - last_acc) / dt)
            last_acc = state.accel
        else:
            pytest.fail("segment never finished")
    return state, max_cte, peak_jerk


@pytest.mark.parametrize(
    "goal",
    [
        Pose(15, 0, 0),  # straight
        Pose(10, 6, math.pi / 2),  # forward turn
        Pose(-8, -3, 0.0),  # reverse with a lateral shift
        Pose(-1.0, -2.6, 0.0),  # parallel-park style multi-cusp
        Pose(2, 7, math.pi),  # turn around
    ],
)
def test_tracks_reeds_shepp_paths_to_goal(goal: Pose) -> None:
    start = Pose(0, 0, 0)
    radius = VEH.min_turn_radius * 1.05
    rs = paths_between(start, goal, radius)[0]
    poses, dirs = sample(start, rs, radius, 0.2)
    final, max_cte, _ = _drive(Path(poses, dirs), start)
    # Paths that end mid-arc leave a residual (the steering wheel cannot unwind in zero
    # distance); the planner therefore ends parking paths with a short straight.
    assert final.pose.distance_to(goal) < 0.4
    assert abs(angle_diff(final.pose.theta, goal.theta)) < math.radians(10)
    assert max_cte < 0.35


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_final_straight_gives_precise_stop(direction: float) -> None:
    start = Pose(0, 0, 0)
    radius = VEH.min_turn_radius * 1.05
    pre = Pose(direction * 6.0, direction * 2.0, 0.0)
    rs = paths_between(start, pre, radius)[0]
    poses, dirs = sample(start, rs, radius, 0.2)
    s = np.linspace(0.2, 1.0, 5) * direction
    tail = np.stack([pre.x + s, np.full_like(s, pre.y), np.zeros_like(s)], axis=1)
    path = Path(np.vstack([poses, tail]), np.concatenate([dirs, np.full(5, direction)]))
    final, _, _ = _drive(path, start)
    goal = Pose(pre.x + direction * 1.0, pre.y, 0.0)
    assert final.pose.distance_to(goal) < 0.15
    assert abs(angle_diff(final.pose.theta, 0.0)) < math.radians(3)


def test_jerk_is_limited_on_straight() -> None:
    xs = np.linspace(0, 20, 101)
    poses = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    _, _, peak = _drive(Path(poses, np.ones_like(xs)), Pose(0, 0, 0))
    assert peak <= SpeedConfig().jerk * 1.6


def test_external_speed_limit_and_inactive() -> None:
    follower = PathFollower(ControlConfig(), SpeedConfig(), VEH)
    stopped = follower.step(VehicleState(Pose(0, 0, 0), speed=1.0), 0.05)
    assert stopped.done is True and stopped.accel < 0
    xs = np.linspace(0, 20, 101)
    poses = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
    follower.set_segment(poses, 1, np.full(101, 3.0))
    cmd = follower.step(VehicleState(Pose(0, 0, 0), speed=2.0), 0.05, speed_limit=0.0)
    assert cmd.accel < -VehicleConfig().max_decel + 1e-9 or cmd.accel <= -2.5
    assert follower.active and follower.gear == 1
    follower.clear()
    assert not follower.active
