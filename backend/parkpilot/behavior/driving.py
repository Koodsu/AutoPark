"""Path execution shared by autonomy stacks: segments, speed profiles, gear changes."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from parkpilot.behavior.types import ControlOutput
from parkpilot.control.tracker import Command, PathFollower
from parkpilot.core.config import ControlConfig, SpeedConfig
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.hybrid_astar import Path
from parkpilot.planning.speed import speed_profile
from parkpilot.vehicle.model import VehicleModel, VehicleState


def route_path(start: Pose, waypoints: list[tuple[float, float]], spacing: float = 0.2) -> Path:
    """Forward polyline path from ``start`` through the remaining route waypoints."""
    pts = [(start.x, start.y)]
    for x, y in waypoints:
        if math.hypot(x - pts[-1][0], y - pts[-1][1]) > 0.5:
            pts.append((x, y))
    poses: list[tuple[float, float, float]] = []
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        ln = math.hypot(x1 - x0, y1 - y0)
        th = math.atan2(y1 - y0, x1 - x0)
        n = max(1, math.ceil(ln / spacing))
        for k in range(n):
            f = k / n
            poses.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), th))
    if len(pts) >= 2:
        poses.append((pts[-1][0], pts[-1][1], poses[-1][2]))
    else:
        poses.append((start.x, start.y, start.theta))
    arr = np.array(poses)
    return Path(arr, np.ones(len(arr)))


@dataclass(slots=True)
class StepResult:
    control: ControlOutput
    command: Command | None
    segment_done: bool
    path_done: bool


class Driver:
    """Executes a :class:`Path` segment by segment with a jerk-limited follower."""

    def __init__(self, control: ControlConfig, speed: SpeedConfig, vehicle: VehicleModel) -> None:
        self.vehicle = vehicle
        self.speed_cfg = speed
        self.follower = PathFollower(control, speed, vehicle)
        self.path: Path | None = None
        self.segments: list[Path] = []
        self.index = 0

    @property
    def active(self) -> bool:
        return self.path is not None and self.index < len(self.segments)

    def set_path(self, path: Path, checker: CollisionChecker | None, v_start: float = 0.0) -> None:
        self.path = path
        self.segments = path.segments() or [path]
        self.index = 0
        self._load(checker, v_start)

    def clear(self) -> None:
        self.path = None
        self.segments = []
        self.index = 0
        self.follower.clear()

    def _load(self, checker: CollisionChecker | None, v_start: float) -> None:
        seg = self.segments[self.index]
        if checker is not None:
            clearance = checker.clearance(seg.poses)
        else:
            clearance = np.full(len(seg.poses), 5.0)
        v = speed_profile(
            seg.poses,
            seg.directions,
            clearance,
            self.speed_cfg,
            v_start=v_start,
            wheelbase=self.vehicle.config.wheelbase,
            max_steer_rate=self.vehicle.config.max_steer_rate,
        )
        self.follower.set_segment(seg.poses, int(seg.directions[0]), v)
        self._checker = checker

    def remaining_poses(self) -> FloatArray:
        """Poses of the current segment from the nearest index on."""
        if not self.active:
            return np.zeros((0, 3))
        seg = self.segments[self.index]
        return np.asarray(seg.poses[self.follower._idx :])

    @property
    def gear(self) -> int:
        return self.follower.gear

    def step(
        self, state: VehicleState, dt: float, speed_limit: float, emergency: bool
    ) -> StepResult:
        if not self.active:
            return StepResult(self.hold(state, dt), None, False, True)
        gear = self.follower.gear
        moving_wrong_way = (state.speed > 0.05 and gear < 0) or (state.speed < -0.05 and gear > 0)
        if moving_wrong_way:
            return StepResult(self.hold(state, dt), None, False, False)
        limit = 0.0 if emergency else speed_limit
        cmd = self.follower.step(state, dt, limit, emergency=emergency)
        accel = cmd.accel
        if emergency and abs(state.speed) > 0.05:
            accel = -math.copysign(self.vehicle.config.emergency_decel, state.speed)
        out = ControlOutput(accel, cmd.steer, cmd.gear)
        if cmd.done:
            self.index += 1
            if self.index < len(self.segments):
                self._load(self._checker, 0.0)
                return StepResult(out, cmd, True, False)
            return StepResult(out, cmd, True, True)
        return StepResult(out, cmd, False, False)

    def hold(self, state: VehicleState, dt: float) -> ControlOutput:
        """Brake smoothly (jerk-limited) to a stop in the current direction and hold."""
        gear = 1 if state.speed >= 0 else -1
        speed = abs(state.speed)
        if speed < 0.05:
            return ControlOutput(gear * -speed / dt, state.steer, gear)
        want = -min(speed / dt, self.vehicle.config.max_decel)
        current = gear * state.accel  # magnitude-signed: negative while slowing down
        a = max(want, current - self.speed_cfg.jerk * dt)
        a = min(a, 0.0)
        return ControlOutput(gear * a, state.steer, gear)
