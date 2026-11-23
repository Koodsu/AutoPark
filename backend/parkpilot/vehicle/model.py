"""Kinematic bicycle model and vehicle footprint.

The reference point is the rear-axle centre, so curvature is ``tan(steer) / wheelbase`` and a
pose moves along exact circular arcs for constant steering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from parkpilot.core.config import VehicleConfig
from parkpilot.core.geometry import FloatArray, Pose, rect_polygon
from parkpilot.core.units import wrap_angle


def arc_poses(start: Pose, curvature: float, distances: FloatArray) -> FloatArray:
    """Poses ``(N, 3)`` reached by driving signed ``distances`` along a constant-curvature arc.

    Negative distances move in reverse. Vectorised and exact (no Euler integration error).
    """
    s = np.asarray(distances, dtype=np.float64)
    th0 = start.theta
    if abs(curvature) < 1e-9:
        x = start.x + s * math.cos(th0)
        y = start.y + s * math.sin(th0)
        th = np.full_like(s, th0)
    else:
        th = th0 + curvature * s
        x = start.x + (np.sin(th) - math.sin(th0)) / curvature
        y = start.y - (np.cos(th) - math.cos(th0)) / curvature
    th = (th + math.pi) % (2.0 * math.pi) - math.pi
    return np.stack([x, y, th], axis=-1)


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Full kinematic state of the ego vehicle."""

    pose: Pose
    speed: float = 0.0
    """Signed longitudinal speed (m/s); negative while reversing."""
    steer: float = 0.0
    """Current road-wheel angle (rad)."""
    accel: float = 0.0
    """Last applied longitudinal acceleration (m/s^2)."""


class VehicleModel:
    """Geometry queries and kinematic integration for one vehicle configuration."""

    def __init__(self, config: VehicleConfig) -> None:
        self.config = config
        c = config
        self.min_turn_radius = c.wheelbase / math.tan(c.max_steer)
        self.max_curvature = 1.0 / self.min_turn_radius
        self.center_offset = c.length / 2.0 - c.rear_overhang
        """Rear axle to geometric centre of the body (m)."""
        mirror_x = c.length - c.rear_overhang - c.mirror_from_front
        mirror_half = (c.mirror_width - c.width) / 2.0
        self._mirror_centers = [
            Pose(mirror_x, side * (c.width / 2.0 + mirror_half / 2.0), 0.0) for side in (1, -1)
        ]
        self._mirror_size = (c.mirror_length, mirror_half)
        n = c.collision_circles
        seg = c.length / n
        xs = -c.rear_overhang + seg * (np.arange(n) + 0.5)
        self.circle_offsets = np.asarray(xs, dtype=np.float64)
        """Longitudinal offsets of the planner's covering disks from the rear axle (m)."""
        body_r = math.hypot(seg / 2.0, c.width / 2.0)
        # Make sure the disk that contains the mirrors also covers their outer corners.
        mirror_r = 0.0
        for xm in (mirror_x - c.mirror_length / 2.0, mirror_x + c.mirror_length / 2.0):
            nearest = float(np.min(np.abs(xs - xm)))
            mirror_r = max(mirror_r, math.hypot(nearest, c.mirror_width / 2.0))
        self.circle_radius = max(body_r, mirror_r)
        """Radius of each covering disk (m); the union covers body and mirrors."""

    def body_center(self, pose: Pose) -> Pose:
        return pose.compose(Pose(self.center_offset, 0.0, 0.0))

    def rear_axle_from_center(self, center: Pose) -> Pose:
        """Inverse of :meth:`body_center`."""
        return center.compose(Pose(-self.center_offset, 0.0, 0.0))

    def body_polygon(self, pose: Pose) -> FloatArray:
        return rect_polygon(self.body_center(pose), self.config.length, self.config.width)

    def mirror_polygons(self, pose: Pose) -> list[FloatArray]:
        length, width = self._mirror_size
        return [rect_polygon(pose.compose(m), length, width) for m in self._mirror_centers]

    def footprint(self, pose: Pose) -> list[FloatArray]:
        """Body plus mirrors, each a convex polygon in world coordinates."""
        return [self.body_polygon(pose), *self.mirror_polygons(pose)]

    def circle_centers(self, poses: FloatArray) -> FloatArray:
        """Covering-disk centres for many poses ``(..., 3)`` -> ``(..., n, 2)``."""
        p = np.asarray(poses, dtype=np.float64)
        c, s = np.cos(p[..., 2:3]), np.sin(p[..., 2:3])
        x = p[..., 0:1] + c * self.circle_offsets
        y = p[..., 1:2] + s * self.circle_offsets
        return np.stack([x, y], axis=-1)

    def charge_port(self, pose: Pose) -> FloatArray:
        c = self.config
        return np.asarray(pose.to_world(np.array([[c.charge_port_x, c.charge_port_y]]))[0])

    def curvature(self, steer: float) -> float:
        return math.tan(steer) / self.config.wheelbase

    def steer_for_curvature(self, curvature: float) -> float:
        steer = math.atan(curvature * self.config.wheelbase)
        return max(-self.config.max_steer, min(self.config.max_steer, steer))

    def step(
        self, state: VehicleState, accel: float, steer_cmd: float, dt: float, gear: int = 1
    ) -> VehicleState:
        """Advance the bicycle model by ``dt`` with actuator limits applied.

        ``accel`` is the signed rate of change of speed. ``gear`` (+1 drive, -1 reverse)
        bounds the speed to that direction, so braking in a gear stops the car instead of
        driving it the other way.
        """
        c = self.config
        target = max(-c.max_steer, min(c.max_steer, steer_cmd))
        max_delta = c.max_steer_rate * dt
        steer = state.steer + max(-max_delta, min(max_delta, target - state.steer))
        accel = max(-c.emergency_decel, min(c.max_accel, accel))
        v0 = state.speed
        v1 = v0 + accel * dt
        if gear >= 0:
            v1 = max(0.0, min(c.max_speed, v1))
        else:
            v1 = max(-c.max_reverse_speed, min(0.0, v1))
        distance = 0.5 * (v0 + v1) * dt
        new = arc_poses(state.pose, self.curvature(steer), np.array([distance]))[0]
        pose = Pose(float(new[0]), float(new[1]), wrap_angle(float(new[2])))
        return replace(state, pose=pose, speed=v1, steer=steer, accel=(v1 - v0) / dt)
