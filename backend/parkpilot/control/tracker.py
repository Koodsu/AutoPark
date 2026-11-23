"""Lateral and longitudinal control along one single-direction path segment.

* Forward: pure pursuit on the rear axle with speed-scheduled lookahead.
* Reverse: Stanley-style control on a *virtual* vehicle. Driving backwards with rear-axle
  reference is kinematically a forward-driving bicycle with heading ``theta + pi`` and
  steering ``-delta``. The law combines path-curvature feedforward (previewed to cover the
  steering actuator lag), heading error and a cross-track term at the rear axle, and the
  command is mirrored back. Referencing the rear axle (not a point a wheelbase away) keeps
  it accurate on the short arcs that parking maneuvers are made of.
* Longitudinal: P control on speed toward the segment's speed profile, capped by the
  stopping distance to the segment end and by an external limit (safety supervisor), with
  acceleration and jerk limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from parkpilot.core.config import ControlConfig, SpeedConfig
from parkpilot.core.geometry import FloatArray
from parkpilot.core.units import wrap_angle
from parkpilot.vehicle.model import VehicleModel, VehicleState


@dataclass(frozen=True, slots=True)
class Command:
    accel: float
    """Signed rate of change of speed (m/s^2)."""
    steer: float
    gear: int
    done: bool
    cross_track: float
    heading_error: float
    target_speed: float
    remaining: float
    """Arc length left on the segment (m)."""


class PathFollower:
    def __init__(self, control: ControlConfig, speed: SpeedConfig, vehicle: VehicleModel) -> None:
        self.cfg = control
        self.speed_cfg = speed
        self.vehicle = vehicle
        self._poses: FloatArray = np.zeros((0, 3))
        self._speeds: FloatArray = np.zeros(0)
        self._s: FloatArray = np.zeros(0)
        self._gear = 1
        self._idx = 0
        self._last_accel = 0.0

    @property
    def active(self) -> bool:
        return len(self._poses) >= 2

    @property
    def gear(self) -> int:
        return self._gear

    def set_segment(self, poses: FloatArray, gear: int, speeds: FloatArray) -> None:
        self._poses = np.asarray(poses, dtype=np.float64)
        self._speeds = np.asarray(speeds, dtype=np.float64)
        ds = np.hypot(*np.diff(self._poses[:, :2], axis=0).T) if len(poses) > 1 else np.zeros(0)
        self._s = np.concatenate([[0.0], np.cumsum(ds)])
        self._gear = 1 if gear >= 0 else -1
        self._idx = 0
        # Straight extension past the end so steering references stay valid while the car
        # settles at the end point (the virtual front axle runs a wheelbase ahead).
        end = self._poses[-1]
        ext_s = np.arange(1, 26) * 0.2
        ext = np.stack(
            [
                end[0] + self._gear * ext_s * math.cos(end[2]),
                end[1] + self._gear * ext_s * math.sin(end[2]),
                np.full_like(ext_s, end[2]),
            ],
            axis=1,
        )
        self._ext = np.vstack([self._poses, ext])
        self._ext_s = np.concatenate([self._s, self._s[-1] + ext_s])
        dth = np.diff(np.unwrap(self._ext[:, 2]))
        dss = np.maximum(np.diff(self._ext_s), 1e-6)
        k = dth / dss
        self._kappa = np.concatenate([k, k[-1:]]) if len(k) else np.zeros(len(self._ext))

    def clear(self) -> None:
        self._poses = np.zeros((0, 3))
        self._idx = 0

    def _nearest(self, x: float, y: float) -> int:
        lo = self._idx
        hi = min(len(self._poses), lo + self.cfg.search_window)
        d = np.hypot(self._poses[lo:hi, 0] - x, self._poses[lo:hi, 1] - y)
        self._idx = lo + int(np.argmin(d))
        return self._idx

    def _point_at(self, s: float) -> FloatArray:
        s = min(max(s, 0.0), float(self._ext_s[-1]))
        i = int(np.searchsorted(self._ext_s, s))
        i = min(max(i, 1), len(self._ext_s) - 1)
        s0, s1 = self._ext_s[i - 1], self._ext_s[i]
        t = 0.0 if s1 - s0 < 1e-9 else (s - s0) / (s1 - s0)
        return np.asarray(self._ext[i - 1] + t * (self._ext[i] - self._ext[i - 1]))

    def step(
        self,
        state: VehicleState,
        dt: float,
        speed_limit: float = math.inf,
        emergency: bool = False,
    ) -> Command:
        """Compute the actuator command for one control period."""
        if not self.active:
            return Command(self._brake(state, dt), 0.0, self._gear, True, 0.0, 0.0, 0.0, 0.0)
        p = state.pose
        i = self._nearest(p.x, p.y)
        ref = self._poses[i]
        # Signed cross-track (left of path positive) and heading error.
        dx, dy = p.x - ref[0], p.y - ref[1]
        cte = -math.sin(ref[2]) * dx + math.cos(ref[2]) * dy
        heading_err = wrap_angle(p.theta - ref[2])
        end = self._poses[-1]
        # Remaining distance: arc length beyond the nearest index plus projection error.
        along_end = (end[0] - p.x) * math.cos(end[2]) + (end[1] - p.y) * math.sin(end[2])
        remaining = max(float(self._s[-1] - self._s[i]), 0.0)
        if remaining < 1.0:
            remaining = max(self._gear * along_end, 0.0)
        speed = abs(state.speed)
        steer = self._steer(state, i, speed)
        if remaining <= self.cfg.stop_tolerance or (
            i == len(self._poses) - 1 and self._gear * along_end <= 0
        ):
            accel = self._brake(state, dt)
            done = speed < 0.05
            return Command(accel, steer, self._gear, done, cte, heading_err, 0.0, remaining)
        target = float(self._speeds[min(i + 2, len(self._speeds) - 1)])
        target = max(target, self.speed_cfg.creep)
        stop_dist = max(remaining - self.cfg.stop_tolerance / 2, 0.01)
        target = min(target, math.sqrt(2.0 * self.speed_cfg.decel * 0.8 * stop_dist), speed_limit)
        a_mag = self.cfg.speed_gain * (target - speed)
        # Feedforward: the constant deceleration that stops exactly at the segment end.
        a_mag = min(a_mag, -(speed**2) / (2.0 * stop_dist) + 0.3)
        a_mag = self._limit(a_mag, dt, emergency=emergency)
        return Command(
            self._gear * a_mag, steer, self._gear, False, cte, heading_err, target, remaining
        )

    def _limit(self, a_mag: float, dt: float, emergency: bool) -> float:
        cfg = self.speed_cfg
        v = self.vehicle.config
        lo = -v.emergency_decel if emergency else -v.max_decel
        a_mag = min(max(a_mag, lo), v.max_accel)
        if not emergency:
            jmax = cfg.jerk * dt
            a_mag = min(max(a_mag, self._last_accel - jmax), self._last_accel + jmax)
        self._last_accel = a_mag
        return a_mag

    def _brake(self, state: VehicleState, dt: float) -> float:
        speed = abs(state.speed)
        a_mag = self._limit(-max(speed / dt, 0.0), dt, emergency=speed > 0.5)
        gear = 1 if state.speed >= 0 else -1
        return gear * a_mag

    def _steer(self, state: VehicleState, i: int, speed: float) -> float:
        p = state.pose
        wb = self.vehicle.config.wheelbase
        if self._gear > 0:
            ld = self.cfg.lookahead_base + self.cfg.lookahead_gain * speed
            tgt = self._point_at(float(self._s[i]) + ld)
            alpha = math.atan2(tgt[1] - p.y, tgt[0] - p.x) - p.theta
            dist = max(math.hypot(tgt[0] - p.x, tgt[1] - p.y), 1e-3)
            delta = math.atan2(2.0 * wb * math.sin(alpha), dist)
        else:
            th_v = p.theta + math.pi
            ref = self._ext[i]
            ref_v = ref[2] + math.pi
            e_head = wrap_angle(ref_v - th_v)
            e_ct = -math.sin(ref_v) * (p.x - ref[0]) + math.cos(ref_v) * (p.y - ref[1])
            preview = self.cfg.steer_preview * max(speed, 0.3)
            j = int(np.searchsorted(self._ext_s, self._ext_s[i] + preview))
            kappa = float(self._kappa[min(j, len(self._kappa) - 1)])
            delta_v = (
                math.atan(wb * kappa)
                + self.cfg.heading_gain * e_head
                + math.atan2(-self.cfg.stanley_gain * e_ct, speed + self.cfg.stanley_softening)
            )
            delta = -delta_v
        mx = self.vehicle.config.max_steer
        return max(-mx, min(mx, delta))
