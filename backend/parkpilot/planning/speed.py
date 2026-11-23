"""Clearance-driven speed profile (C2).

Speed is high in open space and low near obstacles, capped by curvature (lateral
acceleration) and shaped by forward/backward passes so acceleration and deceleration limits
hold, with a smooth stop at every cusp and at the goal. The longitudinal controller adds the
jerk limit when tracking this profile.
"""

from __future__ import annotations

import numpy as np

from parkpilot.core.config import SpeedConfig
from parkpilot.core.geometry import FloatArray


def path_curvature(poses: FloatArray) -> FloatArray:
    """Unsigned curvature per pose from heading change over arc length."""
    if len(poses) < 2:
        return np.zeros(len(poses))
    ds = np.hypot(*np.diff(poses[:, :2], axis=0).T)
    dth = np.diff(np.unwrap(poses[:, 2]))
    k = np.abs(dth) / np.maximum(ds, 1e-6)
    return np.concatenate([k[:1], k])


def _steer_rate_cap(poses: FloatArray, wheelbase: float, max_steer_rate: float) -> FloatArray:
    """Speed at which the steering actuator can keep up with curvature changes.

    Paths made of arcs have curvature steps; the wheel needs time to turn, so the car slows
    where the required steering angle changes quickly per metre (v <= rate / |d delta/ds|).
    """
    ds = np.hypot(*np.diff(poses[:, :2], axis=0).T)
    dth = np.diff(np.unwrap(poses[:, 2]))
    signed_k = dth / np.maximum(ds, 1e-6)
    delta = np.arctan(signed_k * wheelbase)
    s = np.concatenate([[0.0], np.cumsum(ds)])[1:]
    # Change of steering over a 1 m window, assigned to the poses that lead into it.
    window = 1.0
    budget = 0.5  # leave half the actuator rate for feedback corrections
    cap = np.full(len(poses), np.inf)
    for i in range(len(delta)):
        j = int(np.searchsorted(s, s[i] + window))
        j = min(j, len(delta) - 1)
        change = float(np.max(np.abs(delta[i : j + 1] - delta[i])))
        if change > 1e-3:
            lo = int(np.searchsorted(s, s[i] - window))
            cap[lo : i + 2] = np.minimum(cap[lo : i + 2], budget * max_steer_rate * window / change)
    return cap


def speed_profile(
    poses: FloatArray,
    directions: FloatArray,
    clearance: FloatArray,
    cfg: SpeedConfig,
    v_start: float = 0.0,
    wheelbase: float | None = None,
    max_steer_rate: float | None = None,
) -> FloatArray:
    """Target speed magnitude (m/s) at each pose of a single-direction segment or full path.

    Speed reaches zero at the last pose and at every cusp (direction change).
    """
    n = len(poses)
    if n == 0:
        return np.zeros(0)
    reverse = directions < 0
    if cfg.constant_speed is not None:
        v = np.full(n, cfg.constant_speed)
    else:
        frac = np.clip((clearance - cfg.clear_low) / (cfg.clear_high - cfg.clear_low), 0.0, 1.0)
        v = cfg.v_tight + (cfg.v_open - cfg.v_tight) * frac
        kappa = path_curvature(poses)
        v = np.minimum(v, np.sqrt(cfg.lat_accel / np.maximum(kappa, 1e-6)))
        if wheelbase is not None and max_steer_rate is not None and n > 2:
            v = np.minimum(v, _steer_rate_cap(poses, wheelbase, max_steer_rate))
    v = np.where(reverse, np.minimum(v, cfg.v_reverse), v)
    stops = np.zeros(n, dtype=bool)
    stops[-1] = True
    if n > 1:
        cusp = np.flatnonzero(directions[1:] != directions[:-1])
        stops[cusp] = True
    v[stops] = 0.0
    ds = (
        np.concatenate([[0.0], np.hypot(*np.diff(poses[:, :2], axis=0).T)])
        if n > 1
        else np.zeros(1)
    )
    # Forward pass: acceleration limit from the starting speed (and from each cusp).
    out = v.copy()
    out[0] = min(out[0], max(v_start, cfg.creep)) if not stops[0] else 0.0
    for i in range(1, n):
        prev = 0.0 if stops[i - 1] else out[i - 1]
        out[i] = min(out[i], float(np.sqrt(prev**2 + 2 * cfg.accel * ds[i])))
    # Backward pass: deceleration limit toward stops.
    for i in range(n - 2, -1, -1):
        out[i] = min(out[i], np.sqrt(out[i + 1] ** 2 + 2 * cfg.decel * ds[i + 1]))
    return np.asarray(out)
