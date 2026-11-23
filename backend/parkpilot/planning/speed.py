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


def speed_profile(
    poses: FloatArray,
    directions: FloatArray,
    clearance: FloatArray,
    cfg: SpeedConfig,
    v_start: float = 0.0,
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
