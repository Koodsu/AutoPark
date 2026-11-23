"""Safety supervisor with e-stop authority (C7, C8, C12).

Runs every control tick on perception output (never ground truth) and returns a speed
limit, an emergency flag and a human-readable reason:

* **Static path check.** The footprint (mirrors included) is swept along the remaining path
  out to the stopping distance plus a buffer, against the *memory* grid, with a margin that
  widens as perception noise grows (night). A blockage caps the speed so the car stops short
  of it, and is reported so the behavior layer can re-plan.
* **Agents and TTC.** Confirmed tracks are predicted with constant velocity; a car showing
  reverse lights gets an extra keep-out radius even before it moves (intent cue). The time
  to conflict with the ego's predicted motion along its path yields a stop (below
  ``ttc_stop``) or a yield (below ``ttc_yield``).
* **Persistence filter.** Unconfirmed tracks (shadows, puddles, noise) can never trigger a
  stop. With ``persistence_filter=False`` (the baseline) any raw detection in the corridor
  does, which reproduces phantom braking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from parkpilot.core.config import SafetyConfig, TrackerConfig
from parkpilot.core.geometry import FloatArray, Pose, points_in_convex_polygon, rect_polygon
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.perception.tracks import Track
from parkpilot.planning.collision import CollisionChecker
from parkpilot.vehicle.model import VehicleModel, VehicleState


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    speed_limit: float
    emergency: bool
    reason: str
    margin: float
    blocked_at: float | None = None
    """Path distance (m) to a static blockage, if any."""
    yield_track: int | None = None
    yield_kind: str | None = None
    min_ttc: float = math.inf


def track_kind(tr: Track) -> str:
    if tr.reverse_lights:
        return "car backing out"
    if tr.extent < 1.0:
        return "pedestrian"
    if tr.extent < 2.5:
        return "cart"
    return "vehicle"


class SafetySupervisor:
    def __init__(
        self, cfg: SafetyConfig, tracker_cfg: TrackerConfig, vehicle: VehicleModel
    ) -> None:
        self.cfg = cfg
        self.tracker_cfg = tracker_cfg
        self.vehicle = vehicle

    def margin(self, noise_sigma: float) -> float:
        """Uncertainty-aware footprint margin (m)."""
        return self.cfg.base_margin + self.cfg.sigma_gain * noise_sigma

    def evaluate(
        self,
        state: VehicleState,
        path_ahead: FloatArray,
        perc: PerceptionOutput,
        checker: CollisionChecker,
    ) -> SafetyDecision:
        """``path_ahead``: remaining poses (N, 3) of the current segment, nearest first."""
        cfg = self.cfg
        margin = self.margin(perc.noise_sigma)
        speed = abs(state.speed)
        limit = math.inf
        reason = ""
        emergency = False
        blocked_at: float | None = None
        s = _arc_lengths(path_ahead)
        horizon = speed**2 / (2 * cfg.comfort_decel) + cfg.lookahead_buffer
        if len(path_ahead) > 1:
            window = s <= horizon
            free = checker.free(path_ahead[window])
            if not free.all():
                first = int(np.argmin(free))
                blocked_at = float(s[window][first])
                stop = max(blocked_at - cfg.stop_buffer, 0.0)
                limit = math.sqrt(2 * cfg.comfort_decel * stop)
                reason = "obstacle on path"
                if (
                    blocked_at < speed**2 / (2 * self.vehicle.config.max_decel) + 0.1
                    and speed > 0.3
                ):
                    emergency = True
        tracks = perc.confirmed
        yield_track, yield_kind, min_ttc = None, None, math.inf
        for tr in tracks:
            dynamic = tr.is_dynamic(self.tracker_cfg)
            if not dynamic:
                continue
            ttc = self._time_to_conflict(state, path_ahead, s, tr)
            if ttc < min_ttc:
                min_ttc = ttc
                if ttc < cfg.ttc_yield:
                    yield_track, yield_kind = tr.id, track_kind(tr)
        if min_ttc < cfg.ttc_yield:
            if min_ttc < cfg.ttc_stop and speed > 0.3:
                emergency = True
            limit = 0.0
            reason = f"yielding to {yield_kind}"
        # Without the persistence filter any raw detection in the corridor ahead is a hazard.
        if not cfg.persistence_filter and self._raw_hit_in_corridor(
            state, path_ahead, s, perc, margin, horizon
        ):
            limit, emergency = 0.0, speed > 0.05
            reason = "detection ahead"
        return SafetyDecision(
            speed_limit=limit,
            emergency=emergency,
            reason=reason,
            margin=margin,
            blocked_at=blocked_at,
            yield_track=yield_track,
            yield_kind=yield_kind,
            min_ttc=min_ttc,
        )

    def _time_to_conflict(
        self, state: VehicleState, path: FloatArray, s: FloatArray, tr: Track
    ) -> float:
        cfg = self.cfg
        radius = tr.extent / 2 + cfg.agent_margin
        if tr.reverse_lights:
            radius += cfg.backing_radius
        v_plan = max(abs(state.speed), 1.0)
        times = np.arange(0.0, cfg.horizon + 1e-9, cfg.time_step)
        if len(path) > 1:
            idx = np.searchsorted(s, v_plan * times).clip(0, len(path) - 1)
            ego = path[idx]
        else:
            ego = np.repeat([[state.pose.x, state.pose.y, state.pose.theta]], len(times), axis=0)
        centers = self.vehicle.circle_centers(ego)  # (T, C, 2)
        agent = tr.position[None, :] + tr.velocity[None, :] * times[:, None]
        d = np.linalg.norm(centers - agent[:, None, :], axis=2).min(axis=1)
        hit = d < self.vehicle.circle_radius + radius
        return float(times[int(np.argmax(hit))]) if hit.any() else math.inf

    def _raw_hit_in_corridor(
        self,
        state: VehicleState,
        path: FloatArray,
        s: FloatArray,
        perc: PerceptionOutput,
        margin: float,
        horizon: float,
    ) -> bool:
        """Any raw detection inside the inflated footprint swept over the next metres."""
        pts = perc.frame.hit_points
        if len(pts) == 0 or len(path) < 2:
            return False
        v = self.vehicle.config
        reach = min(horizon, 3.0)
        idx = np.unique(
            np.searchsorted(s, np.arange(0.0, reach + 1e-9, 0.4)).clip(0, len(path) - 1)
        )
        near = pts[np.hypot(pts[:, 0] - state.pose.x, pts[:, 1] - state.pose.y) < reach + v.length]
        if len(near) == 0:
            return False
        for x, y, th in path[idx]:
            body = self.vehicle.body_center(Pose(float(x), float(y), float(th)))
            poly = rect_polygon(body, v.length + 2 * margin, v.width + 2 * margin)
            if points_in_convex_polygon(near, poly).any():
                return True
        return False


def _arc_lengths(path: FloatArray) -> FloatArray:
    if len(path) < 2:
        return np.zeros(len(path))
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path[:, :2], axis=0).T))])
