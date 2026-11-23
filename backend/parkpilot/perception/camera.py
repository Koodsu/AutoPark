"""Camera-only sensor simulation (C7, C8).

Each camera casts depth rays across its field of view against ground-truth obstacle edges
and dynamic agents. The model reproduces the failure modes owners complain about:

* **Near-field blind zones.** Nothing is seen within ``near_blind_range`` of the body, and
  *low* objects (curbs, bollards, snow piles) disappear within ``low_blind_range`` because
  they drop below the cameras' view. The ray then "sees past" them, exactly like a camera
  that only sees ground there.
* **Lighting.** Depth noise grows and range and detection probability shrink at night.
* **Occlusion.** Only the first surface along a ray is returned.
* **False positives.** Shadows and puddles (scenario ``phantoms``) and random clutter produce
  short-lived spurious detections.

The output is a :class:`SensorFrame`. Ground-truth labels in it (``truth_*``) exist only for
metrics and the debug layer; the autonomy stack must not read them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from parkpilot.core.config import SensorConfig, VehicleConfig
from parkpilot.core.geometry import FloatArray, Pose, segment_intersections
from parkpilot.world.agents import AgentState
from parkpilot.world.model import Feature, FeatureKind, PaintedLine, World


@dataclass(frozen=True, slots=True)
class LineObservation:
    line_id: str
    """Data association is simulated: the observation carries the painted line's identity."""
    a: tuple[float, float]
    b: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    kind: FeatureKind
    label: str
    position: tuple[float, float]


@dataclass(frozen=True, slots=True)
class CueObservation:
    """A visual intent cue on a detected agent (reverse lights of a car backing out)."""

    position: tuple[float, float]
    reverse_lights: bool


@dataclass(slots=True)
class SensorFrame:
    t: float
    ego: Pose
    origins: FloatArray
    """Ray origins (R, 2)."""
    directions: FloatArray
    """Unit ray directions (R, 2)."""
    ranges: FloatArray
    """Measured range per ray (R,); equals ``max_ranges`` where nothing was returned."""
    max_ranges: FloatArray
    hits: NDArray[np.bool_]
    """True where the ray returned a surface."""
    valid: NDArray[np.bool_]
    """False where depth estimation failed (dropout): no information, not free space."""
    noise_sigma: float
    """Depth noise scale this frame (m at 10 m range); drives uncertainty-aware margins."""
    lighting: float
    lines: list[LineObservation] = field(default_factory=list)
    features: list[FeatureObservation] = field(default_factory=list)
    cues: list[CueObservation] = field(default_factory=list)
    truth_phantom: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    """Ground truth: which hits are false positives (metrics/debug only)."""

    @property
    def hit_points(self) -> FloatArray:
        pts = self.origins + self.directions * self.ranges[:, None]
        return np.asarray(pts[self.hits])


@dataclass(frozen=True, slots=True)
class _Segments:
    a: FloatArray
    b: FloatArray
    low: NDArray[np.bool_]
    agent: NDArray[np.int64]
    """Index into the agent list, or -1 for static obstacles."""


class CameraRig:
    """Simulates every camera on the vehicle for one perception frame."""

    def __init__(self, sensor: SensorConfig, vehicle: VehicleConfig) -> None:
        self.cfg = sensor
        self.vehicle = vehicle
        origins, yaws, ranges = [], [], []
        for cam in sensor.cameras:
            n = max(2, int(cam.fov / sensor.ray_resolution) + 1)
            for a in np.linspace(-cam.fov / 2, cam.fov / 2, n):
                origins.append((cam.x, cam.y))
                yaws.append(cam.yaw + a)
                ranges.append(cam.max_range)
        self._local_origins = np.array(origins, dtype=np.float64)
        self._local_yaws = np.array(yaws, dtype=np.float64)
        self._base_ranges = np.array(ranges, dtype=np.float64)
        self._cams = [(np.array([c.x, c.y]), c.yaw, c.fov) for c in sensor.cameras]
        self._body_cx = vehicle.length / 2 - vehicle.rear_overhang

    def _lighting_scale(self, lighting: float) -> tuple[float, float, float]:
        c = self.cfg
        dark = 1.0 - lighting
        noise_gain = 1.0 + c.night_noise_gain * dark
        range_scale = c.night_range_factor + (1.0 - c.night_range_factor) * lighting
        dropout = c.dropout + c.night_dropout * dark
        return noise_gain, range_scale, dropout

    def _segments(
        self, world: World, agents: list[AgentState], ego: Pose, reach: float
    ) -> _Segments:
        a_list, b_list, low_list, ag_list = [], [], [], []
        polys: list[tuple[FloatArray, bool, int]] = [
            (o.polygon, o.is_low, -1) for o in world.obstacles
        ]
        polys += [(ag.polygon(), False, i) for i, ag in enumerate(agents)]
        for poly, low, idx in polys:
            center = poly.mean(axis=0)
            radius = float(np.max(np.linalg.norm(poly - center, axis=1)))
            if math.hypot(center[0] - ego.x, center[1] - ego.y) - radius > reach:
                continue
            a_list.append(poly)
            b_list.append(np.roll(poly, -1, axis=0))
            low_list.append(np.full(len(poly), low))
            ag_list.append(np.full(len(poly), idx))
        if not a_list:
            z = np.zeros((0, 2))
            return _Segments(z, z, np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int64))
        return _Segments(
            np.concatenate(a_list),
            np.concatenate(b_list),
            np.concatenate(low_list),
            np.concatenate(ag_list).astype(np.int64),
        )

    def _body_distance(self, ego: Pose, points: FloatArray) -> FloatArray:
        """Distance from world points (..., 2) to the ego body rectangle."""
        local = (points - ego.xy) @ ego.rotation()
        dx = np.maximum(np.abs(local[..., 0] - self._body_cx) - self.vehicle.length / 2, 0.0)
        dy = np.maximum(np.abs(local[..., 1]) - self.vehicle.width / 2, 0.0)
        return np.asarray(np.hypot(dx, dy))

    def sense(
        self,
        t: float,
        ego: Pose,
        world: World,
        agents: list[AgentState],
        rng: np.random.Generator,
    ) -> SensorFrame:
        c = self.cfg
        noise_gain, range_scale, dropout = self._lighting_scale(world.lighting)
        origins = ego.to_world(self._local_origins)
        yaws = self._local_yaws + ego.theta
        dirs = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
        max_ranges = self._base_ranges * range_scale
        segs = self._segments(world, agents, ego, float(max_ranges.max()) + 1.0)
        t_mat = segment_intersections(origins, dirs, segs.a, segs.b)  # (R, S)
        if t_mat.shape[1]:
            pts = (
                origins[:, None, :]
                + dirs[:, None, :] * np.where(np.isfinite(t_mat), t_mat, 0.0)[..., None]
            )
            body_d = self._body_distance(ego, pts)
            blind = body_d < c.near_blind_range
            blind |= segs.low[None, :] & (body_d < c.low_blind_range)
            t_mat = np.where(blind, np.inf, t_mat)
            first = np.argmin(t_mat, axis=1)
            true_range = t_mat[np.arange(len(t_mat)), first]
        else:
            first = np.zeros(len(origins), dtype=np.int64)
            true_range = np.full(len(origins), np.inf)
        valid = rng.random(len(origins)) >= dropout
        hits = (true_range <= max_ranges) & valid
        sigma = (
            c.range_noise_base + c.range_noise_per_m * np.where(hits, true_range, 0.0)
        ) * noise_gain
        measured = np.where(hits, true_range + rng.normal(0.0, 1.0, len(hits)) * sigma, max_ranges)
        measured = np.clip(measured, 0.05, max_ranges)
        phantom = np.zeros(len(hits), dtype=bool)
        measured, hits, phantom = self._inject_false_positives(
            world, origins, dirs, measured, hits, phantom, max_ranges, rng
        )
        frame = SensorFrame(
            t=t,
            ego=ego,
            origins=origins,
            directions=dirs,
            ranges=measured,
            max_ranges=max_ranges,
            hits=hits,
            valid=valid | hits,
            noise_sigma=(c.range_noise_base + c.range_noise_per_m * 10.0) * noise_gain,
            lighting=world.lighting,
            truth_phantom=phantom,
        )
        if len(segs.agent):
            ray_agent = np.where(hits & ~phantom, segs.agent[first], -1)
            pts = frame.hit_points
            hit_agent = ray_agent[hits]
            for i in sorted({int(v) for v in ray_agent if v >= 0}):
                centroid = pts[hit_agent == i].mean(axis=0)
                frame.cues.append(
                    CueObservation(
                        (float(centroid[0]), float(centroid[1])), agents[i].reverse_lights
                    )
                )
        self._observe_semantics(frame, ego, world, segs, rng)
        return frame

    def _inject_false_positives(
        self,
        world: World,
        origins: FloatArray,
        dirs: FloatArray,
        measured: FloatArray,
        hits: NDArray[np.bool_],
        phantom: NDArray[np.bool_],
        max_ranges: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, NDArray[np.bool_], NDArray[np.bool_]]:
        measured, hits, phantom = measured.copy(), hits.copy(), phantom.copy()
        sources: list[tuple[FloatArray, float]] = []
        for ph in world.phantoms:
            if rng.random() < ph.rate:
                ang = rng.uniform(0, 2 * math.pi)
                r = ph.radius * math.sqrt(rng.random())
                sources.append(
                    (np.array(ph.position) + r * np.array([math.cos(ang), math.sin(ang)]), 0.0)
                )
        if rng.random() < self.cfg.clutter_rate:
            k = int(rng.integers(len(origins)))
            sources.append((origins[k] + dirs[k] * rng.uniform(2.0, float(max_ranges[k])), 0.0))
        for point, _ in sources:
            rel = point - origins
            dist = np.linalg.norm(rel, axis=1)
            along = np.einsum("ij,ij->i", rel, dirs)
            perp = np.abs(rel[:, 0] * dirs[:, 1] - rel[:, 1] * dirs[:, 0])
            cand = (along > 0) & (perp < 0.25) & (dist < max_ranges) & (dist < measured)
            for k in np.flatnonzero(cand)[:4]:
                measured[k] = dist[k]
                hits[k] = True
                phantom[k] = True
        return measured, hits, phantom

    def _visible(
        self, ego: Pose, points: FloatArray, segs: _Segments, max_range: float
    ) -> NDArray[np.bool_]:
        """Line-of-sight test from any camera to each point (tall obstacles occlude)."""
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        vis = np.zeros(len(points), dtype=bool)
        tall_a = segs.a[~segs.low]
        tall_b = segs.b[~segs.low]
        for cam_xy, yaw, fov in self._cams:
            origin = ego.to_world(cam_xy[None, :])[0]
            rel = points - origin
            dist = np.linalg.norm(rel, axis=1)
            bearing = np.arctan2(rel[:, 1], rel[:, 0]) - (ego.theta + yaw)
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
            ok = (np.abs(bearing) <= fov / 2) & (dist <= max_range) & (dist > 0.5)
            if not ok.any():
                continue
            idx = np.flatnonzero(ok)
            dirs = rel[idx] / dist[idx, None]
            tm = segment_intersections(
                np.repeat(origin[None, :], len(idx), 0), dirs, tall_a, tall_b
            )
            nearest = tm.min(axis=1) if tm.shape[1] else np.full(len(idx), np.inf)
            vis[idx[nearest >= dist[idx] - 0.3]] = True
        return vis

    def _observe_semantics(
        self, frame: SensorFrame, ego: Pose, world: World, segs: _Segments, rng: np.random.Generator
    ) -> None:
        c = self.cfg
        rng_scale = c.night_range_factor + (1.0 - c.night_range_factor) * world.lighting
        feat_range = c.feature_range * rng_scale
        visible_lines: list[PaintedLine] = [ln for ln in world.lines if ln.visible]
        if visible_lines:
            mids = np.array(
                [[(ln.a[0] + ln.b[0]) / 2, (ln.a[1] + ln.b[1]) / 2] for ln in visible_lines]
            )
            vis = self._visible(ego, mids, segs, feat_range * 1.3)
            for ln, v in zip(visible_lines, vis, strict=True):
                if v and rng.random() < 0.9:
                    na = rng.normal(0.0, c.line_noise, 2)
                    nb = rng.normal(0.0, c.line_noise, 2)
                    frame.lines.append(
                        LineObservation(
                            ln.id,
                            (ln.a[0] + float(na[0]), ln.a[1] + float(na[1])),
                            (ln.b[0] + float(nb[0]), ln.b[1] + float(nb[1])),
                        )
                    )
        feats: list[Feature] = world.features
        if feats:
            pos = np.array([f.position for f in feats])
            vis = self._visible(ego, pos, segs, feat_range)
            dist = np.linalg.norm(pos - ego.xy, axis=1)
            for f, v, d in zip(feats, vis, dist, strict=True):
                if not v:
                    continue
                p = c.feature_read_prob * (1.0 - 0.5 * d / feat_range)
                if f.kind in (
                    FeatureKind.PAINT_SYMBOL,
                    FeatureKind.HATCHING,
                    FeatureKind.CURB_PAINT,
                ):
                    p -= c.paint_night_penalty * (1.0 - world.lighting)
                if rng.random() < p:
                    jitter = rng.normal(0.0, 0.1, 2)
                    frame.features.append(
                        FeatureObservation(
                            f.kind,
                            f.label,
                            (f.position[0] + float(jitter[0]), f.position[1] + float(jitter[1])),
                        )
                    )
