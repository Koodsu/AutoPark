"""Detection clustering and track persistence filtering (C8, C12).

Hits are clustered into objects; compact clusters become tracks with an alpha-beta filter on
position and velocity. A track is only *confirmed* after ``confirm_frames`` consecutive
detections. The safety supervisor ignores unconfirmed tracks for hard braking (shadows and
puddles flicker in and out), but uses confirmed moving tracks immediately via TTC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from parkpilot.core.config import TrackerConfig
from parkpilot.core.geometry import FloatArray
from parkpilot.perception.camera import CueObservation


@dataclass(slots=True)
class Track:
    id: int
    position: FloatArray
    velocity: FloatArray
    extent: float
    points: FloatArray
    hits: int = 1
    misses: int = 0
    age: int = 1
    reverse_lights: bool = False
    moving_frames: int = 0
    free_space_frames: int = 0
    """Recent frames in which most hits landed on cells the grid had confidently seen
    free: something moved *into* free space. Static objects revealed by parallax land on
    unknown or occupied cells instead."""
    first_points: FloatArray | None = None
    """Hits when the track was born; both ends of a real mover leave them behind."""
    origin: FloatArray | None = None
    """Where the track was first seen; net displacement separates movers from parallax."""

    def confirmed(self, cfg: TrackerConfig) -> bool:
        return self.hits >= cfg.confirm_frames

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def displacement(self) -> float:
        if self.origin is None:
            return 0.0
        return float(np.hypot(*(self.position - self.origin)))

    def is_dynamic(self, cfg: TrackerConfig) -> bool:
        """Moving agent: sustained speed *and* real net travel, or a reverse-light cue.

        The centroid of a static object drifts as different faces come into view, so speed
        alone would misclassify parked cars as movers and erase them from memory.
        """
        if not self.confirmed(cfg):
            return False
        moving = (
            self.moving_frames >= cfg.moving_frames
            and self.displacement >= cfg.min_travel
            and self.free_space_frames >= cfg.free_space_frames
        )
        return moving or self.reverse_lights

    def both_ends_moved(self, distance: float) -> bool:
        """True if the leading *and* trailing edges advanced along the motion direction.

        A static object seen from a moving car reveals more of itself, so its centroid
        drifts, but the edge that was visible from the start stays put. A real mover
        translates as a whole.
        """
        if self.first_points is None or len(self.first_points) == 0 or len(self.points) == 0:
            return False
        sp = self.speed
        if sp < 1e-6:
            return False
        u = self.velocity / sp
        p0, p1 = self.first_points @ u, self.points @ u
        return bool(p1.min() - p0.min() >= distance and p1.max() - p0.max() >= distance)


def cluster_points(points: FloatArray, radius: float) -> list[NDArray[np.int64]]:
    """Grid-hash clustering: points in 8-adjacent cells of size ``radius`` are linked.

    This is single-linkage with a linkage distance between ``radius`` and
    ``2 * sqrt(2) * radius``; it runs in O(cells) instead of O(points^2).
    """
    n = len(points)
    if n == 0:
        return []
    keys = np.floor(points / radius).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    index = {(int(k[0]), int(k[1])): i for i, k in enumerate(uniq)}
    parent = list(range(len(uniq)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (kx, ky), i in index.items():
        for dx, dy in ((1, -1), (1, 0), (1, 1), (0, 1)):
            j = index.get((kx + dx, ky + dy))
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(len(uniq))])
    labels = roots[inverse]
    order = np.argsort(labels, kind="stable")
    splits = np.flatnonzero(np.diff(labels[order])) + 1
    return [g.astype(np.int64) for g in np.split(order, splits)]


@dataclass(slots=True)
class Tracker:
    cfg: TrackerConfig
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _last_t: float | None = None

    def update(
        self,
        t: float,
        points: FloatArray,
        cues: list[CueObservation],
        prior_free: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        """Update tracks from this frame's hit points.

        ``prior_free`` flags hits on cells the occupancy grid had confidently seen free.

        Returns a mask over ``points`` marking hits that belong to confirmed *moving* tracks,
        so the occupancy grid can avoid smearing moving agents into its memory.
        """
        cfg = self.cfg
        dt = 0.1 if self._last_t is None else max(1e-3, t - self._last_t)
        self._last_t = t
        moving_mask = np.zeros(len(points), dtype=bool)
        clusters = []
        for idx in cluster_points(points, cfg.cluster_radius):
            pts = points[idx]
            extent = float(np.max(np.ptp(pts, axis=0))) if len(pts) > 1 else 0.0
            if extent <= cfg.max_extent and len(pts) >= cfg.min_points:
                clusters.append((idx, pts.mean(axis=0), extent))
        predicted = [tr.position + tr.velocity * dt for tr in self.tracks]
        used: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        if predicted and clusters:
            pred = np.array(predicted)
            cents = np.array([c[1] for c in clusters])
            dist = np.linalg.norm(pred[:, None, :] - cents[None, :, :], axis=2)
            ti_idx, ci_idx = np.nonzero(dist <= cfg.gate)
            pairs = sorted(
                zip(dist[ti_idx, ci_idx].tolist(), ti_idx.tolist(), ci_idx.tolist(), strict=True)
            )
        matched_tracks: set[int] = set()
        for _, ti, ci in pairs:
            if ti in matched_tracks or ci in used:
                continue
            matched_tracks.add(ti)
            used.add(ci)
            tr = self.tracks[ti]
            idx, centroid, extent = clusters[ci]
            residual = centroid - predicted[ti]
            tr.position = predicted[ti] + cfg.alpha * residual
            tr.velocity = tr.velocity + (cfg.beta / dt) * residual
            tr.extent, tr.points = extent, points[idx]
            tr.hits += 1
            tr.misses = 0
            tr.age += 1
            tr.moving_frames = tr.moving_frames + 1 if tr.speed > cfg.dynamic_speed else 0
            into_free = prior_free is not None and float(prior_free[idx].mean()) >= 0.5
            tr.free_space_frames = tr.free_space_frames + 1 if into_free else 0
            if tr.is_dynamic(cfg):
                moving_mask[idx] = True
        for ti, tr in enumerate(self.tracks):
            if ti not in matched_tracks:
                tr.misses += 1
                tr.hits = 0 if tr.hits < cfg.confirm_frames else tr.hits
                tr.age += 1
        self.tracks = [tr for tr in self.tracks if tr.misses <= cfg.max_misses]
        for ci, (idx, centroid, extent) in enumerate(clusters):
            if ci in used:
                continue
            self.tracks.append(
                Track(
                    self._next_id,
                    centroid.copy(),
                    np.zeros(2),
                    extent,
                    points[idx],
                    origin=centroid.copy(),
                    first_points=points[idx].copy(),
                )
            )
            self._next_id += 1
        for tr in self.tracks:
            tr.reverse_lights = any(
                c.reverse_lights
                and math.hypot(c.position[0] - tr.position[0], c.position[1] - tr.position[1])
                <= cfg.cue_radius
                for c in cues
            )
        return moving_mask

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed(self.cfg) and t.misses == 0]
