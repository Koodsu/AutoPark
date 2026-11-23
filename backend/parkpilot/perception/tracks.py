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

    def confirmed(self, cfg: TrackerConfig) -> bool:
        return self.hits >= cfg.confirm_frames

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    def is_dynamic(self, cfg: TrackerConfig) -> bool:
        return self.confirmed(cfg) and (self.moving_frames >= 2 or self.reverse_lights)


def cluster_points(points: FloatArray, radius: float) -> list[NDArray[np.int64]]:
    """Single-linkage clustering using a spatial hash (cells of size ``radius``)."""
    n = len(points)
    if n == 0:
        return []
    keys = np.floor(points / radius).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, (kx, ky) in enumerate(keys):
        buckets.setdefault((int(kx), int(ky)), []).append(i)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    r2 = radius * radius
    for (kx, ky), members in buckets.items():
        neigh: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh.extend(buckets.get((kx + dx, ky + dy), ()))
        nb = np.array(neigh)
        for i in members:
            d2 = np.sum((points[nb] - points[i]) ** 2, axis=1)
            for j in nb[d2 <= r2]:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.array(g, dtype=np.int64) for g in groups.values()]


@dataclass(slots=True)
class Tracker:
    cfg: TrackerConfig
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _last_t: float | None = None

    def update(self, t: float, points: FloatArray, cues: list[CueObservation]) -> NDArray[np.bool_]:
        """Update tracks from this frame's hit points.

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
            if extent <= cfg.max_extent:
                clusters.append((idx, pts.mean(axis=0), extent))
        predicted = [tr.position + tr.velocity * dt for tr in self.tracks]
        used: set[int] = set()
        pairs = []
        for ti, pred in enumerate(predicted):
            for ci, (_, centroid, _) in enumerate(clusters):
                d = float(np.hypot(*(centroid - pred)))
                if d <= cfg.gate:
                    pairs.append((d, ti, ci))
        pairs.sort()
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
                Track(self._next_id, centroid.copy(), np.zeros(2), extent, points[idx])
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
