"""SE(2) poses, convex polygons and separating-axis collision tests.

Polygons are ``(N, 2)`` float arrays with vertices in counter-clockwise order. All functions
are pure and allocate new arrays; nothing mutates its inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from parkpilot.core.units import wrap_angle

FloatArray = NDArray[np.float64]

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class Pose:
    """A rigid-body pose in the plane: position (m) and heading (rad, CCW from +x)."""

    x: float
    y: float
    theta: float

    @property
    def xy(self) -> FloatArray:
        return np.array([self.x, self.y], dtype=np.float64)

    def rotation(self) -> FloatArray:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def compose(self, other: Pose) -> Pose:
        """Return ``self * other`` (``other`` expressed in this pose's frame)."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.theta + other.theta),
        )

    def inverse(self) -> Pose:
        c, s = math.cos(self.theta), math.sin(self.theta)
        return Pose(-c * self.x - s * self.y, s * self.x - c * self.y, wrap_angle(-self.theta))

    def relative_to(self, frame: Pose) -> Pose:
        """Express this pose in ``frame``'s local coordinates."""
        return frame.inverse().compose(self)

    def to_world(self, points: FloatArray) -> FloatArray:
        """Transform local points ``(N, 2)`` into the world frame."""
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.rotation().T + self.xy

    def to_local(self, points: FloatArray) -> FloatArray:
        """Transform world points ``(N, 2)`` into this pose's local frame."""
        pts = np.asarray(points, dtype=np.float64)
        return (pts - self.xy) @ self.rotation()

    def distance_to(self, other: Pose) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def advanced(self, distance: float) -> Pose:
        """Move ``distance`` metres along the heading."""
        return Pose(
            self.x + distance * math.cos(self.theta),
            self.y + distance * math.sin(self.theta),
            self.theta,
        )


def rect_polygon(center: Pose, length: float, width: float) -> FloatArray:
    """Rectangle of ``length`` along the pose heading and ``width`` across it."""
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    return center.to_world(local)


def polygon_area(poly: FloatArray) -> float:
    """Signed area (positive for counter-clockwise polygons)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def ensure_ccw(poly: FloatArray) -> FloatArray:
    return poly if polygon_area(poly) >= 0 else poly[::-1].copy()


def polygon_centroid(poly: FloatArray) -> FloatArray:
    return np.asarray(poly.mean(axis=0), dtype=np.float64)


def _axes(poly: FloatArray) -> FloatArray:
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.asarray(normals / np.maximum(norms, _EPS), dtype=np.float64)


def polygons_intersect(a: FloatArray, b: FloatArray, margin: float = 0.0) -> bool:
    """Separating-axis test for two convex polygons.

    ``margin`` grows both shapes by that distance along every tested axis, which is a
    conservative approximation of a Minkowski inflation (exact for edge-to-edge contact).
    """
    for axis in np.concatenate([_axes(a), _axes(b)]):
        pa, pb = a @ axis, b @ axis
        if pa.max() + margin < pb.min() or pb.max() + margin < pa.min():
            return False
    return True


def points_in_convex_polygon(points: FloatArray, poly: FloatArray) -> NDArray[np.bool_]:
    """Vectorised inside test for a convex CCW polygon (boundary counts as inside)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    p = ensure_ccw(poly)
    edges = np.roll(p, -1, axis=0) - p
    rel = pts[:, None, :] - p[None, :, :]
    cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
    return np.asarray(np.all(cross >= -1e-9, axis=1))


def point_segment_distance(points: FloatArray, a: FloatArray, b: FloatArray) -> FloatArray:
    """Distance from each point ``(N, 2)`` to the segment ``a``-``b``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ab = b - a
    denom = float(ab @ ab)
    if denom < _EPS:
        return np.asarray(np.linalg.norm(pts - a, axis=1))
    t = np.clip((pts - a) @ ab / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.asarray(np.linalg.norm(pts - proj, axis=1))


def points_polygon_distance(points: FloatArray, poly: FloatArray) -> FloatArray:
    """Distance from points to a convex polygon (0 for points inside)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    d = np.full(len(pts), np.inf)
    nxt = np.roll(poly, -1, axis=0)
    for a, b in zip(poly, nxt, strict=True):
        d = np.minimum(d, point_segment_distance(pts, a, b))
    inside = points_in_convex_polygon(pts, poly)
    d[inside] = 0.0
    return d


def polygon_distance(a: FloatArray, b: FloatArray) -> float:
    """Minimum distance between two convex polygons (0 if they overlap)."""
    if polygons_intersect(a, b):
        return 0.0
    return float(
        min(points_polygon_distance(a, b).min(), points_polygon_distance(b, a).min()),
    )


def oriented_bounding_box(points: FloatArray) -> tuple[Pose, float, float]:
    """PCA-aligned bounding box of a point set: (center pose, length, width).

    The heading follows the principal axis, so ``length >= width``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mean = pts.mean(axis=0)
    if len(pts) < 3:
        return Pose(float(mean[0]), float(mean[1]), 0.0), 0.0, 0.0
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    theta = math.atan2(float(major[1]), float(major[0]))
    frame = Pose(float(mean[0]), float(mean[1]), theta)
    local = frame.to_local(pts)
    lo, hi = local.min(axis=0), local.max(axis=0)
    center_local = (lo + hi) / 2.0
    center = frame.to_world(center_local[None, :])[0]
    length, width = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    if width > length:
        length, width = width, length
        theta = wrap_angle(theta + math.pi / 2.0)
    return Pose(float(center[0]), float(center[1]), theta), length, width


def segment_intersections(
    origins: FloatArray, directions: FloatArray, seg_a: FloatArray, seg_b: FloatArray
) -> FloatArray:
    """Ray/segment intersection distances.

    ``origins`` and ``directions`` are ``(R, 2)`` (directions unit length); ``seg_a`` and
    ``seg_b`` are ``(S, 2)``. Returns an ``(R, S)`` array of ray parameters, ``inf`` where the
    ray misses the segment.
    """
    if len(seg_a) == 0:
        return np.full((len(origins), 0), np.inf)
    e = seg_b - seg_a  # (S, 2)
    d = directions[:, None, :]  # (R, 1, 2)
    w = seg_a[None, :, :] - origins[:, None, :]  # (R, S, 2)
    denom = d[..., 0] * e[None, :, 1] - d[..., 1] * e[None, :, 0]  # (R, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / denom
        u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / denom
    valid = (np.abs(denom) > _EPS) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return np.where(valid, t, np.inf)


def polygon_edges(polys: list[FloatArray]) -> tuple[FloatArray, FloatArray]:
    """Stack the edges of many polygons into segment start/end arrays."""
    if not polys:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty
    starts = np.concatenate(polys)
    ends = np.concatenate([np.roll(p, -1, axis=0) for p in polys])
    return starts, ends
