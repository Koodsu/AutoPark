"""Ground-truth geometry used only for metrics and the debug layer (never by autonomy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parkpilot.core.geometry import FloatArray, Pose, polygons_intersect
from parkpilot.vehicle.model import VehicleModel
from parkpilot.world.model import ObstacleKind, World


def _perimeter(polys: list[FloatArray], spacing: float = 0.1) -> FloatArray:
    pts = []
    for poly in polys:
        nxt = np.roll(poly, -1, axis=0)
        for a, b in zip(poly, nxt, strict=True):
            n = max(1, int(np.ceil(np.hypot(*(b - a)) / spacing)))
            t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a))
    return np.vstack(pts)


def _min_dist(points: FloatArray, polys: list[FloatArray]) -> float:
    """Min distance from points to polygon edges (exact for points outside polygons)."""
    if not polys:
        return np.inf
    a = np.concatenate(polys)
    b = np.concatenate([np.roll(p, -1, axis=0) for p in polys])
    ab = b - a
    denom = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    rel = points[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", rel, ab) / denom, 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * ab[None, :, :]
    return float(np.min(np.linalg.norm(points[:, None, :] - proj, axis=2)))


@dataclass(frozen=True, slots=True)
class TruthSample:
    static_clearance: float
    """Footprint (incl. mirrors) to nearest non-curb obstacle (m)."""
    curb_clearance: float
    agent_clearance: float
    collision: bool
    """Footprint intersects a non-curb obstacle or an agent."""
    curb_contact: bool
    agent_collision: bool


class TruthMonitor:
    def __init__(self, world: World, vehicle: VehicleModel, radius: float = 9.0) -> None:
        self.world = world
        self.vehicle = vehicle
        self.radius = radius
        self._obs = [
            (
                o,
                o.polygon.mean(axis=0),
                float(np.max(np.linalg.norm(o.polygon - o.polygon.mean(axis=0), axis=1))),
            )
            for o in world.obstacles
        ]

    def sample(self, pose: Pose, agent_polys: list[FloatArray]) -> TruthSample:
        foot = self.vehicle.footprint(pose)
        center = np.array([pose.x, pose.y])
        near = [o for o, c, r in self._obs if np.hypot(*(c - center)) - r < self.radius]
        static = [o.polygon for o in near if o.kind is not ObstacleKind.CURB]
        curbs = [o.polygon for o in near if o.kind is ObstacleKind.CURB]
        agents = [p for p in agent_polys if np.hypot(*(p.mean(axis=0) - center)) < self.radius]
        pts = _perimeter(foot)
        d_static = _min_dist(pts, static)
        d_curb = _min_dist(pts, curbs)
        d_agent = _min_dist(pts, agents)

        def hits(polys: list[FloatArray], d: float) -> bool:
            if d > 0.05:
                return False
            return any(polygons_intersect(f, p) for f in foot for p in polys)

        collision = hits(static, d_static)
        agent_collision = hits(agents, d_agent)
        return TruthSample(
            static_clearance=0.0 if collision else d_static,
            curb_clearance=0.0 if hits(curbs, d_curb) else d_curb,
            agent_clearance=0.0 if agent_collision else d_agent,
            collision=collision or agent_collision,
            curb_contact=hits(curbs, d_curb),
            agent_collision=agent_collision,
        )
