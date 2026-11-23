"""Collision checking of vehicle poses against the perceived occupancy grid.

The check is exact up to grid resolution: the full footprint (body *and* mirrors, C10),
inflated by a margin, is rasterised into sample points at sub-cell spacing and looked up in
the occupancy grid. This avoids the end-of-car conservatism of disk approximations, which
would otherwise forbid pulling up close to a curb or charger. A disk-based clearance lookup
on the distance field is provided separately for soft costs.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from parkpilot.core.geometry import FloatArray
from parkpilot.perception.grid import OccupancyGrid
from parkpilot.vehicle.model import VehicleModel


def _rect_samples(
    x0: float, x1: float, y0: float, y1: float, spacing: float, band: float | None = None
) -> FloatArray:
    """Grid of points covering a rectangle; with ``band``, only a perimeter band that thick.

    A band is enough when consecutive checked poses are closer than the band plus one cell:
    an obstacle cannot jump from outside the footprint to inside without crossing it.
    """
    nx = max(2, math.ceil((x1 - x0) / spacing) + 1)
    ny = max(2, math.ceil((y1 - y0) / spacing) + 1)
    xs, ys = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    if band is not None:
        edge = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]])
        pts = pts[edge <= band + 1e-9]
    return pts


class CollisionChecker:
    """Footprint-vs-grid checks for one grid snapshot and one margin."""

    def __init__(
        self,
        grid: OccupancyGrid,
        vehicle: VehicleModel,
        margin: float,
        occupied: NDArray[np.bool_] | None = None,
    ) -> None:
        self.grid = grid
        self.vehicle = vehicle
        self.margin = margin
        self.occ = grid.occupied() if occupied is None else occupied
        c = vehicle.config
        # Inflate by the margin plus half a cell so partially covered cells count.
        m = margin + grid.res * 0.5
        spacing = grid.res * 0.9
        band = 1.6 * grid.res
        body = _rect_samples(
            -c.rear_overhang - m,
            c.length - c.rear_overhang + m,
            -c.width / 2 - m,
            c.width / 2 + m,
            spacing,
            band,
        )
        self._fast_radius = vehicle.circle_radius + m + grid.res
        """Disk radius beyond which a pose is certainly free (skips the exact check)."""
        mirror_x = c.length - c.rear_overhang - c.mirror_from_front
        mirror = np.vstack(
            [
                _rect_samples(
                    mirror_x - c.mirror_length / 2 - m,
                    mirror_x + c.mirror_length / 2 + m,
                    side * c.width / 2,
                    side * (c.mirror_width / 2 + m),
                    spacing,
                )
                if side > 0
                else _rect_samples(
                    mirror_x - c.mirror_length / 2 - m,
                    mirror_x + c.mirror_length / 2 + m,
                    -(c.mirror_width / 2 + m),
                    -c.width / 2,
                    spacing,
                )
                for side in (1, -1)
            ]
        )
        self.samples = np.vstack([body, mirror])
        """Footprint sample points in the rear-axle frame (M, 2)."""

    def free(self, poses: FloatArray) -> NDArray[np.bool_]:
        """Per-pose collision-free flags for poses ``(N, 3)``.

        Stage 1: covering disks on the distance field; poses far from everything pass.
        Stage 2: exact footprint rasterisation for the rest.
        """
        p = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            return np.zeros(0, dtype=bool)
        centers = self.vehicle.circle_centers(p)
        sure = self.grid.distance_at(centers).min(axis=1) >= self._fast_radius
        out = sure.copy()
        todo = np.flatnonzero(~sure)
        if len(todo):
            out[todo] = self._exact(p[todo])
        return out

    def _exact(self, p: FloatArray) -> NDArray[np.bool_]:
        c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
        sx, sy = self.samples[:, 0], self.samples[:, 1]
        wx = p[:, 0:1] + c[:, None] * sx - s[:, None] * sy
        wy = p[:, 1:2] + s[:, None] * sx + c[:, None] * sy
        g = self.grid
        cols = np.floor((wx - g.origin[0]) / g.res).astype(np.int64)
        rows = np.floor((wy - g.origin[1]) / g.res).astype(np.int64)
        inside = (rows >= 0) & (rows < g.ny) & (cols >= 0) & (cols < g.nx)
        hit = np.ones_like(inside)
        hit[inside] = self.occ[rows[inside], cols[inside]]
        return np.asarray(~hit.any(axis=1))

    def path_free(self, poses: FloatArray) -> bool:
        return bool(self.free(poses).all())

    def clearance(self, poses: FloatArray) -> FloatArray:
        """Approximate lateral clearance (m): distance-field value at points along the
        centreline minus half the mirror width. Accurate at the sides, conservative at the
        ends; used only for soft costs and speed limits, never for collision decisions."""
        centers = self.vehicle.circle_centers(np.asarray(poses).reshape(-1, 3))
        d = self.grid.distance_at(centers)
        return np.asarray(d.min(axis=-1) - self.vehicle.config.mirror_width / 2)
