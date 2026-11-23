"""Log-odds occupancy grid with temporal memory and a capped Euclidean distance field (C7).

Object permanence comes from two rules:

1. Cells keep their log-odds when they leave the field of view (``memory=True``).
2. Free-space evidence is *not* applied inside the near-field low-object blind zone. A camera
   that cannot see a bollard at 1 m sees ground through it; trusting that "free" reading
   would erase the bollard from memory exactly when it matters.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from parkpilot.core.config import GridConfig, VehicleConfig
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.perception.camera import SensorFrame


class OccupancyGrid:
    def __init__(
        self,
        config: GridConfig,
        bounds: tuple[float, float, float, float],
        vehicle: VehicleConfig,
        low_blind_range: float,
    ) -> None:
        self.cfg = config
        self.vehicle = vehicle
        self.low_blind_range = low_blind_range
        m = config.margin
        self.origin = np.array([bounds[0] - m, bounds[1] - m], dtype=np.float64)
        self.res = config.resolution
        self.nx = math.ceil((bounds[2] - bounds[0] + 2 * m) / self.res)
        self.ny = math.ceil((bounds[3] - bounds[1] + 2 * m) / self.res)
        self.logodds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.observed = np.zeros((self.ny, self.nx), dtype=bool)
        self.transient = np.zeros((self.ny, self.nx), dtype=bool)
        self.tall = np.zeros((self.ny, self.nx), dtype=bool)
        self.low = np.zeros((self.ny, self.nx), dtype=bool)
        """Cells that ever returned a near-ground point (curbs, bollards): never cleared
        from inside the blind zone."""
        """Cells hit from inside the low-object blind zone, hence by something tall."""
        self.version = 0
        self._last_t: float | None = None
        self._dist_cache: tuple[int, float, FloatArray] | None = None

    # -- coordinates -----------------------------------------------------------------------
    def to_cells(self, points: FloatArray) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        idx = np.floor((np.asarray(points) - self.origin) / self.res).astype(np.int64)
        return idx[..., 1], idx[..., 0]

    def cell_centers(self, rows: NDArray[np.int64], cols: NDArray[np.int64]) -> FloatArray:
        return np.stack(
            [self.origin[0] + (cols + 0.5) * self.res, self.origin[1] + (rows + 0.5) * self.res],
            axis=-1,
        )

    def in_bounds(self, rows: NDArray[np.int64], cols: NDArray[np.int64]) -> NDArray[np.bool_]:
        return np.asarray((rows >= 0) & (rows < self.ny) & (cols >= 0) & (cols < self.nx))

    # -- update ----------------------------------------------------------------------------
    def _body_distance(self, ego: Pose, points: FloatArray) -> FloatArray:
        local = (points - ego.xy) @ ego.rotation()
        cx = self.vehicle.length / 2 - self.vehicle.rear_overhang
        dx = np.maximum(np.abs(local[..., 0] - cx) - self.vehicle.length / 2, 0.0)
        dy = np.maximum(np.abs(local[..., 1]) - self.vehicle.width / 2, 0.0)
        return np.asarray(np.hypot(dx, dy))

    def update(self, frame: SensorFrame, transient_hits: NDArray[np.bool_] | None = None) -> None:
        """Fuse one sensor frame.

        ``transient_hits`` flags hits on confirmed *moving* agents. They are inserted like
        any other hit (safety first) but their cells are marked transient: free space may
        clear them even inside the blind zone, so a pedestrian who walked past the bumper
        does not leave a permanent ghost behind.
        """
        c = self.cfg
        if not c.memory:
            self.logodds[:] = 0.0
            self.observed[:] = False
        elif c.decay_per_s > 0 and self._last_t is not None:
            k = math.exp(-c.decay_per_s * max(0.0, frame.t - self._last_t))
            self.logodds *= k
        self._last_t = frame.t
        step = self.res * 0.5
        free_len = np.minimum(frame.ranges - c.free_margin, c.free_trust_range)
        free_len = np.where(frame.hits, free_len, np.minimum(frame.max_ranges, c.free_trust_range))
        free_len = np.where(frame.valid, free_len, 0.0)
        n_steps = math.ceil(float(free_len.max(initial=0.0)) / step)
        if n_steps > 0:
            s = np.arange(n_steps) * step
            pts = frame.origins[:, None, :] + frame.directions[:, None, :] * s[None, :, None]
            valid = s[None, :] < free_len[:, None]
            # Only samples near the body can fall in the blind zone; skip the rest.
            near = int(min(n_steps, math.ceil((self.low_blind_range + 3.5) / step)))
            blind = np.zeros_like(valid)
            blind[:, :near] = self._body_distance(frame.ego, pts[:, :near]) < self.low_blind_range
            rows, cols = self.to_cells(pts[valid & ~blind])
            ok = self.in_bounds(rows, cols)
            free_mask = np.zeros_like(self.observed)
            free_mask[rows[ok], cols[ok]] = True
            # Inside the blind zone, free space only erases cells known to hold *tall* things
            # (a tall object there would have been seen) or moving agents' trails; cells
            # that ever held a low object stay remembered (object permanence).
            rows, cols = self.to_cells(pts[valid & blind])
            ok = self.in_bounds(rows, cols)
            blind_mask = np.zeros_like(self.observed)
            blind_mask[rows[ok], cols[ok]] = True
            free_mask |= blind_mask & (self.transient | self.tall) & ~self.low
        else:
            free_mask = np.zeros_like(self.observed)
        hit_pts = frame.hit_points
        rows, cols = self.to_cells(hit_pts)
        ok = self.in_bounds(rows, cols)
        hit_mask = np.zeros_like(self.observed)
        hit_mask[rows[ok], cols[ok]] = True
        if len(hit_pts) and len(frame.hit_low) == len(frame.hits) and frame.hit_low.any():
            lr, lc = self.to_cells(hit_pts[frame.hit_low[frame.hits]])
            lok = self.in_bounds(lr, lc)
            self.low[lr[lok], lc[lok]] = True
        if len(hit_pts) and len(frame.hit_low) == len(frame.hits):
            # Points measured well above the ground mark cells as holding something tall.
            tall_pts = hit_pts[~frame.hit_low[frame.hits]]
            tr_, tc_ = self.to_cells(tall_pts)
            tok = self.in_bounds(tr_, tc_)
            self.tall[tr_[tok], tc_[tok]] = True
        if len(hit_pts):
            # Only tall objects are reported inside the low-object blind zone.
            near_body = self._body_distance(frame.ego, hit_pts) < self.low_blind_range
            if near_body.any():
                nr, nc = self.to_cells(hit_pts[near_body])
                nok = self.in_bounds(nr, nc)
                self.tall[nr[nok], nc[nok]] = True
        if transient_hits is not None and transient_hits.any():
            tr, tc = self.to_cells(hit_pts[transient_hits])
            tok = self.in_bounds(tr, tc)
            self.transient[tr[tok], tc[tok]] = True
        self.transient &= self.logodds > -0.5  # forget the flag once seen free again
        free_mask &= ~hit_mask
        self.logodds[free_mask] += c.l_free
        self.logodds[hit_mask] += c.l_occ
        # The space the car body occupies right now is free by definition: clear any
        # ghosts there (e.g. the trail of an agent the car has since followed).
        self._clear_body(frame.ego)
        np.clip(self.logodds, c.l_min, c.l_max, out=self.logodds)
        self.observed |= free_mask | hit_mask
        self.version += 1

    def _clear_body(self, ego: Pose) -> None:
        v = self.vehicle
        r = math.hypot(v.length, v.width)
        lo = self.to_cells(np.array([ego.x - r, ego.y - r]))
        hi = self.to_cells(np.array([ego.x + r, ego.y + r]))
        r0, c0 = max(int(lo[0]), 0), max(int(lo[1]), 0)
        r1, c1 = min(int(hi[0]) + 1, self.ny), min(int(hi[1]) + 1, self.nx)
        if r1 <= r0 or c1 <= c0:
            return
        rr, cc = np.mgrid[r0:r1, c0:c1]
        centers = self.cell_centers(rr.ravel(), cc.ravel())
        inside = self._body_distance(ego, centers) <= 0.0
        self.logodds[rr.ravel()[inside], cc.ravel()[inside]] = self.cfg.l_min

    # -- queries ---------------------------------------------------------------------------
    def occupied(self) -> NDArray[np.bool_]:
        return np.asarray(self.logodds > self.cfg.occupied_threshold)

    def probability(self) -> NDArray[np.float32]:
        return np.asarray(1.0 / (1.0 + np.exp(-self.logodds)), dtype=np.float32)

    def occupied_points(self) -> FloatArray:
        rows, cols = np.nonzero(self.occupied())
        return self.cell_centers(rows.astype(np.int64), cols.astype(np.int64))

    def logodds_at(self, points: FloatArray) -> NDArray[np.float32]:
        """Current log-odds at world points (0 outside the grid)."""
        rows, cols = self.to_cells(np.asarray(points, dtype=np.float64).reshape(-1, 2))
        ok = self.in_bounds(rows, cols)
        out = np.zeros(len(rows), dtype=np.float32)
        out[ok] = self.logodds[rows[ok], cols[ok]]
        return out

    def is_occupied_at(self, points: FloatArray) -> NDArray[np.bool_]:
        rows, cols = self.to_cells(np.asarray(points, dtype=np.float64).reshape(-1, 2))
        ok = self.in_bounds(rows, cols)
        out = np.ones(len(rows), dtype=bool)  # outside the grid counts as blocked
        occ = self.occupied()
        out[ok] = occ[rows[ok], cols[ok]]
        return out

    def distance_field(self) -> FloatArray:
        """Distance (m) from each cell centre to the nearest occupied cell centre, capped."""
        cap = self.cfg.distance_cap
        if self._dist_cache is not None and self._dist_cache[0] == self.version:
            return self._dist_cache[2]
        d = edt_capped(self.occupied(), self.res, cap)
        self._dist_cache = (self.version, cap, d)
        return d

    def distance_at(self, points: FloatArray) -> FloatArray:
        """Clearance at arbitrary world points (bilinear-free nearest-cell lookup)."""
        pts = np.asarray(points, dtype=np.float64)
        rows, cols = self.to_cells(pts)
        ok = self.in_bounds(rows, cols)
        field = self.distance_field()
        out = np.zeros(rows.shape, dtype=np.float64)
        out[ok] = field[rows[ok], cols[ok]]
        return out


def edt_capped(occupied: NDArray[np.bool_], res: float, cap: float) -> FloatArray:
    """Exact Euclidean distance transform, saturated at ``cap``.

    Separable two-phase algorithm: per-column distance to the nearest occupied cell (fully
    vectorised with running maxima), then a windowed lower envelope along rows. The row window
    only needs to span ``cap`` because anything farther saturates anyway.
    """
    ny, nx = occupied.shape
    k = math.ceil(cap / res)
    big = float(k + 1)
    idx = np.arange(ny, dtype=np.float64)[:, None]
    marked = np.where(occupied, idx, -np.inf)
    above = np.maximum.accumulate(marked, axis=0)
    marked_rev = np.where(occupied, idx, np.inf)[::-1]
    below = np.minimum.accumulate(marked_rev, axis=0)[::-1]
    g = np.minimum(idx - above, below - idx)
    g = np.minimum(g, big)
    g2 = g * g
    best = np.full((ny, nx), big * big)
    padded = np.full((ny, nx + 2 * k), big * big)
    padded[:, k : k + nx] = g2
    for dx in range(-k, k + 1):
        best = np.minimum(best, padded[:, k + dx : k + dx + nx] + dx * dx)
    return np.asarray(np.minimum(np.sqrt(best) * res, cap))
