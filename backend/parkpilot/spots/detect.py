"""Spot detection from perception output (C6).

Two independent detectors:

* **Lined**: pairs adjacent, parallel painted-line landmarks. The pairing is mutual-nearest
  so lines two stalls apart never form a fake parallel stall.
* **Line-less**: finds gaps between parked vehicles in the occupancy grid. Vehicles are
  connected components of occupied cells; they are grouped into rows along the aisle, and a
  stall is inferred in each sufficiently wide gap, oriented by the row's dominant parking
  orientation (perpendicular or parallel).

Both detectors check occupancy and observation coverage in the memory grid, so a stall is
only offered when the car has actually seen that it is empty (unknown is not free).
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from parkpilot.core.config import SpotDetectionConfig
from parkpilot.core.geometry import (
    FloatArray,
    Pose,
    oriented_bounding_box,
    points_in_convex_polygon,
    rect_polygon,
)
from parkpilot.perception.grid import OccupancyGrid
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.perception.semantic import LineLandmark
from parkpilot.world.model import SpotLayout, SpotType, ZoneType


@dataclass(slots=True)
class SpotCandidate:
    """A perceived parking spot. ``pose`` heading points into the stall from the aisle
    (perpendicular) or along the traffic direction (parallel)."""

    id: str
    pose: Pose
    length: float
    width: float
    layout: SpotLayout
    source: Literal["lines", "gap"]
    pull_through: bool = False
    occupied: bool = False
    observed_fraction: float = 0.0
    inspected: bool = False
    spot_type: SpotType = SpotType.STANDARD
    zone: ZoneType | None = None
    charger: tuple[float, float] | None = None
    legal: bool = True
    reasons: list[str] = field(default_factory=list)
    """Human-readable legality notes."""

    def polygon(self) -> FloatArray:
        return rect_polygon(self.pose, self.length, self.width)


@dataclass(frozen=True, slots=True)
class _Route:
    points: FloatArray

    def nearest(self, p: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        """Closest route point, local unit direction, and distance."""
        q, u, d, _ = self.nearest_segment(p)
        return q, u, d

    def nearest_segment(self, p: FloatArray) -> tuple[FloatArray, FloatArray, float, int]:
        """Like :meth:`nearest` plus the index of the closest segment."""
        best: tuple[FloatArray, FloatArray, float, int] | None = None
        for i, (a, b) in enumerate(zip(self.points[:-1], self.points[1:], strict=True)):
            ab = b - a
            ln = float(np.hypot(*ab))
            u = ab / max(ln, 1e-9)
            t = float(np.clip((p - a) @ u, 0.0, ln))
            q = a + u * t
            d = float(np.hypot(*(p - q)))
            if best is None or d < best[2]:
                best = (q, u, d, i)
        assert best is not None
        return best


def _line_pairs(
    lines: list[LineLandmark], cfg: SpotDetectionConfig
) -> list[tuple[LineLandmark, LineLandmark, float]]:
    """Mutually nearest, parallel, aligned neighbours: (left, right, spacing)."""
    nearest: dict[tuple[str, int], tuple[float, str]] = {}
    by_id = {ln.id: ln for ln in lines}
    for li in lines:
        u = li.direction
        n = np.array([-u[1], u[0]])
        for lj in lines:
            if lj.id == li.id:
                continue
            if abs(u[0] * lj.direction[1] - u[1] * lj.direction[0]) > cfg.parallel_tolerance:
                continue
            rel = lj.mid - li.mid
            if abs(float(rel @ u)) > cfg.line_alignment:
                continue
            d = float(rel @ n)
            side = 1 if d > 0 else -1
            key = (li.id, side)
            if abs(d) > 1e-3 and (key not in nearest or abs(d) < nearest[key][0]):
                nearest[key] = (abs(d), lj.id)
    out = []
    seen: set[tuple[str, str]] = set()
    for (iid, side), (dist, jid) in nearest.items():
        back = nearest.get((jid, -side))
        if back is None or back[1] != iid:
            continue
        key2 = (min(iid, jid), max(iid, jid))
        if key2 in seen:
            continue
        seen.add(key2)
        out.append((by_id[iid], by_id[jid], dist))
    return out


def _components(occ: np.ndarray, max_cells: int = 20000) -> list[np.ndarray]:
    """8-connected components of a boolean grid (BFS over occupied cells only)."""
    rows, cols = np.nonzero(occ)
    if len(rows) > max_cells:
        rows, cols = rows[:max_cells], cols[:max_cells]
    cells = set(zip(rows.tolist(), cols.tolist(), strict=True))
    comps: list[np.ndarray] = []
    while cells:
        start = cells.pop()
        queue = deque([start])
        comp = [start]
        while queue:
            r, c = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nb = (r + dr, c + dc)
                    if nb in cells:
                        cells.remove(nb)
                        queue.append(nb)
                        comp.append(nb)
        comps.append(np.array(comp, dtype=np.int64))
    return comps


@dataclass(slots=True)
class _Vehicle:
    points: FloatArray
    pose: Pose
    length: float
    width: float


class SpotDetector:
    """Stateful detector: keeps stable ids and inspection counters across frames."""

    def __init__(
        self,
        config: SpotDetectionConfig,
        route: list[tuple[float, float]],
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.cfg = config
        self.bounds = bounds
        self.route = _Route(np.array(route, dtype=np.float64))
        self._inspection: dict[str, int] = {}
        self._gap_registry: dict[str, SpotCandidate] = {}
        self._next_gap = 1

    # -- public ----------------------------------------------------------------------------
    def detect(
        self, perc: PerceptionOutput, *, use_lines: bool = True, use_gaps: bool = True
    ) -> list[SpotCandidate]:
        cands: list[SpotCandidate] = []
        if use_lines:
            cands.extend(self._lined(perc))
        if use_gaps:
            gaps = self._gaps(perc)
            # Painted stalls take precedence where both detectors fire.
            for g in gaps:
                if all(g.pose.distance_to(c.pose) > 1.5 for c in cands):
                    cands.append(g)
        for c in cands:
            self._assess(c, perc)
        return cands

    # -- lined -----------------------------------------------------------------------------
    def _lined(self, perc: PerceptionOutput) -> list[SpotCandidate]:
        cfg = self.cfg
        lines = [ln for ln in perc.semantic.lines.values() if ln.count >= 2]
        out: list[SpotCandidate] = []
        for li, lj, spacing in _line_pairs(lines, cfg):
            center = (li.mid + lj.mid) / 2
            u = li.direction
            length = (li.length + lj.length) / 2
            sid = "L:" + "|".join(sorted((li.id, lj.id)))
            # Long lines separate perpendicular stalls, short ones parallel stalls. This also
            # rejects pairs that skip an occluded line (two stalls wide is not one stall).
            long_lines = length >= cfg.long_line
            if (
                long_lines
                and cfg.perpendicular_spacing[0] <= spacing <= cfg.perpendicular_spacing[1]
            ):
                heading = self._into_stall(center, u, length)
                out.append(
                    SpotCandidate(
                        sid,
                        Pose(float(center[0]), float(center[1]), heading),
                        length,
                        spacing,
                        SpotLayout.PERPENDICULAR,
                        "lines",
                    )
                )
            elif not long_lines and cfg.parallel_spacing[0] <= spacing <= cfg.parallel_spacing[1]:
                axis = np.array([-u[1], u[0]])
                heading = self._along_traffic(center, axis)
                out.append(
                    SpotCandidate(
                        sid,
                        Pose(float(center[0]), float(center[1]), heading),
                        spacing,
                        length,
                        SpotLayout.PARALLEL,
                        "lines",
                    )
                )
        return out

    def _into_stall(self, center: FloatArray, axis: FloatArray, length: float) -> float:
        e1, e2 = center - axis * length / 2, center + axis * length / 2
        d1, d2 = self.route.nearest(e1)[2], self.route.nearest(e2)[2]
        h = axis if d1 <= d2 else -axis
        return math.atan2(float(h[1]), float(h[0]))

    def _along_traffic(self, center: FloatArray, axis: FloatArray) -> float:
        _, u, _ = self.route.nearest(center)
        h = axis if float(axis @ u) >= 0 else -axis
        return math.atan2(float(h[1]), float(h[0]))

    # -- line-less -------------------------------------------------------------------------
    def _vehicles(self, perc: PerceptionOutput) -> list[_Vehicle]:
        grid = perc.grid
        # Any positive evidence counts here: a single hit is enough to shape a vehicle
        # outline, and isolated clutter is removed by the component size filter below.
        occ = grid.logodds > self.cfg.vehicle_logodds
        # Work on a window around the ego at half resolution to bound the cost.
        r0, c0 = grid.to_cells(perc.ego.xy)
        half = int(20.0 / grid.res)
        rs, re = max(0, int(r0) - half), min(grid.ny, int(r0) + half)
        cs, ce = max(0, int(c0) - half), min(grid.nx, int(c0) + half)
        win = occ[rs:re, cs:ce]
        h, w = (win.shape[0] // 2) * 2, (win.shape[1] // 2) * 2
        pooled = win[:h, :w].reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
        vehicles: list[_Vehicle] = []
        lo, hi = self.cfg.vehicle_extent
        for comp in _components(pooled):
            rows = comp[:, 0] * 2 + rs + 1
            cols = comp[:, 1] * 2 + cs + 1
            pts = grid.cell_centers(rows, cols)
            if len(pts) < 4:
                continue
            pose, length, width = oriented_bounding_box(pts)
            if lo <= length <= hi:
                vehicles.append(_Vehicle(pts, pose, length, width))
        return vehicles

    def _gaps(self, perc: PerceptionOutput) -> list[SpotCandidate]:
        cfg = self.cfg
        rows: dict[tuple[int, int], list[tuple[_Vehicle, float, float, float]]] = {}
        frames: dict[tuple[int, int], tuple[FloatArray, FloatArray, FloatArray]] = {}
        for v in self._vehicles(perc):
            q, u, _, seg = self.route.nearest_segment(v.pose.xy)
            n = np.array([-u[1], u[0]])
            side = 1 if float((v.pose.xy - q) @ n) > 0 else -1
            seg_key = (seg, side)
            q = self.route.points[seg]
            rel = v.points - q
            along = rel @ u
            off = np.abs(rel @ n)
            rows.setdefault(seg_key, []).append(
                (v, float(along.min()), float(along.max()), float(off.min()))
            )
            frames.setdefault(seg_key, (q, u, n * side))
        out: list[SpotCandidate] = []
        for key, members in rows.items():
            q, u, n_side = frames[key]
            # Split by offset from the aisle into separate rows.
            members.sort(key=lambda m: m[3])
            groups: list[list[tuple[_Vehicle, float, float, float]]] = []
            for m in members:
                if groups and abs(m[3] - groups[-1][0][3]) <= cfg.row_tolerance:
                    groups[-1].append(m)
                else:
                    groups.append([m])
            # Only the row facing this aisle: farther "rows" are usually the rear faces of the
            # same cars seen through gaps.
            for grp in groups[:1]:
                if len(grp) < 2:
                    continue
                grp.sort(key=lambda m: m[1])
                front = float(np.median([m[3] for m in grp]))
                aligned = [
                    abs(math.cos(m[0].pose.theta - math.atan2(u[1], u[0]))) > 0.8
                    and m[0].length > 3.5
                    for m in grp
                ]
                parallel = sum(aligned) > len(grp) / 2
                for a, b in itertools.pairwise(grp):
                    g0, g1 = a[2], b[1]
                    out.extend(self._stalls_in_gap(q, u, n_side, front, g0, g1, parallel))
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            out = [
                c
                for c in out
                if (
                    (poly := c.polygon())[:, 0].min() >= xmin
                    and poly[:, 0].max() <= xmax
                    and poly[:, 1].min() >= ymin
                    and poly[:, 1].max() <= ymax
                )
            ]
        return self._stabilise(out)

    def _stalls_in_gap(
        self,
        q: FloatArray,
        u: FloatArray,
        n_side: FloatArray,
        front: float,
        g0: float,
        g1: float,
        parallel: bool,
    ) -> list[SpotCandidate]:
        cfg = self.cfg
        gap = g1 - g0
        heading_in = math.atan2(float(n_side[1]), float(n_side[0]))
        out: list[SpotCandidate] = []
        if parallel:
            if gap < cfg.gap_parallel_min:
                return out
            length = min(gap, 7.5)
            spans = (
                [(g0, g0 + length)]
                if gap <= 2 * cfg.gap_parallel_min
                else [
                    (g0, g0 + length),
                    (g1 - length, g1),
                ]
            )
            depth = 2.4
            for s0, s1 in spans:
                c = q + u * (s0 + s1) / 2 + n_side * (front + depth / 2)
                out.append(
                    SpotCandidate(
                        "",
                        Pose(float(c[0]), float(c[1]), math.atan2(float(u[1]), float(u[0]))),
                        s1 - s0,
                        depth,
                        SpotLayout.PARALLEL,
                        "gap",
                    )
                )
            return out
        if gap < cfg.gap_min_width:
            return out
        if gap <= cfg.gap_max_width:
            spans = [(g0, g1)]
        else:
            w = cfg.gap_stall_width
            spans = [(g0, g0 + w), (g1 - w, g1)]
        length = cfg.gap_stall_length
        for s0, s1 in spans:
            c = q + u * (s0 + s1) / 2 + n_side * (front + length / 2)
            out.append(
                SpotCandidate(
                    "",
                    Pose(float(c[0]), float(c[1]), heading_in),
                    length,
                    s1 - s0,
                    SpotLayout.PERPENDICULAR,
                    "gap",
                )
            )
        return out

    def _stabilise(self, cands: list[SpotCandidate]) -> list[SpotCandidate]:
        """Give line-less candidates stable ids and smoothed poses across frames.

        Candidates are remembered after they leave the detection window (the grid remembers
        their surroundings too); occupancy is re-checked every frame in :meth:`_assess`.
        """
        for c in cands:
            match = None
            for gid, old in self._gap_registry.items():
                if old.pose.distance_to(c.pose) <= self.cfg.track_merge_distance:
                    match = gid
                    break
            if match is None:
                c.id = f"G{self._next_gap}"
                self._next_gap += 1
            else:
                old_pose = self._gap_registry[match].pose
                k = 0.3
                c.pose = Pose(
                    old_pose.x + k * (c.pose.x - old_pose.x),
                    old_pose.y + k * (c.pose.y - old_pose.y),
                    c.pose.theta,
                )
                c.id = match
            self._gap_registry[c.id] = c
        return list(self._gap_registry.values())

    # -- occupancy, coverage, inspection ---------------------------------------------------
    def _assess(self, c: SpotCandidate, perc: PerceptionOutput) -> None:
        cfg = self.cfg
        grid = perc.grid
        inner = rect_polygon(
            c.pose,
            max(0.5, c.length - 2 * cfg.stall_shrink),
            max(0.5, c.width - 2 * cfg.stall_shrink),
        )
        occ_frac, obs_frac = _coverage(grid, inner)
        c.occupied = occ_frac > cfg.occupied_fraction
        c.observed_fraction = obs_frac
        # A small confirmed object well inside the stall (a cart or a person) also blocks it.
        core = rect_polygon(c.pose, max(0.5, c.length - 1.5), max(0.5, c.width - 1.2))
        for tr in perc.confirmed:
            if tr.extent <= 2.5 and points_in_convex_polygon(tr.position[None, :], core)[0]:
                c.occupied = True
        if c.layout is SpotLayout.PERPENDICULAR and not c.occupied:
            c.pull_through = _far_end_open(grid, c)
        if perc.ego.distance_to(c.pose) <= cfg.inspection_range:
            self._inspection[c.id] = self._inspection.get(c.id, 0) + 1
        c.inspected = self._inspection.get(c.id, 0) >= cfg.inspection_frames


def _coverage(grid: OccupancyGrid, poly: FloatArray) -> tuple[float, float]:
    """Fractions of occupied and observed cells inside a polygon."""
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    rlo, clo = grid.to_cells(lo)
    rhi, chi = grid.to_cells(hi)
    r0, c0 = max(int(rlo), 0), max(int(clo), 0)
    r1, c1 = min(int(rhi) + 1, grid.ny), min(int(chi) + 1, grid.nx)
    if r1 <= r0 or c1 <= c0:
        return 0.0, 0.0
    rr, cc = np.mgrid[r0:r1, c0:c1]
    centers = grid.cell_centers(rr.ravel(), cc.ravel())
    inside = points_in_convex_polygon(centers, poly)
    if not inside.any():
        return 0.0, 0.0
    rows, cols = rr.ravel()[inside], cc.ravel()[inside]
    occ = grid.occupied()[rows, cols]
    obs = grid.observed[rows, cols]
    return float(occ.mean()), float(obs.mean())


def _far_end_open(grid: OccupancyGrid, c: SpotCandidate) -> bool:
    """True if the far end of a perpendicular stall is observed, free drivable space."""
    beyond = c.pose.compose(Pose(c.length / 2 + 2.0, 0.0, 0.0))
    strip = rect_polygon(beyond, 3.0, max(0.5, c.width - 0.6))
    occ, obs = _coverage(grid, strip)
    return occ == 0.0 and obs > 0.6
