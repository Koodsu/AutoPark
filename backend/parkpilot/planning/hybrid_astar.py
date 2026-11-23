"""Hybrid A* (Dolgov et al., 2008) with Reeds-Shepp analytic expansion.

* Continuous-state nodes on a discretised (x, y, heading) closed set.
* Motion primitives are exact constant-curvature arcs, forward and reverse.
* Costs: distance (reverse weighted), **gear changes** (C1), steering effort and changes,
  and a **clearance** term from the distance field that keeps the car away from obstacles
  in open space without forbidding tight final maneuvers.
* Heuristic: max(obstacle-aware 2D Dijkstra distance, Reeds-Shepp length near the goal),
  inflated by ``heuristic_weight`` (weighted A*).
* Analytic expansion: every few expansions, Reeds-Shepp paths to the exact goal are tried in
  order of *planner* cost (gear changes included) and the first collision-free one ends the
  search. This is what solves parallel parking (C9) and tight stalls in few expansions.

Deterministic: the search budget is an expansion count, never wall-clock time.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from parkpilot.core.config import PlannerConfig
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.reeds_shepp import RSPath, paths_between, sample
from parkpilot.vehicle.model import VehicleModel


@dataclass(slots=True)
class Path:
    """A planned path: poses ``(N, 3)`` and travel direction per pose (+1 / -1)."""

    poses: FloatArray
    directions: FloatArray
    cost: float = 0.0

    @property
    def length(self) -> float:
        if len(self.poses) < 2:
            return 0.0
        return float(np.sum(np.hypot(*np.diff(self.poses[:, :2], axis=0).T)))

    @property
    def gear_changes(self) -> int:
        d = self.directions
        return int(np.sum(d[1:] != d[:-1])) if len(d) > 1 else 0

    def segments(self) -> list[Path]:
        """Split at cusps into single-direction pieces (cusp pose shared by both)."""
        if len(self.poses) == 0:
            return []
        cuts = [
            0,
            *(int(i) + 1 for i in np.flatnonzero(self.directions[1:] != self.directions[:-1])),
        ]
        cuts.append(len(self.poses))
        out = []
        for a, b in itertools.pairwise(cuts):
            lo = max(0, a - 1)
            d = np.full(b - lo, self.directions[a])
            out.append(Path(self.poses[lo:b], d))
        return [p for p in out if len(p.poses) >= 2]

    @property
    def end(self) -> Pose:
        x, y, th = self.poses[-1]
        return Pose(float(x), float(y), float(th))


@dataclass(slots=True)
class PlanResult:
    success: bool
    path: Path | None
    expansions: int
    reason: str = ""
    cost_terms: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class _Node:
    pose: tuple[float, float, float]
    g: float
    parent: int
    direction: int
    steer: float
    poses: FloatArray
    """Poses of the primitive that led here (excluding the parent pose)."""


class HybridAStar:
    def __init__(self, config: PlannerConfig, vehicle: VehicleModel) -> None:
        self.cfg = config
        self.vehicle = vehicle
        n = config.steer_samples
        steers = np.linspace(-vehicle.config.max_steer, vehicle.config.max_steer, n)
        self._prims: list[tuple[int, float, float]] = [
            (d, float(st), vehicle.curvature(float(st))) for d in (1, -1) for st in steers
        ]
        k = max(2, math.ceil(config.step / config.collision_step))
        self._s = np.linspace(config.step / k, config.step, k)

    # -- primitives ------------------------------------------------------------------------
    def _primitive_poses(self, pose: tuple[float, float, float]) -> FloatArray:
        """All primitives from ``pose``: array (P, K, 3)."""
        x0, y0, th0 = pose
        out = np.empty((len(self._prims), len(self._s), 3))
        for i, (d, _, kappa) in enumerate(self._prims):
            s = d * self._s
            if abs(kappa) < 1e-9:
                out[i, :, 0] = x0 + s * math.cos(th0)
                out[i, :, 1] = y0 + s * math.sin(th0)
                out[i, :, 2] = th0
            else:
                th = th0 + kappa * s
                out[i, :, 0] = x0 + (np.sin(th) - math.sin(th0)) / kappa
                out[i, :, 1] = y0 - (np.cos(th) - math.cos(th0)) / kappa
                out[i, :, 2] = th
        out[..., 2] = (out[..., 2] + math.pi) % (2 * math.pi) - math.pi
        return out

    def _key(self, x: float, y: float, th: float) -> tuple[int, int, int]:
        r = self.cfg.xy_resolution
        b = int(((th + math.pi) / (2 * math.pi)) * self.cfg.yaw_bins) % self.cfg.yaw_bins
        return math.floor(x / r), math.floor(y / r), b

    # -- heuristic -------------------------------------------------------------------------
    def _holonomic(
        self, checker: CollisionChecker, goal: Pose
    ) -> tuple[FloatArray, float, FloatArray]:
        """Obstacle-aware 2D Dijkstra cost-to-go on a coarse grid around the lot."""
        g = checker.grid
        res = max(self.cfg.xy_resolution, 0.5)
        factor = max(1, round(res / g.res))
        ny, nx = g.ny // factor, g.nx // factor
        occ = (
            checker.occ[: ny * factor, : nx * factor]
            .reshape(ny, factor, nx, factor)
            .any(axis=(1, 3))
        )
        dist = np.full((ny, nx), np.inf)
        cell = g.res * factor
        gc = int((goal.x - g.origin[0]) / cell), int((goal.y - g.origin[1]) / cell)
        if not (0 <= gc[1] < ny and 0 <= gc[0] < nx):
            return dist, cell, g.origin
        dist[gc[1], gc[0]] = 0.0
        heap = [(0.0, gc[1], gc[0])]
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)] + [
            (dr, dc, math.sqrt(2)) for dr in (-1, 1) for dc in (-1, 1)
        ]
        while heap:
            d, r, c = heapq.heappop(heap)
            if d > dist[r, c]:
                continue
            for dr, dc, w in moves:
                rr, cc = r + dr, c + dc
                if 0 <= rr < ny and 0 <= cc < nx and not occ[rr, cc]:
                    nd = d + w * cell
                    if nd < dist[rr, cc]:
                        dist[rr, cc] = nd
                        heapq.heappush(heap, (nd, rr, cc))
        return dist, cell, g.origin

    # -- analytic expansion ----------------------------------------------------------------
    def _rs_cost(self, path: RSPath, radius: float, start_dir: int) -> float:
        c = self.cfg
        cost = 0.0
        prev = start_dir
        for _, v in path.segments():
            d = 1 if v > 0 else -1
            ln = abs(v) * radius
            cost += ln * (c.reverse_multiplier if d < 0 else 1.0)
            if prev not in (0, d):
                cost += c.gear_change_cost
            prev = d
        return cost

    def _analytic(
        self, node: _Node, goal: Pose, checker: CollisionChecker
    ) -> tuple[FloatArray, FloatArray, float] | None:
        start = Pose(*node.pose)
        radius = self.vehicle.min_turn_radius * 1.05
        paths = paths_between(start, goal, radius)
        scored = sorted(paths, key=lambda p: self._rs_cost(p, radius, node.direction))
        for p in scored[: self.cfg.analytic_candidates]:
            poses, dirs = sample(start, p, radius, self.cfg.collision_step)
            if checker.path_free(poses[1:]):
                cost = self._rs_cost(p, radius, node.direction)
                clear = checker.clearance(poses)
                cost += self._clearance_cost(clear, p.length * radius / max(len(poses), 1))
                return poses[1:], dirs[1:], cost
        return None

    def _clearance_cost(self, clearance: FloatArray, ds: float) -> float:
        c = self.cfg
        deficit = np.clip(c.preferred_clearance - np.maximum(clearance, 0.0), 0.0, None)
        return float(c.clearance_cost * np.sum(deficit) * ds)

    # -- search ----------------------------------------------------------------------------
    def plan(
        self,
        start: Pose,
        goal: Pose,
        checker: CollisionChecker,
        final_straight: float = 0.0,
        final_direction: int = 1,
    ) -> PlanResult:
        """Plan from ``start`` to ``goal``.

        With ``final_straight > 0`` the path ends with a straight segment of that length in
        ``final_direction``: the car arrives with the wheels straight, which is what makes
        the final pose precise (a path ending mid-arc leaves the steering no distance to
        unwind).
        """
        if not checker.free(np.array([[goal.x, goal.y, goal.theta]]))[0]:
            return PlanResult(False, None, 0, "goal pose is in collision")
        if final_straight <= 0.0:
            return self._plan(start, goal, checker)
        d = 1 if final_direction >= 0 else -1
        pre = goal.compose(Pose(-d * final_straight, 0.0, 0.0))
        k = max(2, math.ceil(final_straight / self.cfg.collision_step))
        s = np.linspace(final_straight / k, final_straight, k) * d
        tail = np.stack(
            [
                pre.x + s * math.cos(pre.theta),
                pre.y + s * math.sin(pre.theta),
                np.full(k, pre.theta),
            ],
            axis=1,
        )
        if (
            not checker.path_free(tail)
            or not checker.free(np.array([[pre.x, pre.y, pre.theta]]))[0]
        ):
            return self._plan(start, goal, checker)
        res = self._plan(start, pre, checker)
        if not res.success or res.path is None:
            return self._plan(start, goal, checker)
        p = res.path
        extra = final_straight * (self.cfg.reverse_multiplier if d < 0 else 1.0)
        if len(p.directions) and p.directions[-1] != d:
            extra += self.cfg.gear_change_cost
        res.path = Path(
            np.vstack([p.poses, tail]),
            np.concatenate([p.directions, np.full(k, float(d))]),
            p.cost + extra,
        )
        return res

    def _plan(self, start: Pose, goal: Pose, checker: CollisionChecker) -> PlanResult:
        c = self.cfg
        h2d, cell, origin = self._holonomic(checker, goal)
        radius = self.vehicle.min_turn_radius * 1.05
        rs_cache: dict[tuple[int, int, int], float] = {}

        def heuristic(x: float, y: float, th: float) -> float:
            r = int((y - origin[1]) / cell)
            col = int((x - origin[0]) / cell)
            hol = h2d[r, col] if 0 <= r < h2d.shape[0] and 0 <= col < h2d.shape[1] else math.inf
            euclid = math.hypot(goal.x - x, goal.y - y)
            if not math.isfinite(hol):
                hol = euclid * 1.5
            h = max(hol, euclid)
            if euclid < 8.0:
                key = self._key(x, y, th)
                rs = rs_cache.get(key)
                if rs is None:
                    rs = min(p.length for p in paths_between(Pose(x, y, th), goal, radius)) * radius
                    rs_cache[key] = rs
                h = max(h, rs)
            return h * c.heuristic_weight

        root = _Node((start.x, start.y, start.theta), 0.0, -1, 0, 0.0, np.zeros((0, 3)))
        nodes = [root]
        best_g: dict[tuple[int, int, int], float] = {}
        heap: list[tuple[float, int]] = [(heuristic(*root.pose), 0)]
        expansions = 0
        ds = c.step / len(self._s)
        while heap and expansions < c.max_expansions:
            _, idx = heapq.heappop(heap)
            node = nodes[idx]
            key = self._key(*node.pose)
            if best_g.get(key, math.inf) < node.g - 1e-9:
                continue
            expansions += 1
            dist_goal = math.hypot(goal.x - node.pose[0], goal.y - node.pose[1])
            if dist_goal <= c.analytic_range and (
                expansions % c.analytic_every == 1 or dist_goal < 3
            ):
                shot = self._analytic(node, goal, checker)
                if shot is not None:
                    poses, dirs, cost = shot
                    return self._finish(nodes, idx, poses, dirs, node.g + cost, expansions)
            prims = self._primitive_poses(node.pose)
            flat = prims.reshape(-1, 3)
            free = checker.free(flat).reshape(prims.shape[0], prims.shape[1]).all(axis=1)
            clear = checker.clearance(flat).reshape(prims.shape[0], prims.shape[1])
            for i, (d, steer, _) in enumerate(self._prims):
                if not free[i]:
                    continue
                end = prims[i, -1]
                g = node.g + c.step * (c.reverse_multiplier if d < 0 else 1.0)
                if node.direction not in (0, d):
                    g += c.gear_change_cost
                g += c.steer_cost * abs(steer) / self.vehicle.config.max_steer * c.step
                g += c.steer_change_cost * abs(steer - node.steer) / self.vehicle.config.max_steer
                g += self._clearance_cost(clear[i], ds)
                k = self._key(float(end[0]), float(end[1]), float(end[2]))
                if g >= best_g.get(k, math.inf) - 1e-9:
                    continue
                best_g[k] = g
                nodes.append(
                    _Node((float(end[0]), float(end[1]), float(end[2])), g, idx, d, steer, prims[i])
                )
                h = heuristic(float(end[0]), float(end[1]), float(end[2]))
                heapq.heappush(heap, (g + h, len(nodes) - 1))
        reason = "search budget exhausted" if heap else "no collision-free path"
        return PlanResult(False, None, expansions, reason)

    def _finish(
        self,
        nodes: list[_Node],
        idx: int,
        tail_poses: FloatArray,
        tail_dirs: FloatArray,
        cost: float,
        expansions: int,
    ) -> PlanResult:
        chain = []
        i = idx
        while i >= 0:
            chain.append(nodes[i])
            i = nodes[i].parent
        chain.reverse()
        root = chain[0]
        poses = [np.array([root.pose])]
        dirs: list[FloatArray] = []
        for n in chain[1:]:
            poses.append(n.poses)
            dirs.append(np.full(len(n.poses), float(n.direction)))
        poses.append(tail_poses)
        dirs.append(tail_dirs)
        all_dirs = np.concatenate(dirs) if dirs else np.zeros(0)
        first_dir = all_dirs[0] if len(all_dirs) else 1.0
        directions = np.concatenate([[first_dir], all_dirs])
        path = Path(np.concatenate(poses), directions, cost)
        return PlanResult(True, path, expansions, "ok")
