"""Reeds-Shepp curves: shortest paths for a car that can drive forward and in reverse.

Implements all 48 path types of Reeds & Shepp (1990) in the five families CSC, CCC, CCCC,
CCSC/CSCC and CCSCC, using the formulas and time-flip / reflection / backwards symmetries
popularised by OMPL. Unlike the usual "shortest only" API, :func:`all_paths` returns every
valid solution so the planner can try them in order of *its own* cost (which penalises gear
changes) and keep the first collision-free one.

Segment lengths are in units of the turning radius; their sign is the direction of travel
(negative means reverse).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.vehicle.model import arc_poses

PI = math.pi
HALF_PI = 0.5 * math.pi
ZERO = 10 * np.finfo(float).eps
_EPS_LEN = 1e-9


def mod2pi(x: float) -> float:
    """Wrap to [-pi, pi] (OMPL convention)."""
    v = math.fmod(x, 2 * PI)
    if v < -PI:
        v += 2 * PI
    elif v > PI:
        v -= 2 * PI
    return v


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _tau_omega(u: float, v: float, xi: float, eta: float, phi: float) -> tuple[float, float]:
    delta = mod2pi(u - v)
    a = math.sin(u) - math.sin(delta)
    b = math.cos(u) - math.cos(delta) - 1.0
    t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
    t2 = 2.0 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3.0
    tau = mod2pi(t1 + PI) if t2 < 0 else mod2pi(t1)
    omega = mod2pi(tau - u + v - phi)
    return tau, omega


# --- the eight base formulas (each returns None or (t, u, v)) ------------------------------
Sol = tuple[float, float, float] | None


def _lp_sp_lp(x: float, y: float, phi: float) -> Sol:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if t >= -ZERO:
        v = mod2pi(phi - t)
        if v >= -ZERO:
            return t, u, v
    return None


def _lp_sp_rp(x: float, y: float, phi: float) -> Sol:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 = u1 * u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        theta = math.atan2(2.0, u)
        t = mod2pi(t1 + theta)
        v = mod2pi(t - phi)
        if t >= -ZERO and v >= -ZERO:
            return t, u, v
    return None


def _lp_rm_l(x: float, y: float, phi: float) -> Sol:
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    u1, theta = _polar(xi, eta)
    if u1 <= 4.0:
        u = -2.0 * math.asin(0.25 * u1)
        t = mod2pi(theta + 0.5 * u + PI)
        v = mod2pi(phi - t + u)
        if t >= -ZERO and u <= ZERO:
            return t, u, v
    return None


def _lp_rup_lum_rm(x: float, y: float, phi: float) -> Sol:
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = 0.25 * (2.0 + math.hypot(xi, eta))
    if rho <= 1.0:
        u = math.acos(rho)
        t, v = _tau_omega(u, -u, xi, eta, phi)
        if t >= -ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rum_lum_rp(x: float, y: float, phi: float) -> Sol:
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho = (20.0 - xi * xi - eta * eta) / 16.0
    if 0.0 <= rho <= 1.0:
        u = -math.acos(rho)
        if u >= -HALF_PI:
            t, v = _tau_omega(u, u, xi, eta, phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


def _lp_rm_sm_lm(x: float, y: float, phi: float) -> Sol:
    xi, eta = x - math.sin(phi), y - 1.0 + math.cos(phi)
    rho, theta = _polar(xi, eta)
    if rho >= 2.0:
        r = math.sqrt(rho * rho - 4.0)
        u = 2.0 - r
        t = mod2pi(theta + math.atan2(r, -2.0))
        v = mod2pi(phi - HALF_PI - t)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rm_sm_rm(x: float, y: float, phi: float) -> Sol:
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, theta = _polar(-eta, xi)
    if rho >= 2.0:
        t = theta
        u = 2.0 - rho
        v = mod2pi(t + HALF_PI - phi)
        if t >= -ZERO and u <= ZERO and v <= ZERO:
            return t, u, v
    return None


def _lp_rm_s_lm_rp(x: float, y: float, phi: float) -> Sol:
    xi, eta = x + math.sin(phi), y - 1.0 - math.cos(phi)
    rho, _ = _polar(xi, eta)
    if rho >= 2.0:
        u = 4.0 - math.sqrt(rho * rho - 4.0)
        if u <= ZERO:
            t = mod2pi(math.atan2((4.0 - u) * xi - 2.0 * eta, -2.0 * xi + (u - 4.0) * eta))
            v = mod2pi(t - phi)
            if t >= -ZERO and v >= -ZERO:
                return t, u, v
    return None


@dataclass(frozen=True, slots=True)
class RSPath:
    """A Reeds-Shepp path: segment letters (L, R, S) and signed unit-radius lengths."""

    word: str
    lengths: tuple[float, ...]

    @property
    def length(self) -> float:
        return float(sum(abs(v) for v in self.lengths))

    def segments(self) -> list[tuple[str, float]]:
        return [(c, v) for c, v in zip(self.word, self.lengths, strict=True) if abs(v) > _EPS_LEN]

    @property
    def gear_changes(self) -> int:
        signs = [v > 0 for _, v in self.segments()]
        return sum(1 for a, b in itertools.pairwise(signs) if a != b)


def _flip(word: str) -> str:
    return word.translate(str.maketrans("LR", "RL"))


def _add(out: list[RSPath], word: str, lengths: tuple[float, ...]) -> None:
    out.append(RSPath(word, lengths))


def all_paths(x: float, y: float, phi: float) -> list[RSPath]:
    """All valid Reeds-Shepp paths to goal ``(x, y, phi)`` in the unit-radius start frame."""
    out: list[RSPath] = []
    xb = x * math.cos(phi) + y * math.sin(phi)
    yb = x * math.sin(phi) - y * math.cos(phi)

    # CSC
    for fn, word in ((_lp_sp_lp, "LSL"), (_lp_sp_rp, "LSR")):
        if (s := fn(x, y, phi)) is not None:
            _add(out, word, s)
        if (s := fn(-x, y, -phi)) is not None:
            _add(out, word, (-s[0], -s[1], -s[2]))
        if (s := fn(x, -y, -phi)) is not None:
            _add(out, _flip(word), s)
        if (s := fn(-x, -y, phi)) is not None:
            _add(out, _flip(word), (-s[0], -s[1], -s[2]))
    # CCC (forward and backward)
    if (s := _lp_rm_l(x, y, phi)) is not None:
        _add(out, "LRL", s)
    if (s := _lp_rm_l(-x, y, -phi)) is not None:
        _add(out, "LRL", (-s[0], -s[1], -s[2]))
    if (s := _lp_rm_l(x, -y, -phi)) is not None:
        _add(out, "RLR", s)
    if (s := _lp_rm_l(-x, -y, phi)) is not None:
        _add(out, "RLR", (-s[0], -s[1], -s[2]))
    if (s := _lp_rm_l(xb, yb, phi)) is not None:
        _add(out, "LRL", (s[2], s[1], s[0]))
    if (s := _lp_rm_l(-xb, yb, -phi)) is not None:
        _add(out, "LRL", (-s[2], -s[1], -s[0]))
    if (s := _lp_rm_l(xb, -yb, -phi)) is not None:
        _add(out, "RLR", (s[2], s[1], s[0]))
    if (s := _lp_rm_l(-xb, -yb, phi)) is not None:
        _add(out, "RLR", (-s[2], -s[1], -s[0]))
    # CCCC
    if (s := _lp_rup_lum_rm(x, y, phi)) is not None:
        _add(out, "LRLR", (s[0], s[1], -s[1], s[2]))
    if (s := _lp_rup_lum_rm(-x, y, -phi)) is not None:
        _add(out, "LRLR", (-s[0], -s[1], s[1], -s[2]))
    if (s := _lp_rup_lum_rm(x, -y, -phi)) is not None:
        _add(out, "RLRL", (s[0], s[1], -s[1], s[2]))
    if (s := _lp_rup_lum_rm(-x, -y, phi)) is not None:
        _add(out, "RLRL", (-s[0], -s[1], s[1], -s[2]))
    if (s := _lp_rum_lum_rp(x, y, phi)) is not None:
        _add(out, "LRLR", (s[0], s[1], s[1], s[2]))
    if (s := _lp_rum_lum_rp(-x, y, -phi)) is not None:
        _add(out, "LRLR", (-s[0], -s[1], -s[1], -s[2]))
    if (s := _lp_rum_lum_rp(x, -y, -phi)) is not None:
        _add(out, "RLRL", (s[0], s[1], s[1], s[2]))
    if (s := _lp_rum_lum_rp(-x, -y, phi)) is not None:
        _add(out, "RLRL", (-s[0], -s[1], -s[1], -s[2]))
    # CCSC and its backwards CSCC
    for fn, word in ((_lp_rm_sm_lm, "LRSL"), (_lp_rm_sm_rm, "LRSR")):
        if (s := fn(x, y, phi)) is not None:
            _add(out, word, (s[0], -HALF_PI, s[1], s[2]))
        if (s := fn(-x, y, -phi)) is not None:
            _add(out, word, (-s[0], HALF_PI, -s[1], -s[2]))
        if (s := fn(x, -y, -phi)) is not None:
            _add(out, _flip(word), (s[0], -HALF_PI, s[1], s[2]))
        if (s := fn(-x, -y, phi)) is not None:
            _add(out, _flip(word), (-s[0], HALF_PI, -s[1], -s[2]))
        back = word[::-1]
        if (s := fn(xb, yb, phi)) is not None:
            _add(out, back, (s[2], s[1], -HALF_PI, s[0]))
        if (s := fn(-xb, yb, -phi)) is not None:
            _add(out, back, (-s[2], -s[1], HALF_PI, -s[0]))
        if (s := fn(xb, -yb, -phi)) is not None:
            _add(out, _flip(back), (s[2], s[1], -HALF_PI, s[0]))
        if (s := fn(-xb, -yb, phi)) is not None:
            _add(out, _flip(back), (-s[2], -s[1], HALF_PI, -s[0]))
    # CCSCC
    if (s := _lp_rm_s_lm_rp(x, y, phi)) is not None:
        _add(out, "LRSLR", (s[0], -HALF_PI, s[1], -HALF_PI, s[2]))
    if (s := _lp_rm_s_lm_rp(-x, y, -phi)) is not None:
        _add(out, "LRSLR", (-s[0], HALF_PI, -s[1], HALF_PI, -s[2]))
    if (s := _lp_rm_s_lm_rp(x, -y, -phi)) is not None:
        _add(out, "RLSRL", (s[0], -HALF_PI, s[1], -HALF_PI, s[2]))
    if (s := _lp_rm_s_lm_rp(-x, -y, phi)) is not None:
        _add(out, "RLSRL", (-s[0], HALF_PI, -s[1], HALF_PI, -s[2]))
    return out


def paths_between(start: Pose, goal: Pose, radius: float) -> list[RSPath]:
    """All Reeds-Shepp paths from ``start`` to ``goal`` for a turning radius, shortest first."""
    rel = goal.relative_to(start)
    paths = all_paths(rel.x / radius, rel.y / radius, rel.theta)
    return sorted(paths, key=lambda p: p.length)


def shortest_length(start: Pose, goal: Pose, radius: float) -> float:
    """Length (m) of the shortest Reeds-Shepp path: an admissible planner heuristic."""
    paths = paths_between(start, goal, radius)
    return min(p.length for p in paths) * radius if paths else math.inf


def sample(start: Pose, path: RSPath, radius: float, step: float) -> tuple[FloatArray, FloatArray]:
    """Sample poses along ``path`` every ``step`` metres.

    Returns ``(poses (N, 3), directions (N,))`` where direction is +1 forward, -1 reverse for
    the motion arriving at each pose (the first pose takes its segment's direction). The
    final pose is exact.
    """
    poses = [np.array([[start.x, start.y, start.theta]])]
    dirs = []
    cur = start
    first = True
    for letter, unit_len in path.segments():
        dist = unit_len * radius
        kappa = {"L": 1.0 / radius, "R": -1.0 / radius, "S": 0.0}[letter]
        n = max(1, math.ceil(abs(dist) / step))
        s = np.linspace(0.0, dist, n + 1)[1:]
        seg = arc_poses(cur, kappa, s)
        d = 1.0 if dist > 0 else -1.0
        if first:
            dirs.append(d)
            first = False
        poses.append(seg)
        dirs.extend([d] * len(seg))
        cur = Pose(float(seg[-1, 0]), float(seg[-1, 1]), float(seg[-1, 2]))
    if first:
        dirs.append(1.0)
    return np.concatenate(poses), np.array(dirs)
