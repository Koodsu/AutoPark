"""Destination-aware, explainable spot ranking with hard constraints (C4, C5, C15).

The drop-off pin is the scenario's entrance (the building door), never the lot centroid.
Every candidate gets a cost broken down into named terms and a list of plain-language
reasons, so the HMI can show *why* a spot was chosen or rejected. The same cost function is
used by the metrics oracle on ground truth, so "selection matches oracle" is apples to
apples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from parkpilot.core.config import (
    DirectionPreference,
    RankingConfig,
    Unlock,
    UserSettings,
    VehicleConfig,
)
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.spots.detect import SpotCandidate
from parkpilot.world.model import SpotLayout


@dataclass(frozen=True, slots=True)
class SideRoom:
    """Free lateral room from the stall centreline to the nearest neighbor on each side."""

    left: float
    right: float

    @property
    def free_width(self) -> float:
        return self.left + self.right

    def centered_offset(self) -> float:
        """Lateral shift (+left) that centres the car between the actual neighbors."""
        return (self.left - self.right) / 2.0


def side_room(
    points: FloatArray, pose: Pose, length: float, width: float, search: float
) -> SideRoom:
    """Measure neighbor room from obstacle points (occupied cells or sampled truth edges)."""
    cap = width / 2 + search
    if len(points) == 0:
        return SideRoom(cap, cap)
    local = pose.to_local(points)
    band = (np.abs(local[:, 0]) <= length / 2 - 0.3) & (np.abs(local[:, 1]) <= cap)
    ys = local[band, 1]
    left = float(ys[ys > 0].min()) if (ys > 0).any() else cap
    right = float(-ys[ys < 0].max()) if (ys < 0).any() else cap
    return SideRoom(min(left, cap), min(right, cap))


@dataclass(slots=True)
class CostInputs:
    walk_distance: float
    drive_distance: float
    room: SideRoom
    layout: SpotLayout
    pull_through: bool
    remembered: bool
    has_charger: bool
    reverse_in: bool


@dataclass(slots=True)
class RankedSpot:
    candidate: SpotCandidate
    cost: float
    eligible: bool
    terms: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    room: SideRoom = SideRoom(0.0, 0.0)


def spot_cost(
    inp: CostInputs,
    settings: UserSettings,
    cfg: RankingConfig,
    vehicle: VehicleConfig,
) -> tuple[float, dict[str, float], list[str]]:
    """Weighted cost (lower is better), its named terms, and human-readable reasons."""
    w = settings.weights
    terms: dict[str, float] = {}
    reasons: list[str] = []
    terms["walking"] = w.walking * inp.walk_distance
    reasons.append(f"{inp.walk_distance:.0f} m walk to entrance")
    terms["drive"] = w.drive_distance * inp.drive_distance
    if inp.layout is SpotLayout.PARALLEL:
        free_len_proxy = inp.room.free_width
        tight = 0.0 if free_len_proxy >= 2.6 else 1.0
        difficulty = cfg.parallel_difficulty + tight
    else:
        difficulty = max(0.0, (cfg.tight_width - inp.room.free_width) / 0.6)
        difficulty = min(difficulty, 2.0)
    if inp.reverse_in:
        difficulty += cfg.reverse_difficulty
    terms["maneuver"] = w.maneuver * difficulty
    if difficulty >= 1.0:
        reasons.append("tight maneuver")
    body_gap = (inp.room.free_width - vehicle.width) / 2.0
    if inp.layout is SpotLayout.PERPENDICULAR:
        missing = max(0.0, cfg.door_clearance - body_gap)
        terms["clearance"] = w.clearance * missing
        if missing > 0.05:
            reasons.append(f"only {max(body_gap, 0.0):.2f} m door clearance")
        else:
            reasons.append(f"{body_gap:.2f} m door clearance")
    if inp.pull_through:
        if settings.direction is DirectionPreference.PULL_THROUGH:
            terms["pull_through"] = -w.pull_through
            reasons.append("pull-through available")
        elif settings.direction is DirectionPreference.AUTO:
            terms["pull_through"] = -w.pull_through / 2
            reasons.append("pull-through available")
    if inp.remembered:
        terms["memory"] = -w.preference_memory
        reasons.append("your usual spot here")
    if inp.has_charger and Unlock.CHARGING in settings.unlocks:
        terms["charger"] = -w.charger
        reasons.append("charger stall")
    return float(sum(terms.values())), terms, reasons


def wants_reverse_in(cand: SpotCandidate, settings: UserSettings, vehicle: VehicleConfig) -> bool:
    """Entry direction for a perpendicular stall (C11, C15).

    Charger stalls choose the direction that puts the charge port toward the charger.
    Otherwise the user's preference wins; AUTO defaults to reverse-in (easier exit,
    fewer conflicts with pedestrians when leaving) unless a pull-through is available.
    """
    if cand.layout is SpotLayout.PARALLEL:
        return False
    if cand.charger is not None and Unlock.CHARGING in settings.unlocks:
        return charge_port_prefers_reverse(cand, vehicle)
    pref = settings.direction
    if pref is DirectionPreference.NOSE_IN:
        return False
    if pref is DirectionPreference.REVERSE_IN:
        return True
    if pref is DirectionPreference.PULL_THROUGH:
        return False
    return not cand.pull_through


def charge_port_prefers_reverse(cand: SpotCandidate, vehicle: VehicleConfig) -> bool:
    """Pick the entry direction (and the charger, if several) that puts the charge port
    closest to a charger. Updates ``cand.charger`` to the chosen one."""
    assert cand.charger is not None
    options = cand.chargers or [cand.charger]
    best: tuple[float, bool, tuple[float, float]] | None = None
    for ch in options:
        charger = np.array(ch)
        for reverse in (False, True):
            heading = cand.pose.theta + (math.pi if reverse else 0.0)
            center = Pose(cand.pose.x, cand.pose.y, heading)
            rear_axle = center.compose(
                Pose(-(vehicle.length / 2 - vehicle.rear_overhang), 0.0, 0.0)
            )
            port = rear_axle.to_world(np.array([[vehicle.charge_port_x, vehicle.charge_port_y]]))[0]
            d = float(np.hypot(*(port - charger)))
            if best is None or d < best[0]:
                best = (d, reverse, ch)
    assert best is not None
    cand.charger = best[2]
    return best[1]


def rank_spots(
    cands: list[SpotCandidate],
    *,
    ego: Pose,
    entrance: tuple[float, float],
    settings: UserSettings,
    cfg: RankingConfig,
    vehicle: VehicleConfig,
    obstacle_points: FloatArray,
    remembered: list[tuple[float, float]] | None = None,
) -> list[RankedSpot]:
    """Rank candidates. Ineligible ones are kept (with reasons) but sorted last."""
    remembered = remembered or []
    out: list[RankedSpot] = []
    ent = np.array(entrance)
    for c in cands:
        room = side_room(obstacle_points, c.pose, c.length, c.width, cfg.neighbor_search)
        reverse_in = wants_reverse_in(c, settings, vehicle)
        inp = CostInputs(
            walk_distance=float(np.hypot(*(c.pose.xy - ent))),
            drive_distance=ego.distance_to(c.pose),
            room=room,
            layout=c.layout,
            pull_through=c.pull_through,
            remembered=any(
                math.hypot(c.pose.x - x, c.pose.y - y) <= cfg.memory_radius for x, y in remembered
            ),
            has_charger=c.charger is not None,
            reverse_in=reverse_in,
        )
        cost, terms, reasons = spot_cost(inp, settings, cfg, vehicle)
        blockers: list[str] = []
        if c.occupied:
            blockers.append("occupied")
        if not c.legal:
            blockers.extend(r for r in c.reasons if r != "signage not yet inspected")
        if not c.inspected:
            blockers.append("signage not yet inspected")
        if c.observed_fraction < 0.35:
            blockers.append("not fully seen yet")
        if c.layout is SpotLayout.PERPENDICULAR:
            fits = room.free_width >= vehicle.mirror_width + 2 * cfg.min_side_margin
        else:
            fits = c.length >= vehicle.length + 1.0
        if not fits:
            blockers.append("too narrow for the car")
        elif c.legal and c.reasons:
            reasons.extend(r for r in c.reasons if r != "signage not yet inspected")
        out.append(
            RankedSpot(
                candidate=c,
                cost=cost,
                eligible=not blockers,
                terms=terms,
                reasons=blockers + reasons if blockers else reasons,
                room=room,
            )
        )
    out.sort(key=lambda r: (not r.eligible, r.cost))
    return out
