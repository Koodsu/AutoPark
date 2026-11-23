"""Spot-type and surface classification with hard legality constraints (C5).

Evidence comes only from the semantic map (signs, painted symbols, curb paint, curb ramps,
driveway aprons, hatching, chargers). The rules are deliberately conservative: a single
sighting of a restricted cue near a stall is enough to restrict it, and a stall whose
surroundings have not been inspected up close is not yet eligible.
"""

from __future__ import annotations

from parkpilot.core.config import SpotDetectionConfig, Unlock
from parkpilot.core.geometry import points_polygon_distance
from parkpilot.perception.semantic import SemanticMap
from parkpilot.spots.detect import SpotCandidate
from parkpilot.world.model import FeatureKind, SpotType, ZoneType

REQUIRED_UNLOCK: dict[SpotType, Unlock | None] = {
    SpotType.STANDARD: None,
    SpotType.COMPACT: None,
    SpotType.ACCESSIBLE: Unlock.PLACARD,
    SpotType.EV_CHARGING: Unlock.CHARGING,
    SpotType.EV_ONLY: Unlock.CHARGING,
    SpotType.EXPECTANT_MOTHER: Unlock.EXPECTANT,
    SpotType.RESERVED: Unlock.PERMIT,
}
"""Motorcycle stalls are absent on purpose: never legal for a car."""

ZONE_CUES = {
    FeatureKind.CURB_RAMP: ZoneType.SIDEWALK_RAMP,
    FeatureKind.DRIVEWAY_APRON: ZoneType.DRIVEWAY,
    FeatureKind.CURB_PAINT: ZoneType.FIRE_LANE,
    FeatureKind.HATCHING: ZoneType.LOADING_ZONE,
}
_SPOT_LABELS = {t.value: t for t in SpotType}
_ZONE_LABELS = {z.value: z for z in ZoneType}


def classify(
    cand: SpotCandidate,
    semantic: SemanticMap,
    unlocks: frozenset[Unlock],
    cfg: SpotDetectionConfig,
) -> None:
    """Fill in type, zone, charger and legality of ``cand`` in place."""
    poly = cand.polygon()
    type_votes: dict[SpotType, int] = {}
    zone: ZoneType | None = None
    chargers: list[tuple[float, float]] = []
    for ev in semantic.features:
        d = float(points_polygon_distance(ev.position[None, :], poly)[0])
        if ev.kind is FeatureKind.CHARGER and d <= 1.5:
            local = cand.pose.to_local(ev.position[None, :])[0]
            if abs(local[1]) > cand.width / 2 + 0.3 or local[0] < -cand.length / 4:
                continue  # beyond the neighbouring stall, or at the aisle end
            chargers.append((float(ev.position[0]), float(ev.position[1])))
            type_votes[SpotType.EV_CHARGING] = type_votes.get(SpotType.EV_CHARGING, 0) + ev.count
            continue
        if d <= cfg.zone_radius:
            if ev.kind in ZONE_CUES:
                zone = ZONE_CUES[ev.kind]
            elif ev.kind is FeatureKind.SIGN and ev.label in _ZONE_LABELS:
                zone = _ZONE_LABELS[ev.label]
        if d <= cfg.feature_radius and ev.kind in (FeatureKind.SIGN, FeatureKind.PAINT_SYMBOL):
            st = _SPOT_LABELS.get(ev.label)
            if st is not None:
                type_votes[st] = type_votes.get(st, 0) + ev.count
    restricted = {t: n for t, n in type_votes.items() if t not in (SpotType.STANDARD,)}
    cand.spot_type = (
        max(restricted, key=lambda t: restricted[t]) if restricted else SpotType.STANDARD
    )
    cand.zone = zone
    cand.chargers = chargers
    if cand.charger not in chargers:
        cand.charger = chargers[0] if chargers else None
    cand.reasons = []
    legal = True
    if zone is not None:
        legal = False
        cand.reasons.append(f"no parking: {zone.value.replace('_', ' ')}")
    if cand.spot_type is SpotType.MOTORCYCLE:
        legal = False
        cand.reasons.append("motorcycle-only stall")
    else:
        need = REQUIRED_UNLOCK.get(cand.spot_type)
        if need is not None:
            if need in unlocks:
                cand.reasons.append(
                    f"{cand.spot_type.value.replace('_', ' ')} (unlocked: {need.value})"
                )
            else:
                legal = False
                cand.reasons.append(
                    f"{cand.spot_type.value.replace('_', ' ')} stall; requires {need.value}"
                )
    if not cand.inspected:
        cand.reasons.append("signage not yet inspected")
    cand.legal = legal
