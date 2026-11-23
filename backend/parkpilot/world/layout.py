"""Builders that turn compact layout descriptions into ground-truth world objects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from parkpilot.core.geometry import Pose, rect_polygon
from parkpilot.world.model import (
    Charger,
    Feature,
    FeatureKind,
    Obstacle,
    ObstacleKind,
    PaintedLine,
    Spot,
    SpotLayout,
    SpotType,
    Zone,
    ZoneType,
)

SIGNED_TYPES = frozenset(
    {
        SpotType.ACCESSIBLE,
        SpotType.EV_CHARGING,
        SpotType.EV_ONLY,
        SpotType.EXPECTANT_MOTHER,
        SpotType.RESERVED,
    }
)
PAINTED_TYPES = frozenset({SpotType.ACCESSIBLE, SpotType.MOTORCYCLE, SpotType.COMPACT})


@dataclass(frozen=True, slots=True)
class RowSpec:
    """A row of adjacent stalls sharing separator lines."""

    prefix: str
    first_center: Pose
    """Centre of the first stall; heading points into the stall (or along the curb)."""
    count: int
    stall_width: float = 2.8
    stall_length: float = 5.4
    layout: SpotLayout = SpotLayout.PERPENDICULAR
    lines_visible: bool = True
    pull_through: bool = False


def build_row(
    spec: RowSpec, types: dict[int, SpotType] | None = None
) -> tuple[list[Spot], list[PaintedLine]]:
    """Generate the stalls of a row and the ``count + 1`` painted lines between them."""
    types = types or {}
    h = np.array([math.cos(spec.first_center.theta), math.sin(spec.first_center.theta)])
    across = np.array([h[1], -h[0]])  # to the right when facing into the stall
    c0 = spec.first_center.xy
    spots: list[Spot] = []
    lines: list[PaintedLine] = []
    if spec.layout is SpotLayout.PERPENDICULAR:
        step = across * spec.stall_width
        half_sep, half_line = across * spec.stall_width / 2, h * spec.stall_length / 2
    else:
        step = h * spec.stall_length
        half_sep, half_line = h * spec.stall_length / 2, across * spec.stall_width / 2
    for i in range(spec.count + 1):
        mid = c0 + step * i - half_sep
        a, b = mid - half_line, mid + half_line
        lines.append(
            PaintedLine(
                f"{spec.prefix}L{i}",
                (float(a[0]), float(a[1])),
                (float(b[0]), float(b[1])),
                spec.lines_visible,
            )
        )
    for i in range(spec.count):
        c = c0 + step * i
        spots.append(
            Spot(
                id=f"{spec.prefix}{i}",
                pose=Pose(float(c[0]), float(c[1]), spec.first_center.theta),
                length=spec.stall_length,
                width=spec.stall_width,
                layout=spec.layout,
                spot_type=types.get(i, SpotType.STANDARD),
                line_ids=(f"{spec.prefix}L{i}", f"{spec.prefix}L{i + 1}"),
                pull_through=spec.pull_through,
            )
        )
    return spots, lines


def spot_features(spot: Spot) -> list[Feature]:
    """Signs and paint that mark a spot's type (what a driver would look for)."""
    feats: list[Feature] = []
    head = spot.pose.advanced(spot.length / 2 - 0.3)
    if spot.layout is SpotLayout.PARALLEL:
        head = spot.pose.compose(Pose(0.0, -spot.width / 2 + 0.2, 0.0))
    if spot.spot_type in SIGNED_TYPES:
        feats.append(Feature(FeatureKind.SIGN, (head.x, head.y), spot.spot_type.value))
    if spot.spot_type in PAINTED_TYPES:
        feats.append(
            Feature(FeatureKind.PAINT_SYMBOL, (spot.pose.x, spot.pose.y), spot.spot_type.value)
        )
    return feats


def parked_vehicle(
    obstacle_id: str,
    spot: Spot,
    *,
    kind: ObstacleKind = ObstacleKind.PARKED_CAR,
    lateral_offset: float = 0.0,
    yaw_offset: float = 0.0,
    length: float | None = None,
    width: float | None = None,
) -> Obstacle:
    """A vehicle parked in ``spot``, optionally off-centre or crooked (the C3 neighbor)."""
    truck = kind is ObstacleKind.PARKED_TRUCK
    if spot.layout is SpotLayout.PARALLEL:
        ln = length or (4.7 if not truck else 5.6)
        wd = width or (1.9 if not truck else 2.1)
    else:
        ln = length or (4.6 if not truck else 5.8)
        wd = width or (1.9 if not truck else 2.15)
    center = spot.pose.compose(Pose(0.0, -lateral_offset, yaw_offset))
    return Obstacle(obstacle_id, kind, rect_polygon(center, ln, wd), 2.0 if truck else 1.5)


def charger_for_spot(charger_id: str, spot: Spot, side: int, reach: float = 1.2) -> Charger:
    """Charger post at the head of a stall on ``side`` (+1 left, -1 right of the heading)."""
    p = spot.pose.compose(Pose(spot.length / 2 + 0.35, side * (spot.width / 2 - 0.35), 0.0))
    return Charger(charger_id, (p.x, p.y), reach)


def charger_obstacle(charger: Charger) -> Obstacle:
    return Obstacle(
        f"post_{charger.id}",
        ObstacleKind.CHARGER,
        rect_polygon(Pose(charger.position[0], charger.position[1], 0.0), 0.4, 0.4),
        1.4,
    )


def zone(
    zone_id: str, zone_type: ZoneType, center: Pose, length: float, width: float
) -> tuple[Zone, list[Feature]]:
    """A no-parking surface plus the visual cues that reveal it."""
    poly = rect_polygon(center, length, width)
    kind = {
        ZoneType.FIRE_LANE: FeatureKind.CURB_PAINT,
        ZoneType.LOADING_ZONE: FeatureKind.HATCHING,
        ZoneType.SIDEWALK_RAMP: FeatureKind.CURB_RAMP,
        ZoneType.DRIVEWAY: FeatureKind.DRIVEWAY_APRON,
    }[zone_type]
    feats = [Feature(kind, (center.x, center.y), zone_type.value)]
    if zone_type in (ZoneType.FIRE_LANE, ZoneType.LOADING_ZONE):
        sign = center.compose(Pose(0.0, -width / 2, 0.0))
        feats.append(Feature(FeatureKind.SIGN, (sign.x, sign.y), zone_type.value))
    return Zone(zone_id, zone_type, poly), feats
