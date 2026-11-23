"""Ground-truth world model: lot, typed spots, painted lines, legality features, obstacles.

These objects are the simulator's truth. The autonomy stack never reads them directly; it
only sees what :mod:`parkpilot.perception` reports. Metrics and the debug layer use them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from parkpilot.core.geometry import FloatArray, Pose, points_in_convex_polygon, rect_polygon


class SpotType(StrEnum):
    STANDARD = "standard"
    ACCESSIBLE = "accessible"
    EV_CHARGING = "ev_charging"
    EV_ONLY = "ev_only"
    EXPECTANT_MOTHER = "expectant_mother"
    MOTORCYCLE = "motorcycle"
    RESERVED = "reserved"
    COMPACT = "compact"


class ZoneType(StrEnum):
    """Surfaces where stopping to park is never legal, regardless of user unlocks."""

    FIRE_LANE = "fire_lane"
    LOADING_ZONE = "loading_zone"
    SIDEWALK_RAMP = "sidewalk_ramp"
    DRIVEWAY = "driveway"


class SpotLayout(StrEnum):
    PERPENDICULAR = "perpendicular"
    PARALLEL = "parallel"


class FeatureKind(StrEnum):
    """Visual cues a camera can pick up and the spot classifier can reason about."""

    SIGN = "sign"
    PAINT_SYMBOL = "paint_symbol"
    CURB_PAINT = "curb_paint"
    CURB_RAMP = "curb_ramp"
    DRIVEWAY_APRON = "driveway_apron"
    HATCHING = "hatching"
    CHARGER = "charger"


class ObstacleKind(StrEnum):
    WALL = "wall"
    CURB = "curb"
    POST = "post"
    BOLLARD = "bollard"
    PILLAR = "pillar"
    PARKED_CAR = "parked_car"
    PARKED_TRUCK = "parked_truck"
    CHARGER = "charger"
    SNOW_PILE = "snow_pile"
    CART_CORRAL = "cart_corral"


LOW_OBJECT_KINDS = frozenset({ObstacleKind.CURB, ObstacleKind.BOLLARD, ObstacleKind.SNOW_PILE})
"""Obstacle kinds below camera-friendly height; hardest to see up close (the C7 pattern)."""


@dataclass(frozen=True, slots=True)
class PaintedLine:
    """A painted stall line segment. ``visible`` is false when worn, snow-covered or absent."""

    id: str
    a: tuple[float, float]
    b: tuple[float, float]
    visible: bool = True


@dataclass(frozen=True, slots=True)
class Charger:
    id: str
    position: tuple[float, float]
    cable_reach: float
    """Distance from the charger post the cable can comfortably reach (m)."""


@dataclass(frozen=True, slots=True)
class Spot:
    """A parking space.

    ``pose`` is the spot centre. Its heading points *into* the stall from the aisle for
    perpendicular spots (the direction a nose-in car faces) and along the curb for parallel
    spots. ``length`` is measured along the heading, ``width`` across it.
    """

    id: str
    pose: Pose
    length: float
    width: float
    layout: SpotLayout = SpotLayout.PERPENDICULAR
    spot_type: SpotType = SpotType.STANDARD
    line_ids: tuple[str, ...] = ()
    charger_id: str | None = None
    pull_through: bool = False
    """True if the far end opens onto another aisle (drive-through possible)."""

    def polygon(self) -> FloatArray:
        return rect_polygon(self.pose, self.length, self.width)


@dataclass(frozen=True, slots=True)
class Feature:
    """A visual cue in the world (sign, painted symbol, curb paint, ramp, driveway apron)."""

    kind: FeatureKind
    position: tuple[float, float]
    label: str
    """Semantic label, e.g. a :class:`SpotType` or :class:`ZoneType` value."""


@dataclass(frozen=True, slots=True)
class Obstacle:
    id: str
    kind: ObstacleKind
    polygon: FloatArray
    height: float
    """Height above ground (m); low objects are harder for cameras near the bumper."""

    @property
    def is_low(self) -> bool:
        return self.kind in LOW_OBJECT_KINDS or self.height < 0.5


@dataclass(frozen=True, slots=True)
class Zone:
    """A no-parking surface (fire lane, loading zone, sidewalk ramp, driveway)."""

    id: str
    zone_type: ZoneType
    polygon: FloatArray


@dataclass(frozen=True, slots=True)
class Phantom:
    """A spot where shadows or puddles cause spurious camera detections (C8)."""

    position: tuple[float, float]
    radius: float
    rate: float
    """Probability per perception frame of a false detection here."""


@dataclass(slots=True)
class World:
    """Static ground truth of a scenario (dynamic agents live in :mod:`world.agents`)."""

    bounds: tuple[float, float, float, float]
    """xmin, ymin, xmax, ymax of the drivable lot (m)."""
    entrance: tuple[float, float]
    """Pedestrian destination (the store or building door), the 'drop-off pin'."""
    location_id: str
    spots: list[Spot] = field(default_factory=list)
    lines: list[PaintedLine] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    zones: list[Zone] = field(default_factory=list)
    chargers: list[Charger] = field(default_factory=list)
    phantoms: list[Phantom] = field(default_factory=list)
    lighting: float = 1.0
    """1.0 is full daylight, 0.0 is an unlit lot at night."""
    surface: str = "asphalt"

    def spot(self, spot_id: str) -> Spot:
        for s in self.spots:
            if s.id == spot_id:
                return s
        raise KeyError(spot_id)

    def charger(self, charger_id: str) -> Charger:
        for c in self.chargers:
            if c.id == charger_id:
                return c
        raise KeyError(charger_id)

    def occupied_spot_ids(self) -> set[str]:
        """Spots whose centre lies inside a parked vehicle."""
        parked = [
            o
            for o in self.obstacles
            if o.kind in (ObstacleKind.PARKED_CAR, ObstacleKind.PARKED_TRUCK)
        ]
        out: set[str] = set()
        for s in self.spots:
            pt = s.pose.xy[None, :]
            if any(points_in_convex_polygon(pt, o.polygon)[0] for o in parked):
                out.add(s.id)
        return out


def boundary_walls(
    bounds: tuple[float, float, float, float], thickness: float = 0.3
) -> list[Obstacle]:
    """Four wall obstacles enclosing ``bounds`` from the outside."""
    xmin, ymin, xmax, ymax = bounds
    w, h = xmax - xmin, ymax - ymin
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    t = thickness
    specs = [
        (Pose(cx, ymin - t / 2, 0.0), w + 2 * t, t),
        (Pose(cx, ymax + t / 2, 0.0), w + 2 * t, t),
        (Pose(xmin - t / 2, cy, 0.0), t, h),
        (Pose(xmax + t / 2, cy, 0.0), t, h),
    ]
    return [
        Obstacle(f"wall{i}", ObstacleKind.WALL, rect_polygon(p, ln, wd), 2.5)
        for i, (p, ln, wd) in enumerate(specs)
    ]


def box_obstacle(
    obstacle_id: str, kind: ObstacleKind, pose: Pose, length: float, width: float, height: float
) -> Obstacle:
    return Obstacle(obstacle_id, kind, rect_polygon(pose, length, width), height)


def circle_polygon(center: tuple[float, float], radius: float, sides: int = 8) -> FloatArray:
    ang = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    return np.stack([center[0] + radius * np.cos(ang), center[1] + radius * np.sin(ang)], axis=1)
