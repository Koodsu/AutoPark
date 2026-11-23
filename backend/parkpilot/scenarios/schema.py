"""YAML scenario format (pydantic models) and the builder that turns it into a :class:`World`."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from parkpilot.core.config import UserSettings
from parkpilot.core.geometry import Pose
from parkpilot.world.agents import AgentSpec
from parkpilot.world.layout import (
    RowSpec,
    build_row,
    charger_for_spot,
    charger_obstacle,
    parked_vehicle,
    spot_features,
    zone,
)
from parkpilot.world.model import (
    Feature,
    FeatureKind,
    Obstacle,
    ObstacleKind,
    Phantom,
    Spot,
    SpotLayout,
    SpotType,
    World,
    ZoneType,
    boundary_walls,
    box_obstacle,
    circle_polygon,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RowDef(_Model):
    prefix: str
    first_center: tuple[float, float, float]
    """x, y, heading in degrees (heading points into the stall, or along the curb)."""
    count: int = Field(ge=1)
    stall_width: float = 2.8
    stall_length: float = 5.4
    layout: SpotLayout = SpotLayout.PERPENDICULAR
    lines_visible: bool = True
    pull_through: bool = False
    types: dict[int, SpotType] = Field(default_factory=dict)
    occupied: list[int] = Field(default_factory=list)
    trucks: list[int] = Field(default_factory=list)
    offsets: dict[int, tuple[float, float]] = Field(default_factory=dict)
    """Per-stall (lateral m, yaw deg) offsets of the parked vehicle (crooked neighbors)."""
    chargers: dict[int, Literal["left", "right"]] = Field(default_factory=dict)
    """Charger posts at the head of these stalls, on the given side of the stall heading."""
    vehicle_sizes: dict[int, tuple[float, float]] = Field(default_factory=dict)


class ObstacleDef(_Model):
    kind: ObstacleKind
    center: tuple[float, float]
    size: tuple[float, float] | None = None
    """Length and width of a box obstacle; omit and give ``radius`` for round ones."""
    radius: float | None = None
    heading: float = 0.0
    """Degrees."""
    height: float = 1.0


class ZoneDef(_Model):
    type: ZoneType
    center: tuple[float, float, float]
    size: tuple[float, float]


class FeatureDef(_Model):
    kind: FeatureKind
    position: tuple[float, float]
    label: str


class PhantomDef(_Model):
    position: tuple[float, float]
    radius: float = 1.0
    rate: float = 0.3


class ScriptedCommand(_Model):
    """A command injected at sim time ``t`` (used to reproduce overrides headlessly)."""

    t: float
    command: Literal["override_near"]
    position: tuple[float, float]


class AgentDef(_Model):
    """A dynamic agent following waypoints (pedestrian, cart, or a car backing out)."""

    id: str
    kind: Literal["pedestrian", "cart", "car"]
    waypoints: list[tuple[float, float]]
    speed: float = 1.3
    start_time: float = 0.0
    loop: bool = False
    reversing: bool = False
    """Car backs along the first leg (reverse lights on: the "backing out" intent cue)."""
    trigger_distance: float | None = None
    """If set, the agent starts once the ego is within this distance (m) of its start."""


class Scenario(_Model):
    id: str
    title: str
    complaints: list[str]
    description: str
    location_id: str
    bounds: tuple[float, float, float, float]
    entrance: tuple[float, float]
    lighting: float = Field(default=1.0, ge=0.0, le=1.0)
    surface: str = "asphalt"
    ego_start: tuple[float, float, float]
    """x, y, heading in degrees (rear-axle reference)."""
    search_route: list[tuple[float, float]]
    rows: list[RowDef] = Field(default_factory=list)
    obstacles: list[ObstacleDef] = Field(default_factory=list)
    zones: list[ZoneDef] = Field(default_factory=list)
    features: list[FeatureDef] = Field(default_factory=list)
    phantoms: list[PhantomDef] = Field(default_factory=list)
    agents: list[AgentDef] = Field(default_factory=list)
    scripted: list[ScriptedCommand] = Field(default_factory=list)
    user: UserSettings = UserSettings()
    max_time: float = 120.0
    expect_handback: bool = False
    """True when the correct outcome is a safe handback (e.g. a full lot)."""
    notes: str = ""
    """Design notes, including any difficulty reductions and why."""

    def start_pose(self) -> Pose:
        x, y, h = self.ego_start
        return Pose(x, y, math.radians(h))


def _pose(v: tuple[float, float, float]) -> Pose:
    return Pose(v[0], v[1], math.radians(v[2]))


def build_world(sc: Scenario) -> World:
    """Instantiate ground truth for a scenario. Pure: same input, same world."""
    world = World(
        bounds=sc.bounds,
        entrance=sc.entrance,
        location_id=sc.location_id,
        lighting=sc.lighting,
        surface=sc.surface,
    )
    world.obstacles.extend(boundary_walls(sc.bounds))
    for row in sc.rows:
        spec = RowSpec(
            prefix=row.prefix,
            first_center=_pose(row.first_center),
            count=row.count,
            stall_width=row.stall_width,
            stall_length=row.stall_length,
            layout=row.layout,
            lines_visible=row.lines_visible,
            pull_through=row.pull_through,
        )
        spots, lines = build_row(spec, dict(row.types))
        for idx, side in row.chargers.items():
            ch = charger_for_spot(f"ch_{row.prefix}{idx}", spots[idx], 1 if side == "left" else -1)
            world.chargers.append(ch)
            world.obstacles.append(charger_obstacle(ch))
            world.features.append(Feature(FeatureKind.CHARGER, ch.position, "charger"))
            spots[idx] = _with_charger(spots[idx], ch.id)
        for i, s in enumerate(spots):
            world.features.extend(spot_features(s))
            if i in row.occupied or i in row.trucks:
                lat, yaw = row.offsets.get(i, (0.0, 0.0))
                size = row.vehicle_sizes.get(i)
                world.obstacles.append(
                    parked_vehicle(
                        f"veh_{s.id}",
                        s,
                        kind=ObstacleKind.PARKED_TRUCK
                        if i in row.trucks
                        else ObstacleKind.PARKED_CAR,
                        lateral_offset=lat,
                        yaw_offset=math.radians(yaw),
                        length=size[0] if size else None,
                        width=size[1] if size else None,
                    )
                )
        world.spots.extend(spots)
        world.lines.extend(lines)
    for i, ob in enumerate(sc.obstacles):
        world.obstacles.append(_obstacle(f"obs{i}_{ob.kind.value}", ob))
    for i, zd in enumerate(sc.zones):
        z, feats = zone(f"zone{i}_{zd.type.value}", zd.type, _pose(zd.center), *zd.size)
        world.zones.append(z)
        world.features.extend(feats)
    world.features.extend(Feature(f.kind, f.position, f.label) for f in sc.features)
    world.phantoms.extend(Phantom(p.position, p.radius, p.rate) for p in sc.phantoms)
    return world


def _with_charger(spot: Spot, charger_id: str) -> Spot:
    return replace(spot, charger_id=charger_id)


def _obstacle(oid: str, ob: ObstacleDef) -> Obstacle:
    if ob.size is not None:
        pose = Pose(ob.center[0], ob.center[1], math.radians(ob.heading))
        return box_obstacle(oid, ob.kind, pose, ob.size[0], ob.size[1], ob.height)
    radius = ob.radius if ob.radius is not None else 0.15
    return Obstacle(oid, ob.kind, circle_polygon(ob.center, radius), ob.height)


def agent_specs(sc: Scenario) -> list[AgentSpec]:
    """Runtime agent specs for a scenario's dynamic agents."""
    return [
        AgentSpec(
            id=a.id,
            kind=a.kind,
            waypoints=tuple(a.waypoints),
            speed=a.speed,
            start_time=a.start_time,
            loop=a.loop,
            reversing=a.reversing,
            trigger_distance=a.trigger_distance,
        )
        for a in sc.agents
    ]
