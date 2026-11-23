import math

import numpy as np
import pytest

from parkpilot.core.geometry import Pose, points_in_convex_polygon, polygons_intersect
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
    FeatureKind,
    ObstacleKind,
    SpotLayout,
    SpotType,
    World,
    ZoneType,
    boundary_walls,
    circle_polygon,
)


def test_perpendicular_row_shares_lines() -> None:
    spots, lines = build_row(RowSpec("A", Pose(0, 0, math.pi / 2), 4), {1: SpotType.ACCESSIBLE})
    assert len(spots) == 4 and len(lines) == 5
    assert spots[1].spot_type is SpotType.ACCESSIBLE
    assert spots[0].line_ids[1] == spots[1].line_ids[0]
    # Stalls step to the right of the heading by the stall width.
    assert math.isclose(spots[1].pose.x - spots[0].pose.x, 2.8)
    # Each line lies on the boundary between neighbours.
    a, b = np.array(lines[1].a), np.array(lines[1].b)
    assert math.isclose(float(np.linalg.norm(b - a)), 5.4)
    assert math.isclose(float((a[0] + b[0]) / 2), 1.4)


def test_parallel_row() -> None:
    spots, lines = build_row(
        RowSpec(
            "P", Pose(0, 0, 0), 3, stall_width=2.5, stall_length=7.0, layout=SpotLayout.PARALLEL
        )
    )
    assert math.isclose(spots[2].pose.x, 14.0)
    assert math.isclose(lines[0].a[0], -3.5) and math.isclose(lines[0].b[0], -3.5)


def test_parked_vehicle_inside_spot_and_occupancy() -> None:
    spots, lines = build_row(RowSpec("A", Pose(0, 0, math.pi / 2), 3))
    car = parked_vehicle("c0", spots[0])
    assert points_in_convex_polygon(car.polygon, spots[0].polygon()).all()
    crooked = parked_vehicle("c2", spots[2], lateral_offset=0.6, yaw_offset=0.1)
    assert not points_in_convex_polygon(crooked.polygon, spots[2].polygon()).all()
    world = World((-10, -10, 10, 10), (0, 9), "lot", spots, lines, [car, crooked])
    assert world.occupied_spot_ids() == {"A0", "A2"}
    assert world.spot("A1").id == "A1"
    with pytest.raises(KeyError):
        world.spot("nope")
    with pytest.raises(KeyError):
        world.charger("nope")
    truck = parked_vehicle("t", spots[1], kind=ObstacleKind.PARKED_TRUCK)
    assert truck.height > car.height


def test_features_and_charger() -> None:
    spots, _ = build_row(
        RowSpec("A", Pose(0, 0, 0), 3), {0: SpotType.ACCESSIBLE, 1: SpotType.EV_CHARGING}
    )
    kinds = {f.kind for f in spot_features(spots[0])}
    assert kinds == {FeatureKind.SIGN, FeatureKind.PAINT_SYMBOL}
    assert spot_features(spots[2]) == []
    ch = charger_for_spot("ch", spots[1], side=1)
    assert ch.position[0] > spots[1].pose.x + spots[1].length / 2
    post = charger_obstacle(ch)
    assert post.kind is ObstacleKind.CHARGER and not post.is_low
    world = World((-10, -10, 10, 10), (0, 9), "lot", spots, chargers=[ch])
    assert world.charger("ch") is ch
    par, _ = build_row(
        RowSpec("P", Pose(0, 0, 0), 1, layout=SpotLayout.PARALLEL), {0: SpotType.RESERVED}
    )
    assert spot_features(par[0])[0].label == "reserved"


def test_zones_and_walls() -> None:
    z, feats = zone("f", ZoneType.FIRE_LANE, Pose(0, 0, 0), 10, 3)
    assert z.zone_type is ZoneType.FIRE_LANE
    assert {f.kind for f in feats} == {FeatureKind.CURB_PAINT, FeatureKind.SIGN}
    _, ramp = zone("r", ZoneType.SIDEWALK_RAMP, Pose(0, 0, 0), 2, 2)
    assert [f.kind for f in ramp] == [FeatureKind.CURB_RAMP]
    walls = boundary_walls((0, 0, 10, 5))
    assert len(walls) == 4
    inner = circle_polygon((5, 2.5), 1.0)
    assert not any(polygons_intersect(w.polygon, inner) for w in walls)
