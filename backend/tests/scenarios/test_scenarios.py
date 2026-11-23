import itertools

import pytest

from parkpilot.core.config import VehicleConfig
from parkpilot.core.geometry import polygons_intersect
from parkpilot.scenarios.loader import (
    list_scenario_ids,
    list_scenarios,
    load_scenario,
    parse_scenario,
)
from parkpilot.scenarios.schema import agent_specs, build_world
from parkpilot.vehicle.model import VehicleModel
from parkpilot.world.model import ObstacleKind

IDS = list_scenario_ids()


def test_scenarios_discovered() -> None:
    assert len(IDS) >= 6
    assert len(list_scenarios()) == len(IDS)


@pytest.mark.parametrize("sid", IDS)
def test_scenario_builds_consistently(sid: str) -> None:
    sc = load_scenario(sid)
    world = build_world(sc)
    assert world.location_id == sc.location_id
    xmin, ymin, xmax, ymax = sc.bounds
    for x, y in sc.search_route:
        assert xmin <= x <= xmax and ymin <= y <= ymax
    model = VehicleModel(VehicleConfig())
    ego = model.footprint(sc.start_pose())
    for ob in world.obstacles:
        assert not any(polygons_intersect(p, ob.polygon) for p in ego), ob.id
    parked = [
        o for o in world.obstacles if o.kind in (ObstacleKind.PARKED_CAR, ObstacleKind.PARKED_TRUCK)
    ]
    for a, b in itertools.combinations(parked, 2):
        assert not polygons_intersect(a.polygon, b.polygon), (a.id, b.id)
    # Building twice yields identical ground truth (determinism).
    again = build_world(sc)
    assert [s.id for s in again.spots] == [s.id for s in world.spots]
    assert len(world.occupied_spot_ids()) < len(world.spots) or sc.expect_handback


def test_unknown_scenario() -> None:
    with pytest.raises(KeyError):
        load_scenario("does_not_exist")


def test_parse_inline_with_round_obstacle_and_charger() -> None:
    sc = parse_scenario(
        """
id: tiny
title: t
complaints: [C11]
description: d
location_id: here
bounds: [0, 0, 20, 20]
entrance: [10, 19]
ego_start: [2, 10, 0]
search_route: [[3, 10], [18, 10]]
rows:
  - {prefix: A, first_center: [5, 15, 90], count: 2, chargers: {0: right}, types: {0: ev_charging}}
obstacles:
  - {kind: bollard, center: [10, 5], radius: 0.2, height: 0.8}
phantoms:
  - {position: [8, 8]}
"""
    )
    world = build_world(sc)
    assert world.spots[0].charger_id == "ch_A0"
    assert world.chargers[0].id == "ch_A0"
    assert any(o.kind is ObstacleKind.BOLLARD and o.is_low for o in world.obstacles)
    assert len(world.phantoms) == 1


def test_agent_specs_from_yaml() -> None:
    sc = parse_scenario(
        """
id: a
title: t
complaints: [C12]
description: d
location_id: here
bounds: [0, 0, 20, 20]
entrance: [10, 19]
ego_start: [2, 10, 0]
search_route: [[3, 10], [18, 10]]
agents:
  - {id: p1, kind: pedestrian, waypoints: [[5, 2], [5, 18]], speed: 1.4, start_time: 2}
  - {id: c1, kind: car, waypoints: [[8, 14], [8, 10], [18, 10]], reversing: true,
     trigger_distance: 8}
"""
    )
    specs = agent_specs(sc)
    assert [s.id for s in specs] == ["p1", "c1"]
    assert specs[1].reversing and specs[1].trigger_distance == 8
