import math
from collections.abc import Callable

import numpy as np

from parkpilot.core.config import (
    DirectionPreference,
    ParkPilotConfig,
    RankingWeights,
    Unlock,
    UserSettings,
)
from parkpilot.core.geometry import Pose
from parkpilot.spots.classify import classify
from parkpilot.spots.detect import SpotCandidate
from parkpilot.spots.rank import (
    RankedSpot,
    SideRoom,
    charge_port_prefers_reverse,
    rank_spots,
    side_room,
    wants_reverse_in,
)
from parkpilot.world.model import SpotLayout
from tests.helpers import survey

CFG = ParkPilotConfig()


def _rank(
    sid: str, settings: UserSettings, remembered: list[tuple[float, float]] | None = None
) -> tuple[list[RankedSpot], Callable[[SpotCandidate], str]]:
    sc, world, out, cands, _ = survey(sid)
    for c in cands:
        classify(c, out.semantic, settings.unlocks, CFG.spots)
    ranked = rank_spots(
        cands,
        ego=out.ego,
        entrance=sc.entrance,
        settings=settings,
        cfg=CFG.ranking,
        vehicle=CFG.vehicle,
        obstacle_points=out.grid.occupied_points(),
        remembered=remembered,
    )

    def truth(c: SpotCandidate) -> str:
        return min(world.spots, key=lambda s: s.pose.distance_to(c.pose)).id

    return ranked, truth


def test_destination_ranking_prefers_roomy_spot_near_entrance() -> None:
    ranked, truth = _rank("destination_entrance", UserSettings())
    eligible = [r for r in ranked if r.eligible]
    assert truth(eligible[0].candidate) == "A15"
    between_trucks = next(r for r in ranked if truth(r.candidate) == "A12")
    assert between_trucks.terms["clearance"] > 0
    assert any("door clearance" in reason for reason in between_trucks.reasons)
    first_available = next(r for r in ranked if truth(r.candidate) == "A1")
    assert first_available.terms["walking"] > eligible[0].terms["walking"] * 2
    assert all(not r.eligible for r in ranked if r.candidate.occupied)


def test_weights_are_user_adjustable() -> None:
    # A user who only cares about walking accepts the tight spot between trucks.
    settings = UserSettings(weights=RankingWeights(clearance=0.0, maneuver=0.0, drive_distance=0.0))
    ranked, truth = _rank("destination_entrance", settings)
    assert truth(next(r for r in ranked if r.eligible).candidate) == "A12"


def test_memory_bonus_changes_choice() -> None:
    ranked, truth = _rank("destination_entrance", UserSettings(), remembered=[(10.8, 23.4)])
    best = next(r for r in ranked if r.eligible)
    assert truth(best.candidate) == "A1"
    assert "your usual spot here" in best.reasons


def test_side_room_and_centering() -> None:
    pose = Pose(0, 0, math.pi / 2)  # stall heading +y; left is -x
    pts = np.array([[-1.2, 0.0], [1.6, 1.0], [5.0, 0.0]])
    room = side_room(pts, pose, 5.4, 2.8, 2.0)
    assert math.isclose(room.left, 1.2) and math.isclose(room.right, 1.6)
    assert math.isclose(room.centered_offset(), -0.2)
    empty = side_room(np.zeros((0, 2)), pose, 5.4, 2.8, 2.0)
    assert empty == SideRoom(3.4, 3.4)


def test_direction_choice_and_charge_port() -> None:
    cand = SpotCandidate("c", Pose(0, 0, math.pi / 2), 5.4, 2.8, SpotLayout.PERPENDICULAR, "lines")
    veh = CFG.vehicle
    assert wants_reverse_in(cand, UserSettings(), veh)
    assert not wants_reverse_in(cand, UserSettings(direction=DirectionPreference.NOSE_IN), veh)
    assert wants_reverse_in(cand, UserSettings(direction=DirectionPreference.REVERSE_IN), veh)
    # Charger at the head of the stall, on its left: the rear-left port wants reverse-in
    # only if that brings the port to the charger; check consistency with geometry.
    cand.charger = (-1.05, 3.05)
    rev = charge_port_prefers_reverse(cand, veh)
    charging = UserSettings(unlocks=frozenset({Unlock.CHARGING}))
    assert wants_reverse_in(cand, charging, veh) == rev
    assert rev  # port is at the rear: backing in puts it at the head of the stall
    par = SpotCandidate("p", Pose(0, 0, 0), 7.0, 2.5, SpotLayout.PARALLEL, "lines")
    assert not wants_reverse_in(par, UserSettings(), veh)
