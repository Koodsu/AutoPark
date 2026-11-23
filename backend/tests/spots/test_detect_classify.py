import numpy as np

from parkpilot.core.config import SpotDetectionConfig, Unlock
from parkpilot.core.geometry import Pose
from parkpilot.perception.semantic import LineLandmark, SemanticMap
from parkpilot.spots.classify import classify
from parkpilot.spots.detect import SpotCandidate, _line_pairs
from parkpilot.world.model import FeatureKind, Spot, SpotLayout, SpotType, World, ZoneType
from tests.helpers import survey

CFG = SpotDetectionConfig()


def _truth(world: World, cand: SpotCandidate) -> Spot:
    return min(world.spots, key=lambda s: s.pose.distance_to(cand.pose))


def test_lined_detection_and_legality_on_restricted_lot() -> None:
    _, world, out, cands, _ = survey("restricted_spots")
    occupied_truth = world.occupied_spot_ids()
    free_truth = {s.id for s in world.spots if s.id not in occupied_truth}
    free_found = set()
    for c in cands:
        classify(c, out.semantic, frozenset(), CFG)
        t = _truth(world, c)
        assert t.pose.distance_to(c.pose) < 0.3
        if not c.occupied:
            free_found.add(t.id)
            assert t.id not in occupied_truth
            if t.spot_type is not SpotType.STANDARD:
                assert c.spot_type is t.spot_type
                assert not c.legal
    assert free_found == free_truth
    legal = {_truth(world, c).id for c in cands if c.legal and not c.occupied and c.inspected}
    assert legal == {"A14", "B9"}


def test_placard_unlocks_accessible_only() -> None:
    _, world, out, cands, _ = survey("restricted_spots")
    unlocked = set()
    for c in cands:
        classify(c, out.semantic, frozenset({Unlock.PLACARD}), CFG)
        if c.legal and not c.occupied:
            unlocked.add(_truth(world, c).spot_type)
    assert SpotType.ACCESSIBLE in unlocked
    assert SpotType.MOTORCYCLE not in unlocked
    assert SpotType.RESERVED not in unlocked


def test_lineless_detection_on_unmarked_lot() -> None:
    _, world, _, cands, _ = survey("unmarked_lot")
    free = [c for c in cands if not c.occupied and c.observed_fraction >= 0.35]
    assert all(c.source == "gap" for c in cands)
    truth_ids = {_truth(world, c).id for c in free}
    assert truth_ids == {"A6", "A10"}
    for c in free:
        t = _truth(world, c)
        # Centered in the actual gap, oriented like the row.
        assert abs(c.pose.x - t.pose.x) < 0.6
        assert abs(np.cos(c.pose.theta - t.pose.theta) - 1.0) < 0.01
        assert c.width >= CFG.gap_min_width


def test_baseline_without_gaps_finds_nothing_unmarked() -> None:
    _, _, _, cands, _ = survey("unmarked_lot", use_gaps=False)
    assert cands == []


def test_line_pairs_reject_non_adjacent() -> None:
    def ln(i: str, x: float) -> LineLandmark:
        return LineLandmark(i, np.array([x, 0.0]), np.array([x, 5.4]), 3)

    lines = [ln("a", 0.0), ln("b", 2.8), ln("c", 5.6)]
    pairs = {"".join(sorted((p[0].id, p[1].id))) for p in _line_pairs(lines, CFG)}
    assert pairs == {"ab", "bc"}


def test_zone_cues_make_stall_illegal() -> None:
    sem = SemanticMap()
    from parkpilot.perception.semantic import FeatureEvidence

    sem.features.append(
        FeatureEvidence(FeatureKind.CURB_RAMP, "sidewalk_ramp", np.array([0.0, 3.0]))
    )
    cand = SpotCandidate("x", Pose(0, 0, np.pi / 2), 5.4, 2.8, SpotLayout.PERPENDICULAR, "gap")
    cand.inspected = True
    classify(cand, sem, frozenset(Unlock), CFG)
    assert cand.zone is ZoneType.SIDEWALK_RAMP and not cand.legal
    assert "no parking" in cand.reasons[0]
