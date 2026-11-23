import math

import numpy as np

from parkpilot.core.config import GridConfig, SafetyConfig, TrackerConfig, VehicleConfig
from parkpilot.core.geometry import Pose
from parkpilot.perception.camera import SensorFrame
from parkpilot.perception.grid import OccupancyGrid
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.perception.semantic import SemanticMap
from parkpilot.perception.tracks import Track
from parkpilot.planning.collision import CollisionChecker
from parkpilot.safety.supervisor import SafetySupervisor, track_kind
from parkpilot.vehicle.model import VehicleModel, VehicleState

VEH = VehicleModel(VehicleConfig())
TRK = TrackerConfig()


def _perception(
    tracks: list[Track],
    blocks: list[tuple[float, float, float, float]] = (),  # type: ignore[assignment]
    hits: np.ndarray | None = None,
    sigma: float = 0.08,
) -> PerceptionOutput:
    g = OccupancyGrid(GridConfig(), (-20, -20, 40, 20), VEH.config, 1.0)
    g.observed[:] = True
    for x0, y0, x1, y1 in blocks:
        r0, c0 = g.to_cells(np.array([x0, y0]))
        r1, c1 = g.to_cells(np.array([x1, y1]))
        g.logodds[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = 3.0
    g.version += 1
    pts = np.zeros((0, 2)) if hits is None else hits
    n = len(pts)
    frame = SensorFrame(
        t=0.0,
        ego=Pose(0, 0, 0),
        origins=np.zeros((n, 2)),
        directions=pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-9)
        if n
        else np.zeros((0, 2)),
        ranges=np.linalg.norm(pts, axis=1) if n else np.zeros(0),
        max_ranges=np.full(n, 30.0),
        hits=np.ones(n, dtype=bool),
        valid=np.ones(n, dtype=bool),
        noise_sigma=sigma,
        lighting=1.0,
    )
    confirmed = [t for t in tracks if t.confirmed(TRK)]
    return PerceptionOutput(0.0, Pose(0, 0, 0), frame, g, tracks, confirmed, SemanticMap(), sigma)


def _path() -> np.ndarray:
    xs = np.arange(0, 25, 0.2)
    return np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)


def _track(
    pos: tuple[float, float], vel: tuple[float, float], hits: int = 5, **kw: object
) -> Track:
    t = Track(
        1, np.array(pos), np.array(vel), 0.5, np.zeros((0, 2)), hits=hits, origin=np.array(pos)
    )
    t.moving_frames = 5
    t.free_space_frames = 5
    t.innovation = 0.1
    t.origin = np.array(pos) - np.array(vel) * 3
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_static_blockage_limits_speed() -> None:
    sup = SafetySupervisor(SafetyConfig(), TRK, VEH)
    perc = _perception([], blocks=[(6.0, -1.0, 6.4, 1.0)])
    checker = CollisionChecker(perc.grid, VEH, sup.margin(perc.noise_sigma) * 0.5)
    d = sup.evaluate(VehicleState(Pose(0, 0, 0), speed=3.0), _path(), perc, checker)
    assert d.blocked_at is not None and 1.5 < d.blocked_at < 3.0
    assert d.speed_limit < 3.0 and d.reason == "obstacle on path"
    empty = _perception([])
    clear = sup.evaluate(
        VehicleState(Pose(0, 0, 0), speed=3.0),
        _path(),
        empty,
        CollisionChecker(empty.grid, VEH, 0.1),
    )
    assert math.isinf(clear.speed_limit) and clear.blocked_at is None


def test_margin_widens_at_night() -> None:
    sup = SafetySupervisor(SafetyConfig(), TRK, VEH)
    assert sup.margin(0.28) > sup.margin(0.08) + 0.07


def test_yields_to_crossing_pedestrian_and_stops_hard_if_close() -> None:
    sup = SafetySupervisor(SafetyConfig(), TRK, VEH)
    ped = _track((8.0, -4.0), (0.0, 1.5))
    perc = _perception([ped])
    checker = CollisionChecker(perc.grid, VEH, 0.1)
    d = sup.evaluate(VehicleState(Pose(0, 0, 0), speed=1.0), _path(), perc, checker)
    assert d.yield_track == ped.id and d.yield_kind == "pedestrian"
    assert d.speed_limit == 0.0 and d.reason == "yielding to pedestrian"
    close = _track((5.5, -1.5), (0.0, 1.5))
    d2 = sup.evaluate(
        VehicleState(Pose(0, 0, 0), speed=2.0), _path(), _perception([close]), checker
    )
    assert d2.emergency and d2.min_ttc < SafetyConfig().ttc_stop


def test_persistence_filter_ignores_unconfirmed_but_baseline_brakes() -> None:
    ghost = _track((5.0, 0.0), (0.0, 0.0), hits=1)
    hits = np.array([[5.0, 0.0], [5.1, 0.1]])
    perc = _perception([ghost], hits=hits)
    checker = CollisionChecker(perc.grid, VEH, 0.1)
    state = VehicleState(Pose(0, 0, 0), speed=2.0)
    filtered = SafetySupervisor(SafetyConfig(), TRK, VEH).evaluate(state, _path(), perc, checker)
    assert not filtered.emergency and math.isinf(filtered.speed_limit)
    naive = SafetySupervisor(SafetyConfig(persistence_filter=False), TRK, VEH)
    d = naive.evaluate(state, _path(), perc, checker)
    assert d.emergency and d.speed_limit == 0.0


def test_reverse_lights_intent_cue_triggers_yield_before_motion() -> None:
    sup = SafetySupervisor(SafetyConfig(), TRK, VEH)
    car = _track((7.0, 4.0), (0.0, 0.0), reverse_lights=True)
    car.extent = 2.0
    car.moving_frames = 0
    perc = _perception([car])
    checker = CollisionChecker(perc.grid, VEH, 0.1)
    d = sup.evaluate(VehicleState(Pose(0, 0, 0), speed=1.0), _path(), perc, checker)
    assert d.yield_kind == "car backing out"
    assert track_kind(_track((0, 0), (0, 0))) == "pedestrian"
    big = _track((0, 0), (0, 0))
    big.extent = 2.0
    assert track_kind(big) == "cart"
    big.extent = 4.0
    assert track_kind(big) == "vehicle"
