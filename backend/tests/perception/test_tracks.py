import numpy as np

from parkpilot.core.config import TrackerConfig
from parkpilot.perception.camera import CueObservation
from parkpilot.perception.tracks import Tracker, cluster_points

CFG = TrackerConfig()


def _blob(x: float, y: float, n: int = 6) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([x + 0.2 * np.cos(ang), y + 0.2 * np.sin(ang)], axis=1)


def test_clustering_separates_far_blobs() -> None:
    pts = np.vstack([_blob(0, 0), _blob(5, 0), _blob(5.3, 0.1)])
    groups = cluster_points(pts, 0.6)
    assert sorted(len(g) for g in groups) == [6, 12]
    assert cluster_points(np.zeros((0, 2)), 0.6) == []


def test_persistence_filter_confirms_only_after_n_frames() -> None:
    tr = Tracker(CFG)
    for k in range(CFG.confirm_frames - 1):
        tr.update(0.1 * k, _blob(5, 0), [])
        assert tr.confirmed() == []
    tr.update(0.1 * CFG.confirm_frames, _blob(5, 0), [])
    assert len(tr.confirmed()) == 1


def test_flicker_never_confirms() -> None:
    tr = Tracker(CFG)
    for k in range(20):
        pts = _blob(5, 0) if k % 2 == 0 else np.zeros((0, 2))
        tr.update(0.1 * k, pts, [])
        assert tr.confirmed() == []


def test_moving_track_velocity_and_mask() -> None:
    tr = Tracker(CFG)
    mask = np.zeros(0, dtype=bool)
    for k in range(15):
        pts = _blob(0.15 * k, 0)
        mask = tr.update(0.1 * k, pts, [], np.ones(len(pts), dtype=bool))
    track = tr.confirmed()[0]
    assert abs(track.velocity[0] - 1.5) < 0.3
    assert track.is_dynamic(CFG)
    assert mask.all()


def test_reverse_light_cue_attaches_and_big_clusters_ignored() -> None:
    tr = Tracker(CFG)
    wall = np.stack([np.linspace(0, 20, 100), np.full(100, 5.0)], axis=1)
    cue = CueObservation((3.0, 0.0), True)
    for k in range(4):
        tr.update(0.1 * k, np.vstack([_blob(3, 0), wall]), [cue])
    assert len(tr.tracks) == 1
    assert tr.tracks[0].reverse_lights
    assert tr.tracks[0].is_dynamic(CFG)


def test_tracks_deleted_after_misses() -> None:
    tr = Tracker(CFG)
    tr.update(0.0, _blob(1, 1), [])
    for k in range(CFG.max_misses + 1):
        tr.update(0.1 * (k + 1), np.zeros((0, 2)), [])
    assert tr.tracks == []
