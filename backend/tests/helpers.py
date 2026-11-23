"""Shared test helpers: drive perception along a scenario's search route."""

from __future__ import annotations

import numpy as np

from parkpilot.core.config import ParkPilotConfig
from parkpilot.core.geometry import Pose
from parkpilot.perception.pipeline import Perception, PerceptionOutput
from parkpilot.scenarios.loader import load_scenario
from parkpilot.scenarios.schema import Scenario, build_world
from parkpilot.spots.detect import SpotCandidate, SpotDetector
from parkpilot.world.model import World

_CACHE: dict[tuple[str, int, float, bool], object] = {}


def survey(
    scenario_id: str,
    *,
    seed: int = 0,
    step: float = 0.3,
    config: ParkPilotConfig | None = None,
    use_gaps: bool = True,
) -> tuple[Scenario, World, PerceptionOutput, list[SpotCandidate], SpotDetector]:
    """Drive straight along the first search-route leg with perception and spot detection.

    Results for the default config are cached per process (tests only read them, apart
    from re-classifying candidates, which overwrites every classification field).
    """
    key = (scenario_id, seed, step, use_gaps)
    if config is None and key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    sc = load_scenario(scenario_id)
    world = build_world(sc)
    cfg = config or ParkPilotConfig()
    perception = Perception(cfg, sc.bounds)
    detector = SpotDetector(cfg.spots, sc.search_route, sc.bounds)
    rng = np.random.default_rng(seed)
    (x0, y0), (x1, _) = sc.search_route[0], sc.search_route[1]
    out: PerceptionOutput | None = None
    cands: list[SpotCandidate] = []
    for i, x in enumerate(np.arange(x0, x1, step)):
        out = perception.update(i * 0.1, Pose(float(x), y0, 0.0), world, [], rng)
        if i % 5 == 0:
            cands = detector.detect(out, use_gaps=use_gaps)
    assert out is not None
    cands = detector.detect(out, use_gaps=use_gaps)
    result = (sc, world, out, cands, detector)
    if config is None:
        _CACHE[key] = result
    return result
