"""Benchmark runner: every scenario x seeds x systems, in parallel processes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

from parkpilot.metrics.run_metrics import RunMetrics, compute_metrics
from parkpilot.scenarios.loader import list_scenario_ids, load_scenario
from parkpilot.sim.engine import Simulation
from parkpilot.sim.systems import SYSTEMS, system_config

SYSTEM_NAMES = ("parkpilot", "baseline")


def run_one(scenario_id: str, system: str, seed: int) -> RunMetrics:
    """Run one (scenario, system, seed). Memory-protocol scenarios run two visits."""
    sc = load_scenario(scenario_id)
    cfg = system_config(system)
    if not sc.memory_protocol:
        sim = Simulation(sc, SYSTEMS[system], cfg, seed)
        sim.run()
        return compute_metrics(sim, system)
    # Visit 1: the driver overrides; the override is stored as a favorite for this location.
    first = Simulation(sc, SYSTEMS[system], cfg, seed)
    favorites: list[tuple[float, float]] = []
    first.override_listeners.append(lambda x, y: favorites.append((x, y)))
    first.run()
    target = None
    if sc.scripted:
        px, py = sc.scripted[0].position
        target = min(first.world.spots, key=lambda s: math.hypot(s.pose.x - px, s.pose.y - py)).id
    # Visit 2: same lot, no scripted override; only ParkPilot has preference memory.
    second_sc = sc.model_copy(update={"scripted": []})
    remembered = favorites if system == "parkpilot" else []
    second = Simulation(second_sc, SYSTEMS[system], cfg, seed + 1000, remembered=remembered)
    second.run()
    m = compute_metrics(second, system, remembered)
    m.repeat_match = m.parked_spot is not None and m.parked_spot == target
    return m


def _run_tuple(args: tuple[str, str, int]) -> RunMetrics:
    return run_one(*args)


def run_benchmark(
    scenarios: Iterable[str] | None = None,
    seeds: int = 5,
    systems: Iterable[str] = SYSTEM_NAMES,
    workers: int = 4,
) -> list[RunMetrics]:
    jobs = [
        (sid, system, seed)
        for sid in (list(scenarios) if scenarios else list_scenario_ids())
        for system in systems
        for seed in range(seeds)
    ]
    if workers <= 1:
        return [run_one(*j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_tuple, jobs, chunksize=1))
