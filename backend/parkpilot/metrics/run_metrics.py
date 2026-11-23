"""Per-run metrics computed from the run log, ground truth and autonomy telemetry.

Ground truth is used *only* here and in the debug layer. Every number in
``docs/BENCHMARK.md`` comes from these functions; nothing is hard-coded.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from parkpilot.behavior.types import DriveState
from parkpilot.core.config import DirectionPreference, UserSettings
from parkpilot.core.geometry import (
    FloatArray,
    Pose,
    polygon_distance,
    polygons_intersect,
    rect_polygon,
)
from parkpilot.core.units import angle_diff
from parkpilot.sim.engine import DT, Simulation
from parkpilot.spots.classify import REQUIRED_UNLOCK
from parkpilot.spots.detect import SpotCandidate
from parkpilot.spots.rank import CostInputs, side_room, spot_cost, wants_reverse_in
from parkpilot.world.model import ObstacleKind, Spot, SpotLayout, SpotType, World

FALSE_STOP_RADIUS = 6.0
"""A stop with no real agent within this distance and no static obstacle close ahead is a
false stop (m)."""
HAZARD_TTC = 0.5
"""Ground-truth time to contact (with the ego's current deceleration) below which a
still-moving ego counts as a missed hazard (s)."""


class RunMetrics(BaseModel):
    scenario: str
    system: str
    seed: int
    parked: bool
    handback: bool
    abort_with_reason: bool | None
    time_to_park: float | None
    sim_time: float
    gear_changes: int
    replans: int
    spot_switches: int
    mean_speed: float
    peak_jerk: float
    final_lateral_error: float | None
    final_heading_error_deg: float | None
    min_side_clearance: float | None
    min_clearance: float
    collisions: int
    curb_contacts: int
    violations: int
    missed_hazards: int
    false_stops: int
    walking_distance: float | None
    oracle_match: bool | None
    port_error: float | None
    curb_distance_error: float | None
    yield_events: int
    time_lost_yielding: float
    preference_honored: bool | None
    parked_spot: str | None
    repeat_match: bool | None = None
    override_honored: bool | None = None


def gear_changes(speeds: FloatArray, threshold: float = 0.05) -> int:
    last = 0
    changes = 0
    for v in speeds:
        s = 1 if v > threshold else (-1 if v < -threshold else 0)
        if s != 0:
            if last not in (0, s):
                changes += 1
            last = s
    return changes


def _peak_jerk(speeds: FloatArray, accels: FloatArray) -> float:
    if len(accels) < 2:
        return 0.0
    moving = (np.abs(speeds[1:]) > 0.1) & (np.abs(speeds[:-1]) > 0.1)
    jerk = np.abs(np.diff(accels)) / DT
    return float(jerk[moving].max()) if moving.any() else 0.0


def _obstacle_points(world: World, spacing: float = 0.1) -> FloatArray:
    pts = []
    for o in world.obstacles:
        poly = o.polygon
        for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
            n = max(1, math.ceil(float(np.hypot(*(b - a))) / spacing))
            t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
            pts.append(a + t * (b - a))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def spot_is_legal(spot: Spot, world: World, settings: UserSettings) -> bool:
    if spot.spot_type is SpotType.MOTORCYCLE:
        return False
    need = REQUIRED_UNLOCK.get(spot.spot_type)
    if need is not None and need not in settings.unlocks:
        return False
    inner = rect_polygon(spot.pose, spot.length - 0.2, spot.width - 0.2)
    return not any(polygons_intersect(inner, z.polygon) for z in world.zones)


def oracle_best_spot(
    sim: Simulation, settings: UserSettings, remembered: list[tuple[float, float]]
) -> str | None:
    """Best spot by the same cost function, evaluated on ground truth."""
    world = sim.world
    cfg = sim.config
    occupied = world.occupied_spot_ids()
    # Availability is judged when the system first committed to a spot (or at the start):
    # a stall vacated by a car before that moment is a fair choice.
    commit_t = next(
        (tr.t for tr in sim.autonomy.telemetry.transitions if tr.dst is DriveState.APPROACH),
        0.0,
    )
    cars: list[tuple[float, float]] = []
    if sim.log.car_agents:
        _, cars = min(sim.log.car_agents, key=lambda f: abs(f[0] - commit_t))
    agent_spots = {
        s.id for s in world.spots for x, y in cars if s.pose.distance_to(Pose(x, y, 0.0)) < 2.0
    }
    pts = _obstacle_points(world)
    start = sim.scenario.start_pose()
    best: tuple[float, str] | None = None
    for s in world.spots:
        if s.id in occupied or s.id in agent_spots or not spot_is_legal(s, world, settings):
            continue
        room = side_room(pts, s.pose, s.length, s.width, cfg.ranking.neighbor_search)
        if s.layout is SpotLayout.PERPENDICULAR and room.free_width < (
            cfg.vehicle.mirror_width + 2 * cfg.ranking.min_side_margin
        ):
            continue
        cand = SpotCandidate(s.id, s.pose, s.length, s.width, s.layout, "lines")
        if s.charger_id is not None:
            ch = world.charger(s.charger_id)
            cand.charger = ch.position
        inp = CostInputs(
            walk_distance=math.hypot(s.pose.x - world.entrance[0], s.pose.y - world.entrance[1]),
            drive_distance=start.distance_to(s.pose),
            room=room,
            layout=s.layout,
            pull_through=s.pull_through,
            remembered=any(
                math.hypot(s.pose.x - x, s.pose.y - y) <= cfg.ranking.memory_radius
                for x, y in remembered
            ),
            has_charger=s.charger_id is not None,
            reverse_in=wants_reverse_in(cand, settings, cfg.vehicle),
        )
        cost, _, _ = spot_cost(inp, settings, cfg.ranking, cfg.vehicle)
        if best is None or cost < best[0]:
            best = (cost, s.id)
    return best[1] if best else None


def compute_metrics(
    sim: Simulation, system: str, remembered: list[tuple[float, float]] | None = None
) -> RunMetrics:
    log = sim.log
    tel = sim.autonomy.telemetry
    world = sim.world
    settings = sim.settings
    remembered = remembered or []
    ticks = log.ticks
    speeds = np.array([r.speed for r in ticks]) if ticks else np.zeros(0)
    accels = np.array([r.accel for r in ticks]) if ticks else np.zeros(0)
    parked = tel.state is DriveState.PARKED and tel.parked_pose is not None
    t_park = next((r.t for r in ticks if r.state is DriveState.PARKED), None) if parked else None
    end_idx = (
        len(ticks)
        if t_park is None
        else next(i for i, r in enumerate(ticks) if r.state is DriveState.PARKED)
    )
    xy = np.array([[r.x, r.y] for r in ticks[:end_idx]]) if end_idx > 1 else np.zeros((0, 2))
    dist = float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 0.0
    duration = t_park if t_park is not None else sim.t
    truth = [s for _, s in log.truth]
    collisions = _episodes([s.collision for s in truth])
    curb = _episodes([s.curb_contact for s in truth])
    min_clear = min((s.static_clearance for s in truth), default=math.inf)

    # Missed hazards: still moving with real contact imminent, or touching an agent.
    hazard = [abs(v) > 0.3 and ttc < HAZARD_TTC for _, v, ttc in log.hazard_frames]
    agent_hits = [s.agent_collision for s in truth]
    missed = _episodes([h or a for h, a in zip(hazard, agent_hits, strict=False)])

    false_stops = _false_stops(sim)

    final = _final_pose_metrics(sim, parked)
    yield_time = tel.yield_time
    honored = None
    if parked and settings.direction in (
        DirectionPreference.NOSE_IN,
        DirectionPreference.REVERSE_IN,
    ):
        spot = final["spot"]
        if isinstance(spot, Spot) and spot.layout is SpotLayout.PERPENDICULAR and tel.parked_pose:
            nose_in = abs(angle_diff(tel.parked_pose.theta, spot.pose.theta)) < math.pi / 2
            honored = nose_in == (settings.direction is DirectionPreference.NOSE_IN)
    # When the driver picks the spot (scripted override), the system is not choosing.
    oracle = None if sim.scenario.scripted else oracle_best_spot(sim, settings, remembered)
    spot_obj = final["spot"]
    oracle_match = (
        (spot_obj.id == oracle)
        if isinstance(spot_obj, Spot) and oracle
        else (None if oracle is None else False)
    )
    handback = tel.state is DriveState.HANDBACK
    override_honored = None
    if sim.scenario.scripted:
        px, py = sim.scenario.scripted[0].position
        target = min(world.spots, key=lambda s: math.hypot(s.pose.x - px, s.pose.y - py))
        override_honored = isinstance(spot_obj, Spot) and spot_obj.id == target.id
    return RunMetrics(
        override_honored=override_honored,
        scenario=sim.scenario.id,
        system=system,
        seed=sim.seed,
        parked=parked,
        handback=handback,
        abort_with_reason=(tel.handback_reason is not None) if (handback or not parked) else None,
        time_to_park=t_park,
        sim_time=sim.t,
        gear_changes=gear_changes(speeds[:end_idx]),
        replans=tel.replans,
        spot_switches=tel.spot_switches,
        mean_speed=dist / duration if duration > 0 else 0.0,
        peak_jerk=_peak_jerk(speeds[:end_idx], accels[:end_idx]),
        final_lateral_error=final["lateral"],
        final_heading_error_deg=final["heading"],
        min_side_clearance=final["side"],
        min_clearance=float(min_clear),
        collisions=collisions,
        curb_contacts=curb,
        violations=int(final["violation"]),  # type: ignore[call-overload]
        missed_hazards=missed,
        false_stops=false_stops,
        walking_distance=final["walk"],
        oracle_match=oracle_match,
        port_error=final["port"],
        curb_distance_error=final["curb"],
        yield_events=tel.yield_events,
        time_lost_yielding=yield_time,
        preference_honored=honored,
        parked_spot=spot_obj.id if isinstance(spot_obj, Spot) else None,
    )


def _episodes(flags: list[bool]) -> int:
    n, prev = 0, False
    for f in flags:
        if f and not prev:
            n += 1
        prev = f
    return n


def _false_stops(sim: Simulation) -> int:
    """Hard brakes and yields with no real agent nearby and nothing solid close ahead."""
    log = sim.log
    events = list(log.hard_brake_times)
    for tr in sim.autonomy.telemetry.transitions:
        if tr.dst is DriveState.YIELD:
            events.append(tr.t)
    count = 0
    truth_by_t = dict(log.truth)
    hazard_by_t = {round(t, 2): ttc for t, _, ttc in log.hazard_frames}
    for t in sorted(set(round(e, 2) for e in events)):
        near = [k for k in truth_by_t if abs(k - t) <= 0.15]
        sample = truth_by_t[near[0]] if near else None
        ttc = min((v for k, v in hazard_by_t.items() if abs(k - t) <= 0.3), default=math.inf)
        agent_close = sample is not None and sample.agent_clearance < FALSE_STOP_RADIUS
        static_close = sample is not None and sample.static_clearance < 0.3
        if ttc > 3.0 and not agent_close and not static_close:
            count += 1
    return count


def _final_pose_metrics(sim: Simulation, parked: bool) -> dict[str, object]:
    out: dict[str, object] = {
        "spot": None,
        "lateral": None,
        "heading": None,
        "side": None,
        "violation": False,
        "walk": None,
        "port": None,
        "curb": None,
    }
    tel = sim.autonomy.telemetry
    pose = tel.parked_pose if parked else None
    if pose is None:
        return out
    world, veh, cfg = sim.world, sim.vehicle, sim.config
    center = veh.body_center(pose)
    spot = min(world.spots, key=lambda s: s.pose.distance_to(center))
    out["spot"] = spot
    out["walk"] = math.hypot(spot.pose.x - world.entrance[0], spot.pose.y - world.entrance[1])
    rel = center.relative_to(spot.pose)
    pts = _obstacle_points(world)
    if spot.layout is SpotLayout.PERPENDICULAR:
        room = side_room(pts, spot.pose, spot.length, spot.width, cfg.ranking.neighbor_search)
        ideal = room.centered_offset()
        limit = max(0.0, (spot.width - cfg.vehicle.mirror_width) / 2 - cfg.goal.line_margin)
        ideal = max(-limit, min(limit, ideal))
        out["lateral"] = abs(rel.y - ideal)
        out["heading"] = math.degrees(abs(math.remainder(rel.theta, math.pi)))
    else:
        out["heading"] = math.degrees(abs(math.remainder(rel.theta, math.pi)))
        curbs = [o.polygon for o in world.obstacles if o.kind is ObstacleKind.CURB]
        body = veh.body_polygon(pose)
        if curbs:
            gap = min(polygon_distance(body, c) for c in curbs)
            out["curb"] = abs(gap - cfg.goal.curb_gap)
            out["lateral"] = out["curb"]
    body = veh.body_polygon(pose)
    solid = [o.polygon for o in world.obstacles if o.kind is not ObstacleKind.CURB]
    out["side"] = min((polygon_distance(body, p) for p in solid), default=math.inf)
    foot = veh.footprint(pose)
    in_zone = any(polygons_intersect(f, z.polygon) for f in foot for z in world.zones)
    out["violation"] = in_zone or not spot_is_legal(spot, world, sim.settings)
    if spot.charger_id is not None:
        ch = world.charger(spot.charger_id)
        port = veh.charge_port(pose)
        out["port"] = float(np.hypot(port[0] - ch.position[0], port[1] - ch.position[1]))
    return out
