"""Final-pose refinement: where exactly the car should stop in a chosen spot (C3, C9, C11).

* Perpendicular stalls: centre between the *actual* neighbors measured in the occupancy grid
  (not just the painted lines), but never so far that the mirrors cross a painted line;
  pull in toward the stall head up to a target bumper gap.
* Parallel stalls: target a fixed body-to-curb gap measured from the perceived curb, and
  centre between the front and rear neighbors.
* Charger stalls: the entry direction puts the charge port toward the charger (chosen in
  ranking), and the car pulls in until the port is within reach of the charger.

Finally a small local search guarantees the goal is collision-free with the planner margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from parkpilot.core.config import GoalConfig, RankingConfig
from parkpilot.core.geometry import FloatArray, Pose
from parkpilot.planning.collision import CollisionChecker
from parkpilot.spots.detect import SpotCandidate
from parkpilot.spots.rank import side_room
from parkpilot.vehicle.model import VehicleModel
from parkpilot.world.model import SpotLayout


@dataclass(slots=True)
class GoalResult:
    ok: bool
    pose: Pose
    """Rear-axle goal pose for the planner."""
    center: Pose
    """Body-centre goal pose."""
    lateral_offset: float = 0.0
    """Shift from the stall centre, + to the stall's left (m)."""
    longitudinal_offset: float = 0.0
    """Shift from the stall centre toward the stall head (m)."""
    port_distance: float | None = None
    reasons: list[str] = field(default_factory=list)


def _local_points(points: FloatArray, frame: Pose, half_len: float, half_wid: float) -> FloatArray:
    if len(points) == 0:
        return np.zeros((0, 2))
    loc = frame.to_local(points)
    keep = (np.abs(loc[:, 0]) <= half_len) & (np.abs(loc[:, 1]) <= half_wid)
    return np.asarray(loc[keep])


def compute_goal(
    cand: SpotCandidate,
    *,
    reverse_in: bool,
    checker: CollisionChecker,
    vehicle: VehicleModel,
    cfg: GoalConfig,
    ranking: RankingConfig,
    obstacle_points: FloatArray,
    use_neighbors: bool = True,
) -> GoalResult:
    """Refined goal for ``cand``. ``use_neighbors=False`` centres on painted lines only."""
    if cand.layout is SpotLayout.PARALLEL:
        lat, lon, reasons = _parallel_offsets(cand, vehicle, cfg, obstacle_points, use_neighbors)
        heading = cand.pose.theta
        head_sign = 1.0
    else:
        lat, lon, reasons = _perpendicular_offsets(
            cand, vehicle, cfg, ranking, obstacle_points, use_neighbors
        )
        heading = cand.pose.theta + (math.pi if reverse_in else 0.0)
        head_sign = 1.0
    port_distance = None
    if cand.charger is not None and cand.layout is SpotLayout.PERPENDICULAR:
        lon, port_distance, msg = _charger_depth(cand, heading, lat, lon, vehicle, cfg, checker)
        reasons.append(msg)

    def center_for(dlat: float, dlon: float) -> Pose:
        c = cand.pose.compose(Pose(head_sign * dlon, dlat, 0.0))
        return Pose(c.x, c.y, heading)

    best: tuple[float, float, float] | None = None
    step, ext = cfg.search_step, cfg.search_extent
    n = round(ext / step)
    offsets = sorted(
        ((i * step, j * step) for i in range(-n, n + 1) for j in range(-n, n + 1)),
        key=lambda d: abs(d[0]) + 1.5 * abs(d[1]),
    )
    for dlat, dlon in offsets:
        c = center_for(lat + dlat, lon + dlon)
        rear = vehicle.rear_axle_from_center(c)
        if checker.free(np.array([[rear.x, rear.y, rear.theta]]))[0]:
            best = (lat + dlat, lon + dlon, abs(dlat) + abs(dlon))
            break
    if best is None:
        c = center_for(lat, lon)
        return GoalResult(
            False,
            vehicle.rear_axle_from_center(c),
            c,
            lat,
            lon,
            None,
            [
                *reasons,
                "no collision-free final pose in this spot",
            ],
        )
    lat, lon, moved = best
    if moved > 1e-9:
        reasons.append(f"nudged {moved:.2f} m to keep the safety margin")
    c = center_for(lat, lon)
    if port_distance is not None and cand.charger is not None:
        port = vehicle.charge_port(vehicle.rear_axle_from_center(c))
        port_distance = float(np.hypot(*(port - np.array(cand.charger))))
    return GoalResult(True, vehicle.rear_axle_from_center(c), c, lat, lon, port_distance, reasons)


def _perpendicular_offsets(
    cand: SpotCandidate,
    vehicle: VehicleModel,
    cfg: GoalConfig,
    ranking: RankingConfig,
    points: FloatArray,
    use_neighbors: bool,
) -> tuple[float, float, list[str]]:
    v = vehicle.config
    reasons: list[str] = []
    lat = 0.0
    if use_neighbors:
        room = side_room(points, cand.pose, cand.length, cand.width, ranking.neighbor_search)
        lat = room.centered_offset()
        limit = max(0.0, (cand.width - v.mirror_width) / 2 - cfg.line_margin)
        if cand.source == "lines" and cfg.stay_within_lines and abs(lat) > limit:
            lat = math.copysign(limit, lat)
            reasons.append("centered between neighbors, limited by painted lines")
        elif abs(lat) > 0.03:
            reasons.append(f"centered between actual neighbors ({lat:+.2f} m)")
        body_gap = (room.free_width - v.width) / 2
        reasons.append(f"door clearance {body_gap:.2f} m each side")
    else:
        reasons.append("centered on painted lines")
    # Pull in toward the stall head: stop ``head_gap`` short of the first obstacle.
    loc = _local_points(points, cand.pose, cand.length / 2 + 1.5, v.width / 2)
    lon = 0.0
    if use_neighbors:
        ahead = loc[loc[:, 0] > 0, 0] if len(loc) else np.zeros(0)
        if len(ahead):
            lon = min(cfg.max_pull_in, float(ahead.min()) - cfg.head_gap - v.length / 2)
            lon = max(lon, -0.3)
    return lat, lon, reasons


def _parallel_offsets(
    cand: SpotCandidate,
    vehicle: VehicleModel,
    cfg: GoalConfig,
    points: FloatArray,
    use_neighbors: bool,
) -> tuple[float, float, list[str]]:
    v = vehicle.config
    if not use_neighbors:
        return 0.0, 0.0, ["centered on painted lines"]
    reasons: list[str] = []
    # Curb is on the right (-y) side of the traffic direction.
    side = _local_points(points, cand.pose, v.length / 2 - 0.2, cand.width / 2 + 1.5)
    right = side[side[:, 1] < 0, 1] if len(side) else np.zeros(0)
    lat = 0.0
    if len(right):
        curb = float(right.max())
        lat = curb + cfg.curb_gap + v.width / 2
        reasons.append(f"targeting {cfg.curb_gap:.2f} m curb gap")
    along = _local_points(points, cand.pose, cand.length / 2 + 2.0, v.width / 2 - 0.1)
    front = along[along[:, 0] > 0, 0] if len(along) else np.zeros(0)
    rear = along[along[:, 0] < 0, 0] if len(along) else np.zeros(0)
    f = float(front.min()) if len(front) else cand.length / 2 + 2.0
    r = float(-rear.max()) if len(rear) else cand.length / 2 + 2.0
    lon = (f - r) / 2
    reasons.append(f"{min(f, r) - v.length / 2 + abs(lon):.2f} m to neighbors front and rear")
    return lat, lon, reasons


def _charger_depth(
    cand: SpotCandidate,
    heading: float,
    lat: float,
    lon: float,
    vehicle: VehicleModel,
    cfg: GoalConfig,
    checker: CollisionChecker,
) -> tuple[float, float, str]:
    """Pull in until the charge port is within ``port_target`` of the charger, if possible."""
    assert cand.charger is not None
    charger = np.array(cand.charger)
    best_lon, best_d = lon, math.inf
    for k in range(25):
        dlon = lon + 0.05 * k
        c = cand.pose.compose(Pose(dlon, lat, 0.0))
        rear = vehicle.rear_axle_from_center(Pose(c.x, c.y, heading))
        if not checker.free(np.array([[rear.x, rear.y, rear.theta]]))[0]:
            break
        d = float(np.hypot(*(vehicle.charge_port(rear) - charger)))
        if d < best_d:
            best_lon, best_d = dlon, d
        if d <= cfg.port_target:
            break
    return best_lon, best_d, f"charge port {best_d:.2f} m from charger"
