"""ParkPilot's hierarchical decision state machine.

States: SEARCH -> SELECT -> APPROACH -> MANEUVER -> REFINE -> PARKED, plus YIELD, REPLAN,
FALLBACK and HANDBACK. Every transition is logged with a plain-language reason.

Design rules that address the complaints:

* **Commitment with hysteresis (C1).** Once a spot is chosen, a different spot is only
  taken if it is cheaper by an absolute *and* a relative margin, the commitment is at least
  ``min_commit_time`` old, the car is not yet maneuvering, and a plan to the new spot
  exists. The current plan is only recomputed when it becomes invalid (blocked, tracking
  lost), never on a timer.
* **Destination-aware search (C4).** The search continues past the first free spot until
  the area near the entrance has been seen, unless a near-optimal spot is already known.
* **Fallback ladder (C14).** Plan failure -> next-best spot; nothing left -> circle the
  search route again; still nothing -> pull over to the aisle edge, stop and hand back with
  a stated reason. The car never just freezes without saying why.
* **Yield and resume (C12).** A predicted conflict with a moving agent stops the car with a
  visible reason; it resumes as soon as the agent has been clear for ``clear_time``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from parkpilot.behavior.driving import Driver, route_path
from parkpilot.behavior.types import (
    TERMINAL,
    Abort,
    AutonomyCommand,
    ControlOutput,
    DriveState,
    Telemetry,
    Transition,
)
from parkpilot.core.config import ParkPilotConfig, Unlock, UserSettings
from parkpilot.core.geometry import Pose
from parkpilot.core.units import angle_diff
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.goal import GoalResult, compute_goal
from parkpilot.planning.hybrid_astar import HybridAStar, Path
from parkpilot.safety.supervisor import SafetyDecision, SafetySupervisor
from parkpilot.spots.classify import classify
from parkpilot.spots.detect import SpotCandidate, SpotDetector
from parkpilot.spots.rank import RankedSpot, rank_spots, wants_reverse_in
from parkpilot.vehicle.model import VehicleModel, VehicleState
from parkpilot.world.model import SpotLayout


def clearance_readouts(
    perc: PerceptionOutput, ego: Pose, vehicle: VehicleModel
) -> dict[str, float]:
    """Distances (m) from the body to the nearest remembered obstacle in four sectors."""
    v = vehicle.config
    pts = perc.grid.occupied_points()
    cap = 5.0
    out = {"front": cap, "rear": cap, "left": cap, "right": cap}
    if len(pts) == 0:
        return out
    near = pts[np.hypot(pts[:, 0] - ego.x, pts[:, 1] - ego.y) < cap + v.length]
    if len(near) == 0:
        return out
    loc = ego.to_local(near)
    front_x, rear_x, half = v.length - v.rear_overhang, -v.rear_overhang, v.width / 2
    lat = np.abs(loc[:, 1]) <= half + 0.3
    along = (loc[:, 0] >= rear_x) & (loc[:, 0] <= front_x)
    sel = lat & (loc[:, 0] > front_x)
    if sel.any():
        out["front"] = float(min(cap, (loc[sel, 0] - front_x).min()))
    sel = lat & (loc[:, 0] < rear_x)
    if sel.any():
        out["rear"] = float(min(cap, (rear_x - loc[sel, 0]).min()))
    sel = along & (loc[:, 1] > half)
    if sel.any():
        out["left"] = float(min(cap, (loc[sel, 1] - half).min()))
    sel = along & (loc[:, 1] < -half)
    if sel.any():
        out["right"] = float(min(cap, (-loc[sel, 1] - half).min()))
    return out


class ParkPilotAutonomy:
    name = "parkpilot"

    def __init__(
        self,
        config: ParkPilotConfig,
        *,
        route: list[tuple[float, float]],
        entrance: tuple[float, float],
        settings: UserSettings,
        remembered: list[tuple[float, float]] | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> None:
        self.cfg = config
        self.vehicle = VehicleModel(config.vehicle)
        self.route = list(route)
        self.entrance = entrance
        self.settings = settings
        self.remembered = list(remembered or [])
        self.detector = SpotDetector(config.spots, route, bounds)
        self.planner = HybridAStar(config.planner, self.vehicle)
        self.supervisor = SafetySupervisor(config.safety, config.tracker, self.vehicle)
        self.driver = Driver(config.control, config.speed, self.vehicle)
        self.telemetry = Telemetry()
        self._state = DriveState.SEARCH
        self._t = 0.0
        self._perc: PerceptionOutput | None = None
        self._last_spot_update = -math.inf
        self._committed: RankedSpot | None = None
        self._commit_time = 0.0
        self._failures: dict[str, int] = {}
        self._replans_here = 0
        self._override: str | None = None
        self._circles = 0
        self._refines = 0
        self._resume_state = DriveState.APPROACH
        self._clear_since: float | None = None
        self._yield_started = 0.0
        self._goal_refreshed = False
        self._pending_reason = ""
        self._route_progress_target = self._entrance_coverage_s()
        self._route_s = 0.0
        self._start_pose: Pose | None = None
        self._last_emergency = False

    # ------------------------------------------------------------------ public API
    @property
    def state(self) -> DriveState:
        return self._state

    @property
    def done(self) -> bool:
        return self._state in TERMINAL

    def command(self, cmd: AutonomyCommand) -> str:
        if isinstance(cmd, Abort):
            if self._state is not DriveState.HANDBACK:
                self._handback("aborted by the driver")
            return "aborting: stopping and handing back control"
        if self._state is DriveState.HANDBACK:
            return "control already handed back"
        ranked = {r.candidate.id: r for r in self.telemetry.ranked}
        target = ranked.get(cmd.spot_id)
        if target is None:
            msg = f"spot {cmd.spot_id} is not known"
        elif target.candidate.occupied:
            msg = "that spot is occupied"
        elif not target.candidate.legal:
            msg = "cannot park there: " + "; ".join(target.candidate.reasons[:1])
        else:
            self._override = cmd.spot_id
            self._failures.pop(cmd.spot_id, None)
            self._goto(DriveState.SELECT, f"driver picked spot {cmd.spot_id}")
            msg = f"rerouting to spot {cmd.spot_id}"
        self.telemetry.override_ack = msg
        return msg

    def step(
        self, t: float, dt: float, state: VehicleState, perc: PerceptionOutput | None
    ) -> ControlOutput:
        self._t = t
        if self._start_pose is None:
            self._start_pose = state.pose
            self._set_route_path(state)
        if perc is not None:
            self._perc = perc
            self.telemetry.clearances = clearance_readouts(perc, state.pose, self.vehicle)
            if t - self._last_spot_update >= self.cfg.behavior.spot_update_period:
                self._update_spots(perc)
                self._last_spot_update = t
        if self._perc is None:
            return self.driver.hold(state, dt)
        handler = {
            DriveState.SEARCH: self._search,
            DriveState.SELECT: self._select,
            DriveState.APPROACH: self._follow,
            DriveState.MANEUVER: self._follow,
            DriveState.REFINE: self._follow,
            DriveState.REPLAN: self._replan,
            DriveState.FALLBACK: self._fallback,
            DriveState.YIELD: self._yield,
            DriveState.PARKED: self._hold_terminal,
            DriveState.HANDBACK: self._hold_terminal,
        }[self._state]
        return handler(state, dt)

    # ------------------------------------------------------------------ perception -> spots
    def _update_spots(self, perc: PerceptionOutput) -> None:
        cands = self.detector.detect(perc)
        for c in cands:
            classify(c, perc.semantic, self.settings.unlocks, self.cfg.spots)
        ranked = rank_spots(
            cands,
            ego=perc.ego,
            entrance=self.entrance,
            settings=self.settings,
            cfg=self.cfg.ranking,
            vehicle=self.cfg.vehicle,
            obstacle_points=perc.grid.occupied_points(),
            remembered=self.remembered,
        )
        for r in ranked:
            if self._failures.get(r.candidate.id, 0) >= self.cfg.behavior.max_spot_failures:
                r.eligible = False
                r.reasons.insert(0, "no feasible maneuver found")
        ranked.sort(key=lambda r: (not r.eligible, r.cost))
        self.telemetry.candidates = cands
        self.telemetry.ranked = ranked
        if self._committed is not None:
            fresh = next(
                (r for r in ranked if r.candidate.id == self._committed.candidate.id), None
            )
            if fresh is not None:
                self._committed = fresh

    def _best(self) -> RankedSpot | None:
        if self._override is not None:
            for r in self.telemetry.ranked:
                if r.candidate.id == self._override and not r.candidate.occupied:
                    return r
        return next((r for r in self.telemetry.ranked if r.eligible), None)

    # ------------------------------------------------------------------ helpers
    def _goto(self, dst: DriveState, reason: str) -> None:
        if dst is self._state and reason == self.telemetry.reason:
            return
        self.telemetry.transitions.append(Transition(self._t, self._state, dst, reason))
        self._state = dst
        self.telemetry.state = dst
        self.telemetry.reason = reason

    def _margin(self) -> float:
        assert self._perc is not None
        return self.supervisor.margin(self._perc.noise_sigma)

    def _checker(self, fraction: float = 1.0) -> CollisionChecker:
        assert self._perc is not None
        return CollisionChecker(self._perc.grid, self.vehicle, self._margin() * fraction)

    def _set_route_path(self, state: VehicleState) -> None:
        remaining = self._route_remaining(state.pose)
        self.driver.set_path(route_path(state.pose, remaining), None, abs(state.speed))
        self.telemetry.path = self.driver.path

    def _route_remaining(self, pose: Pose) -> list[tuple[float, float]]:
        pts = np.array(self.route)
        if len(pts) < 2:
            return list(self.route)
        d = np.hypot(pts[:, 0] - pose.x, pts[:, 1] - pose.y)
        i = int(np.argmin(d))
        if i < len(pts) - 1:
            a, b = pts[i], pts[i + 1]
            ahead = (b - a) @ (np.array([pose.x, pose.y]) - a) > 0
            if ahead:
                i += 1
        return [(float(p[0]), float(p[1])) for p in pts[i:]]

    def _entrance_coverage_s(self) -> float:
        pts = np.array(self.route, dtype=float)
        if len(pts) < 2:
            return 0.0
        best_s, best_d, s_acc = 0.0, math.inf, 0.0
        for a, b in itertools.pairwise(pts):
            seg = b - a
            ln = float(np.hypot(*seg))
            for f in np.linspace(0, 1, 50):
                p = a + f * seg
                d = math.hypot(p[0] - self.entrance[0], p[1] - self.entrance[1])
                if d < best_d:
                    best_d, best_s = d, s_acc + f * ln
            s_acc += ln
        self._min_route_walk = best_d
        return min(best_s + self.cfg.behavior.entrance_coverage, s_acc)

    def _route_progress(self, pose: Pose) -> float:
        pts = np.array(self.route, dtype=float)
        s_acc, best, best_d = 0.0, 0.0, math.inf
        for a, b in itertools.pairwise(pts):
            seg = b - a
            ln = float(np.hypot(*seg))
            u = seg / max(ln, 1e-9)
            t = float(np.clip((np.array([pose.x, pose.y]) - a) @ u, 0, ln))
            q = a + u * t
            d = math.hypot(pose.x - q[0], pose.y - q[1])
            if d < best_d:
                best_d, best = d, s_acc + t
            s_acc += ln
        return best

    def _safety(self, state: VehicleState) -> SafetyDecision:
        assert self._perc is not None
        decision = self.supervisor.evaluate(
            state,
            self.driver.remaining_poses(),
            self._perc,
            self._checker(self.cfg.safety.check_margin_fraction),
        )
        self.telemetry.safety = decision
        if decision.emergency and not self._last_emergency:
            self.telemetry.hard_brakes += 1
        self._last_emergency = decision.emergency
        return decision

    def _drive(
        self, state: VehicleState, dt: float, decision: SafetyDecision
    ) -> tuple[ControlOutput, bool]:
        res = self.driver.step(state, dt, decision.speed_limit, decision.emergency)
        if res.segment_done:
            self.telemetry.segment_index = self.driver.index
        return res.control, res.path_done

    # ------------------------------------------------------------------ states
    def _search(self, state: VehicleState, dt: float) -> ControlOutput:
        self._route_s = self._route_progress(state.pose)
        best = self._best()
        if best is not None:
            covered = self._route_s >= self._route_progress_target - 1.0 or not self.driver.active
            # Near-optimal: the whole cost (not just walking) is within reach of the best
            # walking distance achievable anywhere along the route.
            bound = self.settings.weights.walking * (
                self._min_route_walk + self.cfg.behavior.near_optimal_walk
            )
            near_opt = best.cost - best.terms.get("drive", 0.0) <= bound
            if Unlock.CHARGING in self.settings.unlocks and best.candidate.charger is None:
                near_opt = False  # a charger stall may still be ahead
            if self.remembered and "memory" not in best.terms:
                near_opt = False  # the usual spot may still be ahead
            if covered or near_opt or self._override is not None or self._circles > 0:
                why = "near-optimal spot found" if near_opt and not covered else "lot searched"
                self._goto(DriveState.SELECT, f"{why}: {best.candidate.id}")
                return self.driver.hold(state, dt)
        decision = self._safety(state)
        if decision.yield_kind is not None:
            return self._enter_yield(state, dt, decision)
        if self.telemetry.reason.startswith("starting") or self._state is DriveState.SEARCH:
            self.telemetry.reason = (
                "searching near the entrance"
                if self._route_s < self._route_progress_target
                else "searching for a legal spot"
            )
        control, done = self._drive(state, dt, decision)
        stopped = abs(state.speed) < self.cfg.behavior.stop_speed
        blocked = decision.blocked_at is not None and decision.blocked_at < 1.0
        if stopped and (done or blocked):
            why = (
                "search route blocked"
                if blocked and not done
                else "reached the end of the search route"
            )
            return self._exhausted(state, dt, why)
        return control

    def _exhausted(self, state: VehicleState, dt: float, why: str) -> ControlOutput:
        if self._circles < self.cfg.behavior.max_circles:
            self._circles += 1
            self._goto(DriveState.FALLBACK, f"{why}; circling the aisle to look again")
            return self._plan_circle(state, dt)
        self._handback(f"no legal, reachable spot found after circling {self._circles}x", state)
        return self.driver.hold(state, dt)

    def _plan_circle(self, state: VehicleState, dt: float) -> ControlOutput:
        """Turn around where we are and search the route again in the other direction."""
        rel_route = self._route_remaining(state.pose)
        _, u = self._route_direction(state.pose)
        th = math.atan2(-u[1], -u[0])
        goal = Pose(state.pose.x, state.pose.y, th).compose(Pose(4.0, 0.0, 0.0))
        res = self.planner.plan(state.pose, goal, self._checker())
        self.telemetry.plans += 1
        if not res.success or res.path is None:
            self._handback("cannot turn around to circle the lot: " + res.reason, state)
            return self.driver.hold(state, dt)
        del rel_route
        self.route = list(reversed(self.route))
        self.detector.route = type(self.detector.route)(np.array(self.route, dtype=float))
        self._route_progress_target = self._entrance_coverage_s()
        self.driver.set_path(res.path, self._checker(), abs(state.speed))
        self.telemetry.path = res.path
        return self.driver.hold(state, dt)

    def _route_direction(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        pts = np.array(self.route, dtype=float)
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for a, b in itertools.pairwise(pts):
            seg = b - a
            ln = float(np.hypot(*seg))
            u = seg / max(ln, 1e-9)
            t = float(np.clip((np.array([pose.x, pose.y]) - a) @ u, 0, ln))
            q = a + u * t
            d = math.hypot(pose.x - q[0], pose.y - q[1])
            if best is None or d < best[0]:
                best = (d, q, u)
        assert best is not None
        return best[1], best[2]

    def _fallback(self, state: VehicleState, dt: float) -> ControlOutput:
        """Executing a fallback maneuver (circling back to the start of the route)."""
        decision = self._safety(state)
        if decision.yield_kind is not None:
            return self._enter_yield(state, dt, decision)
        if decision.blocked_at is not None and decision.blocked_at < 1.0:
            self._handback("route back is blocked", state)
            return self.driver.hold(state, dt)
        control, done = self._drive(state, dt, decision)
        if done and abs(state.speed) < self.cfg.behavior.stop_speed:
            self._set_route_path(state)
            self._goto(DriveState.SEARCH, "searching again")
        return control

    def _select(self, state: VehicleState, dt: float) -> ControlOutput:
        if abs(state.speed) > self.cfg.behavior.stop_speed:
            # Come to a smooth stop on the current path before planning the maneuver.
            res = self.driver.step(state, dt, 0.0, False) if self.driver.active else None
            return res.control if res is not None else self.driver.hold(state, dt)
        for _ in range(8):
            best = self._best()
            if best is None:
                self._override = None
                return self._exhausted(state, dt, "no eligible spot left")
            if self._try_commit(best, state):
                return self.driver.hold(state, dt)
            if self._override == best.candidate.id:
                self.telemetry.override_ack = f"cannot reach spot {best.candidate.id}"
                self._override = None
        return self.driver.hold(state, dt)

    def _try_commit(self, best: RankedSpot, state: VehicleState) -> bool:
        cand = best.candidate
        path, goal, why = self._plan_to(cand, state.pose)
        if path is None or goal is None:
            self._failures[cand.id] = self._failures.get(cand.id, 0) + 1
            best.eligible = False
            best.reasons.insert(0, why)
            self.telemetry.ranked.sort(key=lambda r: (not r.eligible, r.cost))
            self._goto(DriveState.FALLBACK, f"spot {cand.id}: {why}; trying the next-best spot")
            self._state = DriveState.SELECT
            return False
        switching = self._committed is not None and self._committed.candidate.id != cand.id
        if switching:
            self.telemetry.spot_switches += 1
        self._committed = best
        self._commit_time = self._t
        self._replans_here = 0
        self._refines = 0
        self._goal_refreshed = False
        self.telemetry.chosen_spot = cand.id
        self.telemetry.goal = goal
        self.driver.set_path(path, self._checker(), abs(state.speed))
        self.telemetry.path = path
        self.telemetry.segment_index = 0
        reason = "; ".join(best.reasons[:2])
        self._goto(DriveState.APPROACH, f"going to spot {cand.id} ({reason})")
        return True

    def _plan_to(
        self, cand: SpotCandidate, start: Pose
    ) -> tuple[Path | None, GoalResult | None, str]:
        assert self._perc is not None
        reverse = wants_reverse_in(cand, self.settings, self.cfg.vehicle)
        checker = self._checker()
        if not checker.free(np.array([[start.x, start.y, start.theta]]))[0]:
            # Already inside the full planning margin (e.g. mid-maneuver next to a curb):
            # plan with the supervisor's reduced margin so the car can still get out.
            checker = self._checker(self.cfg.safety.check_margin_fraction)
        # The final pose gets extra slack: depth noise keeps adding cells to low objects
        # (curbs) as the car closes in, and a goal that is only just free would soon be
        # "blocked" by them.
        goal_checker = CollisionChecker(
            self._perc.grid, self.vehicle, checker.margin + self.cfg.goal.goal_slack
        )
        goal = compute_goal(
            cand,
            reverse_in=reverse,
            checker=goal_checker,
            vehicle=self.vehicle,
            cfg=self.cfg.goal,
            ranking=self.cfg.ranking,
            obstacle_points=self._perc.grid.occupied_points(),
        )
        if not goal.ok:
            goal = compute_goal(
                cand,
                reverse_in=reverse,
                checker=checker,
                vehicle=self.vehicle,
                cfg=self.cfg.goal,
                ranking=self.cfg.ranking,
                obstacle_points=self._perc.grid.occupied_points(),
            )
        if not goal.ok:
            return None, None, "no collision-free final pose"
        straight = (
            self.cfg.planner.final_straight if cand.layout is SpotLayout.PERPENDICULAR else 0.0
        )
        res = self.planner.plan(start, goal.pose, checker, straight, -1 if reverse else 1)
        self.telemetry.plans += 1
        if not res.success or res.path is None:
            if reverse and cand.charger is None:
                # Preference is soft: try the other entry direction before giving up.
                goal2 = compute_goal(
                    cand,
                    reverse_in=False,
                    checker=checker,
                    vehicle=self.vehicle,
                    cfg=self.cfg.goal,
                    ranking=self.cfg.ranking,
                    obstacle_points=self._perc.grid.occupied_points(),
                )
                if goal2.ok:
                    res = self.planner.plan(start, goal2.pose, checker, straight, 1)
                    self.telemetry.plans += 1
                    if res.success and res.path is not None:
                        goal2.reasons.append("nose-in: reversing in was not feasible")
                        return res.path, goal2, ""
            return None, None, f"no maneuver found ({res.reason})"
        return res.path, goal, ""

    def _follow(self, state: VehicleState, dt: float) -> ControlOutput:
        assert self._committed is not None and self.telemetry.goal is not None
        b = self.cfg.behavior
        cand = self._committed.candidate
        goal = self.telemetry.goal
        dist_goal = state.pose.distance_to(goal.pose)
        decision = self._safety(state)
        if decision.yield_kind is not None:
            return self._enter_yield(state, dt, decision)
        if (
            decision.blocked_at is not None
            and self._state is DriveState.REFINE
            and abs(state.speed) < b.stop_speed
            and decision.blocked_at < 1.2
        ):
            self._refines = b.refine_attempts
            return self._arrived(state, dt)
        if self._state is not DriveState.REFINE:
            if cand.occupied:
                return self._spot_lost(state, dt, f"spot {cand.id} became occupied")
            if not cand.legal:
                return self._spot_lost(state, dt, f"spot {cand.id} turned out restricted")
            # Commitment hysteresis: only a clearly better, reachable spot replaces this one.
            if self._state is DriveState.APPROACH and self._override is None:
                switched = self._consider_switch(state, dist_goal)
                if switched is not None:
                    return switched
            if decision.blocked_at is not None and decision.blocked_at < 1.2:
                remaining = self._remaining_path_length()
                if remaining - decision.blocked_at < 0.6 and remaining < 1.5:
                    # The blockage is at the very end of the path: stop here and let REFINE
                    # judge the pose instead of re-planning the whole maneuver.
                    if abs(state.speed) < b.stop_speed:
                        return self._arrived(state, dt)
                    return self.driver.hold(state, dt)
                return self._request_replan(state, dt, "path blocked by a newly seen obstacle")
            if self._state is DriveState.APPROACH and dist_goal < b.maneuver_distance:
                self._goto(DriveState.MANEUVER, f"maneuvering into spot {cand.id}")
                if not self._goal_refreshed:
                    self._goal_refreshed = True
                    refreshed = self._refresh_goal(state, dt)
                    if refreshed is not None:
                        return refreshed
        control, done = self._drive(state, dt, decision)
        last = self.driver.follower
        if last.active and self.driver.active:
            seg = self.driver.segments[self.driver.index]
            i = last._idx
            ref = seg.poses[min(i, len(seg.poses) - 1)]
            cte = abs(
                -math.sin(ref[2]) * (state.pose.x - ref[0])
                + math.cos(ref[2]) * (state.pose.y - ref[1])
            )
            if cte > b.replan_tracking_error:
                return self._request_replan(state, dt, "tracking error too large")
        if done and abs(state.speed) < b.stop_speed:
            return self._arrived(state, dt)
        return control

    def _remaining_path_length(self) -> float:
        """Arc length left on the whole path (current segment remainder plus later ones)."""
        rem = self.driver.remaining_poses()
        total = float(np.sum(np.hypot(*np.diff(rem[:, :2], axis=0).T))) if len(rem) > 1 else 0.0
        for seg in self.driver.segments[self.driver.index + 1 :]:
            total += seg.length
        return total

    def _consider_switch(self, state: VehicleState, dist_goal: float) -> ControlOutput | None:
        b = self.cfg.behavior
        assert self._committed is not None
        if self._t - self._commit_time < b.min_commit_time or dist_goal < b.maneuver_lock_distance:
            return None
        best = self._best()
        if best is None or best.candidate.id == self._committed.candidate.id:
            return None
        cur = self._committed.cost
        if not (
            best.cost < cur - b.switch_abs_margin
            and best.cost < cur - b.switch_rel_margin * abs(cur)
        ):
            return None
        path, goal, _ = self._plan_to(best.candidate, state.pose)
        if path is None or goal is None:
            return None
        old = self._committed.candidate.id
        self.telemetry.spot_switches += 1
        self.telemetry.replans += 1
        self._committed = best
        self._commit_time = self._t
        self._replans_here = 0
        self._goal_refreshed = False
        self.telemetry.chosen_spot = best.candidate.id
        self.telemetry.goal = goal
        self.driver.set_path(path, self._checker(), abs(state.speed))
        self.telemetry.path = path
        self._goto(
            DriveState.APPROACH,
            f"switched from {old} to {best.candidate.id}: clearly better "
            f"(cost {best.cost:.0f} vs {self._committed.cost:.0f})",
        )
        return None

    def _refresh_goal(self, state: VehicleState, dt: float) -> ControlOutput | None:
        """Re-measure neighbors up close; re-plan only if the goal moved noticeably."""
        assert self._committed is not None and self.telemetry.goal is not None and self._perc
        cand = self._committed.candidate
        reverse = wants_reverse_in(cand, self.settings, self.cfg.vehicle)
        old = self.telemetry.goal
        new = compute_goal(
            cand,
            reverse_in=abs(angle_diff(old.pose.theta, cand.pose.theta)) > math.pi / 2 or reverse,
            checker=self._checker(),
            vehicle=self.vehicle,
            cfg=self.cfg.goal,
            ranking=self.cfg.ranking,
            obstacle_points=self._perc.grid.occupied_points(),
        )
        if not new.ok or new.pose.distance_to(old.pose) < 0.15:
            return None
        if abs(angle_diff(new.pose.theta, old.pose.theta)) > 0.5:
            return None
        straight = (
            self.cfg.planner.final_straight if cand.layout is SpotLayout.PERPENDICULAR else 0.0
        )
        direction = 1 if abs(angle_diff(new.pose.theta, cand.pose.theta)) < math.pi / 2 else -1
        res = self.planner.plan(state.pose, new.pose, self._checker(), straight, direction)
        self.telemetry.plans += 1
        if res.success and res.path is not None:
            self.telemetry.goal = new
            self.telemetry.replans += 1
            self.driver.set_path(res.path, self._checker(), abs(state.speed))
            self.telemetry.path = res.path
            self.telemetry.reason = f"maneuvering into spot {cand.id} (goal refined up close)"
        return None

    def _arrived(self, state: VehicleState, dt: float) -> ControlOutput:
        assert self._committed is not None and self.telemetry.goal is not None and self._perc
        b = self.cfg.behavior
        goal = self.telemetry.goal
        if self._refines < b.refine_attempts:
            # Re-measure the neighbors and the curb from up close before judging the pose.
            cand = self._committed.candidate
            reverse = abs(angle_diff(goal.pose.theta, cand.pose.theta)) > math.pi / 2
            fresh = compute_goal(
                cand,
                reverse_in=reverse,
                checker=self._checker(self.cfg.safety.check_margin_fraction),
                vehicle=self.vehicle,
                cfg=self.cfg.goal,
                ranking=self.cfg.ranking,
                obstacle_points=self._perc.grid.occupied_points(),
            )
            if fresh.ok:
                goal = fresh
                self.telemetry.goal = fresh
        rel = state.pose.relative_to(goal.pose)
        lat_err, head_err = abs(rel.y), abs(angle_diff(state.pose.theta, goal.pose.theta))
        lon_err = abs(rel.x)
        needs = (
            lat_err > self.cfg.goal.refine_tolerance
            or head_err > self.cfg.goal.refine_heading_tolerance
            or lon_err > 0.3
        )
        if needs and self._refines < b.refine_attempts:
            self._refines += 1
            checker = self._checker()
            if not checker.free(np.array([[state.pose.x, state.pose.y, state.pose.theta]]))[0]:
                checker = self._checker(self.cfg.safety.check_margin_fraction)
            cand = self._committed.candidate
            straight = (
                self.cfg.planner.final_straight if cand.layout is SpotLayout.PERPENDICULAR else 0.0
            )
            direction = 1 if abs(angle_diff(goal.pose.theta, cand.pose.theta)) < math.pi / 2 else -1
            res = self.planner.plan(state.pose, goal.pose, checker, straight, direction)
            self.telemetry.plans += 1
            if res.success and res.path is not None and res.path.length > 0.05:
                self.driver.set_path(res.path, checker, 0.0)
                self.telemetry.path = res.path
                self._goto(
                    DriveState.REFINE,
                    f"straightening up: {lat_err:.2f} m off-center, "
                    f"{math.degrees(head_err):.0f} deg",
                )
                return self.driver.hold(state, dt)
        self.telemetry.parked_pose = state.pose
        msg = f"parked in spot {self._committed.candidate.id}"
        if goal.port_distance is not None:
            msg += f"; charge port {goal.port_distance:.2f} m from charger"
        self._goto(DriveState.PARKED, msg)
        return self.driver.hold(state, dt)

    def _spot_lost(self, state: VehicleState, dt: float, why: str) -> ControlOutput:
        assert self._committed is not None
        self._failures[self._committed.candidate.id] = self.cfg.behavior.max_spot_failures
        self._override = None
        self._goto(DriveState.FALLBACK, f"{why}; trying the next-best spot")
        self._state = DriveState.SELECT
        return self.driver.hold(state, dt)

    def _request_replan(self, state: VehicleState, dt: float, why: str) -> ControlOutput:
        self._goto(DriveState.REPLAN, why)
        return self.driver.hold(state, dt)

    def _replan(self, state: VehicleState, dt: float) -> ControlOutput:
        if abs(state.speed) > self.cfg.behavior.stop_speed:
            return self.driver.hold(state, dt)
        assert self._committed is not None
        self._replans_here += 1
        self.telemetry.replans += 1
        if self._replans_here > self.cfg.behavior.max_replans_per_spot:
            return self._spot_lost(state, dt, "too many re-plans toward this spot")
        path, goal, why = self._plan_to(self._committed.candidate, state.pose)
        if path is None or goal is None:
            return self._spot_lost(state, dt, why)
        self.telemetry.goal = goal
        self.driver.set_path(path, self._checker(), 0.0)
        self.telemetry.path = path
        self._goto(DriveState.APPROACH, f"re-planned to spot {self._committed.candidate.id}")
        return self.driver.hold(state, dt)

    def _enter_yield(
        self, state: VehicleState, dt: float, decision: SafetyDecision
    ) -> ControlOutput:
        if self._state is not DriveState.YIELD:
            self._resume_state = self._state
            self._yield_started = self._t
            self.telemetry.yield_events += 1
            self._goto(DriveState.YIELD, f"yielding to {decision.yield_kind}")
        self._clear_since = None
        res = self.driver.step(state, dt, 0.0, decision.emergency)
        return res.control

    def _yield(self, state: VehicleState, dt: float) -> ControlOutput:
        decision = self._safety(state)
        self.telemetry.yield_time += dt
        if decision.yield_kind is not None:
            self._clear_since = None
            self.telemetry.reason = f"yielding to {decision.yield_kind}"
            if self._t - self._yield_started > self.cfg.behavior.yield_timeout and self._committed:
                return self._request_replan(state, dt, "agent not moving; re-planning around it")
            return self.driver.step(state, dt, 0.0, decision.emergency).control
        if self._clear_since is None:
            self._clear_since = self._t
        if self._t - self._clear_since >= self.cfg.safety.clear_time:
            self._goto(self._resume_state, "path clear, resuming")
            return self.driver.step(state, dt, decision.speed_limit, False).control
        self.telemetry.reason = "agent clearing, about to resume"
        return self.driver.step(state, dt, 0.0, False).control

    def _handback(self, why: str, state: VehicleState | None = None) -> None:
        """Last rung of the fallback ladder: pull over to the aisle edge, stop, hand back."""
        self.telemetry.handback_reason = why
        self.driver.clear()
        if state is not None and self._perc is not None:
            target = state.pose.compose(Pose(5.0, -self.cfg.behavior.pullover_offset, 0.0))
            checker = self._checker()
            if checker.free(np.array([[target.x, target.y, target.theta]]))[0]:
                res = self.planner.plan(state.pose, target, checker)
                self.telemetry.plans += 1
                if res.success and res.path is not None and res.path.gear_changes == 0:
                    self.driver.set_path(res.path, checker, abs(state.speed))
                    self.telemetry.path = res.path
                    why += "; pulling over to the aisle edge"
        self._goto(DriveState.HANDBACK, f"handing back control: {why}")

    def _hold_terminal(self, state: VehicleState, dt: float) -> ControlOutput:
        if self._state is DriveState.HANDBACK and self.driver.active:
            decision = self._safety(state)
            return self.driver.step(state, dt, decision.speed_limit, decision.emergency).control
        return self.driver.hold(state, dt)

    @property
    def settled(self) -> bool:
        """Terminal and no longer executing a pull-over maneuver."""
        return self.done and not (self._state is DriveState.HANDBACK and self.driver.active)
