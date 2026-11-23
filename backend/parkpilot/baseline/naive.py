"""A deliberately naive, complaint-reproducing parking stack for A/B comparison.

It shares the simulator, sensors, controller and vehicle with ParkPilot, but makes the
choices the owner complaints describe:

* takes the **first** free painted stall it sees (no destination ranking, no legality
  checks beyond painted lines, no line-less inference) (C4, C5, C6);
* plans with **no gear-change penalty** and **re-plans on a fixed timer** from wherever
  it is, whether or not the old plan was still fine (C1);
* only sees the **current** camera frame: no occupancy memory, so low objects vanish once
  inside the near-field blind zone (C7);
* stops for **any** raw detection in its corridor: no persistence filter (C8);
* crawls at a constant speed (C2) and centres on the painted lines only, nose-in (C3, C11);
* has no fallback ladder: if nothing works it just stops where it is (C14).
"""

from __future__ import annotations

import math

from parkpilot.behavior.driving import Driver, route_path
from parkpilot.behavior.machine import clearance_readouts
from parkpilot.behavior.types import (
    TERMINAL,
    Abort,
    AutonomyCommand,
    ControlOutput,
    DriveState,
    OverrideSpot,
    Telemetry,
    Transition,
)
from parkpilot.core.config import ParkPilotConfig, UserSettings
from parkpilot.core.geometry import Pose
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.planning.collision import CollisionChecker
from parkpilot.planning.goal import GoalResult
from parkpilot.planning.hybrid_astar import HybridAStar
from parkpilot.safety.supervisor import SafetySupervisor
from parkpilot.spots.detect import SpotCandidate, SpotDetector
from parkpilot.spots.rank import RankedSpot, SideRoom
from parkpilot.vehicle.model import VehicleModel, VehicleState

REPLAN_PERIOD = 2.0
"""The baseline re-plans on this timer whether or not its plan is still valid (s)."""


def baseline_config(base: ParkPilotConfig | None = None) -> ParkPilotConfig:
    """ParkPilot's configuration with the naive choices switched on."""
    cfg = base or ParkPilotConfig()
    return cfg.model_copy(
        update={
            # Memoryless grid: rebuilt from the current frame, so a single hit is occupied.
            "grid": cfg.grid.model_copy(update={"memory": False, "occupied_threshold": 0.5}),
            "safety": cfg.safety.model_copy(
                update={"persistence_filter": False, "sigma_gain": 0.0, "base_margin": 0.05}
            ),
            "speed": cfg.speed.model_copy(update={"constant_speed": 1.0}),
        }
    )


class NaiveAutonomy:
    name = "baseline"

    def __init__(
        self,
        config: ParkPilotConfig,
        *,
        route: list[tuple[float, float]],
        settings: UserSettings,
    ) -> None:
        self.cfg = config
        self.vehicle = VehicleModel(config.vehicle)
        self.route = route
        self.settings = settings
        self.detector = SpotDetector(config.spots, route)
        self.planner = HybridAStar(
            config.planner.model_copy(
                update={
                    "gear_change_cost": 0.0,
                    "reverse_multiplier": 1.0,
                    "clearance_cost": 0.0,
                    "steer_change_cost": 0.0,
                    "max_expansions": 3000,
                }
            ),
            self.vehicle,
        )
        self._last_plan = -1e9
        self.supervisor = SafetySupervisor(config.safety, config.tracker, self.vehicle)
        self.driver = Driver(config.control, config.speed, self.vehicle)
        self.telemetry = Telemetry()
        self._state = DriveState.SEARCH
        self._t = 0.0
        self._perc: PerceptionOutput | None = None
        self._target: SpotCandidate | None = None
        self._goal: Pose | None = None
        self._seen_order: list[str] = []
        self._started = False
        self._last_emergency = False

    @property
    def done(self) -> bool:
        return self._state in TERMINAL

    def command(self, cmd: AutonomyCommand) -> str:
        if isinstance(cmd, Abort):
            self._goto(DriveState.HANDBACK, "aborted")
            self.telemetry.handback_reason = "aborted by the driver"
            self.driver.clear()
            return "aborting"
        assert isinstance(cmd, OverrideSpot)
        for c in self.telemetry.candidates:
            if c.id == cmd.spot_id:
                self._set_target(c)
                return f"rerouting to spot {cmd.spot_id}"
        return f"spot {cmd.spot_id} is not known"

    def _goto(self, dst: DriveState, reason: str) -> None:
        if dst is not self._state:
            self.telemetry.transitions.append(Transition(self._t, self._state, dst, reason))
        self._state = dst
        self.telemetry.state = dst
        self.telemetry.reason = reason

    def _set_target(self, cand: SpotCandidate) -> None:
        self._target = cand
        center = Pose(cand.pose.x, cand.pose.y, cand.pose.theta)  # nose-in, painted centre
        self._goal = self.vehicle.rear_axle_from_center(center)
        self.telemetry.chosen_spot = cand.id
        self.telemetry.goal = GoalResult(True, self._goal, center, reasons=["centered on lines"])
        self._goto(DriveState.MANEUVER, f"going to first free spot {cand.id}")

    def step(
        self, t: float, dt: float, state: VehicleState, perc: PerceptionOutput | None
    ) -> ControlOutput:
        self._t = t
        if not self._started:
            self._started = True
            self.driver.set_path(route_path(state.pose, list(self.route[1:])), None)
        if self._state in TERMINAL:
            return self.driver.hold(state, dt)
        if perc is not None:
            self._perc = perc
            self.telemetry.clearances = clearance_readouts(perc, state.pose, self.vehicle)
            self._perceive(perc)
            if self._state is DriveState.MANEUVER:
                self._replan(state)
        if self._perc is None:
            return self.driver.hold(state, dt)
        checker = CollisionChecker(self._perc.grid, self.vehicle, self.cfg.safety.base_margin)
        decision = self.supervisor.evaluate(
            state, self.driver.remaining_poses(), self._perc, checker
        )
        self.telemetry.safety = decision
        if decision.emergency and not self._last_emergency:
            self.telemetry.hard_brakes += 1
        self._last_emergency = decision.emergency
        res = self.driver.step(state, dt, decision.speed_limit, decision.emergency)
        if self._state is DriveState.MANEUVER and self._goal is not None:
            if state.pose.distance_to(self._goal) < 0.25 and abs(state.speed) < 0.05:
                self.telemetry.parked_pose = state.pose
                self._goto(
                    DriveState.PARKED, f"parked in spot {self._target.id if self._target else '?'}"
                )
        elif self._state is DriveState.SEARCH and res.path_done and abs(state.speed) < 0.05:
            # No fallback ladder: it simply stops at the end of the aisle.
            self.telemetry.reason = "stopped: no spot found"
        return res.control

    def _perceive(self, perc: PerceptionOutput) -> None:
        cands = self.detector.detect(perc, use_gaps=False)
        for c in cands:
            if c.id not in self._seen_order:
                self._seen_order.append(c.id)
        order = {cid: i for i, cid in enumerate(self._seen_order)}
        cands.sort(key=lambda c: order[c.id])
        self.telemetry.candidates = cands
        self.telemetry.ranked = [
            RankedSpot(
                c,
                float(order[c.id]),
                not c.occupied and c.observed_fraction >= 0.35,
                {},
                ["first seen"] if not c.occupied else ["occupied"],
                SideRoom(c.width / 2, c.width / 2),
            )
            for c in cands
        ]
        if self._state is DriveState.SEARCH:
            first = next((r.candidate for r in self.telemetry.ranked if r.eligible), None)
            if first is not None:
                self._set_target(first)
        elif self._target is not None:
            fresh = next((c for c in cands if c.id == self._target.id), None)
            if fresh is not None and fresh.occupied:
                # Greedy: hop to whatever free spot comes first now.
                nxt = next((r.candidate for r in self.telemetry.ranked if r.eligible), None)
                if nxt is not None:
                    self.telemetry.spot_switches += 1
                    self._set_target(nxt)

    def _replan(self, state: VehicleState) -> None:
        """Greedy re-planning on a fixed timer, toward the painted-line centre.

        Hybrid A* with the naive cost (no gear-change penalty, no reverse penalty, no
        clearance term) on a memoryless grid. Re-planning every ``REPLAN_PERIOD`` seconds
        regardless of whether the old plan was fine is what produces the "parking dance".
        (Re-planning every perception frame would be even worse, but makes benchmark runs
        too slow in pure Python; see docs/BENCHMARK.md.)
        """
        assert self._perc is not None and self._goal is not None
        if state.pose.distance_to(self._goal) < 0.25:
            return
        if self._t - self._last_plan < REPLAN_PERIOD and self.driver.active:
            return
        self._last_plan = self._t
        checker = CollisionChecker(self._perc.grid, self.vehicle, self.cfg.safety.base_margin)
        res = self.planner.plan(state.pose, self._goal, checker)
        self.telemetry.plans += 1
        if res.success and res.path is not None:
            self.driver.set_path(res.path, None, abs(state.speed))
            self.telemetry.path = res.path
            self.telemetry.replans += 1
            return
        # Nothing works from here: keep crawling along the aisle and try again.
        self.driver.set_path(
            route_path(state.pose, self._route_ahead(state.pose)), None, abs(state.speed)
        )
        self.telemetry.path = self.driver.path

    def _route_ahead(self, pose: Pose) -> list[tuple[float, float]]:
        ahead = [
            p
            for p in self.route
            if (p[0] - pose.x) * math.cos(pose.theta) + (p[1] - pose.y) * math.sin(pose.theta) > 1.0
        ]
        return ahead or [self.route[-1]]
