"""Fixed-timestep simulation tying the world, agents, perception and an autonomy stack.

Fully headless and deterministic for a given (scenario, system, seed): every random draw
comes from one seeded generator, and planning has expansion budgets instead of wall-clock
limits. The API server and the benchmark are both just consumers of this class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from parkpilot.behavior.types import (
    Abort,
    AutonomyCommand,
    AutonomyStack,
    DriveState,
    OverrideSpot,
)
from parkpilot.core.config import ParkPilotConfig, UserSettings
from parkpilot.core.geometry import Pose, polygons_intersect, rect_polygon
from parkpilot.perception.pipeline import Perception, PerceptionOutput
from parkpilot.scenarios.schema import Scenario, agent_specs, build_world
from parkpilot.sim.truth import TruthMonitor, TruthSample
from parkpilot.vehicle.model import VehicleModel, VehicleState
from parkpilot.world.agents import AgentSet, AgentState

DT = 0.05
"""Control and physics period (s)."""
PERCEPTION_EVERY = 2
"""Perception runs every N ticks (10 Hz)."""


@dataclass(slots=True)
class TickRecord:
    t: float
    x: float
    y: float
    theta: float
    speed: float
    accel: float
    steer: float
    state: DriveState


@dataclass(slots=True)
class RunLog:
    ticks: list[TickRecord] = field(default_factory=list)
    truth: list[tuple[float, TruthSample]] = field(default_factory=list)
    phantom_frames: list[float] = field(default_factory=list)
    """Times of perception frames that contained injected false positives."""
    hazard_frames: list[tuple[float, float, float]] = field(default_factory=list)
    """(t, ego speed, ground-truth time-to-contact with the nearest agent ahead)."""
    hard_brake_times: list[float] = field(default_factory=list)
    overrides: list[tuple[float, str]] = field(default_factory=list)


AutonomyFactory = Callable[
    [ParkPilotConfig, Scenario, UserSettings, list[tuple[float, float]]], AutonomyStack
]


class Simulation:
    def __init__(
        self,
        scenario: Scenario,
        autonomy_factory: AutonomyFactory,
        config: ParkPilotConfig,
        seed: int = 0,
        settings: UserSettings | None = None,
        remembered: list[tuple[float, float]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.world = build_world(scenario)
        self.agents = AgentSet.from_specs(config.agents, agent_specs(scenario))
        self.vehicle = VehicleModel(config.vehicle)
        self.state = VehicleState(scenario.start_pose())
        self.settings = settings or scenario.user
        self.perception = Perception(config, scenario.bounds)
        self.autonomy = autonomy_factory(config, scenario, self.settings, list(remembered or []))
        self.truth = TruthMonitor(self.world, self.vehicle)
        self.log = RunLog()
        self.t = 0.0
        self.tick = 0
        self.last_perception: PerceptionOutput | None = None
        self._scripted = sorted(scenario.scripted, key=lambda c: c.t)
        self._last_hard_brakes = 0
        self.override_listeners: list[Callable[[float, float], None]] = []

    # ------------------------------------------------------------------ control
    @property
    def done(self) -> bool:
        settled = bool(getattr(self.autonomy, "settled", self.autonomy.done))
        stopped = abs(self.state.speed) < 0.05
        return (settled and stopped) or self.t >= self.scenario.max_time

    def command(self, cmd: AutonomyCommand) -> str:
        msg = self.autonomy.command(cmd)
        if isinstance(cmd, OverrideSpot):
            self.log.overrides.append((self.t, cmd.spot_id))
            if msg.startswith("rerouting"):
                cand = next(
                    (c for c in self.autonomy.telemetry.candidates if c.id == cmd.spot_id), None
                )
                if cand is not None:
                    for listener in self.override_listeners:
                        listener(cand.pose.x, cand.pose.y)
        return msg

    def abort(self) -> str:
        return self.command(Abort())

    def _run_scripts(self) -> None:
        while self._scripted and self._scripted[0].t <= self.t:
            sc = self._scripted.pop(0)
            px, py = sc.position
            cands = self.autonomy.telemetry.candidates
            if not cands:
                self._scripted.insert(0, sc)
                return
            best = min(cands, key=lambda c: (c.pose.x - px) ** 2 + (c.pose.y - py) ** 2)
            if (best.pose.x - px) ** 2 + (best.pose.y - py) ** 2 > 2.0**2:
                # The driver taps a spot only once the car has seen it: retry next tick.
                self._scripted.insert(0, sc)
                return
            msg = self.command(OverrideSpot(best.id))
            if not msg.startswith("rerouting") and self.t < sc.t + 15.0:
                self._scripted.insert(0, sc)  # not seen well enough yet: tap again later
                return

    # ------------------------------------------------------------------ stepping
    def step(self) -> None:
        if self.done:
            return
        ego_polys = self.vehicle.footprint(self.state.pose)
        self.agents.step(self.t, DT, self.state.pose, ego_polys)
        perc = None
        if self.tick % PERCEPTION_EVERY == 0:
            perc = self.perception.update(
                self.t, self.state.pose, self.world, self.agents.active(), self.rng
            )
            self.last_perception = perc
            self._record_truth(perc)
        self._run_scripts()
        ctrl = self.autonomy.step(self.t, DT, self.state, perc)
        self.state = self.vehicle.step(self.state, ctrl.accel, ctrl.steer, DT, ctrl.gear)
        hb = self.autonomy.telemetry.hard_brakes
        if hb > self._last_hard_brakes:
            self.log.hard_brake_times.append(self.t)
            self._last_hard_brakes = hb
        self.t = round(self.t + DT, 6)
        self.tick += 1
        p = self.state.pose
        self.log.ticks.append(
            TickRecord(
                self.t,
                p.x,
                p.y,
                p.theta,
                self.state.speed,
                self.state.accel,
                self.state.steer,
                self.autonomy.telemetry.state,
            )
        )

    def _record_truth(self, perc: PerceptionOutput) -> None:
        agents = self.agents.active()
        sample = self.truth.sample(self.state.pose, [a.polygon() for a in agents])
        self.log.truth.append((self.t, sample))
        if perc.frame.truth_phantom.any():
            self.log.phantom_frames.append(self.t)
        self.log.hazard_frames.append((self.t, self.state.speed, self._truth_ttc(agents)))

    def _truth_ttc(self, agents: list[AgentState]) -> float:
        """Ground-truth time to contact with an agent along the ego's current motion."""
        if not agents:
            return float("inf")
        v = self.state.speed
        acc = self.state.accel
        braking = v * acc < 0
        best = float("inf")
        for k in range(0, 31):
            tt = k * 0.1
            # Constant-acceleration prediction: a car that is already braking stops.
            if braking:
                t_stop = abs(v / acc)
                te = min(tt, t_stop)
                travel = v * te + 0.5 * acc * te * te
            else:
                travel = v * tt
            pose = self.state.pose.advanced(travel)
            foot = self.vehicle.footprint(pose)
            for ag in agents:
                vel = ag.velocity
                ap = Pose(ag.pose.x + vel[0] * tt, ag.pose.y + vel[1] * tt, ag.pose.theta)
                poly = rect_polygon(ap, ag.length, ag.width)
                if any(polygons_intersect(f, poly) for f in foot):
                    return tt
        return best

    def run(self, max_ticks: int | None = None) -> RunLog:
        n = 0
        while not self.done and (max_ticks is None or n < max_ticks):
            self.step()
            n += 1
        return self.log
