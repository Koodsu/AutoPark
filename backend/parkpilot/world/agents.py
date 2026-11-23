"""Dynamic agents: pedestrians, carts and cars backing out of stalls (C12).

Agents follow scripted waypoints. They are deliberately simple but not adversarial: as a
last resort they stop if the ego vehicle is right in front of them, so a stationary ego is
never "hit" by a pedestrian walking through it. The ego must still predict and yield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from parkpilot.core.config import AgentConfig
from parkpilot.core.geometry import FloatArray, Pose, polygons_intersect, rect_polygon

AgentKind = Literal["pedestrian", "cart", "car"]


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str
    kind: AgentKind
    waypoints: tuple[tuple[float, float], ...]
    speed: float
    start_time: float = 0.0
    loop: bool = False
    reversing: bool = False
    trigger_distance: float | None = None


@dataclass(slots=True)
class AgentState:
    spec: AgentSpec
    pose: Pose
    length: float
    width: float
    speed: float = 0.0
    target: int = 1
    started: bool = False
    active: bool = True
    waiting: bool = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def reverse_lights(self) -> bool:
        """The visual intent cue of a car backing out."""
        return self.spec.kind == "car" and self.spec.reversing and self.started and self.target == 1

    @property
    def velocity(self) -> FloatArray:
        if not self.started or self.waiting or not self.active:
            return np.zeros(2)
        direction = -1.0 if self.reverse_lights else 1.0
        return np.array(
            [
                direction * self.speed * math.cos(self.pose.theta),
                direction * self.speed * math.sin(self.pose.theta),
            ]
        )

    def polygon(self) -> FloatArray:
        return rect_polygon(self.pose, self.length, self.width)


def _initial_heading(spec: AgentSpec) -> float:
    (x0, y0), (x1, y1) = spec.waypoints[0], spec.waypoints[1]
    th = math.atan2(y1 - y0, x1 - x0)
    return th + math.pi if spec.reversing else th


@dataclass(slots=True)
class AgentSet:
    config: AgentConfig
    agents: list[AgentState] = field(default_factory=list)

    @classmethod
    def from_specs(cls, config: AgentConfig, specs: list[AgentSpec]) -> AgentSet:
        sizes = {
            "pedestrian": config.pedestrian_size,
            "cart": config.cart_size,
            "car": config.car_size,
        }
        out = cls(config)
        for spec in specs:
            if len(spec.waypoints) < 2:
                raise ValueError(f"agent {spec.id} needs at least two waypoints")
            ln, wd = sizes[spec.kind]
            x, y = spec.waypoints[0]
            out.agents.append(AgentState(spec, Pose(x, y, _initial_heading(spec)), ln, wd))
        return out

    def active(self) -> list[AgentState]:
        return [a for a in self.agents if a.active]

    def step(self, t: float, dt: float, ego_pose: Pose, ego_polys: list[FloatArray]) -> None:
        for a in self.agents:
            if a.active:
                self._step_agent(a, t, dt, ego_pose, ego_polys)

    def _step_agent(
        self, a: AgentState, t: float, dt: float, ego_pose: Pose, ego_polys: list[FloatArray]
    ) -> None:
        spec = a.spec
        if not a.started:
            near = spec.trigger_distance is None or (
                ego_pose.distance_to(Pose(*spec.waypoints[0], 0.0)) <= spec.trigger_distance
            )
            if t >= spec.start_time and near:
                a.started = True
                a.speed = spec.speed
            else:
                return
        tx, ty = spec.waypoints[a.target]
        dx, dy = tx - a.pose.x, ty - a.pose.y
        dist = math.hypot(dx, dy)
        move_dir = math.atan2(dy, dx) if dist > 1e-9 else a.pose.theta
        backing = a.reverse_lights
        heading = move_dir + math.pi if backing else move_dir
        # Polite last resort: don't walk or drive into the ego vehicle.
        step_len = min(dist, spec.speed * dt)
        reach = self.config.yield_lookahead + step_len
        look = Pose(a.pose.x, a.pose.y, move_dir).compose(Pose(a.length / 2 + reach / 2, 0.0, 0.0))
        probe = rect_polygon(look, reach, a.width)
        a.waiting = any(polygons_intersect(probe, p) for p in ego_polys)
        if a.waiting:
            return
        nx = a.pose.x + step_len * math.cos(move_dir)
        ny = a.pose.y + step_len * math.sin(move_dir)
        a.pose = Pose(nx, ny, heading if dist > 1e-9 else a.pose.theta)
        if dist - step_len <= self.config.waypoint_tolerance:
            a.target += 1
            if a.target >= len(spec.waypoints):
                if spec.loop:
                    a.target = 0
                else:
                    a.active = False
