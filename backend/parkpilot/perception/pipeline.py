"""Perception pipeline: camera rig -> tracks -> occupancy grid and semantic map.

This is the only place ground truth enters the autonomy stack, and only through the camera
simulation. Everything downstream consumes :class:`PerceptionOutput`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parkpilot.core.config import ParkPilotConfig
from parkpilot.core.geometry import Pose
from parkpilot.perception.camera import CameraRig, SensorFrame
from parkpilot.perception.grid import OccupancyGrid
from parkpilot.perception.semantic import SemanticMap
from parkpilot.perception.tracks import Track, Tracker
from parkpilot.world.agents import AgentState
from parkpilot.world.model import World


@dataclass(slots=True)
class PerceptionOutput:
    t: float
    ego: Pose
    frame: SensorFrame
    grid: OccupancyGrid
    tracks: list[Track]
    """All live tracks, confirmed or not."""
    confirmed: list[Track]
    semantic: SemanticMap
    noise_sigma: float
    """Current depth-noise scale: the uncertainty signal used to widen safety margins."""


class Perception:
    def __init__(self, config: ParkPilotConfig, bounds: tuple[float, float, float, float]) -> None:
        self.cfg = config
        self.rig = CameraRig(config.sensor, config.vehicle)
        self.grid = OccupancyGrid(
            config.grid, bounds, config.vehicle, config.sensor.low_blind_range
        )
        self.tracker = Tracker(config.tracker)
        self.semantic = SemanticMap()

    def update(
        self,
        t: float,
        ego: Pose,
        world: World,
        agents: list[AgentState],
        rng: np.random.Generator,
    ) -> PerceptionOutput:
        frame = self.rig.sense(t, ego, world, agents, rng)
        pts = frame.hit_points
        prior = self.grid.logodds_at(pts)
        prior_free = prior <= self.cfg.tracker.free_logodds
        prior_occupied = prior > self.cfg.grid.occupied_threshold
        moving = self.tracker.update(t, pts, frame.cues, prior_free, prior_occupied)
        # Every hit goes into the grid, movers included (wrongly excluding a static object
        # would erase it); movers' cells are marked transient so their trail fades.
        self.grid.update(frame, transient_hits=moving)
        self.semantic.update(frame)
        return PerceptionOutput(
            t=t,
            ego=ego,
            frame=frame,
            grid=self.grid,
            tracks=list(self.tracker.tracks),
            confirmed=self.tracker.confirmed(),
            semantic=self.semantic,
            noise_sigma=frame.noise_sigma,
        )
