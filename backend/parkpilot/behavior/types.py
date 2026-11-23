"""Types shared by every autonomy stack (ParkPilot and the naive baseline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from parkpilot.core.geometry import Pose
from parkpilot.perception.pipeline import PerceptionOutput
from parkpilot.planning.goal import GoalResult
from parkpilot.planning.hybrid_astar import Path
from parkpilot.safety.supervisor import SafetyDecision
from parkpilot.spots.detect import SpotCandidate
from parkpilot.spots.rank import RankedSpot
from parkpilot.vehicle.model import VehicleState


class DriveState(StrEnum):
    SEARCH = "SEARCH"
    SELECT = "SELECT"
    APPROACH = "APPROACH"
    MANEUVER = "MANEUVER"
    REFINE = "REFINE"
    PARKED = "PARKED"
    YIELD = "YIELD"
    REPLAN = "REPLAN"
    FALLBACK = "FALLBACK"
    HANDBACK = "HANDBACK"


TERMINAL = frozenset({DriveState.PARKED, DriveState.HANDBACK})


@dataclass(frozen=True, slots=True)
class Transition:
    t: float
    src: DriveState
    dst: DriveState
    reason: str


@dataclass(frozen=True, slots=True)
class ControlOutput:
    accel: float
    steer: float
    gear: int


@dataclass(frozen=True, slots=True)
class OverrideSpot:
    """The driver picked a spot on the HMI."""

    spot_id: str


@dataclass(frozen=True, slots=True)
class Abort:
    """The driver aborted: stop safely and hand back control."""


AutonomyCommand = OverrideSpot | Abort


@dataclass(slots=True)
class Telemetry:
    state: DriveState = DriveState.SEARCH
    reason: str = "starting"
    chosen_spot: str | None = None
    goal: GoalResult | None = None
    candidates: list[SpotCandidate] = field(default_factory=list)
    ranked: list[RankedSpot] = field(default_factory=list)
    path: Path | None = None
    segment_index: int = 0
    transitions: list[Transition] = field(default_factory=list)
    replans: int = 0
    spot_switches: int = 0
    plans: int = 0
    clearances: dict[str, float] = field(default_factory=dict)
    safety: SafetyDecision | None = None
    handback_reason: str | None = None
    override_ack: str | None = None
    hard_brakes: int = 0
    """Emergency stops commanded by the safety layer."""
    yield_events: int = 0
    yield_time: float = 0.0
    parked_pose: Pose | None = None


class AutonomyStack(Protocol):
    name: str
    telemetry: Telemetry

    def step(
        self, t: float, dt: float, state: VehicleState, perc: PerceptionOutput | None
    ) -> ControlOutput: ...

    def command(self, cmd: AutonomyCommand) -> str: ...

    @property
    def done(self) -> bool: ...
