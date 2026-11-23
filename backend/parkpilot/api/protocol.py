"""Wire protocol: the single source of truth for messages between backend and HMI.

``parkpilot export-schema`` writes the JSON Schema of :class:`ProtocolSchema`; the frontend
generates its TypeScript types from it (``frontend/src/generated``) and CI fails on drift.
Bump :data:`PROTOCOL_VERSION` on any breaking change.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from parkpilot.core.config import UserSettings

PROTOCOL_VERSION = 1

Vec2 = tuple[float, float]
Polygon = list[Vec2]


class _Wire(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PoseModel(_Wire):
    x: float
    y: float
    theta: float


class ScenarioSummary(_Wire):
    id: str
    title: str
    complaints: list[str]
    description: str
    location_id: str
    lighting: float


class GridInfo(_Wire):
    origin_x: float
    origin_y: float
    resolution: float
    nx: int
    ny: int


class VehicleInfo(_Wire):
    length: float
    width: float
    mirror_width: float
    wheelbase: float
    rear_overhang: float


class SceneMessage(_Wire):
    """Sent once when a client connects: everything static about the run."""

    type: Literal["scene"] = "scene"
    protocol_version: int = PROTOCOL_VERSION
    run_id: str
    system: str
    scenario: ScenarioSummary
    bounds: tuple[float, float, float, float]
    entrance: Vec2
    grid: GridInfo
    vehicle: VehicleInfo


class EgoView(_Wire):
    pose: PoseModel
    speed: float
    steer: float
    footprint: list[Polygon]


class SpotView(_Wire):
    id: str
    polygon: Polygon
    layout: str
    spot_type: str
    legal: bool
    occupied: bool
    eligible: bool
    rank: int
    cost: float
    reasons: list[str]
    chosen: bool
    source: str


class TrackView(_Wire):
    id: int
    x: float
    y: float
    vx: float
    vy: float
    extent: float
    confirmed: bool
    dynamic: bool
    kind: str


class GridPatch(_Wire):
    """Occupancy grid as base64 uint8, row-major from the origin (0 unknown, 1-255 = p)."""

    version: int
    data: str


class TransitionView(_Wire):
    t: float
    src: str
    dst: str
    reason: str


class LiveMetrics(_Wire):
    elapsed: float
    speed: float
    gear_changes: int
    replans: int
    spot_switches: int
    yield_events: int
    hard_brakes: int


class DebugObject(_Wire):
    id: str
    kind: str
    polygon: Polygon


class DebugLayer(_Wire):
    """Ground truth, only sent when the HMI's debug toggle is on."""

    obstacles: list[DebugObject]
    agents: list[DebugObject]
    spots: list[DebugObject]
    zones: list[DebugObject]


class FrameMessage(_Wire):
    type: Literal["frame"] = "frame"
    protocol_version: int = PROTOCOL_VERSION
    seq: int
    t: float
    paused: bool
    speed_factor: float
    system: str
    state: str
    reason: str
    ego: EgoView
    chosen_spot: str | None
    goal: PoseModel | None
    path: list[tuple[float, float, int]]
    spots: list[SpotView]
    tracks: list[TrackView]
    grid: GridPatch | None
    clearances: dict[str, float]
    metrics: LiveMetrics
    transitions: list[TransitionView]
    safety_reason: str
    handback_reason: str | None
    override_ack: str | None
    done: bool
    debug: DebugLayer | None


class AckMessage(_Wire):
    type: Literal["ack"] = "ack"
    message: str


class ErrorMessage(_Wire):
    type: Literal["error"] = "error"
    message: str


ServerMessage = Annotated[
    SceneMessage | FrameMessage | AckMessage | ErrorMessage, Field(discriminator="type")
]


class OverrideSpotCommand(_Wire):
    type: Literal["override_spot"] = "override_spot"
    spot_id: str


class PauseCommand(_Wire):
    type: Literal["pause"] = "pause"


class ResumeCommand(_Wire):
    type: Literal["resume"] = "resume"


class AbortCommand(_Wire):
    type: Literal["abort"] = "abort"


class SetSpeedCommand(_Wire):
    type: Literal["set_speed"] = "set_speed"
    factor: float = Field(gt=0.0, le=16.0)


class SetDebugCommand(_Wire):
    type: Literal["set_debug"] = "set_debug"
    enabled: bool


ClientCommand = Annotated[
    OverrideSpotCommand
    | PauseCommand
    | ResumeCommand
    | AbortCommand
    | SetSpeedCommand
    | SetDebugCommand,
    Field(discriminator="type"),
]


class StartRunRequest(_Wire):
    scenario_id: str
    system: Literal["parkpilot", "baseline"] = "parkpilot"
    seed: int = 0
    settings: UserSettings | None = None


class RunInfo(_Wire):
    run_id: str
    ws_path: str
    scenario_id: str
    system: str


class ConfigResponse(_Wire):
    protocol_version: int
    systems: list[str]
    config: dict[str, object]


class FavoriteView(_Wire):
    x: float
    y: float
    count: int


class LocationPreferenceView(_Wire):
    location_id: str
    favorites: list[FavoriteView]


class ProtocolSchema(_Wire):
    """Envelope that pulls every wire type into one JSON Schema document."""

    server_message: ServerMessage
    client_command: ClientCommand
    start_run_request: StartRunRequest
    run_info: RunInfo
    scenario_summary: ScenarioSummary
    config_response: ConfigResponse
    user_settings: UserSettings
    location_preference: LocationPreferenceView


def export_schema() -> str:
    schema = ProtocolSchema.model_json_schema()
    schema["title"] = "ParkPilotProtocol"
    schema["description"] = f"ParkPilot wire protocol v{PROTOCOL_VERSION}"
    return json.dumps(schema, indent=2, sort_keys=True)
