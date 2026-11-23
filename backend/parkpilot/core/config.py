"""Typed configuration models.

Every tunable in the stack lives in one of these pydantic models with a documented default,
so there are no magic numbers in algorithm code. ``ParkPilotConfig`` aggregates them; each
subsystem receives only the section it needs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    """Immutable, strict base for config sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class VehicleConfig(FrozenModel):
    """Geometry and limits of the ego vehicle. Reference point is the rear-axle centre."""

    length: float = 4.75
    """Bumper-to-bumper length (m)."""
    width: float = 1.92
    """Body width without mirrors (m)."""
    wheelbase: float = 2.88
    """Rear axle to front axle (m)."""
    rear_overhang: float = 0.95
    """Rear axle to rear bumper (m)."""
    mirror_width: float = 2.12
    """Tip-to-tip width across the folded-out mirrors (m)."""
    mirror_from_front: float = 1.75
    """Front bumper to the mirror centre line (m)."""
    mirror_length: float = 0.22
    """Longitudinal size of a mirror housing (m)."""
    max_steer: float = 0.60
    """Maximum road-wheel steering angle (rad); gives ~4.2 m minimum rear-axle radius."""
    max_steer_rate: float = 0.9
    """Steering actuator rate limit (rad/s)."""
    max_speed: float = 4.5
    """Speed cap inside a parking lot (m/s)."""
    max_reverse_speed: float = 2.0
    """Speed cap while reversing (m/s)."""
    max_accel: float = 1.5
    """Comfortable acceleration limit (m/s^2)."""
    max_decel: float = 2.5
    """Comfortable deceleration limit (m/s^2)."""
    emergency_decel: float = 6.0
    """Deceleration available to the safety supervisor (m/s^2)."""
    charge_port_x: float = -0.62
    """Charge port longitudinal position relative to the rear axle (m, + forward)."""
    charge_port_y: float = 0.96
    """Charge port lateral position (m, + left). Default: rear-left, like many EVs."""
    collision_circles: int = 5
    """Number of disks used to approximate the footprint in the planner's fast check."""


class AgentConfig(FrozenModel):
    """Ground-truth behavior of simulated pedestrians, carts and cars."""

    pedestrian_size: tuple[float, float] = (0.5, 0.5)
    cart_size: tuple[float, float] = (1.0, 0.6)
    car_size: tuple[float, float] = (4.6, 1.9)
    yield_lookahead: float = 0.8
    """Agents stop if the ego is within this distance ahead of them (polite last resort)."""
    waypoint_tolerance: float = 0.05
    """Distance at which a waypoint counts as reached (m)."""


class CameraConfig(FrozenModel):
    """One camera rigidly mounted on the ego (vehicle frame, rear-axle origin)."""

    name: str
    x: float
    y: float
    yaw: float
    """Optical axis direction relative to the vehicle heading (rad)."""
    fov: float
    """Horizontal field of view (rad)."""
    max_range: float
    """Daylight detection range (m)."""


def _default_cameras() -> tuple[CameraConfig, ...]:
    return (
        CameraConfig(name="front", x=2.6, y=0.0, yaw=0.0, fov=2.1, max_range=32.0),
        CameraConfig(name="rear", x=-0.9, y=0.0, yaw=3.14159, fov=2.1, max_range=20.0),
        CameraConfig(name="left", x=1.2, y=0.95, yaw=1.5708, fov=1.9, max_range=16.0),
        CameraConfig(name="right", x=1.2, y=-0.95, yaw=-1.5708, fov=1.9, max_range=16.0),
    )


class SensorConfig(FrozenModel):
    """Camera-only perception model (C7, C8). No ultrasonics, no lidar."""

    cameras: tuple[CameraConfig, ...] = Field(default_factory=_default_cameras)
    ray_resolution: float = 0.025
    """Angular spacing of depth rays (rad, ~1.4 deg)."""
    near_blind_range: float = 0.25
    """Nothing is seen closer than this to the body: below the cameras' view (m)."""
    low_blind_range: float = 1.4
    """Low objects (curbs, bollards, snow piles) vanish inside this distance of the body (m)."""
    range_noise_base: float = 0.02
    """Depth noise standard deviation at zero range in daylight (m)."""
    range_noise_per_m: float = 0.006
    """Additional depth noise per metre of range (stereo-like growth)."""
    night_noise_gain: float = 2.5
    """Noise multiplier added at lighting 0 (1 + gain * (1 - lighting))."""
    night_range_factor: float = 0.55
    """Fraction of daylight range available at lighting 0."""
    dropout: float = 0.03
    """Probability that a ray returns nothing in daylight."""
    night_dropout: float = 0.12
    """Additional dropout probability at lighting 0."""
    clutter_rate: float = 0.04
    """Per-frame probability of one random clutter detection (noise, glare)."""
    feature_read_prob: float = 0.85
    """Probability of reading a visible sign or paint symbol in daylight at close range."""
    feature_range: float = 14.0
    """Range beyond which signs and paint cannot be read (m)."""
    line_noise: float = 0.05
    """Painted-line endpoint noise (m)."""
    paint_night_penalty: float = 0.35
    """Read-probability reduction for paint (not signs) at lighting 0."""


class GridConfig(FrozenModel):
    """Log-odds occupancy grid with temporal memory (C7)."""

    resolution: float = 0.15
    """Cell size (m)."""
    margin: float = 2.0
    """Extra border around the lot bounds (m)."""
    l_occ: float = 0.85
    """Log-odds increment for a cell containing a hit."""
    l_free: float = -0.35
    """Log-odds increment for a cell a ray passed through."""
    l_min: float = -2.0
    l_max: float = 3.5
    occupied_threshold: float = 1.0
    """Log-odds above which a cell counts as occupied (two consistent hits)."""
    free_margin: float = 0.35
    """Free-space carving stops this far short of a hit, so depth noise cannot erode thin
    surfaces such as bumpers and pillars (m)."""
    free_trust_range: float = 12.0
    """Free-space evidence beyond this range is ignored (stereo depth is unreliable)."""
    memory: bool = True
    """Keep cells when they leave the field of view (object permanence). The baseline
    disables this and rebuilds the grid from the current frame only."""
    decay_per_s: float = 0.0
    """Optional log-odds decay toward unknown for unobserved cells (0 = perfect memory)."""
    distance_cap: float = 3.0
    """The distance field is exact up to this distance and saturates beyond it (m)."""


class TrackerConfig(FrozenModel):
    """Clustering and persistence filtering of detections into tracks (C8, C12)."""

    cluster_radius: float = 0.6
    """Hits closer than this belong to the same cluster (m)."""
    max_extent: float = 6.5
    """Clusters larger than this are structure (walls, rows), not trackable objects (m)."""
    gate: float = 1.5
    """Association gate around a track's predicted position (m)."""
    confirm_frames: int = 3
    """Consecutive frames a track must be seen before it is confirmed (persistence filter)."""
    max_misses: int = 4
    """Frames a track may go unseen before deletion."""
    alpha: float = 0.5
    beta: float = 0.25
    dynamic_speed: float = 0.35
    """Speed above which a confirmed track is treated as a moving agent (m/s)."""
    moving_frames: int = 3
    """Consecutive frames above ``dynamic_speed`` needed to call a track moving."""
    min_travel: float = 1.2
    """Net displacement a track needs before it is treated as a moving agent (m)."""
    cue_radius: float = 2.5
    """Max distance between a visual cue and a track centroid to attach it (m)."""


class SpotDetectionConfig(FrozenModel):
    """Lined and line-less spot detection plus legality classification (C5, C6)."""

    perpendicular_spacing: tuple[float, float] = (2.2, 3.8)
    """Line spacing range that forms a perpendicular stall (m)."""
    parallel_spacing: tuple[float, float] = (5.2, 8.5)
    """Line spacing range that forms a parallel stall (m)."""
    parallel_tolerance: float = 0.17
    """Max |sin| of the angle between two lines treated as parallel (~10 deg)."""
    line_alignment: float = 1.2
    """Max along-line offset between paired line midpoints (m)."""
    occupied_fraction: float = 0.04
    """Fraction of occupied cells inside a stall above which it is taken."""
    stall_shrink: float = 0.3
    """Inset applied to stall polygons before checking occupancy (m)."""
    gap_min_width: float = 2.9
    """Minimum gap between parked vehicles to infer a line-less perpendicular stall (m)."""
    gap_max_width: float = 9.0
    """Gaps wider than this are split: the stall is placed beside the nearer vehicle (m)."""
    gap_stall_width: float = 3.0
    """Width assigned to an inferred stall carved from a wide gap (m)."""
    gap_stall_length: float = 5.2
    """Depth assumed for line-less perpendicular stalls (m)."""
    gap_parallel_min: float = 6.0
    """Minimum gap for a line-less parallel stall (m)."""
    vehicle_extent: tuple[float, float] = (1.2, 7.0)
    """Size range of occupied-grid components treated as parked vehicles (m)."""
    vehicle_logodds: float = 0.5
    """Log-odds above which a cell contributes to parked-vehicle outlines."""
    row_tolerance: float = 1.6
    """Max offset across the row for two vehicles to be in the same row (m)."""
    min_observed_fraction: float = 0.35
    """A stall must be at least this well observed before it is offered (unknown != free)."""
    inspection_range: float = 11.0
    """The ego must come this close for signs and paint to count as inspected (m)."""
    inspection_frames: int = 4
    """Frames within inspection range before a stall's type is considered verified."""
    feature_radius: float = 0.9
    """Features within this distance of a stall polygon are attributed to it (m)."""
    zone_radius: float = 1.8
    """No-parking surface cues within this distance make a stall candidate illegal (m)."""
    track_merge_distance: float = 1.2
    """Line-less candidates closer than this are the same spot across frames (m)."""


class DirectionPreference(StrEnum):
    """How the user likes to end up in a perpendicular stall (C15)."""

    AUTO = "auto"
    NOSE_IN = "nose_in"
    REVERSE_IN = "reverse_in"
    PULL_THROUGH = "pull_through"


class Unlock(StrEnum):
    """User-declared entitlements that make a restricted spot type legal (C5)."""

    PLACARD = "placard"
    """Accessible-parking placard: unlocks accessible spots."""
    CHARGING = "charging"
    """The driver intends to charge: unlocks charger and EV-only stalls."""
    EXPECTANT = "expectant"
    """Unlocks expectant-mother spots."""
    PERMIT = "permit"
    """Holds a permit for reserved spots."""


class RankingWeights(FrozenModel):
    """User-adjustable weights of the destination-aware spot cost (C4). Units: cost per unit."""

    walking: float = 1.0
    """Cost per metre of walking from the spot to the entrance."""
    maneuver: float = 6.0
    """Cost per unit of maneuver difficulty (0 easy .. ~3 hard)."""
    clearance: float = 25.0
    """Cost per metre of missing neighbor clearance below the door-opening target."""
    pull_through: float = 4.0
    """Bonus (subtracted) when a pull-through is available and the user likes it."""
    preference_memory: float = 30.0
    """Bonus (subtracted) for a spot matching a remembered per-location preference (C15)."""
    drive_distance: float = 0.15
    """Cost per metre of driving needed to reach the spot."""


class UserSettings(FrozenModel):
    """What the driver asked for on this trip."""

    direction: DirectionPreference = DirectionPreference.AUTO
    unlocks: frozenset[Unlock] = Field(default_factory=frozenset)
    weights: RankingWeights = RankingWeights()


class ParkPilotConfig(FrozenModel):
    """Root configuration for one autonomy stack instance."""

    vehicle: VehicleConfig = VehicleConfig()
    agents: AgentConfig = AgentConfig()
    sensor: SensorConfig = SensorConfig()
    grid: GridConfig = GridConfig()
    tracker: TrackerConfig = TrackerConfig()
    spots: SpotDetectionConfig = SpotDetectionConfig()
