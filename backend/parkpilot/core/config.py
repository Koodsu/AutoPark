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
    min_points: int = 2
    """Clusters with fewer hits are noise, not objects."""
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
    min_travel: float = 0.6
    """Net displacement a track needs before it is treated as a moving agent (m)."""
    free_space_frames: int = 2
    """Consecutive frames a mover's hits must land on confidently free cells."""
    max_innovation: float = 0.25
    """Movers must be predictable: average residual at most this (m)."""
    free_logodds: float = -1.4
    """Log-odds at or below which a cell counts as confidently free (seen free often)."""
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
    long_line: float = 4.0
    """Lines at least this long bound perpendicular stalls; shorter ones parallel stalls (m)."""
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
    painted_area_radius: float = 8.0
    """Line-less candidates within this distance of a painted line are discarded (m)."""
    track_merge_distance: float = 1.2
    """Line-less candidates closer than this are the same spot across frames (m)."""


class RankingConfig(FrozenModel):
    """Non-user tunables of spot ranking (C3, C4)."""

    door_clearance: float = 0.7
    """Body-to-neighbor gap that lets a door open comfortably (m)."""
    min_side_margin: float = 0.2
    """Hard minimum mirror-to-neighbor gap for a stall to count as physically usable (m)."""
    neighbor_search: float = 2.0
    """How far beyond the stall edges to look for neighbors (m)."""
    tight_width: float = 3.4
    """Free width at or above which maneuver difficulty from tightness is zero (m)."""
    parallel_difficulty: float = 1.0
    """Extra difficulty for parallel stalls."""
    reverse_difficulty: float = 0.4
    """Extra difficulty when the chosen entry direction requires reversing in."""
    memory_radius: float = 1.5
    """A remembered favorite matches a stall whose centre is this close (m)."""


class PlannerConfig(FrozenModel):
    """Hybrid A* with Reeds-Shepp analytic expansion (C1, C9, C10)."""

    xy_resolution: float = 0.5
    """Cell size of the closed set (m)."""
    yaw_bins: int = 48
    """Heading discretisation of the closed set (7.5 deg)."""
    step: float = 1.0
    """Arc length of one motion primitive (m)."""
    steer_samples: int = 5
    """Steering angles per direction, spread over [-max_steer, max_steer]."""
    collision_step: float = 0.2
    """Spacing of collision-checked poses along primitives and analytic paths (m)."""
    reverse_multiplier: float = 1.4
    """Cost multiplier for reverse distance."""
    gear_change_cost: float = 9.0
    """Cost of each forward/reverse switch, in metre-equivalents (the C1 knob)."""
    steer_cost: float = 0.3
    """Cost per metre at full steering lock."""
    steer_change_cost: float = 0.4
    """Cost per change of steering between consecutive primitives (full lock to lock = 2x)."""
    clearance_cost: float = 2.0
    """Cost per metre driven per metre of clearance below ``preferred_clearance``."""
    preferred_clearance: float = 0.9
    """Clearance (body disk to obstacle) below which the clearance cost applies (m)."""
    heuristic_weight: float = 2.0
    """Weighted A* inflation (>1 trades optimality for speed)."""
    analytic_every: int = 4
    """Attempt a Reeds-Shepp shot to the goal every N expansions."""
    analytic_range: float = 18.0
    """Only attempt analytic shots within this distance of the goal (m)."""
    analytic_candidates: int = 12
    """Number of Reeds-Shepp paths (cheapest first) tried per analytic shot."""
    max_expansions: int = 6000
    """Search budget; deterministic (no wall-clock limits)."""
    safety_margin: float = 0.15
    """Base clearance added around the footprint (incl. mirrors) for collision checks (m)."""
    final_straight: float = 1.2
    """Length of the straight segment that ends perpendicular parking paths (m)."""


class GoalConfig(FrozenModel):
    """Final-pose refinement (C3, C9, C11)."""

    stay_within_lines: bool = True
    """Never let centering push the mirrors across painted lines (lined stalls)."""
    line_margin: float = 0.05
    """Mirror-to-line margin kept when clamping to painted lines (m)."""
    head_gap: float = 0.35
    """Target gap between the leading bumper and the stall-head obstacle (m)."""
    max_pull_in: float = 0.6
    """How far beyond the stall centre the car may pull in toward the head (m)."""
    curb_gap: float = 0.4
    """Target body-to-curb gap for parallel parking (m)."""
    port_target: float = 0.5
    """Desired charge-port-to-charger distance (m); the cable reach is ~1.2 m."""
    search_step: float = 0.05
    """Step of the local search for a collision-free goal near the ideal one (m)."""
    search_extent: float = 0.4
    """Max lateral / longitudinal deviation explored by that search (m)."""
    goal_slack: float = 0.1
    """Extra margin the final pose must have beyond the planning margin (m)."""
    refine_tolerance: float = 0.12
    """Residual lateral error that triggers a REFINE correction maneuver (m)."""
    refine_heading_tolerance: float = 0.05
    """Residual heading error that triggers a REFINE correction maneuver (rad)."""


class SpeedConfig(FrozenModel):
    """Clearance-driven speed profile with acceleration and jerk limits (C2)."""

    v_open: float = 4.0
    """Forward speed in open space (m/s)."""
    v_reverse: float = 1.6
    """Reverse speed cap (m/s)."""
    v_tight: float = 0.6
    """Speed near obstacles (m/s)."""
    clear_low: float = 0.25
    """Clearance at or below which ``v_tight`` applies (m)."""
    clear_high: float = 1.6
    """Clearance at or above which full speed applies (m)."""
    lat_accel: float = 1.2
    """Lateral acceleration limit used for the curvature speed cap (m/s^2)."""
    accel: float = 1.2
    """Planned acceleration (m/s^2)."""
    decel: float = 1.2
    """Planned deceleration toward cusps and the goal (m/s^2)."""
    jerk: float = 2.0
    """Jerk limit applied by the longitudinal controller (m/s^3)."""
    creep: float = 0.25
    """Minimum commanded speed while not yet at a stop point (m/s)."""
    constant_speed: float | None = None
    """If set, ignore clearance and drive at this speed (the baseline's crawl)."""


class ControlConfig(FrozenModel):
    """Path tracking and longitudinal control."""

    lookahead_base: float = 1.0
    """Pure-pursuit lookahead at standstill (m)."""
    lookahead_gain: float = 0.5
    """Additional lookahead per m/s of speed (s)."""
    stanley_gain: float = 2.5
    """Cross-track gain of the reverse Stanley controller."""
    stanley_softening: float = 0.4
    """Speed softening term of the Stanley controller (m/s)."""
    heading_gain: float = 2.0
    """Heading-error gain of the reverse controller."""
    steer_preview: float = 0.5
    """Curvature feedforward preview time, covering steering actuator lag (s)."""
    speed_gain: float = 1.8
    """Proportional gain from speed error to acceleration (1/s)."""
    stop_tolerance: float = 0.06
    """Distance to the segment end at which the car is considered there (m)."""
    search_window: int = 40
    """Path indices searched ahead of the last nearest index."""
    max_tracking_error: float = 0.5
    """Cross-track error beyond which the behavior layer requests a re-plan (m)."""


class SafetyConfig(FrozenModel):
    """Safety supervisor: TTC, uncertainty-inflated margins, persistence filter (C7, C8, C12)."""

    base_margin: float = 0.10
    """Footprint margin in perfect conditions (m)."""
    sigma_gain: float = 0.4
    """Extra margin per metre of depth-noise sigma (night widens margins)."""
    check_margin_fraction: float = 0.5
    """Fraction of the planning margin the supervisor requires before it intervenes."""
    horizon: float = 4.0
    """Prediction horizon for agents (s)."""
    time_step: float = 0.2
    """Prediction time step (s)."""
    ttc_stop: float = 1.6
    """Time to conflict below which the supervisor commands an immediate stop (s)."""
    ttc_yield: float = 3.5
    """Time to conflict below which the car yields to a moving agent (s)."""
    agent_margin: float = 0.6
    """Extra radius kept around predicted agents (m)."""
    backing_radius: float = 2.0
    """Extra keep-out radius around a car showing reverse lights (intent cue) (m)."""
    stop_buffer: float = 0.3
    """Distance kept before a static blockage when stopping (m)."""
    comfort_decel: float = 1.5
    """Deceleration used to compute speed limits ahead of hazards (m/s^2)."""
    lookahead_buffer: float = 2.5
    """Path distance checked beyond the stopping distance (m)."""
    clear_time: float = 0.8
    """An agent must be clear this long before the car resumes after yielding (s)."""
    persistence_filter: bool = True
    """Only confirmed tracks may trigger stops. The baseline disables this."""


class BehaviorConfig(FrozenModel):
    """Decision state machine: commitment hysteresis and the fallback ladder (C1, C14)."""

    spot_update_period: float = 0.5
    """How often spots are re-detected and re-ranked (s)."""
    switch_abs_margin: float = 8.0
    """A new spot must be cheaper than the committed one by at least this much..."""
    switch_rel_margin: float = 0.2
    """...and by at least this fraction of the committed cost."""
    min_commit_time: float = 2.0
    """No spot switch within this time of committing (s)."""
    maneuver_lock_distance: float = 9.0
    """No spot switch once this close to the goal (m)."""
    entrance_coverage: float = 10.0
    """Search continues until the car has passed the route point nearest the entrance by
    this much, unless a near-optimal spot is already known (m)."""
    near_optimal_walk: float = 6.0
    """A spot within this walking distance of the best possible on the route ends the
    search early (m)."""
    max_spot_failures: int = 1
    """Plan failures tolerated per spot before it is skipped (fallback ladder step 1)."""
    max_circles: int = 1
    """Times the car circles the search route before handing back (ladder step 2)."""
    max_replans_per_spot: int = 4
    """Re-plans tolerated toward one spot before trying the next one."""
    refine_attempts: int = 1
    """Correction maneuvers allowed in REFINE (C3)."""
    stop_speed: float = 0.05
    """Speed below which the car counts as stopped (m/s)."""
    maneuver_distance: float = 7.0
    """Distance to the goal at which APPROACH becomes MANEUVER (m)."""
    replan_tracking_error: float = 0.6
    """Cross-track error that invalidates the current plan (m)."""
    yield_timeout: float = 25.0
    """Yielding longer than this triggers a re-plan around the agent (s)."""
    blocked_patience: float = 5.0
    """How long to wait for a blocked aisle to clear before moving down the ladder (s)."""
    pullover_offset: float = 1.2
    """Lateral offset toward the aisle edge for the safe handback stop (m)."""


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
    charger: float = 40.0
    """Bonus (subtracted) for a charger stall when the user intends to charge."""


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
    ranking: RankingConfig = RankingConfig()
    planner: PlannerConfig = PlannerConfig()
    goal: GoalConfig = GoalConfig()
    speed: SpeedConfig = SpeedConfig()
    control: ControlConfig = ControlConfig()
    safety: SafetyConfig = SafetyConfig()
    behavior: BehaviorConfig = BehaviorConfig()
