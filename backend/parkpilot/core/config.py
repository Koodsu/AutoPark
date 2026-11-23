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
