"""Typed configuration models.

Every tunable in the stack lives in one of these pydantic models with a documented default,
so there are no magic numbers in algorithm code. ``ParkPilotConfig`` aggregates them; each
subsystem receives only the section it needs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Immutable, strict base for config sections."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ParkPilotConfig(FrozenModel):
    """Root configuration for one autonomy stack instance."""
