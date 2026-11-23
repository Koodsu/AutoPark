"""Unit helpers. Internally everything is SI: metres, seconds, radians."""

from __future__ import annotations

import math

KPH_TO_MPS = 1000.0 / 3600.0


def deg(value: float) -> float:
    """Degrees to radians."""
    return math.radians(value)


def kph(value: float) -> float:
    """Kilometres per hour to metres per second."""
    return value * KPH_TO_MPS


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b, in [-pi, pi)."""
    return wrap_angle(a - b)
