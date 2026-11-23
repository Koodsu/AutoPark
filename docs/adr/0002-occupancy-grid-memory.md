# ADR 0002: Occupancy grid with temporal memory for object permanence (C7)

Status: accepted

## Context
Camera-only systems lose low objects (bollards, curbs, snow) when they enter the near-field
blind zone, which is exactly when they matter. Objects behind the car are out of view.

## Decision
A log-odds occupancy grid that keeps cell values when they leave the field of view, with
three rules about free-space evidence:
1. Free space is never carved inside the low-object blind zone, except in cells known to
   hold something tall (hit from inside the blind zone, where only tall objects are
   reported) or a moving agent (transient cells), so pedestrians do not leave ghosts.
2. Dropout rays carry no free-space evidence (unknown is not free).
3. Carving stops `free_margin` short of a hit so depth noise does not erode thin surfaces.
Planning, refinement and the safety supervisor all use the memory grid, and margins widen
with the current depth-noise level (night).

## Consequences
- ParkPilot keeps bollards it saw from a distance; the baseline (memoryless grid) does not.
- A remembered obstacle that was removed would persist until seen free again outside the
  blind zone (acceptable for parking; documented).
