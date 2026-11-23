# ADR 0001: Hybrid A* with Reeds-Shepp expansion instead of RRT*

Status: accepted

## Context
Parking needs kinematically feasible, forward-and-reverse paths in tight spaces, with
explicit control over the number of gear changes (C1) and repeatable results for the
benchmark.

## Decision
Hybrid A* on a discretised (x, y, heading) closed set with exact arc primitives, a cost
that penalises reverse distance, gear changes, steering effort and low clearance, and a
Reeds-Shepp analytic expansion that tries all RS words to the exact goal in order of
planner cost. The search budget is an expansion count, never wall-clock time.

## Consequences
- Deterministic for a given grid; the benchmark is reproducible.
- Gear changes are a first-class cost, which is what fixes the "parking dance".
- RS expansion solves parallel parking and exact goal poses in few expansions.
- Python makes each expansion ~1-2 ms; nose-in into a 6.4 m garage aisle can exhaust the
  budget, so the behavior layer falls back to the other entry direction or the next spot.
- RRT* would give anytime asymptotic optimality but noisy, non-repeatable paths and no
  natural handle on gear changes.
