# ADR 0004: Graceful degradation ladder (C14)

Status: accepted

## Decision
1. Plan failure or a spot that became occupied/restricted -> next-best spot (bounded
   failures per spot, bounded re-plans per spot).
2. No eligible spot -> turn around locally and search the route again (`max_circles`).
3. Still nothing, or the way is blocked longer than `blocked_patience` -> pull over to the
   aisle edge if a short forward path exists, stop, and HANDBACK with a stated reason.
Every rung is a logged transition with a reason that the HMI shows.

## Consequences
The car never silently freezes in the aisle; the benchmark measures abort-with-reason.
