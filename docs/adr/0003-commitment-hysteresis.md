# ADR 0003: Commitment with hysteresis in the decision layer (C1)

Status: accepted

## Decision
Once a spot is chosen the car only switches if another spot is cheaper by an absolute
margin *and* a relative margin, the commitment is at least `min_commit_time` old, the car
is not yet within `maneuver_lock_distance` of the goal, and a plan to the new spot exists.
The current plan is recomputed only when it becomes invalid (blocked, tracking lost,
goal refined up close), never on a timer.

## Consequences
Fewer re-plans and gear changes; a spot that opens up mid-maneuver is ignored unless it
is clearly better. The driver can always override explicitly from the HMI.
