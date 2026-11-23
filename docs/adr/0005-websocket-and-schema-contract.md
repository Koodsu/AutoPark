# ADR 0005: WebSocket streaming with a pydantic-to-TypeScript schema contract

Status: accepted

## Decision
Runs are created over REST and streamed over one WebSocket per run (frames at ~30 Hz,
grid at <= 4 Hz when changed, commands upstream). The pydantic models are the only
definition of the protocol: JSON Schema is exported in serialization mode and compiled to
TypeScript; CI fails if the generated files drift. Messages carry `protocol_version`.

## Consequences
One source of truth, typed end to end, cheap to evolve. Frames are JSON (simple to debug);
the grid is the only bulky payload and is rate-limited and base64-packed.
