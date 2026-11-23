# Wire protocol (v1)

The backend's pydantic models in `backend/parkpilot/api/protocol.py` are the single source
of truth. `make schema` exports `frontend/schema/protocol.schema.json` and regenerates
`frontend/src/generated/protocol.ts`; CI runs `make schema-check` and fails on drift.
Every scene and frame carries `protocol_version`; the HMI refuses mismatched versions.

## REST

| Method | Path | Body / result |
|---|---|---|
| GET | `/api/health` | `{status: "ok"}` |
| GET | `/api/scenarios` | `ScenarioSummary[]` |
| GET | `/api/config` | `ConfigResponse` (protocol version, systems, full default config) |
| POST | `/api/runs` | `StartRunRequest {scenario_id, system, seed, settings?}` -> `RunInfo {run_id, ws_path}` |
| DELETE | `/api/runs/{run_id}` | stops a run |
| GET / PUT | `/api/preferences` | `UserSettings` (direction, unlocks, ranking weights) |
| GET | `/api/preferences/locations` | per-location favorites |
| DELETE | `/api/preferences/locations/{id}` | forget a location |

## WebSocket `/ws/runs/{run_id}`

Server -> client (discriminated by `type`):

- `scene` once on connect: scenario summary, lot bounds, entrance, grid geometry, vehicle
  dimensions.
- `frame` at about 30 Hz: sim time, state and its reason, ego pose and footprint
  (mirrors included), chosen spot and goal, planned path with direction per point,
  ranked spots (polygon, type, legality, eligibility, rank, cost, reasons), tracks,
  clearance readouts, live metrics, the last decisions, safety and hand-back reasons.
  `grid` (base64 uint8, 0 = unknown, 1-255 = occupancy probability) is included at most
  4 Hz and only when it changed. `debug` (ground truth) only while the debug toggle is on.
- `ack` / `error` replies to commands.

Client -> server: `override_spot {spot_id}`, `pause`, `resume`, `abort`,
`set_speed {factor}` (0 < factor <= 16), `set_debug {enabled}`.

Frames contain perceived state only (plus the optional debug layer): the HMI shows what
the car believes, not what is true.
