/* eslint-disable */
/**
 * GENERATED from backend pydantic models by scripts/gen-types.mjs. Do not edit.
 * Regenerate with `make schema`.
 */

export type ClientCommand =
  | OverrideSpotCommand
  | PauseCommand
  | ResumeCommand
  | AbortCommand
  | SetSpeedCommand
  | SetDebugCommand;
export type SpotId = string;
export type Type = "override_spot";
export type Type1 = "pause";
export type Type2 = "resume";
export type Type3 = "abort";
export type Factor = number;
export type Type4 = "set_speed";
export type Enabled = boolean;
export type Type5 = "set_debug";
export type ProtocolVersion = number;
export type Systems = string[];
export type Count = number;
export type X = number;
export type Y = number;
export type Favorites = FavoriteView[];
export type LocationId = string;
export type RunId = string;
export type ScenarioId = string;
export type System = string;
export type WsPath = string;
export type Complaints = string[];
export type Description = string;
export type Id = string;
export type Lighting = number;
export type LocationId1 = string;
export type Title = string;
export type ServerMessage = SceneMessage | FrameMessage | AckMessage | ErrorMessage;
/**
 * @minItems 4
 * @maxItems 4
 */
export type Bounds = [number, number, number, number];
/**
 * @minItems 2
 * @maxItems 2
 */
export type Entrance = [number, number];
export type Nx = number;
export type Ny = number;
export type OriginX = number;
export type OriginY = number;
export type Resolution = number;
export type ProtocolVersion1 = number;
export type RunId1 = string;
export type System1 = string;
export type Type6 = "scene";
export type Length = number;
export type MirrorWidth = number;
export type RearOverhang = number;
export type Wheelbase = number;
export type Width = number;
export type ChosenSpot = string | null;
export type Id1 = string;
export type Kind = string;
export type Polygon = [number, number][];
export type Agents = DebugObject[];
export type Obstacles = DebugObject[];
export type Spots = DebugObject[];
export type Zones = DebugObject[];
export type Done = boolean;
export type Footprint = [number, number][][];
export type Theta = number;
export type X1 = number;
export type Y1 = number;
export type Speed = number;
export type Steer = number;
export type Data = string;
export type Version = number;
export type HandbackReason = string | null;
export type Elapsed = number;
export type GearChanges = number;
export type HardBrakes = number;
export type Replans = number;
export type Speed1 = number;
export type SpotSwitches = number;
export type YieldEvents = number;
export type OverrideAck = string | null;
export type Path = [number, number, number][];
export type Paused = boolean;
export type ProtocolVersion2 = number;
export type Reason = string;
export type SafetyReason = string;
export type Seq = number;
export type SpeedFactor = number;
export type Chosen = boolean;
export type Cost = number;
export type Eligible = boolean;
export type Id2 = string;
export type Layout = string;
export type Legal = boolean;
export type Occupied = boolean;
export type Polygon1 = [number, number][];
export type Rank = number;
export type Reasons = string[];
export type Source = string;
export type SpotType = string;
export type Spots1 = SpotView[];
export type State = string;
export type System2 = string;
export type T = number;
export type Confirmed = boolean;
export type Dynamic = boolean;
export type Extent = number;
export type Id3 = number;
export type Kind1 = string;
export type Vx = number;
export type Vy = number;
export type X2 = number;
export type Y2 = number;
export type Tracks = TrackView[];
export type Dst = string;
export type Reason1 = string;
export type Src = string;
export type T1 = number;
export type Transitions = TransitionView[];
export type Type7 = "frame";
export type Message = string;
export type Type8 = "ack";
export type Message1 = string;
export type Type9 = "error";
export type ScenarioId1 = string;
export type Seed = number;
/**
 * How the user likes to end up in a perpendicular stall (C15).
 */
export type DirectionPreference = "auto" | "nose_in" | "reverse_in" | "pull_through";
/**
 * User-declared entitlements that make a restricted spot type legal (C5).
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "Unlock".
 */
export type Unlock = "placard" | "charging" | "expectant" | "permit";
export type Unlocks = Unlock[];
export type Charger = number;
export type Clearance = number;
export type DriveDistance = number;
export type Maneuver = number;
export type PreferenceMemory = number;
export type PullThrough = number;
export type Walking = number;
export type System3 = "parkpilot" | "baseline";
/**
 * How the user likes to end up in a perpendicular stall (C15).
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "DirectionPreference".
 */
export type DirectionPreference1 = "auto" | "nose_in" | "reverse_in" | "pull_through";

/**
 * ParkPilot wire protocol v1
 */
export interface ParkPilotProtocol {
  client_command: ClientCommand;
  config_response: ConfigResponse;
  location_preference: LocationPreferenceView;
  run_info: RunInfo;
  scenario_summary: ScenarioSummary;
  server_message: ServerMessage;
  start_run_request: StartRunRequest;
  user_settings: UserSettings;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "OverrideSpotCommand".
 */
export interface OverrideSpotCommand {
  spot_id: SpotId;
  type: Type;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "PauseCommand".
 */
export interface PauseCommand {
  type: Type1;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "ResumeCommand".
 */
export interface ResumeCommand {
  type: Type2;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "AbortCommand".
 */
export interface AbortCommand {
  type: Type3;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "SetSpeedCommand".
 */
export interface SetSpeedCommand {
  factor: Factor;
  type: Type4;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "SetDebugCommand".
 */
export interface SetDebugCommand {
  enabled: Enabled;
  type: Type5;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "ConfigResponse".
 */
export interface ConfigResponse {
  config: Config;
  protocol_version: ProtocolVersion;
  systems: Systems;
}
export interface Config {
  [k: string]: unknown;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "LocationPreferenceView".
 */
export interface LocationPreferenceView {
  favorites: Favorites;
  location_id: LocationId;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "FavoriteView".
 */
export interface FavoriteView {
  count: Count;
  x: X;
  y: Y;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "RunInfo".
 */
export interface RunInfo {
  run_id: RunId;
  scenario_id: ScenarioId;
  system: System;
  ws_path: WsPath;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "ScenarioSummary".
 */
export interface ScenarioSummary {
  complaints: Complaints;
  description: Description;
  id: Id;
  lighting: Lighting;
  location_id: LocationId1;
  title: Title;
}
/**
 * Sent once when a client connects: everything static about the run.
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "SceneMessage".
 */
export interface SceneMessage {
  bounds: Bounds;
  entrance: Entrance;
  grid: GridInfo;
  protocol_version: ProtocolVersion1;
  run_id: RunId1;
  scenario: ScenarioSummary;
  system: System1;
  type: Type6;
  vehicle: VehicleInfo;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "GridInfo".
 */
export interface GridInfo {
  nx: Nx;
  ny: Ny;
  origin_x: OriginX;
  origin_y: OriginY;
  resolution: Resolution;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "VehicleInfo".
 */
export interface VehicleInfo {
  length: Length;
  mirror_width: MirrorWidth;
  rear_overhang: RearOverhang;
  wheelbase: Wheelbase;
  width: Width;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "FrameMessage".
 */
export interface FrameMessage {
  chosen_spot: ChosenSpot;
  clearances: Clearances;
  debug: DebugLayer | null;
  done: Done;
  ego: EgoView;
  goal: PoseModel | null;
  grid: GridPatch | null;
  handback_reason: HandbackReason;
  metrics: LiveMetrics;
  override_ack: OverrideAck;
  path: Path;
  paused: Paused;
  protocol_version: ProtocolVersion2;
  reason: Reason;
  safety_reason: SafetyReason;
  seq: Seq;
  speed_factor: SpeedFactor;
  spots: Spots1;
  state: State;
  system: System2;
  t: T;
  tracks: Tracks;
  transitions: Transitions;
  type: Type7;
}
export interface Clearances {
  [k: string]: number;
}
/**
 * Ground truth, only sent when the HMI's debug toggle is on.
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "DebugLayer".
 */
export interface DebugLayer {
  agents: Agents;
  obstacles: Obstacles;
  spots: Spots;
  zones: Zones;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "DebugObject".
 */
export interface DebugObject {
  id: Id1;
  kind: Kind;
  polygon: Polygon;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "EgoView".
 */
export interface EgoView {
  footprint: Footprint;
  pose: PoseModel;
  speed: Speed;
  steer: Steer;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "PoseModel".
 */
export interface PoseModel {
  theta: Theta;
  x: X1;
  y: Y1;
}
/**
 * Occupancy grid as base64 uint8, row-major from the origin (0 unknown, 1-255 = p).
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "GridPatch".
 */
export interface GridPatch {
  data: Data;
  version: Version;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "LiveMetrics".
 */
export interface LiveMetrics {
  elapsed: Elapsed;
  gear_changes: GearChanges;
  hard_brakes: HardBrakes;
  replans: Replans;
  speed: Speed1;
  spot_switches: SpotSwitches;
  yield_events: YieldEvents;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "SpotView".
 */
export interface SpotView {
  chosen: Chosen;
  cost: Cost;
  eligible: Eligible;
  id: Id2;
  layout: Layout;
  legal: Legal;
  occupied: Occupied;
  polygon: Polygon1;
  rank: Rank;
  reasons: Reasons;
  source: Source;
  spot_type: SpotType;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "TrackView".
 */
export interface TrackView {
  confirmed: Confirmed;
  dynamic: Dynamic;
  extent: Extent;
  id: Id3;
  kind: Kind1;
  vx: Vx;
  vy: Vy;
  x: X2;
  y: Y2;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "TransitionView".
 */
export interface TransitionView {
  dst: Dst;
  reason: Reason1;
  src: Src;
  t: T1;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "AckMessage".
 */
export interface AckMessage {
  message: Message;
  type: Type8;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "ErrorMessage".
 */
export interface ErrorMessage {
  message: Message1;
  type: Type9;
}
/**
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "StartRunRequest".
 */
export interface StartRunRequest {
  scenario_id: ScenarioId1;
  seed: Seed;
  settings: UserSettings | null;
  system: System3;
}
/**
 * What the driver asked for on this trip.
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "UserSettings".
 */
export interface UserSettings {
  direction?: DirectionPreference;
  unlocks?: Unlocks;
  weights?: RankingWeights;
}
/**
 * User-adjustable weights of the destination-aware spot cost (C4). Units: cost per unit.
 */
export interface RankingWeights {
  charger?: Charger;
  clearance?: Clearance;
  drive_distance?: DriveDistance;
  maneuver?: Maneuver;
  preference_memory?: PreferenceMemory;
  pull_through?: PullThrough;
  walking?: Walking;
}
/**
 * User-adjustable weights of the destination-aware spot cost (C4). Units: cost per unit.
 *
 * This interface was referenced by `ParkPilotProtocol`'s JSON-Schema
 * via the `definition` "RankingWeights".
 */
export interface RankingWeights1 {
  charger?: Charger;
  clearance?: Clearance;
  drive_distance?: DriveDistance;
  maneuver?: Maneuver;
  preference_memory?: PreferenceMemory;
  pull_through?: PullThrough;
  walking?: Walking;
}
