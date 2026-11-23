import type {
  AckMessage,
  ClientCommand,
  ErrorMessage,
  FrameMessage,
  SceneMessage,
  ServerMessage,
} from "../generated/protocol";

/** Protocol version this HMI was built against (must match the backend). */
export const PROTOCOL_VERSION = 1;

export type { ClientCommand, FrameMessage, SceneMessage, ServerMessage };

export class ProtocolError extends Error {}

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/**
 * Decode one WebSocket text message. Throws ProtocolError on malformed input or a protocol
 * version mismatch, so the UI can say so instead of rendering garbage.
 */
export function decodeServerMessage(raw: string): ServerMessage {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new ProtocolError("message is not JSON");
  }
  if (!isObject(data) || typeof data.type !== "string") {
    throw new ProtocolError("message has no type");
  }
  switch (data.type) {
    case "scene":
    case "frame": {
      if (data.protocol_version !== PROTOCOL_VERSION) {
        throw new ProtocolError(
          `protocol version ${String(data.protocol_version)} != ${PROTOCOL_VERSION}`,
        );
      }
      if (data.type === "frame" && (!isObject(data.ego) || !Array.isArray(data.spots))) {
        throw new ProtocolError("frame is missing ego or spots");
      }
      return data as unknown as SceneMessage | FrameMessage;
    }
    case "ack":
    case "error":
      if (typeof data.message !== "string") throw new ProtocolError("ack without message");
      return data as unknown as AckMessage | ErrorMessage;
    default:
      throw new ProtocolError(`unknown message type ${data.type}`);
  }
}

export function encodeCommand(cmd: ClientCommand): string {
  return JSON.stringify(cmd);
}
