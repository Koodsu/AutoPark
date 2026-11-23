import { describe, expect, it } from "vitest";
import {
  decodeServerMessage,
  encodeCommand,
  PROTOCOL_VERSION,
  ProtocolError,
} from "../../src/net/protocol";
import { backoffDelay } from "../../src/net/client";

describe("protocol decoding", () => {
  it("accepts acks and well-formed frames", () => {
    expect(decodeServerMessage('{"type":"ack","message":"paused"}')).toEqual({
      type: "ack",
      message: "paused",
    });
    const frame = { type: "frame", protocol_version: PROTOCOL_VERSION, ego: {}, spots: [] };
    expect(decodeServerMessage(JSON.stringify(frame)).type).toBe("frame");
  });

  it("rejects garbage, unknown types and version mismatches", () => {
    expect(() => decodeServerMessage("not json")).toThrow(ProtocolError);
    expect(() => decodeServerMessage('{"type":"nope"}')).toThrow(/unknown message type/);
    expect(() => decodeServerMessage('{"type":"scene","protocol_version":99}')).toThrow(/version/);
    expect(() => decodeServerMessage('{"type":"frame","protocol_version":1}')).toThrow(/missing/);
    expect(() => decodeServerMessage('{"type":"ack"}')).toThrow(ProtocolError);
    expect(() => decodeServerMessage("[1]")).toThrow(/no type/);
  });

  it("encodes commands and backs off exponentially with a cap", () => {
    expect(encodeCommand({ type: "set_speed", factor: 2 })).toBe('{"type":"set_speed","factor":2}');
    expect([0, 1, 2, 3].map((a) => backoffDelay(a))).toEqual([250, 500, 1000, 2000]);
    expect(backoffDelay(10)).toBe(4000);
  });
});
