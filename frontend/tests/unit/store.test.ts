import { describe, expect, it } from "vitest";
import { createAppStore } from "../../src/store/store";
import type { FrameMessage, SceneMessage } from "../../src/net/protocol";

const scene = {
  type: "scene",
  protocol_version: 1,
  run_id: "r1",
  system: "parkpilot",
  scenario: { id: "s", title: "t", complaints: [], description: "", location_id: "l", lighting: 1 },
  bounds: [0, 0, 10, 10],
  entrance: [5, 9],
  grid: { origin_x: 0, origin_y: 0, resolution: 1, nx: 2, ny: 1 },
  vehicle: { length: 4.75, width: 1.92, mirror_width: 2.12, wheelbase: 2.88, rear_overhang: 0.95 },
} as unknown as SceneMessage;

function frame(seq: number, grid: { version: number; data: string } | null): FrameMessage {
  return { type: "frame", protocol_version: 1, seq, grid } as unknown as FrameMessage;
}

describe("app store", () => {
  it("keeps the last grid until a newer version arrives", () => {
    const store = createAppStore();
    store.getState().applyMessage(scene);
    const g1 = Buffer.from([1, 200]).toString("base64");
    store.getState().applyMessage(frame(1, { version: 1, data: g1 }));
    expect(Array.from(store.getState().grid?.data ?? [])).toEqual([1, 200]);
    store.getState().applyMessage(frame(2, null));
    expect(store.getState().grid?.version).toBe(1);
    expect(store.getState().frame?.seq).toBe(2);
    store.getState().applyMessage({ type: "ack", message: "paused" });
    expect(store.getState().lastAck).toBe("paused");
    store.getState().applyMessage({ type: "error", message: "bad" });
    expect(store.getState().lastAck).toBe("error: bad");
    store.getState().setStatus("open", "x");
    store.getState().setDebug(true);
    expect(store.getState().status).toBe("open");
    expect(store.getState().debug).toBe(true);
    store.getState().reset();
    expect(store.getState().scene).toBeNull();
  });
});
