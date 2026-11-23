import { describe, expect, it } from "vitest";
import { Camera } from "../../src/render/camera";
import { pointInPolygon, spotAt } from "../../src/render/hit";
import { decodeGrid, gridToRgba } from "../../src/render/grid";

describe("camera", () => {
  it("round-trips world and screen coordinates with y flipped", () => {
    const c = new Camera();
    c.resize(800, 600);
    c.cx = 10;
    c.cy = 5;
    c.scale = 20;
    expect(c.worldToScreen(10, 5)).toEqual([400, 300]);
    const [sx, sy] = c.worldToScreen(11, 6);
    expect(sx).toBe(420);
    expect(sy).toBe(280);
    const [wx, wy] = c.screenToWorld(sx, sy);
    expect(wx).toBeCloseTo(11);
    expect(wy).toBeCloseTo(6);
  });

  it("zooms about the cursor and pans by screen deltas", () => {
    const c = new Camera();
    c.resize(800, 600);
    const before = c.screenToWorld(100, 100);
    c.zoomAt(100, 100, 2);
    const after = c.screenToWorld(100, 100);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
    expect(c.scale).toBe(20);
    c.zoomAt(0, 0, 1e6);
    expect(c.scale).toBe(c.maxScale);
    const cx = c.cx;
    c.pan(c.scale, 0);
    expect(c.cx).toBeCloseTo(cx - 1);
  });

  it("fits bounds into the view", () => {
    const c = new Camera();
    c.resize(1000, 500);
    c.fit([0, 0, 60, 30], 0);
    expect(c.scale).toBeCloseTo(500 / 30);
    expect([c.cx, c.cy]).toEqual([30, 15]);
  });
});

describe("hit testing and grid decoding", () => {
  const square: [number, number][] = [
    [0, 0],
    [2, 0],
    [2, 2],
    [0, 2],
  ];
  it("finds the spot under a point", () => {
    expect(pointInPolygon([1, 1], square)).toBe(true);
    expect(pointInPolygon([3, 1], square)).toBe(false);
    expect(spotAt([{ id: "a", polygon: square }], [1, 1])?.id).toBe("a");
    expect(spotAt([{ id: "a", polygon: square }], [5, 5])).toBeUndefined();
  });

  it("decodes the grid and flips rows to world orientation", () => {
    const raw = Buffer.from(Uint8Array.from([0, 255, 1, 128])).toString("base64");
    const data = decodeGrid(raw, 4);
    expect(Array.from(data)).toEqual([0, 255, 1, 128]);
    expect(() => decodeGrid(raw, 5)).toThrow(/expected 5/);
    const px = gridToRgba(data, 2, 2);
    // Row 0 (bottom of the world) lands on the last image row.
    expect(px[(2 + 0) * 4 + 3]).toBe(0); // unknown -> transparent
    expect(px[(2 + 1) * 4]).toBe(230); // occupied -> orange
    expect(px[0 * 4 + 3]).toBe(255); // free -> opaque grey
  });
});
