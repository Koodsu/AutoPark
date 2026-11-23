/** Decode the base64 uint8 occupancy grid (0 = unknown, 1..255 = occupancy probability). */
export function decodeGrid(b64: string, expected: number): Uint8Array {
  const bin =
    typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
  if (bin.length !== expected) {
    throw new Error(`grid has ${bin.length} cells, expected ${expected}`);
  }
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/**
 * RGBA pixels for the grid, flipped so row 0 (the grid origin, smallest y) is at the
 * bottom of the image like the world. Unknown cells are transparent.
 */
export function gridToRgba(data: Uint8Array, nx: number, ny: number): Uint8ClampedArray {
  const px = new Uint8ClampedArray(nx * ny * 4);
  for (let row = 0; row < ny; row++) {
    const dst = (ny - 1 - row) * nx;
    for (let col = 0; col < nx; col++) {
      const v = data[row * nx + col] ?? 0;
      const o = (dst + col) * 4;
      if (v === 0) continue;
      const p = (v - 1) / 254;
      if (p > 0.6) {
        px[o] = 230;
        px[o + 1] = 120;
        px[o + 2] = 60;
        px[o + 3] = 235;
      } else {
        const shade = Math.round(70 + (1 - p) * 40);
        px[o] = shade;
        px[o + 1] = shade + 6;
        px[o + 2] = shade + 14;
        px[o + 3] = 255;
      }
    }
  }
  return px;
}
