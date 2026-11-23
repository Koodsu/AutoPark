import type { Vec2 } from "./camera";

/** Ray-casting point-in-polygon test (works for any simple polygon). */
export function pointInPolygon(p: Vec2, poly: readonly (readonly [number, number])[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (!a || !b) continue;
    const [xi, yi] = a;
    const [xj, yj] = b;
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

export interface HasPolygon {
  id: string;
  polygon: (readonly [number, number])[];
}

/** The first spot whose polygon contains the world point, if any. */
export function spotAt<T extends HasPolygon>(spots: readonly T[], p: Vec2): T | undefined {
  return spots.find((s) => pointInPolygon(p, s.polygon));
}
