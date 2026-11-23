export type Vec2 = readonly [number, number];

/**
 * 2D camera mapping world metres (y up) to canvas pixels (y down), with pan and zoom
 * about a screen point. Pure math, no DOM access.
 */
export class Camera {
  /** Pixels per metre. */
  scale = 10;
  /** World point at the canvas centre. */
  cx = 0;
  cy = 0;
  width = 800;
  height = 600;
  minScale = 2;
  maxScale = 200;

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
  }

  worldToScreen(x: number, y: number): Vec2 {
    return [
      this.width / 2 + (x - this.cx) * this.scale,
      this.height / 2 - (y - this.cy) * this.scale,
    ];
  }

  screenToWorld(sx: number, sy: number): Vec2 {
    return [
      this.cx + (sx - this.width / 2) / this.scale,
      this.cy - (sy - this.height / 2) / this.scale,
    ];
  }

  /** Pan by a screen-space drag delta. */
  pan(dxPx: number, dyPx: number): void {
    this.cx -= dxPx / this.scale;
    this.cy += dyPx / this.scale;
  }

  /** Zoom by ``factor`` keeping the world point under (sx, sy) fixed on screen. */
  zoomAt(sx: number, sy: number, factor: number): void {
    const [wx, wy] = this.screenToWorld(sx, sy);
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, this.scale * factor));
    const [nx, ny] = this.screenToWorld(sx, sy);
    this.cx += wx - nx;
    this.cy += wy - ny;
  }

  /** Fit a world rectangle (xmin, ymin, xmax, ymax) into the view with a pixel margin. */
  fit(bounds: readonly [number, number, number, number], marginPx = 24): void {
    const [xmin, ymin, xmax, ymax] = bounds;
    const w = Math.max(xmax - xmin, 1e-6);
    const h = Math.max(ymax - ymin, 1e-6);
    const sx = (this.width - 2 * marginPx) / w;
    const sy = (this.height - 2 * marginPx) / h;
    this.scale = Math.min(this.maxScale, Math.max(this.minScale, Math.min(sx, sy)));
    this.cx = (xmin + xmax) / 2;
    this.cy = (ymin + ymax) / 2;
  }
}
