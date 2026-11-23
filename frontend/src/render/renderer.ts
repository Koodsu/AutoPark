import type { FrameMessage, SceneMessage } from "../net/protocol";
import type { AppState, AppStore } from "../store/store";
import { Camera } from "./camera";
import { gridToRgba } from "./grid";

type Poly = readonly (readonly [number, number])[];

export const COLORS = {
  background: "#1b1f24",
  lot: "#262b31",
  spotEligible: "#3fb27f",
  spotIneligible: "#8a6d3b",
  spotIllegal: "#c0504d",
  spotOccupied: "#555d66",
  spotChosen: "#4aa3ff",
  pathForward: "#4aa3ff",
  pathReverse: "#ff9f43",
  ego: "#f5f7fa",
  mirror: "#c9d1d9",
  track: "#ffd166",
  trackDynamic: "#ff4d6d",
  entrance: "#b388ff",
  debug: "rgba(255, 64, 129, 0.8)",
};

/**
 * Draws the world on a canvas from the store, every animation frame. React never
 * renders world geometry: this class owns the canvas and the camera.
 */
export class WorldRenderer {
  readonly camera = new Camera();
  private raf = 0;
  private gridCanvas: HTMLCanvasElement | null = null;
  private gridVersion = -1;
  private fittedRun = "";

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly store: AppStore,
  ) {}

  start(): void {
    const loop = () => {
      this.draw();
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  stop(): void {
    cancelAnimationFrame(this.raf);
  }

  resize(width: number, height: number, dpr: number): void {
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;
    this.camera.resize(width, height);
    const ctx = this.canvas.getContext("2d");
    ctx?.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  private ensureFit(scene: SceneMessage): void {
    if (this.fittedRun === scene.run_id) return;
    this.fittedRun = scene.run_id;
    this.camera.fit(scene.bounds as [number, number, number, number]);
  }

  draw(): void {
    const ctx = this.canvas.getContext("2d");
    if (!ctx) return;
    const s = this.store.getState();
    const cam = this.camera;
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, cam.width, cam.height);
    if (!s.scene) return;
    this.ensureFit(s.scene);
    this.drawLot(ctx, s.scene);
    this.drawGrid(ctx, s);
    const f = s.frame;
    if (f) {
      this.drawSpots(ctx, f);
      if (f.debug) this.drawDebug(ctx, f);
      this.drawPath(ctx, f);
      this.drawTracks(ctx, f);
      this.drawEgo(ctx, f);
    }
    this.drawEntrance(ctx, s.scene);
  }

  private poly(ctx: CanvasRenderingContext2D, pts: Poly): void {
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const [sx, sy] = this.camera.worldToScreen(x, y);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.closePath();
  }

  private drawLot(ctx: CanvasRenderingContext2D, scene: SceneMessage): void {
    const [xmin, ymin, xmax, ymax] = scene.bounds as [number, number, number, number];
    this.poly(ctx, [
      [xmin, ymin],
      [xmax, ymin],
      [xmax, ymax],
      [xmin, ymax],
    ]);
    ctx.fillStyle = COLORS.lot;
    ctx.fill();
  }

  private drawGrid(ctx: CanvasRenderingContext2D, s: AppState): void {
    if (!s.grid || !s.scene) return;
    const g = s.scene.grid;
    if (s.grid.version !== this.gridVersion || !this.gridCanvas) {
      const c = this.gridCanvas ?? document.createElement("canvas");
      c.width = g.nx;
      c.height = g.ny;
      const gctx = c.getContext("2d");
      if (!gctx) return;
      const img = gctx.createImageData(g.nx, g.ny);
      img.data.set(gridToRgba(s.grid.data, g.nx, g.ny));
      gctx.putImageData(img, 0, 0);
      this.gridCanvas = c;
      this.gridVersion = s.grid.version;
    }
    const [x0, y1] = this.camera.worldToScreen(g.origin_x, g.origin_y + g.ny * g.resolution);
    const w = g.nx * g.resolution * this.camera.scale;
    const h = g.ny * g.resolution * this.camera.scale;
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = 0.9;
    ctx.drawImage(this.gridCanvas, x0, y1, w, h);
    ctx.globalAlpha = 1;
  }

  private drawSpots(ctx: CanvasRenderingContext2D, f: FrameMessage): void {
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const sp of f.spots) {
      let color = COLORS.spotIneligible;
      if (sp.occupied) color = COLORS.spotOccupied;
      else if (!sp.legal) color = COLORS.spotIllegal;
      else if (sp.eligible) color = COLORS.spotEligible;
      if (sp.chosen) color = COLORS.spotChosen;
      this.poly(ctx, sp.polygon);
      ctx.fillStyle = color + (sp.chosen ? "55" : "22");
      ctx.fill();
      ctx.lineWidth = sp.chosen ? 3 : 1.5;
      ctx.setLineDash(sp.source === "gap" ? [5, 4] : []);
      ctx.strokeStyle = color;
      ctx.stroke();
      ctx.setLineDash([]);
      if (!sp.occupied) {
        const cx = sp.polygon.reduce((a, p) => a + p[0], 0) / sp.polygon.length;
        const cy = sp.polygon.reduce((a, p) => a + p[1], 0) / sp.polygon.length;
        const [sx, sy] = this.camera.worldToScreen(cx, cy);
        ctx.fillStyle = "#e6edf3";
        ctx.fillText(sp.eligible ? `#${sp.rank}` : sp.legal ? "?" : "X", sx, sy);
      }
    }
  }

  private drawPath(ctx: CanvasRenderingContext2D, f: FrameMessage): void {
    if (f.path.length < 2) return;
    ctx.lineWidth = 2.5;
    for (let i = 1; i < f.path.length; i++) {
      const a = f.path[i - 1];
      const b = f.path[i];
      if (!a || !b) continue;
      const [ax, ay] = this.camera.worldToScreen(a[0], a[1]);
      const [bx, by] = this.camera.worldToScreen(b[0], b[1]);
      ctx.strokeStyle = b[2] < 0 ? COLORS.pathReverse : COLORS.pathForward;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    if (f.goal) {
      const [gx, gy] = this.camera.worldToScreen(f.goal.x, f.goal.y);
      ctx.strokeStyle = COLORS.spotChosen;
      ctx.beginPath();
      ctx.arc(gx, gy, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  private drawTracks(ctx: CanvasRenderingContext2D, f: FrameMessage): void {
    for (const t of f.tracks) {
      // Confirmed movers stand out; static clutter (parked-car outlines) stays in the grid.
      if (!t.confirmed || (!t.dynamic && t.extent > 1.0)) continue;
      const [sx, sy] = this.camera.worldToScreen(t.x, t.y);
      const r = Math.max(3, (t.extent / 2) * this.camera.scale);
      ctx.strokeStyle = t.dynamic ? COLORS.trackDynamic : COLORS.track;
      ctx.lineWidth = t.dynamic ? 2.5 : 1;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.stroke();
      if (t.dynamic) {
        const [ex, ey] = this.camera.worldToScreen(t.x + t.vx, t.y + t.vy);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
      }
    }
  }

  private drawEgo(ctx: CanvasRenderingContext2D, f: FrameMessage): void {
    f.ego.footprint.forEach((poly, i) => {
      this.poly(ctx, poly);
      ctx.fillStyle = i === 0 ? COLORS.ego : COLORS.mirror;
      ctx.fill();
    });
    const p = f.ego.pose;
    const [sx, sy] = this.camera.worldToScreen(p.x, p.y);
    const [hx, hy] = this.camera.worldToScreen(
      p.x + 2.5 * Math.cos(p.theta),
      p.y + 2.5 * Math.sin(p.theta),
    );
    ctx.strokeStyle = "#0d1117";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(hx, hy);
    ctx.stroke();
  }

  private drawEntrance(ctx: CanvasRenderingContext2D, scene: SceneMessage): void {
    const [x, y] = scene.entrance as [number, number];
    const [sx, sy] = this.camera.worldToScreen(x, y);
    ctx.fillStyle = COLORS.entrance;
    ctx.beginPath();
    ctx.moveTo(sx, sy - 9);
    ctx.lineTo(sx + 8, sy + 6);
    ctx.lineTo(sx - 8, sy + 6);
    ctx.closePath();
    ctx.fill();
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText("entrance", sx, sy + 16);
  }

  private drawDebug(ctx: CanvasRenderingContext2D, f: FrameMessage): void {
    const d = f.debug;
    if (!d) return;
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.debug;
    for (const o of [...d.obstacles, ...d.agents, ...d.zones]) {
      this.poly(ctx, o.polygon);
      ctx.stroke();
    }
  }
}
