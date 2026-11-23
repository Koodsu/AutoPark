import { useEffect, useRef } from "react";
import { WorldRenderer } from "../render/renderer";
import { spotAt } from "../render/hit";
import { send } from "../session";
import { appStore } from "../store/store";

/** Canvas host: owns the renderer, pan/zoom and click-to-override. */
export function WorldView() {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const host = wrap.current;
    if (!el || !host) return;
    const r = new WorldRenderer(el, appStore);
    const fit = () => r.resize(host.clientWidth, host.clientHeight, window.devicePixelRatio || 1);
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(host);
    r.start();
    let drag: { x: number; y: number; moved: boolean } | null = null;
    const down = (e: PointerEvent) => {
      drag = { x: e.offsetX, y: e.offsetY, moved: false };
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.offsetX - drag.x;
      const dy = e.offsetY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) {
        r.camera.pan(dx, dy);
        drag.x = e.offsetX;
        drag.y = e.offsetY;
      }
    };
    const up = (e: PointerEvent) => {
      if (drag && !drag.moved) {
        const w = r.camera.screenToWorld(e.offsetX, e.offsetY);
        const spots = appStore.getState().frame?.spots ?? [];
        const hit = spotAt(spots, w);
        if (hit && !hit.occupied) send({ type: "override_spot", spot_id: hit.id });
      }
      drag = null;
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      r.camera.zoomAt(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.15 : 1 / 1.15);
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("wheel", wheel, { passive: false });
    return () => {
      r.stop();
      ro.disconnect();
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("wheel", wheel);
    };
  }, []);

  return (
    <div className="world" ref={wrap}>
      <canvas ref={canvas} data-testid="world-canvas" />
    </div>
  );
}
