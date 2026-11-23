import { useApp } from "../store/store";

const SIDES = ["front", "left", "right", "rear"] as const;

export function gaugeLevel(m: number): "ok" | "warn" | "danger" {
  if (m < 0.3) return "danger";
  if (m < 0.8) return "warn";
  return "ok";
}

export function ClearanceGauges() {
  const c = useApp((s) => s.frame?.clearances);
  const metrics = useApp((s) => s.frame?.metrics);
  const safety = useApp((s) => s.frame?.safety_reason ?? "");
  if (!c || !metrics) return null;
  return (
    <section className="panel">
      <h3>Clearance & status</h3>
      <div className="gauges">
        {SIDES.map((side) => {
          const v = c[side] ?? 5;
          return (
            <div key={side} className={`gauge ${gaugeLevel(v)}`} data-testid={`gauge-${side}`}>
              <span>{side}</span>
              <div className="bar">
                <div style={{ width: `${Math.min(100, (v / 5) * 100)}%` }} />
              </div>
              <span>{v >= 5 ? ">5" : v.toFixed(2)} m</span>
            </div>
          );
        })}
      </div>
      <dl className="metrics">
        <dt>time</dt>
        <dd>{metrics.elapsed.toFixed(1)} s</dd>
        <dt>speed</dt>
        <dd>{metrics.speed.toFixed(1)} m/s</dd>
        <dt>gear changes</dt>
        <dd>{metrics.gear_changes}</dd>
        <dt>re-plans</dt>
        <dd>{metrics.replans}</dd>
        <dt>yields</dt>
        <dd>{metrics.yield_events}</dd>
        <dt>hard brakes</dt>
        <dd>{metrics.hard_brakes}</dd>
      </dl>
      {safety && <p className="muted">safety: {safety}</p>}
    </section>
  );
}
