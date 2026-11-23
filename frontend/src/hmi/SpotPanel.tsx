import { send } from "../session";
import { useApp } from "../store/store";

export function SpotPanel() {
  const spots = useApp((s) => s.frame?.spots);
  const ack = useApp((s) => s.frame?.override_ack ?? null);
  if (!spots) return null;
  const free = spots.filter((s) => !s.occupied).slice(0, 8);
  return (
    <section className="panel">
      <h3>Spots (ranked)</h3>
      {free.length === 0 && <p className="muted">No free spot seen yet.</p>}
      <ol className="spots" data-testid="spot-list">
        {free.map((sp) => (
          <li key={sp.id} className={sp.chosen ? "chosen" : sp.eligible ? "" : "ineligible"}>
            <button
              type="button"
              onClick={() => send({ type: "override_spot", spot_id: sp.id })}
              title="Park here instead"
              data-testid={`spot-${sp.id}`}
            >
              <span className="rank">{sp.eligible ? `#${sp.rank}` : "-"}</span>
              <span className="sid">{sp.id}</span>
              <span className="cost">{sp.eligible ? sp.cost.toFixed(1) : ""}</span>
            </button>
            <div className="reasons">{sp.reasons.join(" · ")}</div>
          </li>
        ))}
      </ol>
      {ack && <p className="ack">{ack}</p>}
    </section>
  );
}
