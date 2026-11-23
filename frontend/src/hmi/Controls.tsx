import { send } from "../session";
import { appStore, useApp } from "../store/store";

const SPEEDS = [0.5, 1, 2, 4, 8];

export function Controls() {
  const paused = useApp((s) => s.frame?.paused ?? false);
  const factor = useApp((s) => s.frame?.speed_factor ?? 1);
  const debug = useApp((s) => s.debug);
  const ack = useApp((s) => s.lastAck);
  const live = useApp((s) => s.status === "open");
  return (
    <section className="panel controls">
      <div className="row">
        <button
          type="button"
          disabled={!live}
          onClick={() => send({ type: paused ? "resume" : "pause" })}
          data-testid="pause"
        >
          {paused ? "Resume" : "Pause"}
        </button>
        <button
          type="button"
          className="danger"
          disabled={!live}
          onClick={() => send({ type: "abort" })}
          data-testid="abort"
        >
          Abort
        </button>
        <label>
          speed
          <select
            value={factor}
            disabled={!live}
            onChange={(e) => send({ type: "set_speed", factor: Number(e.target.value) })}
            data-testid="speed"
          >
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                x{s}
              </option>
            ))}
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            checked={debug}
            onChange={(e) => {
              appStore.getState().setDebug(e.target.checked);
              send({ type: "set_debug", enabled: e.target.checked });
            }}
            data-testid="debug"
          />
          ground truth
        </label>
      </div>
      {ack && <p className="muted">{ack}</p>}
      <p className="muted hint">Click a spot on the map or in the list to park there instead.</p>
    </section>
  );
}
