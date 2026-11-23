import { useEffect, useState } from "react";
import type { ScenarioSummary, UserSettings } from "../generated/protocol";
import { client } from "../session";
import { appStore } from "../store/store";

interface Props {
  settings: UserSettings | null;
}

export function ScenarioPicker({ settings }: Props) {
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [scenario, setScenario] = useState("");
  const [system, setSystem] = useState<"parkpilot" | "baseline">("parkpilot");
  const [seed, setSeed] = useState(0);
  const [error, setError] = useState("");

  useEffect(() => {
    client
      .listScenarios()
      .then((list) => {
        setScenarios(list);
        const fromUrl = new URLSearchParams(window.location.search).get("scenario");
        setScenario(fromUrl && list.some((s) => s.id === fromUrl) ? fromUrl : (list[0]?.id ?? ""));
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const start = () => {
    setError("");
    appStore.getState().reset();
    const req = { scenario_id: scenario, system, seed, settings: settings ?? null };
    client.startRun(req).catch((e: unknown) => setError(String(e)));
  };

  const current = scenarios.find((s) => s.id === scenario);
  return (
    <section className="panel">
      <h3>Scenario</h3>
      <div className="row">
        <select
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          data-testid="scenario"
        >
          {scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.id} ({s.complaints.join(", ")})
            </option>
          ))}
        </select>
        <div className="toggle" role="radiogroup" aria-label="system">
          {(["parkpilot", "baseline"] as const).map((s) => (
            <button
              key={s}
              type="button"
              className={s === system ? "on" : ""}
              onClick={() => setSystem(s)}
              data-testid={`system-${s}`}
            >
              {s === "parkpilot" ? "ParkPilot" : "Baseline"}
            </button>
          ))}
        </div>
        <label>
          seed
          <input
            type="number"
            min={0}
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{ width: "4em" }}
          />
        </label>
        <button
          type="button"
          className="primary"
          onClick={start}
          disabled={!scenario}
          data-testid="start"
        >
          Start
        </button>
      </div>
      {current && <p className="muted">{current.description}</p>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}
