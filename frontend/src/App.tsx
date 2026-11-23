import { useEffect } from "react";
import { ClearanceGauges } from "./hmi/ClearanceGauges";
import { SpotPanel } from "./hmi/SpotPanel";
import { StateBanner } from "./hmi/StateBanner";
import { TransitionLog } from "./hmi/TransitionLog";
import { WorldView } from "./hmi/WorldView";
import { client } from "./session";
import "./styles.css";

export function App() {
  useEffect(() => {
    const scenario = new URLSearchParams(window.location.search).get("scenario");
    if (scenario)
      client.startRun({ scenario_id: scenario, system: "parkpilot", seed: 0, settings: null });
    return () => client.close();
  }, []);
  return (
    <div className="app">
      <header>
        <h1>ParkPilot</h1>
        <StateBanner />
      </header>
      <main>
        <WorldView />
        <aside>
          <SpotPanel />
          <ClearanceGauges />
          <TransitionLog />
        </aside>
      </main>
    </div>
  );
}
