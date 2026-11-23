import { useEffect, useState } from "react";
import type { UserSettings } from "./generated/protocol";
import { ClearanceGauges } from "./hmi/ClearanceGauges";
import { Controls } from "./hmi/Controls";
import { PreferencesPanel } from "./hmi/PreferencesPanel";
import { ScenarioPicker } from "./hmi/ScenarioPicker";
import { SpotPanel } from "./hmi/SpotPanel";
import { StateBanner } from "./hmi/StateBanner";
import { TransitionLog } from "./hmi/TransitionLog";
import { WorldView } from "./hmi/WorldView";
import { client } from "./session";
import "./styles.css";

export function App() {
  const [settings, setSettings] = useState<UserSettings | null>(null);
  useEffect(() => {
    client
      .getPreferences()
      .then(setSettings)
      .catch(() => setSettings(null));
    return () => client.close();
  }, []);
  const update = (s: UserSettings) => {
    setSettings(s);
    client.savePreferences(s).catch(() => undefined);
  };
  return (
    <div className="app">
      <header>
        <h1>ParkPilot</h1>
        <StateBanner />
      </header>
      <main>
        <WorldView />
        <aside>
          <ScenarioPicker settings={settings} />
          <Controls />
          <SpotPanel />
          <ClearanceGauges />
          <TransitionLog />
          <PreferencesPanel settings={settings} onChange={update} />
        </aside>
      </main>
    </div>
  );
}
