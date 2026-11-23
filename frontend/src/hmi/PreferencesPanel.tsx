import type { UserSettings } from "../generated/protocol";

const DIRECTIONS = ["auto", "nose_in", "reverse_in", "pull_through"] as const;
const UNLOCKS = ["placard", "charging", "expectant", "permit"] as const;

interface Props {
  settings: UserSettings | null;
  onChange: (s: UserSettings) => void;
}

export function PreferencesPanel({ settings, onChange }: Props) {
  if (!settings) return null;
  const unlocks = new Set(settings.unlocks ?? []);
  const toggle = (u: (typeof UNLOCKS)[number]) => {
    const next = new Set(unlocks);
    if (next.has(u)) next.delete(u);
    else next.add(u);
    onChange({ ...settings, unlocks: [...next] as UserSettings["unlocks"] });
  };
  const weights = settings.weights;
  return (
    <section className="panel">
      <h3>Preferences (apply on next start)</h3>
      <div className="row">
        <label>
          entry
          <select
            value={settings.direction ?? "auto"}
            onChange={(e) =>
              onChange({ ...settings, direction: e.target.value as UserSettings["direction"] })
            }
          >
            {DIRECTIONS.map((d) => (
              <option key={d} value={d}>
                {d.replace("_", "-")}
              </option>
            ))}
          </select>
        </label>
        {UNLOCKS.map((u) => (
          <label key={u}>
            <input type="checkbox" checked={unlocks.has(u)} onChange={() => toggle(u)} />
            {u}
          </label>
        ))}
      </div>
      {weights && (
        <div className="row weights">
          {(["walking", "maneuver", "clearance"] as const).map((k) => (
            <label key={k}>
              {k}
              <input
                type="range"
                min={0}
                max={k === "clearance" ? 60 : 12}
                step={0.5}
                value={weights[k] ?? 0}
                onChange={(e) =>
                  onChange({ ...settings, weights: { ...weights, [k]: Number(e.target.value) } })
                }
              />
            </label>
          ))}
        </div>
      )}
    </section>
  );
}
