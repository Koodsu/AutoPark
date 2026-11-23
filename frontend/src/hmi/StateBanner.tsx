import { useApp } from "../store/store";

const TONE: Record<string, string> = {
  PARKED: "ok",
  HANDBACK: "warn",
  YIELD: "warn",
  REPLAN: "info",
  FALLBACK: "warn",
};

export function StateBanner() {
  const state = useApp((s) => s.frame?.state ?? "");
  const reason = useApp((s) => s.frame?.reason ?? "");
  const handback = useApp((s) => s.frame?.handback_reason ?? null);
  const status = useApp((s) => s.status);
  const paused = useApp((s) => s.frame?.paused ?? false);
  if (!state) {
    return (
      <div className="banner" data-testid="state-banner">
        <span className="state">{status === "idle" ? "Pick a scenario" : status}</span>
      </div>
    );
  }
  return (
    <div className={`banner ${TONE[state] ?? ""}`} data-testid="state-banner" data-state={state}>
      <span className="state">{state}</span>
      <span className="reason">{reason}</span>
      {paused && <span className="chip">paused</span>}
      {handback && <span className="chip warn">hand-back: {handback}</span>}
    </div>
  );
}
