import { useApp } from "../store/store";

export function TransitionLog() {
  const transitions = useApp((s) => s.frame?.transitions);
  if (!transitions || transitions.length === 0) return null;
  return (
    <section className="panel">
      <h3>Decisions</h3>
      <ul className="log">
        {[...transitions].reverse().map((t, i) => (
          <li key={`${t.t}-${i}`}>
            <span className="t">{t.t.toFixed(1)}s</span> {t.src} → <b>{t.dst}</b>: {t.reason}
          </li>
        ))}
      </ul>
    </section>
  );
}
