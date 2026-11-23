"""Turn benchmark results into ``docs/BENCHMARK.md`` plus PNG plots."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from parkpilot.metrics.run_metrics import RunMetrics
from parkpilot.scenarios.loader import load_scenario

Getter = Callable[[RunMetrics], float | int | bool | None]


def _mean(vals: Sequence[float]) -> float | None:
    return statistics.fmean(vals) if vals else None


def _fmt(v: float | None, digits: int = 2, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if pct:
        return f"{100 * v:.0f}%"
    return f"{v:.{digits}f}"


def _vals(runs: list[RunMetrics], get: Getter) -> list[float]:
    out = []
    for r in runs:
        v = get(r)
        if v is not None:
            out.append(float(v))
    return out


# (label, getter, aggregation, digits, lower_is_better, only_parked)
HEADLINE: list[tuple[str, Getter, str, int, bool | None, bool]] = [
    ("Park success rate (excl. expected handbacks)", lambda r: r.parked, "rate", 0, False, False),
    ("Collisions (total)", lambda r: r.collisions, "sum", 0, True, False),
    ("Curb contacts (total)", lambda r: r.curb_contacts, "sum", 0, True, False),
    ("Legality violations (total)", lambda r: r.violations, "sum", 0, True, False),
    ("Missed hazards (total)", lambda r: r.missed_hazards, "sum", 0, True, False),
    ("False stops (total)", lambda r: r.false_stops, "sum", 0, True, False),
    ("Gear changes per park", lambda r: r.gear_changes, "mean", 1, True, True),
    ("Re-plans per park", lambda r: r.replans, "mean", 1, True, True),
    ("Time to park (s)", lambda r: r.time_to_park, "mean", 1, True, True),
    ("Mean speed (m/s)", lambda r: r.mean_speed, "mean", 2, False, True),
    ("Peak jerk, raw (m/s^3)", lambda r: r.peak_jerk, "mean", 1, True, True),
    (
        "Peak jerk excl. emergency stops (m/s^3)",
        lambda r: r.peak_jerk_comfort,
        "mean",
        1,
        True,
        True,
    ),
    ("Final lateral error (m)", lambda r: r.final_lateral_error, "mean", 2, True, True),
    ("Final heading error (deg)", lambda r: r.final_heading_error_deg, "mean", 1, True, True),
    ("Min side clearance at rest (m)", lambda r: r.min_side_clearance, "mean", 2, False, True),
    ("Walking distance (m)", lambda r: r.walking_distance, "mean", 1, True, True),
    ("Selection matches oracle", lambda r: r.oracle_match, "rate", 0, False, True),
    ("Abort/handback with reason", lambda r: r.abort_with_reason, "rate", 0, False, False),
]


def _agg(runs: list[RunMetrics], get: Getter, how: str, only_parked: bool) -> float | None:
    pool = [r for r in runs if r.parked] if only_parked else runs
    vals = _vals(pool, get)
    if how == "sum":
        return float(sum(vals))
    if how == "rate":
        return _mean(vals)
    return _mean(vals)


def headline_rows(results: list[RunMetrics]) -> list[tuple[str, str, str]]:
    by_sys: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in results:
        by_sys[r.system].append(r)
    rows = []
    for label, get, how, digits, _, only_parked in HEADLINE:
        cells = []
        for system in ("baseline", "parkpilot"):
            runs = by_sys.get(system, [])
            if label.startswith("Park success"):
                runs = [r for r in runs if not load_scenario(r.scenario).expect_handback]
            v = _agg(runs, get, how, only_parked)
            cells.append(_fmt(v, digits, pct=how == "rate"))
        rows.append((label, cells[0], cells[1]))
    return rows


# Per-complaint focus metrics: (complaint, scenario, label, getter, digits, pct)
FOCUS: list[tuple[str, str, str, Getter, int, bool]] = [
    ("C1", "dance_spot_opens", "gear changes", lambda r: r.gear_changes, 1, False),
    ("C1", "dance_spot_opens", "re-plans", lambda r: r.replans, 1, False),
    ("C2", "open_lot_speed", "mean speed (m/s)", lambda r: r.mean_speed, 2, False),
    ("C2", "open_lot_speed", "time to park (s)", lambda r: r.time_to_park, 1, False),
    ("C3", "crooked_neighbor", "lateral error (m)", lambda r: r.final_lateral_error, 2, False),
    ("C3", "crooked_neighbor", "min side clearance (m)", lambda r: r.min_side_clearance, 2, False),
    ("C4", "destination_entrance", "oracle match", lambda r: r.oracle_match, 0, True),
    ("C4", "destination_entrance", "walking distance (m)", lambda r: r.walking_distance, 1, False),
    ("C5", "restricted_spots", "violations", lambda r: r.violations, 1, False),
    ("C6", "unmarked_lot", "success rate", lambda r: r.parked, 0, True),
    ("C7", "night_bollards", "collisions", lambda r: r.collisions, 1, False),
    ("C7", "night_bollards", "min clearance (m)", lambda r: r.min_clearance, 2, False),
    ("C8", "phantom_shadows", "false stops", lambda r: r.false_stops, 1, False),
    ("C8", "phantom_shadows", "missed hazards", lambda r: r.missed_hazards, 1, False),
    ("C9", "parallel_curb", "success rate", lambda r: r.parked, 0, True),
    ("C9", "parallel_curb", "curb distance error (m)", lambda r: r.curb_distance_error, 2, False),
    ("C10", "garage_pillars", "success rate", lambda r: r.parked, 0, True),
    ("C10", "garage_pillars", "min clearance (m)", lambda r: r.min_clearance, 2, False),
    ("C11", "charger_stall", "port-to-charger (m)", lambda r: r.port_error, 2, False),
    ("C11", "home_garage", "port-to-charger (m)", lambda r: r.port_error, 2, False),
    ("C12", "dynamic_agents", "collisions", lambda r: r.collisions, 1, False),
    ("C12", "dynamic_agents", "yield events", lambda r: r.yield_events, 1, False),
    ("C12", "dynamic_agents", "time lost yielding (s)", lambda r: r.time_lost_yielding, 1, False),
    ("C13", "override_spot", "parked in driver's choice", lambda r: r.override_honored, 0, True),
    ("C14", "full_lot_fallback", "handback with reason", lambda r: r.abort_with_reason, 0, True),
    ("C14", "full_lot_fallback", "violations", lambda r: r.violations, 1, False),
    ("C15", "repeat_visit", "repeat-visit match", lambda r: r.repeat_match, 0, True),
    ("C15", "repeat_visit", "preference honored", lambda r: r.preference_honored, 0, True),
]


def write_report(results: list[RunMetrics], out_md: Path, img_dir: Path, seeds: int) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark",
        "",
        "Generated by `make bench` (`parkpilot bench`). Every scenario runs for both systems",
        f"over {seeds} seeds; numbers are computed from ground truth by",
        "`parkpilot/metrics/run_metrics.py`. Nothing here is hand-edited.",
        "",
        "Means over *parked* runs are marked; totals and rates are over all runs. Expected",
        "handbacks (`full_lot_fallback`) are excluded from the success rate.",
        "",
        "## Headline",
        "",
        "| Metric | Baseline | ParkPilot |",
        "|---|---:|---:|",
    ]
    for label, b, p in headline_rows(results):
        lines.append(f"| {label} | {b} | {p} |")
    lines += [
        "",
        "## Per complaint",
        "",
        "| # | Scenario | Metric | Baseline | ParkPilot |",
        "|---|---|---|---:|---:|",
    ]
    by_key: dict[tuple[str, str], list[RunMetrics]] = defaultdict(list)
    for r in results:
        by_key[(r.scenario, r.system)].append(r)
    for comp, sid, label, get, digits, pct in FOCUS:
        cells = []
        for system in ("baseline", "parkpilot"):
            vals = _vals(by_key.get((sid, system), []), get)
            cells.append(_fmt(_mean(vals), digits, pct))
        lines.append(f"| {comp} | `{sid}` | {label} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Per scenario", ""]
    cols = [
        ("parked", lambda r: r.parked, 0, True),
        ("t park", lambda r: r.time_to_park, 1, False),
        ("gears", lambda r: r.gear_changes, 1, False),
        ("replans", lambda r: r.replans, 1, False),
        ("lat err", lambda r: r.final_lateral_error, 2, False),
        ("hdg err", lambda r: r.final_heading_error_deg, 1, False),
        ("port", lambda r: r.port_error, 2, False),
        ("jerk", lambda r: r.peak_jerk, 0, False),
        ("jerk excl. e-stop", lambda r: r.peak_jerk_comfort, 1, False),
        ("oracle", lambda r: r.oracle_match, 0, True),
        ("min clr", lambda r: r.min_clearance, 2, False),
        ("coll", lambda r: r.collisions, 1, False),
        ("viol", lambda r: r.violations, 1, False),
        ("false stops", lambda r: r.false_stops, 1, False),
        ("missed", lambda r: r.missed_hazards, 1, False),
    ]
    lines.append("| Scenario | System | " + " | ".join(c[0] for c in cols) + " |")
    lines.append("|---|---|" + "---:|" * len(cols))
    for sid in sorted({r.scenario for r in results}):
        for system in ("baseline", "parkpilot"):
            runs = by_key.get((sid, system), [])
            cells = [_fmt(_mean(_vals(runs, g)), d, pct) for _, g, d, pct in cols]
            lines.append(f"| `{sid}` | {system} | " + " | ".join(cells) + " |")
    lines += ["", "## Plots", ""]
    for name in _plots(results, img_dir):
        rel = Path("img") / name
        lines.append(f"![{name}]({rel.as_posix()})")
        lines.append("")
    lines += [
        "## Notes on method",
        "",
        "- *False stop*: a hard brake or a yield with no real agent within 6 m, no ground-truth",
        "  time-to-contact under 3 s and no solid obstacle within 0.3 m.",
        "- *Missed hazard*: an episode where the ego kept moving (|v| > 0.3 m/s) with a",
        "  ground-truth time-to-contact with an agent under 0.5 s (predicted with the ego's",
        "  current deceleration, so braking counts), or touched an agent.",
        "- *Final lateral error*: distance from the ideal lateral position (centred between",
        "  the actual neighbours, clamped to the painted lines) for perpendicular stalls, and",
        "  the curb-gap error for parallel stalls.",
        "- *Oracle*: the cheapest legal free stall under the same cost function, evaluated",
        "  on ground truth, with availability judged when the system first commits. Not",
        "  defined where the driver picks the spot (`override_spot`).",
        "- *Peak jerk excl. emergency stops*: as raw peak jerk, ignoring 0.5 s around ticks",
        "  where the safety supervisor commanded an emergency stop. Emergency braking ramps",
        "  at 40 m/s^3 (brake build-up), comfort driving at 2 m/s^3.",
        "- The baseline re-plans on a 2 s timer rather than every perception frame; every",
        "  frame would make the benchmark impractically slow in pure Python and would only",
        "  make the baseline worse.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _plots(results: list[RunMetrics], img_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    metrics: list[tuple[str, Getter, str]] = [
        ("gear_changes", lambda r: r.gear_changes, "Gear changes per park"),
        ("time_to_park", lambda r: r.time_to_park, "Time to park (s)"),
        ("lateral_error", lambda r: r.final_lateral_error, "Final lateral error (m)"),
        ("false_stops", lambda r: r.false_stops, "False stops per run"),
    ]
    scenarios = sorted({r.scenario for r in results})
    colors = {"baseline": "#b4563f", "parkpilot": "#2f6db3"}
    for key, get, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.38
        for k, system in enumerate(("baseline", "parkpilot")):
            ys = []
            for sid in scenarios:
                runs = [r for r in results if r.scenario == sid and r.system == system]
                if key != "false_stops":
                    runs = [r for r in runs if r.parked]
                v = _mean(_vals(runs, get))
                ys.append(v if v is not None else 0.0)
            xs = [i + (k - 0.5) * width for i in range(len(scenarios))]
            ax.bar(xs, ys, width, label=system, color=colors[system])
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        name = f"bench_{key}.png"
        fig.savefig(img_dir / name, dpi=110)
        plt.close(fig)
        names.append(name)
    return names


def save_results(results: list[RunMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([r.model_dump() for r in results], indent=1) + "\n", encoding="utf-8"
    )


def load_results(path: Path) -> list[RunMetrics]:
    return [RunMetrics.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8"))]
