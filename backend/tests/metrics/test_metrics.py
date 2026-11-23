from pathlib import Path

import numpy as np

from parkpilot.metrics.bench import run_benchmark, run_one
from parkpilot.metrics.report import headline_rows, load_results, save_results, write_report
from parkpilot.metrics.run_metrics import gear_changes, spot_is_legal
from parkpilot.scenarios.loader import load_scenario
from parkpilot.scenarios.schema import build_world


def test_gear_change_counting_ignores_jitter() -> None:
    v = np.array([0.0, 0.5, 1.0, 0.02, -0.01, 0.0, -0.4, -0.2, 0.0, 0.3])
    assert gear_changes(v) == 2
    assert gear_changes(np.zeros(5)) == 0


def test_legality_truth() -> None:
    sc = load_scenario("restricted_spots")
    world = build_world(sc)
    legal = {s.id for s in world.spots if spot_is_legal(s, world, sc.user)}
    assert "A3" not in legal and "A8" not in legal  # accessible, motorcycle
    assert "A11" not in legal  # loading zone
    assert {"A10", "A12", "A14", "B9"} <= legal  # neighbours of the zone stay legal


def test_run_one_parkpilot_charger() -> None:
    m = run_one("charger_stall", "parkpilot", 0)
    assert m.parked and m.collisions == 0 and m.violations == 0
    # C11 regression: the port must end within 0.5 m of the charger, not "a foot short".
    assert m.port_error is not None and m.port_error <= 0.5
    assert m.oracle_match is True
    assert m.time_to_park is not None and m.mean_speed > 0.5
    assert m.final_heading_error_deg is not None and m.final_heading_error_deg <= 1.5


def test_home_garage_wall_charger_within_half_a_metre() -> None:
    m = run_one("home_garage", "parkpilot", 1)
    assert m.parked and m.collisions == 0 and m.curb_contacts == 0
    assert m.port_error is not None and m.port_error <= 0.5


def test_memory_protocol_repeat_visit() -> None:
    pp = run_one("repeat_visit", "parkpilot", 0)
    assert pp.repeat_match is True and pp.preference_honored is True
    assert pp.parked_spot == "A3"


def test_report_generation(tmp_path: Path) -> None:
    results = run_benchmark(["home_garage"], seeds=1, workers=1)
    assert {r.system for r in results} == {"parkpilot", "baseline"}
    path = tmp_path / "results.json"
    save_results(results, path)
    again = load_results(path)
    assert again == results
    md = tmp_path / "docs" / "BENCHMARK.md"
    write_report(results, md, tmp_path / "docs" / "img", 1)
    text = md.read_text()
    assert "| Metric | Baseline | ParkPilot |" in text
    assert (tmp_path / "docs" / "img" / "bench_gear_changes.png").is_file()
    rows = dict((label, (b, p)) for label, b, p in headline_rows(results))
    assert rows["Collisions (total)"][1] == "0"


def test_open_lot_picks_the_oracle_spot_at_the_row_end() -> None:
    # Regression: the search used to stop at the first "near-optimal" stall before the
    # roomier, closer stall at the row end (A23) had been seen.
    m = run_one("open_lot_speed", "parkpilot", 0)
    assert m.parked_spot == "A23" and m.oracle_match is True


def test_override_scenario_has_no_oracle() -> None:
    m = run_one("override_spot", "parkpilot", 0)
    assert m.oracle_match is None and m.override_honored is True


def test_parallel_park_ends_straight() -> None:
    # Regression: parallel paths ended on an arc and left ~3.5 deg of heading error.
    m = run_one("parallel_curb", "parkpilot", 0)
    assert m.parked and m.curb_contacts == 0
    assert m.final_heading_error_deg is not None and m.final_heading_error_deg <= 1.5


def test_crooked_stop_is_straightened_at_night() -> None:
    # Regression: this seed stopped 10 deg crooked, wedged inside the margin, and REFINE
    # gave up because every correction path looked blocked at distance zero.
    m = run_one("night_bollards", "parkpilot", 2)
    assert m.parked and m.collisions == 0
    assert m.final_heading_error_deg is not None and m.final_heading_error_deg <= 1.5


def test_emergency_braking_is_rate_limited_and_yields_are_smooth() -> None:
    # Regression: raw peak jerk was ~125 m/s^3 from instant 6 m/s^2 brake steps and
    # instant releases. Emergency stops now ramp at the actuator limit; everything else
    # stays at the comfort limit.
    m = run_one("dynamic_agents", "parkpilot", 0)
    assert m.collisions == 0 and m.missed_hazards == 0
    assert m.peak_jerk <= 41.0
    assert m.peak_jerk_comfort <= 5.0


def test_garage_seed_that_used_to_hand_back_now_parks() -> None:
    m = run_one("garage_pillars", "parkpilot", 1)
    assert m.parked and m.collisions == 0 and not m.handback
